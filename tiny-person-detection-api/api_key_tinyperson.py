import os
import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from ultralytics import YOLO

app = FastAPI(title="Tiny Person Detection API")

import json
import uuid

# ===============================
# API KEY MANAGEMENT
# ===============================
KEYS_FILE = "api_keys.json"

def load_keys():
    if not os.path.exists(KEYS_FILE):
        # Initial default key
        default_key = os.environ.get("DEFAULT_API_KEY") or f"SPI_{uuid.uuid4().hex[:12].upper()}"
        print(f"Created {KEYS_FILE} with initial API key: {default_key}")
        initial_keys = {default_key: {"email": "admin@example.com", "company": "SPILAB"}}
        with open(KEYS_FILE, "w") as f:
            json.dump(initial_keys, f, indent=4)
        return initial_keys
    try:
        with open(KEYS_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_keys(keys):
    with open(KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=4)

def verify_api_key(authorization: str = Header(None)):
    """
    Authorization must be:
    Authorization: Bearer <API_KEY>
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.replace("Bearer ", "").strip()
    keys = load_keys()
    if token not in keys:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    return token


# ===============================
# Load YOLO Model
# ===============================
MODEL_PATH = "tiny_person_best.pt"
if not os.path.exists(MODEL_PATH):
    print(f"{MODEL_PATH} not found, trying tiny_person_best.pt")
    MODEL_PATH = "tiny_person_best.pt"

if not os.path.exists(MODEL_PATH):
    print("❌ Warning: No model found.")
    MODEL_PATH = None

# Custom class mapping
CLASS_MAPPING = {
    0: "person",
    1: "person"
}

print(f"Loading model from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH) if MODEL_PATH else None
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


# ===============================
# Helper: Process Frame
# ===============================
def process_frame(frame, conf_threshold=0.4):
    """
    Common logic for detection and annotation
    """
    if model is None:
        return frame, {}, []

    results = model.predict(source=frame, conf=conf_threshold, verbose=False)
    detections = []
    label_counts = {}
    annotated_img = frame.copy()

    h, w = frame.shape[:2]
    thickness = max(2, int(w / 500))
    font_scale = w / 1500

    res = results[0]
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = CLASS_MAPPING.get(cls_id, res.names.get(cls_id, str(cls_id)))
        
        detections.append({
            "label": label,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })
        label_counts[label] = label_counts.get(label, 0) + 1

        # Draw
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(annotated_img, f"{label} {conf:.2f}", (x1, max(int(40 * font_scale), y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    # Summary box
    y_offset = int(50 * font_scale)
    for label, count in label_counts.items():
        cv2.putText(annotated_img, f"{label}: {count}", (int(20 * font_scale), y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (0, 0, 255), thickness)
        y_offset += int(60 * font_scale)
    
    return annotated_img, label_counts, detections


# ===============================
# Detection Endpoint
# ===============================
@app.post("/detection")
async def detection(
    image: UploadFile = File(None),
    rtsp_url: str = Form(None),
    return_image: bool = Form(False),  # ✅ default False for API calls
    api_key: str = Depends(verify_api_key)
):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    frame = None
    if image:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: raise HTTPException(status_code=400, detail="Invalid image")
    elif rtsp_url:
        cap = cv2.VideoCapture(rtsp_url)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None: raise HTTPException(status_code=400, detail="RTSP Error")
    else:
        raise HTTPException(status_code=400, detail="No source")

    annotated_img, label_counts, detections = process_frame(frame)

    response_data = {"label_counts": label_counts, "detections": detections}
    if return_image:
        _, buffer = cv2.imencode(".jpg", annotated_img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        response_data["image"] = f"data:image/jpeg;base64,{img_base64}"

    return JSONResponse(content=response_data)


# ===============================
# Live Streaming Logic
# ===============================
def gen_frames(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Process every frame
        annotated_frame, _, _ = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret: continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.get("/video_feed")
async def video_feed(rtsp_url: str, token: str = Query(...)):
    keys = load_keys()
    if token not in keys:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    
    return StreamingResponse(
        gen_frames(rtsp_url),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ===============================
# API Key Generation Endpoints
# ===============================
@app.get("/generateapi", response_class=HTMLResponse)
async def generate_api_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Generate API Key | Tiny Object Detection</title>
        <style>
            :root {
                --primary: #00d2ff;
                --secondary: #3a7bd5;
                --bg: #0f172a;
                --card-bg: #1e293b;
                --text: #f8fafc;
            }
            body {
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                background: radial-gradient(circle at top left, #1e293b, #0f172a);
                color: var(--text);
                margin: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }
            .container {
                max-width: 450px;
                width: 90%;
                background: var(--card-bg);
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
                border: 1px solid rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                animation: fadeIn 0.6s ease-out;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            h1 {
                margin-top: 0;
                font-size: 28px;
                background: linear-gradient(to right, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
            }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; color: #94a3b8; font-size: 14px; }
            input {
                width: 100%;
                padding: 12px 16px;
                border-radius: 10px;
                border: 1px solid #334155;
                background: #0f172a;
                color: white;
                box-sizing: border-box;
                font-size: 16px;
                transition: border-color 0.3s, box-shadow 0.3s;
            }
            input:focus {
                outline: none;
                border-color: var(--primary);
                box-shadow: 0 0 0 3px rgba(0, 210, 255, 0.2);
            }
            .btn {
                width: 100%;
                background: linear-gradient(to right, var(--primary), var(--secondary));
                color: white;
                border: none;
                padding: 14px;
                border-radius: 10px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                margin-top: 10px;
                transition: transform 0.2s, opacity 0.2s;
            }
            .btn:hover { transform: translateY(-1px); opacity: 0.9; }
            .btn:active { transform: translateY(0); }

            #result-card {
                margin-top: 25px;
                padding: 20px;
                background: rgba(0, 210, 255, 0.1);
                border: 1px dashed var(--primary);
                border-radius: 12px;
                display: none;
                word-break: break-all;
                text-align: center;
                animation: slideIn 0.4s ease-out;
            }
            @keyframes slideIn {
                from { opacity: 0; transform: scale(0.95); }
                to { opacity: 1; transform: scale(1); }
            }
            .api-key-text {
                font-family: monospace;
                font-size: 18px;
                color: var(--primary);
                margin: 10px 0;
                display: block;
            }
            .copy-hint { font-size: 12px; color: #64748b; }
            .back-link {
                display: block;
                text-align: center;
                margin-top: 20px;
                color: #64748b;
                text-decoration: none;
                font-size: 14px;
            }
            .back-link:hover { color: var(--text); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Register API</h1>
            <p style="text-align: center; color: #94a3b8; font-size: 14px; margin-bottom: 30px;">
                Enter your details to generate a unique API access key.
            </p>
            
            <div class="form-group">
                <label for="email">Email Address</label>
                <input type="email" id="email" placeholder="name@company.com">
            </div>
            
            <div class="form-group">
                <label for="company">Company Name</label>
                <input type="text" id="company" placeholder="ACME Corp">
            </div>
            
            <button class="btn" onclick="generateKey()">Generate API Key</button>
            
            <div id="result-card">
                <p style="margin: 0; font-size: 14px; color: #94a3b8;">Your New API Key:</p>
                <span class="api-key-text" id="apiKeyDisplay"></span>
                <p class="copy-hint" id="copyHint">Click to copy</p>
            </div>

            <a href="/" class="back-link">← Back to Detection UI</a>
        </div>

        <script>
            async function generateKey() {
                const email = document.getElementById('email').value.trim();
                const company = document.getElementById('company').value.trim();
                
                if(!email || !company) {
                    alert("Please fill in all fields.");
                    return;
                }

                try {
                    const response = await fetch('/generate-key', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ email, company })
                    });
                    
                    const data = await response.json();
                    
                    if(response.ok) {
                        const display = document.getElementById('apiKeyDisplay');
                        display.textContent = data.api_key;
                        document.getElementById('result-card').style.display = 'block';
                        
                        display.onclick = () => {
                            navigator.clipboard.writeText(data.api_key);
                            document.getElementById('copyHint').textContent = "Copied!";
                            setTimeout(() => {
                                document.getElementById('copyHint').textContent = "Click to copy";
                            }, 2000);
                        };
                    } else {
                        alert(data.detail || "Error generating key");
                    }
                } catch(e) {
                    alert("Network error");
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/generate-key")
async def generate_key(data: dict):
    email = data.get("email")
    company = data.get("company")
    
    if not email or not company:
        raise HTTPException(status_code=400, detail="Email and Company are required")
    
    # Generate a unique key
    new_key = f"SPI_{uuid.uuid4().hex[:12].upper()}"
    
    keys = load_keys()
    keys[new_key] = {
        "email": email,
        "company": company,
        "created_at": str(np.datetime64('now'))
    }
    save_keys(keys)
    
    return {"api_key": new_key}


# ===============================
# Web UI
# ===============================
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Tiny Object Detection API</title>
        <style>
            :root {
                --primary: #ff4b2b;
                --secondary: #ff416c;
                --bg: #1a1a1a;
                --text: #ffffff;
            }
            body {
                font-family: 'Segoe UI', sans-serif;
                background: var(--bg);
                color: var(--text);
                margin: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 900px;
                width: 100%;
                background: #2d2d2d;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                text-align: center;
            }
            h1 {
                background: linear-gradient(to right, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 15px;
            }
            input {
                width: 90%;
                padding: 12px;
                border-radius: 10px;
                border: none;
                margin-bottom: 15px;
                outline: none;
            }
            .upload-section {
                border: 2px dashed #555;
                padding: 35px;
                border-radius: 10px;
                cursor: pointer;
                margin-bottom: 15px;
            }
            .upload-section:hover {
                border-color: var(--primary);
            }
            input[type="file"] {
                display: none;
            }
            .btn {
                background: linear-gradient(to right, var(--primary), var(--secondary));
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
            }
            .btn:hover { transform: scale(1.05); }
            #result { margin-top: 20px; display: none; }
            #result img { max-width: 100%; border-radius: 10px; }
            .stats { display: flex; justify-content: center; gap: 15px; margin-top: 15px; flex-wrap: wrap;}
            .stat-card { background: #3d3d3d; padding: 12px; border-radius: 10px; min-width: 110px; }
            .stat-card h3 { margin: 0; font-size: 13px; color: #aaa; }
            .stat-card p { margin: 5px 0 0; font-size: 22px; font-weight: bold; color: var(--primary); }
            .loading { display: none; margin: 20px 0; }
            .spinner {
                border: 4px solid rgba(255,255,255,0.1);
                border-left: 4px solid var(--primary);
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin { 0%{transform:rotate(0deg);} 100%{transform:rotate(360deg);} }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Tiny Person Detection System</h1>
            <p>Upload image OR provide RTSP URL</p>

            <input type="text" id="apiKey" placeholder="Enter API Key (Bearer token)">
            
            <p style="font-size: 12px; margin-top: -10px; margin-bottom: 20px;">
                Don't have a key? <a href="/generateapi" style="color: var(--primary); text-decoration: none; font-weight: bold;">Generate one here</a>
            </p>

            <div class="upload-section" onclick="document.getElementById('fileInput').click()">
                <p id="fileName">Click to upload an image</p>
                <input type="file" id="fileInput" accept="image/*" onchange="updateFileName(this)">
            </div>

            <input type="text" id="rtspUrl" placeholder="OR enter RTSP URL (rtsp://...)">

            <button class="btn" onclick="runDetection()">Run Detection</button>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing...</p>
            </div>

            <div id="result">
                <div class="stats" id="stats"></div>
                <h3>Result Image:</h3>
                <img id="resultImage" src="" alt="Result">
            </div>
        </div>

        <script>
            function updateFileName(input) {
                if (input.files[0]) {
                    document.getElementById('fileName').textContent = input.files[0].name;
                }
            }

            async function runDetection() {
                const apiKey = document.getElementById("apiKey").value.trim();
                if (!apiKey) {
                    alert("Please enter API Key");
                    return;
                }

                const fileInput = document.getElementById("fileInput");
                const rtspUrl = document.getElementById("rtspUrl").value.trim();

                if (!fileInput.files[0] && !rtspUrl) {
                    alert("Upload an image OR enter RTSP URL");
                    return;
                }

                const loading = document.getElementById("loading");
                const resultDiv = document.getElementById("result");
                const statsDiv = document.getElementById("stats");
                const resultImage = document.getElementById("resultImage");

                // --- CASE 1: RTSP Live Stream ---
                if (rtspUrl && !fileInput.files[0]) {
                    loading.style.display = "block";
                    resultDiv.style.display = "none";
                    
                    // Direct stream URL as image src
                    const streamUrl = `/video_feed?rtsp_url=${encodeURIComponent(rtspUrl)}&token=${encodeURIComponent(apiKey)}`;
                    
                    resultImage.onload = () => {
                        loading.style.display = "none";
                        resultDiv.style.display = "block";
                    };
                    resultImage.onerror = () => {
                        loading.style.display = "none";
                        alert("Failed to connect to RTSP stream. Check URL and API Key.");
                    };
                    
                    resultImage.src = streamUrl;
                    statsDiv.innerHTML = '<div class="stat-card"><h3>Mode</h3><p>Live Stream</p></div><div class="stat-card"><h3>Status</h3><p>Active</p></div>';
                    return;
                }

                // --- CASE 2: Image Upload (Single Shot) ---
                loading.style.display = "block";
                resultDiv.style.display = "none";
                statsDiv.innerHTML = "";

                const formData = new FormData();
                if (fileInput.files[0]) {
                    formData.append("image", fileInput.files[0]);
                }

                // ✅ WebUI tells API to return the image
                formData.append("return_image", "true");

                try {
                    const response = await fetch("/detection", {
                        method: "POST",
                        headers: { "Authorization": "Bearer " + apiKey },
                        body: formData
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        alert("Error: " + (data.detail || "Request failed"));
                        return;
                    }

                    for (const [label, count] of Object.entries(data.label_counts)) {
                        const card = document.createElement("div");
                        card.className = "stat-card";
                        card.innerHTML = `<h3>${label}</h3><p>${count}</p>`;
                        statsDiv.appendChild(card);
                    }

                    if (Object.keys(data.label_counts).length === 0) {
                        statsDiv.innerHTML = '<div class="stat-card"><h3>No objects</h3><p>0</p></div>';
                    }

                    // ✅ Image will only exist if return_image=true
                    resultImage.src = data.image || "";
                    resultDiv.style.display = "block";

                } catch (error) {
                    console.error(error);
                    alert("Failed request");
                } finally {
                    loading.style.display = "none";
                }
            }
        </script>
    </body>
    </html>
    """


# ===============================
# Run Server
# ===============================
if __name__ == "__main__":
    import uvicorn
    print("✅ Starting server on http://0.0.0.0:8083")
    uvicorn.run(app, host="0.0.0.0", port=8083)
