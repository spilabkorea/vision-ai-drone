# Tiny Object Detection API

REST API and web UI that detect small vehicles in drone and aerial imagery with a custom YOLOv8 model (`best_tinyobject.pt`).

**Classes:** `Car`, `Truck`, `Bus`

## Features

- `POST /detection`: detect objects in an uploaded image or a single frame grabbed from an RTSP stream
- `GET /video_feed`: annotated live MJPEG stream from an RTSP source
- API key issuance page (`/generateapi`) and a browser test UI (`/`)
- Bearer-token authentication backed by a local `api_keys.json`

## Requirements

Tested with Python 3.12.

```bash
pip install -r requirements.txt
```

For GPU inference, install the PyTorch build that matches your CUDA version first (see [pytorch.org](https://pytorch.org/get-started/locally/)).

## Run

```bash
python api_key.py
```

The server listens on `http://0.0.0.0:8082`.

On first start, `api_keys.json` is created with one initial key, which is printed to the console.
To choose that key yourself, set `DEFAULT_API_KEY` before starting.

| URL | Purpose |
|---|---|
| `http://localhost:8082/generateapi` | Issue a new API key (email + company) |
| `http://localhost:8082/` | Test UI: upload an image or enter an RTSP URL |
| `http://localhost:8082/docs` | Interactive OpenAPI docs (FastAPI) |

## API usage

**Detect objects in an image**

```bash
curl -X POST http://localhost:8082/detection \
  -H "Authorization: Bearer <API_KEY>" \
  -F "image=@test.jpg" \
  -F "return_image=false"
```

Use `-F "rtsp_url=rtsp://<camera-ip>:<port>/<path>"` instead of `image` to analyze one frame from a stream.

**Response**

```json
{
  "label_counts": {"Car": 3, "Truck": 1},
  "detections": [
    {"label": "Car", "confidence": 0.91, "bbox": [x1, y1, x2, y2]}
  ]
}
```

With `return_image=true`, the response also contains `image`: the annotated frame as a base64 JPEG data URL.

**Live stream**

```
GET /video_feed?rtsp_url=<RTSP_URL>&token=<API_KEY>
```

## Deploying behind Cloudflare Tunnel (optional)

To expose a locally running server on your own domain:

```bash
cloudflared tunnel login
cloudflared tunnel create <tunnel-name>
cloudflared tunnel route dns <tunnel-name> <api.your-domain.com>
cloudflared tunnel run --url http://localhost:8082 <tunnel-name>
```

## Security notes

- `api_keys.json` stores issued keys and requester emails. Keep it out of version control (it is in `.gitignore`).
- `/generate-key` lets anyone issue a key. Restrict it before exposing the API publicly.
