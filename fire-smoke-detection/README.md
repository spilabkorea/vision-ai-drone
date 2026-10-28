# Fire & Smoke Detection Dashboard

Real-time fire and smoke monitoring for drone and CCTV video streams.
A YOLOv8 model (`best_eo.pt`, trained on electro-optical camera footage) runs on each frame of an RTSP stream, and a PyQt5 dashboard shows the results.

## Features

- Live RTSP video with bounding boxes and a heatmap overlay on detected regions
- Pie charts comparing current fire/smoke counts with a normal baseline
- Time-series graph and statistics table (current and cumulative counts)
- Alert dialog when cumulative smoke detections exceed a threshold (default 150)
- Detection history exported to `detection_counts.csv` when you press **Stop**

## Requirements

- Python 3.8+
- An RTSP stream (drone or CCTV camera), or a local video file / webcam (see the guide)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Run

Set the camera address in the `RTSP_URL` environment variable, then start one of the two variants.

```bash
# Linux / macOS
RTSP_URL="rtsp://<camera-ip>:<port>/<path>" python main_drone.py

# Windows (PowerShell)
$env:RTSP_URL = "rtsp://<camera-ip>:<port>/<path>"
python main_drone.py
```

| Script | Use case | Frame interval | Video panel |
|---|---|---|---|
| `main_drone.py` | Drone camera (primary) | 250 ms (~4 FPS) | 480 × 270 |
| `main.py` | Fixed camera | 30 ms (~33 FPS) | 780 × 570 |

If `RTSP_URL` is not set, the app tries `rtsp://127.0.0.1:8554/live0`.

## Configuration

Apart from `RTSP_URL`, settings such as the confidence threshold (0.97), alert threshold and baselines are constants in the source.
[DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) lists every parameter with its line number and suggested range.

## Known issues

The guide documents known issues and fixes, including:

- YOLO inference runs twice per frame (performance)
- The CSV "Count" and "Total" columns contain the same cumulative values
- No automatic reconnection when the RTSP stream drops
- The start button icon `play.png` is not included

See [DEVELOPMENT_GUIDE.md §12](DEVELOPMENT_GUIDE.md#12-known-issues--bugs) and §14 for recommended improvements.
