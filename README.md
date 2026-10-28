# Vision AI Drone

Vision AI modules for drone and CCTV video, developed by **SPILab**.
Each module runs a custom-trained YOLOv8 model on live RTSP streams or still images.

| Module | What it detects | Interface | Default port |
|---|---|---|---|
| [fire-smoke-detection](fire-smoke-detection/) | Fire, smoke | PyQt5 desktop dashboard | – |
| [tiny-object-detection-api](tiny-object-detection-api/) | Car, truck, bus (small objects in aerial views) | REST API + web UI (FastAPI) | 8082 |
| [tiny-person-detection-api](tiny-person-detection-api/) | Person (small objects in aerial views) | REST API + web UI (FastAPI) | 8083 |

Trained weights (`.pt`) are included in each module folder.

## Repository layout

```
vision-ai-drone/
├── fire-smoke-detection/        # Real-time fire & smoke monitoring dashboard
│   ├── main_drone.py            # Drone camera variant (~4 FPS)
│   ├── main.py                  # Fixed camera variant (~33 FPS)
│   ├── best_eo.pt               # YOLOv8 weights (EO camera)
│   └── DEVELOPMENT_GUIDE.md     # Architecture, configuration, known issues
├── tiny-object-detection-api/   # Vehicle detection API
│   ├── api_key.py
│   └── best_tinyobject.pt
└── tiny-person-detection-api/   # Person detection API
    ├── api_key_tinyperson.py
    └── tiny_person_best.pt
```

## Quick start

Requires Python 3.8+ (the APIs were tested on Python 3.12). A CUDA GPU is recommended but not required.

```bash
git clone https://github.com/spilabkorea/vision-ai-drone.git
cd vision-ai-drone
```

**Fire & smoke dashboard**

```bash
cd fire-smoke-detection
pip install -r requirements.txt
RTSP_URL="rtsp://<camera-ip>:<port>/<path>" python main_drone.py
```

**Detection APIs**

```bash
cd tiny-person-detection-api        # or tiny-object-detection-api
pip install -r requirements.txt
python api_key_tinyperson.py        # or: python api_key.py
```

Then open `http://localhost:8083/generateapi` to issue an API key and `http://localhost:8083/` for the test UI.
See each module's README for details.

## Tech stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) – detection models
- OpenCV – video capture (RTSP, files, webcams)
- PyQt5 + Matplotlib – desktop dashboard
- FastAPI + Uvicorn – REST API

## Security notes

- Never commit `api_keys.json`; it holds issued keys and requester emails and is listed in `.gitignore`.
- Camera addresses are read from the `RTSP_URL` environment variable instead of being hardcoded.
- The `/generate-key` endpoint is open by design for demos. Put it behind authentication before exposing an API to the internet.

## License

See [LICENSE](LICENSE). These modules depend on Ultralytics YOLOv8, which is licensed under AGPL-3.0.

---

## 한국어 요약

SPILab이 개발한 드론·CCTV 영상용 비전 AI 모듈 모음입니다. 직접 학습한 YOLOv8 모델로 RTSP 실시간 영상이나 이미지에서 객체를 탐지합니다.

- **fire-smoke-detection**: 화재·연기를 실시간 탐지하고 그래프·경보를 보여 주는 PyQt5 대시보드
- **tiny-object-detection-api**: 항공 영상의 작은 차량(승용차·트럭·버스)을 탐지하는 REST API와 웹 UI
- **tiny-person-detection-api**: 항공 영상의 작은 사람을 탐지하는 REST API와 웹 UI

설치와 실행 방법은 각 폴더의 README를 참고하세요.
