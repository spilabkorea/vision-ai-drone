# Fire & Smoke Detection Drone Application - Development Guide

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Project Structure](#3-project-structure)
4. [Architecture](#4-architecture)
5. [Application Lifecycle](#5-application-lifecycle)
6. [Class Reference: FireSmokeDetectionApp](#6-class-reference-firesmokedetectionapp)
7. [UI Layout Structure](#7-ui-layout-structure)
8. [Detection Pipeline](#8-detection-pipeline)
9. [Visualization System](#9-visualization-system)
10. [Configuration Reference](#10-configuration-reference)
11. [Data Export](#11-data-export)
12. [Known Issues & Bugs](#12-known-issues--bugs)
13. [Extension Guide](#13-extension-guide)
14. [Recommended Improvements](#14-recommended-improvements)

---

## 1. Project Overview

This application is a **real-time fire and smoke detection dashboard** that:

- Captures live video from an RTSP camera stream (drone/CCTV)
- Runs YOLOv8 object detection on each frame to identify fire and smoke
- Draws bounding boxes and heatmap overlays on detected regions
- Displays real-time analytics: pie charts, time-series graphs, and a statistics table
- Triggers alerts when smoke levels exceed a safety threshold
- Exports detection history to CSV on stop

**Tech Stack:**

| Layer         | Technology                     |
|---------------|--------------------------------|
| Language      | Python 3.8+                    |
| GUI Framework | PyQt5                          |
| AI/ML Model   | YOLOv8 (ultralytics)           |
| Video         | OpenCV (cv2)                   |
| Charts        | Matplotlib (embedded in PyQt5) |
| Data Export   | Python csv module              |

---

## 2. Environment Setup

### 2.1 Prerequisites

- Python 3.8 or higher
- pip package manager
- Access to an RTSP camera stream (or local video file)
- The trained YOLO model file `best_eo.pt`

### 2.2 Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install required packages
pip install PyQt5 opencv-python numpy ultralytics matplotlib
```

### 2.3 Required Packages Summary

| Package           | Minimum Version | Purpose                                |
|-------------------|-----------------|----------------------------------------|
| `PyQt5`           | 5.15+           | GUI framework, widgets, signals/slots  |
| `opencv-python`   | 4.x             | Video capture, image processing        |
| `numpy`           | 1.x             | Array operations, heatmap generation   |
| `ultralytics`     | 8.x             | YOLOv8 model loading and inference     |
| `matplotlib`      | 3.x             | Pie charts, line graphs in Qt canvas   |

### 2.4 Running the Application

```bash
python main_drone.py
```

---

## 3. Project Structure

```
fire-smoke-detection/
├── main.py                  # Fixed-camera variant (alternate RTSP source)
├── main_drone.py            # Drone-camera variant (primary entry point)
├── best_eo.pt               # YOLOv8 trained model weights (~41.5 MB)
├── detection_counts.csv     # Auto-generated CSV output (created on Stop)
├── play.png                 # Start button icon (referenced but missing)
└── DEVELOPMENT_GUIDE.md     # This file
```

### File Descriptions

| File                   | Description                                                              |
|------------------------|--------------------------------------------------------------------------|
| `main_drone.py`        | Primary application. Connects to drone RTSP stream at 4 FPS.            |
| `main.py`              | Alternate version for fixed camera with different RTSP URL and 33 FPS.  |
| `best_eo.pt`           | Pre-trained YOLOv8 model for fire/smoke detection (electro-optical).     |
| `detection_counts.csv` | Frame-by-frame detection log, written when detection is stopped.         |

### Differences Between `main.py` and `main_drone.py`

| Property          | `main.py`                                     | `main_drone.py`                                     |
|-------------------|-----------------------------------------------|------------------------------------------------------|
| RTSP URL          | `RTSP_URL` env var (fixed camera)             | `RTSP_URL` env var (drone camera)                    |
| Video Label Size  | 780 x 570 px                                  | 480 x 270 px                                        |
| Timer Interval    | 30 ms (~33 FPS)                               | 250 ms (~4 FPS)                                     |

---

## 4. Architecture

### 4.1 High-Level Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    FireSmokeDetectionApp                        │
│                      (QMainWindow)                              │
│                                                                  │
│  ┌─────────────┐     QTimer (250ms)     ┌──────────────────┐   │
│  │ RTSP Stream  │ ───────────────────►  │  update_frame()  │   │
│  │ cv2.Video    │                        │  Main Pipeline   │   │
│  │ Capture      │                        └────────┬─────────┘   │
│  └─────────────┘                                  │             │
│                                                    ▼             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                  Frame Processing Pipeline                  │ │
│  │                                                             │ │
│  │  1. capture_video_frame()   ← Read frame from stream       │ │
│  │           │                                                 │ │
│  │           ▼                                                 │ │
│  │  2. detect_objects(frame)   ← YOLO inference + filtering   │ │
│  │           │                                                 │ │
│  │           ▼                                                 │ │
│  │  3. apply_gradcam(frame)    ← Draw boxes + heatmap overlay │ │
│  │           │                                                 │ │
│  │           ▼                                                 │ │
│  │  4. display_frame()         ← BGR→RGB → QImage → QLabel   │ │
│  │           │                                                 │ │
│  │           ▼                                                 │ │
│  │  5. update_detection_counts() ← Accumulate + alert check  │ │
│  │           │                                                 │ │
│  │           ▼                                                 │ │
│  │  6. update_graphs()         ← Redraw all 3 charts          │ │
│  │           │                                                 │ │
│  │           ▼                                                 │ │
│  │  7. update_table()          ← Refresh statistics table     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                      UI Components                          │ │
│  │  ┌──────────────────┬─────────────────────────┐             │ │
│  │  │  Video Feed      │  Pie Charts             │  Grid(0,x) │ │
│  │  │  (QLabel)        │  (Fire | Smoke)         │             │ │
│  │  ├──────────────────┼─────────────────────────┤             │ │
│  │  │  Line Graph      │  Statistics Table       │  Grid(1,x) │ │
│  │  │  (Matplotlib)    │  (QTableWidget 2x2)    │             │ │
│  │  └──────────────────┴─────────────────────────┘             │ │
│  │  [ Start Button ]  [ Stop Button ]                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow

```
RTSP Camera Stream
       │
       ▼
cv2.VideoCapture.read()
       │
       ▼
YOLOv8 Inference ──► confidence > 0.97 filter
       │
       ├──► Bounding boxes + heatmap overlay ──► QLabel (live video)
       │
       ├──► fire_count / smoke_count ──► QTableWidget (statistics)
       │
       ├──► total_fire / total_smoke ──► Pie charts (matplotlib)
       │
       ├──► fire_history / smoke_history ──► Line graph (matplotlib)
       │
       └──► total_smoke > 150 ? ──► QMessageBox alert (one-time)
                                          │
                            [Stop button] ▼
                              save_detection_counts_to_csv()
                                          │
                                          ▼
                              detection_counts.csv
```

---

## 5. Application Lifecycle

```
App Launch
    │
    ▼
__init__()
    ├── init_ui()          ← Build all widgets
    ├── Create QTimer      ← Not started yet
    ├── Open VideoCapture  ← Connect to RTSP
    ├── Initialize counters
    └── Load YOLO model    ← best_eo.pt
    │
    ▼
window.show()              ← Display GUI (idle, no detection)
    │
    ▼
User clicks [Start]
    │
    ▼
start_detection()
    └── timer.start(250)   ← Begin 250ms loop
    │
    ▼
┌─► update_frame()         ← Called every 250ms
│       ├── capture → detect → draw → display → count → graph → table
│       └── If smoke > 150: show alert (once)
│
└── Repeats until...
    │
    ▼
User clicks [Stop]
    │
    ▼
stop_detection()
    ├── timer.stop()
    └── save_detection_counts_to_csv()
    │
    ▼
User closes window → app.exec_() returns → sys.exit()
```

---

## 6. Class Reference: FireSmokeDetectionApp

### 6.1 Instance Variables

| Variable               | Type              | Default Value | Description                                      |
|------------------------|-------------------|---------------|--------------------------------------------------|
| `self.cap`             | `cv2.VideoCapture`| RTSP stream   | Video source handle                              |
| `self.model`           | `YOLO`            | `best_eo.pt`  | YOLOv8 detection model                           |
| `self.class_names`     | `dict`            | from model    | Maps class IDs → names (e.g., 0→"fire")          |
| `self.timer`           | `QTimer`          | —             | Drives the frame update loop                     |
| `self.fire_count`      | `int`             | `0`           | Fire detections in **current** frame             |
| `self.smoke_count`     | `int`             | `0`           | Smoke detections in **current** frame            |
| `self.total_fire`      | `int`             | `0`           | Accumulated fire detections across all frames    |
| `self.total_smoke`     | `int`             | `0`           | Accumulated smoke detections across all frames   |
| `self.fire_history`    | `list[int]`       | `[]`          | Cumulative fire count at each frame              |
| `self.smoke_history`   | `list[int]`       | `[]`          | Cumulative smoke count at each frame             |
| `self.normal_fire_count`  | `int`          | `10`          | Fixed baseline for pie chart "Normal" slice      |
| `self.normal_smoke_count` | `int`          | `15`          | Fixed baseline for pie chart "Normal" slice      |
| `self.alert_shown`     | `bool`            | `False`       | Prevents repeated smoke alert popups             |
| `self.total_smoke_count` | `int`           | `0`           | **Unused** — declared but never referenced       |
| `self.video_path`      | `str`             | `output_video.mp4` | **Unused** — declared but never referenced  |

### 6.2 Method Reference

#### `__init__(self)` — Line 17-41

Constructor. Initializes the window, loads the YOLO model, opens the RTSP stream, and sets up all state variables.

**Important:** The YOLO model is loaded synchronously at startup, which will block the UI until loading completes.

---

#### `init_ui(self)` — Line 43-152

Builds the entire GUI programmatically. No `.ui` designer files are used.

**Widget Tree:**
```
QMainWindow
└── QWidget (centralWidget)
    └── QVBoxLayout (main_layout)
        ├── QLabel (smoke_confidence_label)
        ├── QGridLayout (grid_layout)
        │   ├── [0,0] QLabel (video_label)              ← Live video
        │   ├── [0,1] QWidget (pie_container)
        │   │         └── QHBoxLayout (pie_layout)
        │   │             ├── FigureCanvas (fire pie)
        │   │             └── FigureCanvas (smoke pie)
        │   ├── [1,0] FigureCanvas (line graph)
        │   └── [1,1] QTableWidget (2x2 stats table)
        └── QVBoxLayout (button_layout)
            ├── QPushButton ("Start")
            └── QPushButton ("Stop")
```

---

#### `start_detection(self)` — Line 154-155

Starts the `QTimer` with a 250ms interval. Each tick fires `update_frame()`.

```python
def start_detection(self):
    self.timer.start(250)   # ~4 frames per second
```

---

#### `stop_detection(self)` — Line 157-159

Stops the timer and exports all accumulated data to CSV.

---

#### `update_frame(self)` — Line 161-171

**Core pipeline method.** Called every 250ms by the timer. Orchestrates the full detection → display → analytics cycle.

```
capture_video_frame() → detect_objects() → apply_gradcam()
     → display_frame() → update_detection_counts()
     → update_graphs() → update_table()
```

---

#### `capture_video_frame(self)` — Line 173-178

Reads one frame from `self.cap`. If the read fails (stream ended or dropped), it resets `CAP_PROP_POS_FRAMES` to 0 and retries.

**Note:** The reset-to-frame-0 approach works for local video files (loops them) but does **not** reconnect a dropped RTSP stream.

**Returns:** `(bool, np.ndarray)` — success flag and frame data.

---

#### `detect_objects(self, frame)` — Line 180-191

Runs YOLOv8 inference on the frame. Filters results by confidence threshold.

**Parameters:**
- `frame` (`np.ndarray`): BGR image from OpenCV

**Returns:** `list[tuple[str, float]]` — List of `(class_name, confidence)` tuples

**Confidence Threshold:** `> 0.97` (97%)
**Note:** The inline comment says "85%" but the actual code uses `0.97`. This is a bug in the comment.

---

#### `apply_gradcam(self, frame, detections)` — Line 193-227

Draws bounding boxes and heatmap overlays onto the frame.

**IMPORTANT PERFORMANCE ISSUE:** This method calls `self.model(frame)[0]` a **second time**, running YOLO inference again on the same frame. This doubles the computational cost per frame.

**Bounding Box Colors (BGR format):**
| Class   | Color          | BGR Value       |
|---------|----------------|-----------------|
| fire    | Red            | `(0, 0, 255)`   |
| smoke   | Blue           | `(255, 0, 0)`   |
| other   | Green          | `(0, 255, 0)`   |

**Heatmap Logic:** For fire/smoke detections, extracts the ROI (region of interest), generates a Gaussian heatmap, and blends it at 40% opacity.

---

#### `display_frame(self, frame)` — Line 229-234

Converts an OpenCV BGR frame to a Qt-compatible RGB `QImage` and renders it on the video label.

**Conversion chain:** `BGR (cv2) → RGB → QImage → QPixmap → QLabel.setPixmap()`

---

#### `update_detection_counts(self, detections)` — Line 236-255

- Counts fire and smoke in the current frame
- Adds to cumulative totals
- Appends cumulative totals to history lists
- Updates the smoke confidence label
- Checks if `total_smoke > 150` to trigger a one-time alert

**Smoke Confidence Display:** Only shown when exactly 1 smoke detection exists in the frame. Otherwise displays "N/A".

---

#### `update_graphs(self)` — Line 257-278

Clears and redraws all three matplotlib charts:

1. **Fire Pie Chart** — Fixed "Normal" value (10) vs accumulated fire count
2. **Smoke Pie Chart** — Fixed "Normal" value (15) vs accumulated smoke count
3. **Line Graph** — Cumulative fire and smoke counts over frame number

**Performance Note:** Full clear + redraw every frame is expensive. At 4 FPS this is acceptable but will become a bottleneck at higher frame rates.

---

#### `update_table(self)` — Line 280-293

Updates the 2x2 `QTableWidget`:

|          | Current (col 0)  | Total (col 1)    |
|----------|------------------|------------------|
| Row 0    | `fire_count`     | `total_fire`     |
| Row 1    | `smoke_count`    | `total_smoke`    |

Each cell uses Arial 14pt Bold, center-aligned, with 170px row height.

---

#### `show_bar_chart(self, row, column)` — Line 294-309

Opens a popup `QDialog` with a bar chart comparing total fire vs total smoke.

**STATUS: Dead code.** This method is defined but never connected to any signal or called anywhere. It was likely intended to be triggered by clicking on the table (`self.table.cellClicked.connect(self.show_bar_chart)`).

---

#### `show_alert(self, title, message)` — Line 311-316

Displays a modal `QMessageBox` warning dialog. Currently triggered only when `total_smoke > 150`.

---

#### `generate_gradcam_heatmap(self, height, width)` — Line 318-324

Generates a 2D Gaussian radial gradient centered on the ROI.

**IMPORTANT:** Despite the name, this is **not** a real GradCAM (Gradient-weighted Class Activation Map). A true GradCAM requires backpropagation through the model. This is a visual approximation using a mathematical Gaussian function.

**Algorithm:**
```
distance = sqrt((x - center_x)^2 + (y - center_y)^2)
heatmap  = exp(-distance / (height / 4))
heatmap  = clip(heatmap, 0, 1)
```

---

#### `save_detection_counts_to_csv(self)` — Line 326-336

Exports frame-by-frame detection history to `detection_counts.csv`.

**CSV Columns:**

| Column       | Content                      |
|--------------|------------------------------|
| Frame        | Frame index (1-based)        |
| Fire Count   | `fire_history[frame_idx]`    |
| Smoke Count  | `smoke_history[frame_idx]`   |
| Total Fire   | `fire_history[frame_idx]`    |
| Total Smoke  | `smoke_history[frame_idx]`   |

**BUG:** "Fire Count" and "Total Fire" columns contain identical data (`fire_history[frame_idx]`), and same for smoke. The `fire_history` list stores cumulative totals, not per-frame counts. See [Known Issues](#12-known-issues--bugs).

---

## 7. UI Layout Structure

### 7.1 Visual Layout

```
┌────────────────────────────────────────────────────────────────┐
│              Smoke Confidence: 0.98                            │
├───────────────────────────┬────────────────────────────────────┤
│                           │  ┌────────────┐ ┌──────────────┐  │
│    Live Video Feed        │  │  Normal vs  │ │  Normal vs   │  │
│    (480 x 270 min)        │  │  Fire Ratio │ │  Smoke Ratio │  │
│                           │  │  (Pie)      │ │  (Pie)       │  │
│                           │  └────────────┘ └──────────────┘  │
├───────────────────────────┼────────────────────────────────────┤
│                           │                                    │
│  Time Series Graph        │     ┌──────────┬──────────┐       │
│  Fire & Smoke Detection   │     │ Current  │  Total   │       │
│  over Frame Number        │     ├──────────┼──────────┤       │
│  (Line chart)             │     │   0      │   42     │ Fire  │
│                           │     ├──────────┼──────────┤       │
│                           │     │   1      │   87     │ Smoke │
│                           │     └──────────┴──────────┘       │
├───────────────────────────┴────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                      Start                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                      Stop                                │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### 7.2 Styling

| Element          | Background     | Text Color | Border                   |
|------------------|----------------|------------|--------------------------|
| Main window      | `#F5F7FA`      | —          | —                        |
| Video label      | —              | —          | `3px solid #2E3B55`      |
| Pie container    | —              | —          | `3px solid #2E3B55`      |
| Line graph       | —              | —          | `3px solid #2E3B55`      |
| Table header     | `#2E3B55`      | white      | —                        |
| Table body       | `#fff`         | `#333`     | `1px solid #ccc`         |
| Start button     | `#4CAF50`      | white      | `border-radius: 8px`    |
| Stop button      | `#f44336`      | white      | `border-radius: 10px`   |

---

## 8. Detection Pipeline

### 8.1 YOLO Model

- **Model:** YOLOv8 (loaded via `ultralytics.YOLO`)
- **Weights File:** `best_eo.pt` (electro-optical, custom trained)
- **Classes:** Defined in model metadata (`self.model.names`), expected: `fire`, `smoke`
- **Inference:** Single-frame, synchronous on each timer tick
- **Confidence Filter:** Only detections with confidence > 0.97 are processed

### 8.2 Frame Processing Steps

```python
# Step 1: Read frame
ret, frame = self.cap.read()

# Step 2: YOLO inference (first call)
results = self.model(frame)[0]
detections = [(class_name, confidence), ...]  # filtered by > 0.97

# Step 3: YOLO inference (second call - redundant)
results = self.model(frame)[0]  # BUG: runs model again

# Step 4: For each detection:
#   - Draw bounding box (colored by class)
#   - Add label text with confidence
#   - If fire/smoke: overlay Gaussian heatmap at 40% opacity

# Step 5: Convert BGR → RGB → QImage → display

# Step 6: Update counts, graphs, table
```

### 8.3 Heatmap Overlay

The pseudo-GradCAM overlay blends with the detected ROI:

```python
overlay = cv2.addWeighted(roi, 0.6, heatmap, 0.4, 0)
# 60% original image + 40% heatmap
```

Color mapping uses OpenCV `COLORMAP_JET` (blue → green → yellow → red).

---

## 9. Visualization System

### 9.1 Pie Charts

Two pie charts compare fixed "Normal" baselines against accumulated detection counts:

| Chart       | Segments                               | Colors          |
|-------------|----------------------------------------|-----------------|
| Fire Pie    | Normal (fixed=10) vs Fire (cumulative) | Default palette |
| Smoke Pie   | Normal (fixed=15) vs Smoke (cumulative)| Default palette |

**Note:** As detections accumulate, the "Normal" slice becomes vanishingly small, making the chart less informative over time.

### 9.2 Line Graph

Plots cumulative fire and smoke counts vs. frame number:

| Series         | Color  | Label                  |
|----------------|--------|------------------------|
| Fire history   | Red    | "Fire Count per frame" |
| Smoke history  | Gray   | "Smoke Count per frame"|

**Note:** Despite the label saying "per frame", the data is **cumulative** (monotonically increasing), not per-frame counts.

### 9.3 Statistics Table

2x2 `QTableWidget` showing current frame counts and running totals.

---

## 10. Configuration Reference

All values below except `RTSP_URL` are **hardcoded** and must be changed in source code:

| Parameter              | Location        | Current Value                                     | Suggested Range         |
|------------------------|-----------------|---------------------------------------------------|-------------------------|
| Window size            | Line 20         | `1400 x 800`                                     | Depends on display      |
| RTSP URL               | Line 28         | `RTSP_URL` env var (default `rtsp://127.0.0.1:8554/live0`) | Any valid RTSP URL |
| YOLO model path        | Line 40         | `best_eo.pt`                                     | Any `.pt` file          |
| Confidence threshold   | Line 188, 200   | `0.97`                                            | `0.25` to `0.70` typical|
| Timer interval (ms)    | Line 155        | `250`                                             | `30` (fast) to `500`    |
| Smoke alert threshold  | Line 253        | `150`                                             | Application-dependent   |
| Normal fire baseline   | Line 35         | `10`                                              | Application-dependent   |
| Normal smoke baseline  | Line 36         | `15`                                              | Application-dependent   |
| Video label min size   | Line 49         | `480 x 270`                                       | Match stream resolution |
| CSV output filename    | Line 327        | `detection_counts.csv`                            | Any valid path          |

---

## 11. Data Export

### CSV Format: `detection_counts.csv`

Generated when the user clicks "Stop". Contains one row per processed frame.

```csv
Frame,Fire Count,Smoke Count,Total Fire,Total Smoke
1,0,0,0,0
2,0,1,0,1
3,0,2,0,2
...
```

**Warning:** Due to a bug, "Fire Count" and "Total Fire" columns are identical (both contain cumulative values). Same for smoke columns. See [Known Issues](#12-known-issues--bugs).

---

## 12. Known Issues & Bugs

### BUG-001: Duplicate YOLO Inference (Performance)
- **Location:** `detect_objects()` (line 181) and `apply_gradcam()` (line 194)
- **Issue:** The YOLO model runs inference **twice** on the same frame. `detect_objects()` runs the model and returns filtered detections. Then `apply_gradcam()` runs the model **again** to draw bounding boxes.
- **Impact:** Doubles GPU/CPU cost per frame.
- **Fix:** Pass the `results` object from `detect_objects()` into `apply_gradcam()` instead of re-running inference.

### BUG-002: CSV Export Writes Identical Columns
- **Location:** `save_detection_counts_to_csv()` (line 331)
- **Issue:** `fire_history` and `smoke_history` store **cumulative** totals, but the CSV uses them for both "Count" and "Total" columns, producing duplicate data.
- **Fix:** Store per-frame counts in separate lists (`fire_per_frame`, `smoke_per_frame`) and use those for the "Count" columns.

### BUG-003: Misleading Comment on Confidence Threshold
- **Location:** Line 188 and Line 200
- **Issue:** Comment says `"confidence > 85%"` but code uses `0.97` (97%).
- **Fix:** Update comments to match actual threshold, or change threshold to match intent.

### BUG-004: RTSP Stream Reconnection Not Handled
- **Location:** `capture_video_frame()` (line 173-178)
- **Issue:** Uses `CAP_PROP_POS_FRAMES = 0` to retry, which only works for local video files. RTSP streams cannot seek to frame 0.
- **Impact:** If the RTSP stream drops, the application silently fails to display frames with no recovery.
- **Fix:** Implement reconnection logic with `cv2.VideoCapture()` re-initialization and retry delay.

### BUG-005: Misleading Graph Label
- **Location:** `update_graphs()` (line 272-273)
- **Issue:** Line graph label says "Fire Count per frame" but data is cumulative, not per-frame.
- **Fix:** Either change label to "Cumulative Fire Count" or plot per-frame differences instead.

### BUG-006: Dead Code
- **Location:** `show_bar_chart()` (line 294-309)
- **Issue:** Method exists but is never called or connected to any signal.
- **Fix:** Connect it (e.g., `self.table.cellClicked.connect(self.show_bar_chart)`) or remove it.

### BUG-007: Unused Variables
- **Location:** `self.total_smoke_count` (line 28), `self.video_path` (line 26)
- **Issue:** Declared in `__init__` but never used anywhere.
- **Fix:** Remove them.

### BUG-008: Missing Icon File
- **Location:** Line 113 references `"play.png"`
- **Issue:** The file does not exist in the project directory.
- **Impact:** Start button renders without an icon (no crash, just missing visual).

### BUG-009: Misleading Function Name
- **Location:** `generate_gradcam_heatmap()` (line 318)
- **Issue:** This generates a simple Gaussian radial gradient, not a true GradCAM (which requires backpropagation through the model).
- **Fix:** Rename to `generate_gaussian_heatmap()` or implement actual GradCAM using `pytorch-grad-cam`.

---

## 13. Extension Guide

### 13.1 Adding a New Detection Class

1. **Retrain the YOLO model** to include the new class and replace `best_eo.pt`
2. **Add bounding box color** in `apply_gradcam()` (line 204-209):
   ```python
   elif class_name == "explosion":
       color = (0, 165, 255)  # Orange
   ```
3. **Add counter variables** in `__init__()`:
   ```python
   self.explosion_count = 0
   self.total_explosion = 0
   self.explosion_history = []
   ```
4. **Update `update_detection_counts()`** to track the new class
5. **Expand the `QTableWidget`** from `(2, 2)` to `(3, 2)` and add the new row
6. **Add a pie chart** for the new class in `update_graphs()`
7. **Update CSV export** with new columns

### 13.2 Switching Video Source

Change the `cv2.VideoCapture()` call in `__init__()` (line 27):

```python
# Local video file
self.cap = cv2.VideoCapture("path/to/video.mp4")

# USB webcam (device index 0)
self.cap = cv2.VideoCapture(0)

# IP camera / RTSP stream
self.cap = cv2.VideoCapture("rtsp://ip:port/path")

# HTTP stream (MJPEG)
self.cap = cv2.VideoCapture("http://ip:port/stream")
```

### 13.3 Adjusting Detection Sensitivity

Lower the confidence threshold at line 188 and line 200:

```python
# Current: very strict (may miss valid detections)
if confidence > 0.97:

# Recommended: balanced sensitivity
if confidence > 0.50:

# Aggressive: catch more detections with more false positives
if confidence > 0.25:
```

**Important:** Both thresholds in `detect_objects()` and `apply_gradcam()` must match.

### 13.4 Changing Frame Rate

Modify the timer interval in `start_detection()` (line 155):

```python
self.timer.start(30)    # ~33 FPS (high CPU/GPU load)
self.timer.start(100)   # ~10 FPS (moderate)
self.timer.start(250)   # ~4 FPS (current, low load)
self.timer.start(1000)  # ~1 FPS (minimal load)
```

### 13.5 Adding RTSP Reconnection

Replace `capture_video_frame()` with reconnection logic:

```python
def capture_video_frame(self):
    ret, frame = self.cap.read()
    if not ret:
        # Attempt reconnection
        self.cap.release()
        self.cap = cv2.VideoCapture(
            os.environ.get("RTSP_URL", "rtsp://127.0.0.1:8554/live0"),
            cv2.CAP_FFMPEG
        )
        ret, frame = self.cap.read()
    return ret, frame
```

---

## 14. Recommended Improvements

### Priority 1 — Bug Fixes

| #   | Task                                       | Effort |
|-----|--------------------------------------------|--------|
| 1   | Eliminate duplicate YOLO inference          | Small  |
| 2   | Fix CSV export to separate counts/totals   | Small  |
| 3   | Fix comment mismatch at line 188/200       | Trivial|
| 4   | Remove unused variables                    | Trivial|
| 5   | Rename `generate_gradcam_heatmap`          | Trivial|

### Priority 2 — Reliability

| #   | Task                                       | Effort |
|-----|--------------------------------------------|--------|
| 6   | Add RTSP reconnection logic                | Small  |
| 7   | Add Python `logging` module throughout     | Small  |
| 8   | Handle `VideoCapture` open failure at init | Small  |

### Priority 3 — Architecture

| #   | Task                                                     | Effort  |
|-----|----------------------------------------------------------|---------|
| 9   | Create `config.yaml` for all hardcoded parameters        | Medium  |
| 10  | Separate into modules: `detection.py`, `ui.py`, `data.py`| Medium  |
| 11  | Add `requirements.txt` with pinned versions              | Trivial |
| 12  | Move YOLO inference to a background thread (QThread)     | Medium  |

### Priority 4 — Features

| #   | Task                                               | Effort |
|-----|----------------------------------------------------|--------|
| 13  | Connect `show_bar_chart()` to table click signal   | Small  |
| 14  | Add pause/resume button                            | Small  |
| 15  | Add real GradCAM using `pytorch-grad-cam`          | Large  |
| 16  | Add per-frame count vs cumulative toggle on graph  | Medium |
| 17  | Save annotated frames/video as evidence            | Medium |

---

## Appendix: Signal/Slot Connections

| Signal                           | Slot                            |
|----------------------------------|---------------------------------|
| `timer.timeout`                  | `update_frame()`                |
| `start_button.clicked`           | `start_detection()`             |
| `stop_button.clicked`            | `stop_detection()`              |
| *(not connected)*                | `show_bar_chart(row, column)`   |
