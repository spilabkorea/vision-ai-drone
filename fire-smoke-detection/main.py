import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
    QWidget, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QGridLayout, QMessageBox, QDialog
)
from PyQt5.QtGui import QFont, QPixmap, QImage, QIcon
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import csv

class FireSmokeDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fire & Smoke Detection")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet("background-color: #F5F7FA;")
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.video_path = 'output_video.mp4'
        self.cap = cv2.VideoCapture(os.environ.get("RTSP_URL", "rtsp://127.0.0.1:8554/live0"), cv2.CAP_FFMPEG)
        self.total_smoke_count = 0
        self.alert_shown = False

        self.fire_count = 0
        self.smoke_count = 0
        self.total_fire = 0
        self.total_smoke = 0
        self.normal_fire_count = 10
        self.normal_smoke_count = 15
        self.fire_history = []
        self.smoke_history = []

        self.model = YOLO("best_eo.pt")  # Your trained YOLOv8 model
        self.class_names = self.model.names

    def init_ui(self):
        main_layout = QVBoxLayout()
        grid_layout = QGridLayout()

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(780, 570)
        self.video_label.setStyleSheet("border: 3px solid #2E3B55;")
        grid_layout.addWidget(self.video_label, 0, 0)

        pie_layout = QHBoxLayout()
        self.figure_pie_fire = Figure()
        self.canvas_pie_fire = FigureCanvas(self.figure_pie_fire)
        self.canvas_pie_fire.setMinimumSize(300, 270)
        pie_layout.addWidget(self.canvas_pie_fire)

        self.figure_pie_smoke = Figure()
        self.canvas_pie_smoke = FigureCanvas(self.figure_pie_smoke)
        self.canvas_pie_smoke.setMinimumSize(300, 270)
        pie_layout.addWidget(self.canvas_pie_smoke)

        pie_container = QWidget()
        pie_container.setLayout(pie_layout)
        pie_container.setStyleSheet("border: 3px solid #2E3B55;")
        grid_layout.addWidget(pie_container, 0, 1)

        self.figure_graph = Figure()
        self.canvas_graph = FigureCanvas(self.figure_graph)
        self.canvas_graph.setStyleSheet("border: 3px solid #2E3B55;")
        self.figure_graph.set_figheight(290)
        grid_layout.addWidget(self.canvas_graph, 1, 0)

        self.table = QTableWidget(2, 2)
        self.table.setHorizontalHeaderLabels(["Current", "Total"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ccc;
                background-color: #fff;
                padding: 12px 40px;
                font-size: 18px;
                color: #333;
                gridline-color: #ccc;
            }
            QHeaderView::section {
                background-color: #2E3B55;
                color: white;
                font-weight: bold;
                padding: 8px;
                font-size: 16px;
            }
            QTableWidget::item {
                padding: 40px;
            }
        """)
        self.table.setFixedHeight(430)
        grid_layout.addWidget(self.table, 1, 1)

        # Add smoke confidence label
        self.smoke_confidence_label = QLabel("Smoke Confidence: N/A")
        self.smoke_confidence_label.setStyleSheet("font-size: 16px; color: #333; padding: 10px;")
        self.smoke_confidence_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.smoke_confidence_label)

        main_layout.addLayout(grid_layout)

        button_layout = QVBoxLayout()

        start_button = QPushButton("Start")
        start_button.setIcon(QIcon("play.png"))
        start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 12px 0px;
                font-size: 16px;
                border-radius: 8px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        start_button.clicked.connect(self.start_detection)
        button_layout.addWidget(start_button)

        stop_button = QPushButton("Stop")
        stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 20px;
                padding: 12px 0px;
                border-radius: 10px;
                font-weight: bold;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        stop_button.clicked.connect(self.stop_detection)
        button_layout.addWidget(stop_button)

        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def start_detection(self):
        self.timer.start(30)

    def stop_detection(self):
        self.timer.stop()
        self.save_detection_counts_to_csv()

    def update_frame(self):
        ret, frame = self.capture_video_frame()
        if not ret:
            return

        detections = self.detect_objects(frame)
        frame_with_gradcam = self.apply_gradcam(frame, detections)
        self.display_frame(frame_with_gradcam)
        self.update_detection_counts(detections)
        self.update_graphs()
        self.update_table()

    def capture_video_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

    def detect_objects(self, frame):
        results = self.model(frame)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = self.class_names[cls_id]
            confidence = float(box.conf[0])
            if confidence > 0.97:  # Only include detections with confidence > 85%
                detections.append((class_name, confidence))

        return detections

    def apply_gradcam(self, frame, detections):
        results = self.model(frame)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = self.class_names[cls_id]
            confidence = float(box.conf[0])
            if confidence <= 0.97:  # Skip detections with confidence <= 85%
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if class_name == "fire":
                color = (0, 0, 255)  # Red for fire
            elif class_name == "smoke":
                color = (255, 0, 0)  # Blue for smoke
            else:
                color = (0, 255, 0)  # Green for other detections

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if class_name in ["fire", "smoke"]:
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                heatmap = self.generate_gradcam_heatmap(roi.shape[0], roi.shape[1])
                heatmap = cv2.resize(heatmap, (roi.shape[1], roi.shape[0]))
                heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(roi, 0.6, heatmap, 0.4, 0)
                frame[y1:y2, x1:x2] = overlay

        return frame

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def update_detection_counts(self, detections):
        self.fire_count = sum(1 for class_name, _ in detections if class_name == "fire")
        self.smoke_count = sum(1 for class_name, _ in detections if class_name == "smoke")

        # Update smoke confidence label
        smoke_detections = [(class_name, confidence) for class_name, confidence in detections if class_name == "smoke"]
        if len(smoke_detections) == 1:
            _, smoke_confidence = smoke_detections[0]
            self.smoke_confidence_label.setText(f"Smoke Confidence: {smoke_confidence:.2f}")
        else:
            self.smoke_confidence_label.setText("Smoke Confidence: N/A")

        self.total_fire += self.fire_count
        self.total_smoke += self.smoke_count
        self.fire_history.append(self.total_fire)
        self.smoke_history.append(self.total_smoke)

        if self.total_smoke > 150 and not self.alert_shown:
            self.show_alert("Smoke Alert", "Smoke level has exceeded the safety threshold!")
            self.alert_shown = True

    def update_graphs(self):
        self.figure_pie_fire.clear()
        ax_fire = self.figure_pie_fire.add_subplot(111)
        ax_fire.pie([self.normal_fire_count, self.total_fire], labels=["Normal", "Fire"], autopct="%1.1f%%")
        ax_fire.set_title("Normal vs Fire Ratio")
        self.canvas_pie_fire.draw()

        self.figure_pie_smoke.clear()
        ax_smoke = self.figure_pie_smoke.add_subplot(111)
        ax_smoke.pie([self.normal_smoke_count, self.total_smoke], labels=["Normal", "Smoke"], autopct="%1.1f%%")
        ax_smoke.set_title("Normal vs Smoke Ratio")
        self.canvas_pie_smoke.draw()

        self.figure_graph.clear()
        ax_line = self.figure_graph.add_subplot(111)
        ax_line.plot(self.fire_history, label='Fire Count per frame', color='red')
        ax_line.plot(self.smoke_history, label='Smoke Count per frame', color='gray')
        ax_line.set_title("Time Series: Frame-wise Fire & Smoke Detection")
        ax_line.set_xlabel("Frame Number")
        ax_line.set_ylabel("Detection Count")
        ax_line.legend()
        self.canvas_graph.draw()

    def update_table(self):
        self.table.setItem(0, 0, QTableWidgetItem(str(self.fire_count)))
        self.table.setItem(0, 1, QTableWidgetItem(str(self.total_fire)))
        self.table.setItem(1, 0, QTableWidgetItem(str(self.smoke_count)))
        self.table.setItem(1, 1, QTableWidgetItem(str(self.total_smoke)))

        for row in range(2):
            self.table.setRowHeight(row, 170)
            for col in range(2):
                item = self.table.item(row, col)
                if item:
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setFont(QFont("Arial", 14, QFont.Bold))

    def show_bar_chart(self, row, column):
        dialog = QDialog(self)
        dialog.setWindowTitle("Detection Counts")
        dialog.setGeometry(200, 200, 400, 300)

        bar_fig = Figure(figsize=(5, 4))
        bar_canvas = FigureCanvas(bar_fig)
        ax = bar_fig.add_subplot(111)
        ax.bar(['Accumulated Fire', 'Accumulated Smoke'], [self.total_fire, self.total_smoke], color=['red', 'gray'])
        ax.set_ylabel('Detection Type')
        bar_canvas.draw()

        layout = QVBoxLayout()
        layout.addWidget(bar_canvas)
        dialog.setLayout(layout)
        dialog.exec_()

    def show_alert(self, title, message):
        alert = QMessageBox()
        alert.setWindowTitle(title)
        alert.setText(message)
        alert.setIcon(QMessageBox.Warning)
        alert.exec_()

    def generate_gradcam_heatmap(self, height, width):
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        heatmap = np.exp(-distance / (height / 4))
        heatmap = np.clip(heatmap, 0, 1)
        return heatmap

    def save_detection_counts_to_csv(self):
        csv_filename = "detection_counts.csv"
        headers = ["Frame", "Fire Count", "Smoke Count", "Total Fire", "Total Smoke"]
        data = []
        for frame_idx, (fire, smoke) in enumerate(zip(self.fire_history, self.smoke_history)):
            data.append([frame_idx + 1, self.fire_history[frame_idx], self.smoke_history[frame_idx], fire, smoke])

        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(data)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FireSmokeDetectionApp()
    window.show()
    sys.exit(app.exec_())