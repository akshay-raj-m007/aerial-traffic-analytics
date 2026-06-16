"""
interfaces/traffic_ui.py
========================
PyQt5 UI with three tabs:
  Tab 1 — ROI Setup    : load video frame, click to draw polygon ROI, save it
  Tab 2 — Run Analysis : pick video + calibration, run pipeline with ROI, progress bar
  Tab 3 — Trajectories : load gps_trails.csv or summary CSV, pick Track ID,
                         view trajectory on embedded map (folium → QWebEngineView)
                         + stats card (class, speed, distance, duration)

Run standalone:
    python interfaces/traffic_ui.py

Or launched from main.py --gui (add that flag yourself if needed).
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# On Windows, importing PyQt before Torch can put Qt DLL directories ahead of
# PyTorch's DLLs and make torch fail with WinError 1114 while loading c10.dll.
# Preload torch before any PyQt imports so the detector can be imported later
# from the worker thread without hitting that DLL initialization error.
try:
    import torch  # noqa: F401
except Exception:
    # Keep startup alive; the worker will report the full traceback if Torch is
    # genuinely unavailable. This block is only a DLL load-order guard.
    pass

from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QPointF, QRectF, QTimer
)
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont, QPolygonF
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QComboBox,
    QProgressBar, QTextEdit, QSplitter, QFrame,
    QSlider, QGroupBox, QMessageBox, QSizePolicy,
    QScrollArea, QSpinBox
)

# Optional — graceful fallback if QtWebEngineWidgets not installed
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False

# Optional folium for map rendering
try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Color palette matching existing project
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "car":        "#00FF00",
    "motorcycle": "#FFFF00",
    "rikshaw":    "#00FFFF",
    "HMV":        "#FF0000",
    "pedestrian": "#FF00FF",
    "unknown":    "#FFFFFF",
}
CLASS_COLORS_BGR = {
    "car":        (0, 255, 0),
    "motorcycle": (0, 255, 255),
    "rikshaw":    (255, 255, 0),
    "HMV":        (0, 0, 255),
    "pedestrian": (255, 0, 255),
    "unknown":    (255, 255, 255),
}

DARK_BG    = "#1e1e2e"
PANEL_BG   = "#2a2a3e"
ACCENT     = "#7c6af7"
TEXT_WHITE = "#e0e0e0"
TEXT_GRAY  = "#888888"
BTN_STYLE  = (
    "QPushButton {"
    "  background:" + ACCENT + "; color:white; border-radius:6px;"
    "  padding:7px 16px; font-weight:bold; font-size:13px;"
    "}"
    "QPushButton:hover { background:#9d8df8; }"
    "QPushButton:disabled { background:#444; color:#777; }"
)
COMBO_STYLE = (
    "QComboBox {"
    "  background:" + PANEL_BG + "; color:" + TEXT_WHITE + "; border:1px solid #555;"
    "  border-radius:4px; padding:4px 8px; font-size:13px;"
    "}"
    "QComboBox QAbstractItemView { background:" + PANEL_BG + "; color:" + TEXT_WHITE + "; }"
)
LABEL_STYLE  = "color:" + TEXT_WHITE + "; font-size:13px;"
HEADER_STYLE = "color:" + TEXT_WHITE + "; font-size:15px; font-weight:bold;"
GROUP_STYLE  = (
    "QGroupBox { color:" + TEXT_WHITE + "; font-weight:bold; font-size:13px;"
    "  border:1px solid #555; border-radius:6px; margin-top:8px; padding:8px; }"
    "QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }"
)


# ---------------------------------------------------------------------------
# ROI Canvas Widget — draw polygon on a video frame
# ---------------------------------------------------------------------------
class ROICanvas(QLabel):
    """
    Displays a video frame and lets the user click to place polygon vertices.
    Right-click removes the last point. Double-click closes the polygon.
    """
    polygon_changed = pyqtSignal(list)   # emits list of (x,y) pixel tuples

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background:{DARK_BG}; border:2px dashed #555;")
        self.setText("Load a video frame to start drawing ROI")
        self.setFont(QFont("Arial", 12))

        self._base_pixmap: Optional[QPixmap] = None
        self._points: list[tuple[int, int]] = []   # in *image* coordinates
        self._closed = False
        self._scale  = 1.0
        self._offset = QPointF(0, 0)

    # --- public API --------------------------------------------------------

    def set_frame(self, frame: np.ndarray) -> None:
        """Feed a BGR numpy frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._base_pixmap = QPixmap.fromImage(qimg)
        self._points.clear()
        self._closed = False
        self._redraw()

    def clear_roi(self) -> None:
        self._points.clear()
        self._closed = False
        self._redraw()
        self.polygon_changed.emit([])

    def close_polygon(self) -> None:
        if len(self._points) >= 3:
            self._closed = True
            self._redraw()
            self.polygon_changed.emit(list(self._points))

    @property
    def roi_points(self) -> list[tuple[int, int]]:
        return list(self._points)

    # --- internal ----------------------------------------------------------

    def _img_to_widget(self, x: int, y: int) -> QPointF:
        return QPointF(x * self._scale + self._offset.x(),
                       y * self._scale + self._offset.y())

    def _widget_to_img(self, x: float, y: float) -> tuple[int, int]:
        return (int((x - self._offset.x()) / self._scale),
                int((y - self._offset.y()) / self._scale))

    def _compute_transform(self) -> None:
        if self._base_pixmap is None:
            return
        pw, ph = self.width(), self.height()
        iw, ih = self._base_pixmap.width(), self._base_pixmap.height()
        self._scale = min(pw / iw, ph / ih)
        disp_w = iw * self._scale
        disp_h = ih * self._scale
        self._offset = QPointF((pw - disp_w) / 2, (ph - disp_h) / 2)

    def _redraw(self) -> None:
        if self._base_pixmap is None:
            return
        self._compute_transform()
        canvas = QPixmap(self.size())
        canvas.fill(QColor(DARK_BG))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw scaled image
        scaled = self._base_pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(int(self._offset.x()), int(self._offset.y()), scaled)

        if self._points:
            pts_w = [self._img_to_widget(x, y) for x, y in self._points]

            # Fill if closed
            if self._closed and len(pts_w) >= 3:
                poly = QPolygonF(pts_w)
                painter.setBrush(QBrush(QColor(124, 106, 247, 60)))
                painter.setPen(Qt.NoPen)
                painter.drawPolygon(poly)

            # Draw edges
            pen = QPen(QColor(ACCENT), 2, Qt.SolidLine)
            painter.setPen(pen)
            for i in range(1, len(pts_w)):
                painter.drawLine(pts_w[i-1], pts_w[i])
            if self._closed and len(pts_w) >= 2:
                painter.drawLine(pts_w[-1], pts_w[0])

            # Draw closing dashed line preview (not closed yet)
            if not self._closed and len(pts_w) >= 2:
                pen2 = QPen(QColor("#aaa"), 1, Qt.DashLine)
                painter.setPen(pen2)
                painter.drawLine(pts_w[-1], pts_w[0])

            # Draw vertex dots + labels
            for i, pt in enumerate(pts_w):
                color = QColor("#ffcc00") if i == 0 else QColor(ACCENT)
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(Qt.white, 1))
                painter.drawEllipse(pt, 6, 6)
                painter.setPen(QPen(Qt.white))
                painter.setFont(QFont("Arial", 8, QFont.Bold))
                painter.drawText(
                    QRectF(pt.x()+8, pt.y()-8, 24, 16),
                    Qt.AlignLeft, str(i+1))

        # Instruction overlay
        if not self._closed:
            painter.setPen(QPen(QColor(TEXT_GRAY)))
            painter.setFont(QFont("Arial", 9))
            painter.drawText(
                QRectF(8, 8, 400, 20), Qt.AlignLeft,
                "Left-click: add point  |  Right-click: undo  |  Double-click / 'Close ROI': finish"
            )

        painter.end()
        self.setPixmap(canvas)

    # --- events ------------------------------------------------------------

    def mousePressEvent(self, event):
        if self._base_pixmap is None or self._closed:
            return
        ix, iy = self._widget_to_img(event.x(), event.y())
        iw = self._base_pixmap.width()
        ih = self._base_pixmap.height()
        if not (0 <= ix < iw and 0 <= iy < ih):
            return
        if event.button() == Qt.LeftButton:
            self._points.append((ix, iy))
            self._redraw()
        elif event.button() == Qt.RightButton:
            if self._points:
                self._points.pop()
                self._redraw()

    def mouseDoubleClickEvent(self, event):
        if self._base_pixmap is not None and not self._closed:
            self.close_polygon()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._redraw()


# ---------------------------------------------------------------------------
# Worker thread — runs the pipeline without freezing the UI
# ---------------------------------------------------------------------------
class PipelineWorker(QThread):
    progress  = pyqtSignal(int)       # 0-100
    log       = pyqtSignal(str)
    finished  = pyqtSignal(str)       # output dir path
    error     = pyqtSignal(str)

    def __init__(self, video_path: str, calib_path: str,
                 output_dir: str, roi_points: list,
                 write_video: bool, write_map: bool):
        super().__init__()
        self.video_path  = video_path
        self.calib_path  = calib_path
        self.output_dir  = output_dir
        self.roi_points  = roi_points   # list of (x,y) pixel coords or []
        self.write_video = write_video
        self.write_map   = write_map

    def run(self):
        try:
            # Force venv site-packages into thread on Windows
            import sys as _sys, os as _os
            _here = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            _site = _os.path.join(_here, 'venv', 'Lib', 'site-packages')
            if _os.path.exists(_site) and _site not in _sys.path:
                _sys.path.insert(0, _site)
            if _here not in _sys.path:
                _sys.path.insert(0, _here)

            import cv2, pandas as pd
            from core.detector   import Detector
            from core.tracker    import TrajectoryTracker
            from core.homography import Homography
            from core.exporter   import export_all

            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = Path(self.video_path)
            stem = video_path.stem

            self.log.emit(f"[Homography] Loading: {self.calib_path}")
            hom = Homography.from_json(self.calib_path)
            gps_enabled = hom.origin_gps is not None
            if not gps_enabled:
                self.log.emit("[WARNING] No origin_gps — GPS exports skipped.")

            # ROI polygon filter
            roi_poly = None
            if self.roi_points and len(self.roi_points) >= 3:
                roi_poly = np.array(self.roi_points, dtype=np.float32)
                self.log.emit(f"[ROI] Active — {len(self.roi_points)} vertices")
            else:
                self.log.emit("[ROI] None set — processing full frame")

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit(f"Cannot open video: {self.video_path}")
                return

            fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            self.log.emit(f"[Video] {width}x{height}  {fps:.1f}fps  {total} frames")

            writer = None
            if self.write_video:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_dir / f"{stem}_annotated.mp4"),
                    fourcc, fps, (width, height))

            detector = Detector()
            tracker  = TrajectoryTracker(fps=fps)
            all_detections = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = detector.detect(frame, frame_idx=frame_idx)

                # ROI filter
                if roi_poly is not None:
                    detections = [
                        d for d in detections
                        if cv2.pointPolygonTest(
                            roi_poly, (float(d.cx), float(d.cy)), False) >= 0
                    ]

                for det in detections:
                    det.wx, det.wy = hom.transform(det.cx, det.cy)
                    if gps_enabled:
                        det.lat, det.lon = hom.world_to_gps(det.wx, det.wy)
                    else:
                        det.lat = det.lon = None

                tracker.update(detections, frame_idx=frame_idx)

                if self.write_video:
                    annotated   = detector.annotate_frame(
                        frame, detections, tracker.get_pixel_trails(),
                        tracker.get_track_classes())
                    live_speeds = tracker.get_live_speeds()

                    # Draw ROI polygon on video
                    if roi_poly is not None:
                        pts = roi_poly.astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated, [pts], True, (124, 106, 247), 2)

                    for det in detections:
                        if det.track_id is None:
                            continue
                        if det.wx is not None:
                            cv2.putText(annotated,
                                        f"({det.wx:.1f}m,{det.wy:.1f}m)",
                                        (det.x1, det.y2+14),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        (200,200,200), 1)
                        speed = live_speeds.get(det.track_id)
                        if speed is not None:
                            sc = ((0,255,0) if speed<30 else
                                  (0,165,255) if speed<60 else (0,0,255))
                            lbl = f"{speed:.1f} km/h"
                            (tw,th),_ = cv2.getTextSize(
                                lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                            cv2.rectangle(annotated,
                                          (det.x1, det.y2+18),
                                          (det.x1+tw+4, det.y2+22+th),
                                          (30,30,30), -1)
                            cv2.putText(annotated, lbl,
                                        (det.x1+2, det.y2+18+th),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, sc, 1)
                    writer.write(annotated)

                for det in detections:
                    row = det.to_dict()
                    if gps_enabled:
                        row["latitude"]  = round(det.lat, 8) if det.lat else None
                        row["longitude"] = round(det.lon, 8) if det.lon else None
                    all_detections.append(row)

                frame_idx += 1
                pct = int(frame_idx / total * 85)
                self.progress.emit(pct)

            cap.release()
            if writer:
                writer.release()

            pd.DataFrame(all_detections).to_csv(
                output_dir / f"{stem}_detections.csv", index=False)
            summary_rows = tracker.summary()
            pd.DataFrame(summary_rows).to_csv(
                output_dir / f"{stem}_summary.csv", index=False)

            self.log.emit(f"[Done] Frames: {frame_idx} | Tracks: {tracker.total_tracks}")
            self.progress.emit(90)

            if gps_enabled:
                export_all(
                    summary=summary_rows,
                    gps_trail_rows=tracker.gps_trail_rows(),
                    output_dir=output_dir,
                    stem=stem,
                    skip_map=(not self.write_map),
                )
                self.log.emit("[Exports] GPS CSV, GeoJSON, KML saved.")

            self.progress.emit(100)
            self.finished.emit(str(output_dir))

        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Tab 1 — ROI Setup
# ---------------------------------------------------------------------------
class ROITab(QWidget):
    roi_saved = pyqtSignal(list, str)   # (roi_points, video_path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path = ""
        self._frame_count = 0
        self._current_frame = None
        self._cap = None
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(f"background:{DARK_BG};")
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # --- top controls ---
        ctrl = QHBoxLayout()

        self.btn_load = QPushButton("📂  Load Video")
        self.btn_load.setStyleSheet(BTN_STYLE)
        self.btn_load.clicked.connect(self._load_video)

        self.lbl_video = QLabel("No video loaded")
        self.lbl_video.setStyleSheet(LABEL_STYLE)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._seek_frame)
        self.slider.setStyleSheet(
            "QSlider::groove:horizontal { height:6px; background:#444; border-radius:3px; }"
            f"QSlider::handle:horizontal {{ background:{ACCENT}; width:14px; height:14px;"
            "  margin:-4px 0; border-radius:7px; }"
        )

        self.lbl_frame = QLabel("Frame: 0")
        self.lbl_frame.setStyleSheet(LABEL_STYLE)
        self.lbl_frame.setFixedWidth(80)

        ctrl.addWidget(self.btn_load)
        ctrl.addWidget(self.lbl_video, 1)
        ctrl.addWidget(QLabel("Frame:"))
        ctrl.addWidget(self.slider, 2)
        ctrl.addWidget(self.lbl_frame)
        root.addLayout(ctrl)

        # --- splitter: canvas left, instructions right ---
        splitter = QSplitter(Qt.Horizontal)

        self.canvas = ROICanvas()
        self.canvas.polygon_changed.connect(self._on_polygon_changed)
        splitter.addWidget(self.canvas)

        # right panel
        right = QWidget()
        right.setStyleSheet(f"background:{PANEL_BG}; border-radius:8px;")
        right.setFixedWidth(240)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(12, 12, 12, 12)
        rv.setSpacing(10)

        rv.addWidget(QLabel("ROI Controls"), )
        rv.addWidget(self._divider())

        self.btn_close_roi = QPushButton("✅  Close ROI Polygon")
        self.btn_close_roi.setStyleSheet(BTN_STYLE)
        self.btn_close_roi.clicked.connect(self.canvas.close_polygon)
        self.btn_close_roi.setEnabled(False)

        self.btn_clear_roi = QPushButton("Clear ROI")
        self.btn_clear_roi.setStyleSheet(
            "QPushButton { background:#c0392b; color:white; border-radius:6px;"
            "  padding:7px 16px; font-weight:bold; font-size:13px; }"
            "QPushButton:hover { background:#e74c3c; }"
            "QPushButton:disabled { background:#444; color:#777; }"
        )
        self.btn_clear_roi.clicked.connect(self.canvas.clear_roi)

        self.btn_save_roi = QPushButton("Save ROI & Continue")
        self.btn_save_roi.setStyleSheet(
            "QPushButton { background:#27ae60; color:white; border-radius:6px;"
            "  padding:7px 16px; font-weight:bold; font-size:13px; }"
            "QPushButton:hover { background:#2ecc71; }"
            "QPushButton:disabled { background:#444; color:#777; }"
        )
        self.btn_save_roi.clicked.connect(self._save_roi)
        self.btn_save_roi.setEnabled(False)

        rv.addWidget(self.btn_close_roi)
        rv.addWidget(self.btn_clear_roi)
        rv.addWidget(self._divider())

        self.lbl_points = QLabel("Points: 0")
        self.lbl_points.setStyleSheet(LABEL_STYLE)
        rv.addWidget(self.lbl_points)

        self.txt_coords = QTextEdit()
        self.txt_coords.setReadOnly(True)
        self.txt_coords.setStyleSheet(
            f"background:#1a1a2e; color:{TEXT_WHITE}; font-size:11px;"
            "border:1px solid #444; border-radius:4px;")
        self.txt_coords.setFixedHeight(160)
        rv.addWidget(self.txt_coords)

        rv.addStretch()
        rv.addWidget(self.btn_save_roi)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        root.addWidget(splitter, 1)

    def _divider(self):
        d = QFrame()
        d.setFrameShape(QFrame.HLine)
        d.setStyleSheet("color:#555;")
        return d

    def _load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", str(ROOT),
            "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        self._video_path = path
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(path)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setMaximum(max(self._frame_count - 1, 0))
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self.lbl_video.setText(Path(path).name)
        self._seek_frame(0)

    def _seek_frame(self, idx: int):
        if not self._cap:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if ret:
            self._current_frame = frame
            self.canvas.set_frame(frame)
            self.lbl_frame.setText(f"Frame: {idx}")

    def _on_polygon_changed(self, pts: list):
        self.lbl_points.setText(f"Points: {len(pts)}")
        self.btn_close_roi.setEnabled(len(pts) >= 3)
        self.btn_save_roi.setEnabled(len(pts) >= 3)
        lines = "\n".join(f"  P{i+1}: ({x}, {y})" for i, (x, y) in enumerate(pts))
        self.txt_coords.setText(lines if pts else "No points yet")

    def _save_roi(self):
        pts = self.canvas.roi_points
        if len(pts) < 3:
            QMessageBox.warning(self, "ROI", "Draw at least 3 points first.")
            return
        if not self.canvas._closed:
            self.canvas.close_polygon()
        self.roi_saved.emit(pts, self._video_path)
        QMessageBox.information(
            self, "ROI Saved",
            f"ROI saved with {len(pts)} points.\nSwitch to 'Run Analysis' tab.")

    def get_roi(self) -> list:
        return self.canvas.roi_points

    def get_video_path(self) -> str:
        return self._video_path


# ---------------------------------------------------------------------------
# Tab 2 — Run Analysis
# ---------------------------------------------------------------------------
class AnalysisTab(QWidget):
    analysis_done = pyqtSignal(str, str)   # output_dir, video_path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._roi_points: list = []
        self._worker: Optional[PipelineWorker] = None
        self._build_ui()

    def set_roi(self, pts: list, video_path: str = ""):
        self._roi_points = pts
        n = len(pts)
        self.lbl_roi_status.setText(
            f"✅  ROI active — {n} points" if n >= 3 else "⚠️  No ROI set (full frame)")
        if video_path:
            self.txt_video.setText(video_path)

    def _build_ui(self):
        self.setStyleSheet(f"background:{DARK_BG};")
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        # --- inputs group ---
        grp_in = QGroupBox("Inputs")
        grp_in.setStyleSheet(GROUP_STYLE)
        g = QGridLayout(grp_in)
        g.setSpacing(8)

        g.addWidget(QLabel("Video File:"), 0, 0)
        self.txt_video = QLabel("—")
        self.txt_video.setStyleSheet(f"color:{TEXT_WHITE};")
        btn_vid = QPushButton("Browse")
        btn_vid.setStyleSheet(BTN_STYLE)
        btn_vid.setFixedWidth(90)
        btn_vid.clicked.connect(self._browse_video)
        g.addWidget(self.txt_video, 0, 1)
        g.addWidget(btn_vid, 0, 2)

        g.addWidget(QLabel("Calibration JSON:"), 1, 0)
        self.txt_calib = QLabel("—")
        self.txt_calib.setStyleSheet(f"color:{TEXT_WHITE};")
        btn_cal = QPushButton("Browse")
        btn_cal.setStyleSheet(BTN_STYLE)
        btn_cal.setFixedWidth(90)
        btn_cal.clicked.connect(self._browse_calib)
        g.addWidget(self.txt_calib, 1, 1)
        g.addWidget(btn_cal, 1, 2)

        g.addWidget(QLabel("Output Dir:"), 2, 0)
        self.txt_output = QLabel(str(ROOT / "outputs"))
        self.txt_output.setStyleSheet(f"color:{TEXT_WHITE};")
        btn_out = QPushButton("Browse")
        btn_out.setStyleSheet(BTN_STYLE)
        btn_out.setFixedWidth(90)
        btn_out.clicked.connect(self._browse_output)
        g.addWidget(self.txt_output, 2, 1)
        g.addWidget(btn_out, 2, 2)

        for lbl in grp_in.findChildren(QLabel):
            if lbl.text() in ("Video File:", "Calibration JSON:", "Output Dir:"):
                lbl.setStyleSheet(LABEL_STYLE)

        root.addWidget(grp_in)

        # --- options group ---
        grp_opt = QGroupBox("Options")
        grp_opt.setStyleSheet(GROUP_STYLE)
        oh = QHBoxLayout(grp_opt)

        from PyQt5.QtWidgets import QCheckBox
        self.chk_video = QCheckBox("Write annotated video")
        self.chk_video.setChecked(True)
        self.chk_video.setStyleSheet(f"color:{TEXT_WHITE};")

        self.chk_map = QCheckBox("Generate static map (PNG)")
        self.chk_map.setChecked(True)
        self.chk_map.setStyleSheet(f"color:{TEXT_WHITE};")

        self.lbl_roi_status = QLabel("⚠️  No ROI set (full frame)")
        self.lbl_roi_status.setStyleSheet(f"color:#f39c12; font-size:13px;")

        oh.addWidget(self.chk_video)
        oh.addWidget(self.chk_map)
        oh.addStretch()
        oh.addWidget(self.lbl_roi_status)
        root.addWidget(grp_opt)

        # --- run button ---
        self.btn_run = QPushButton("▶   Run Analysis")
        self.btn_run.setStyleSheet(BTN_STYLE)
        self.btn_run.setMinimumHeight(42)
        self.btn_run.clicked.connect(self._run)
        root.addWidget(self.btn_run)

        # --- progress ---
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setStyleSheet(
            "QProgressBar { background:#333; border-radius:5px; color:white;"
            "  text-align:center; height:22px; }"
            "QProgressBar::chunk { background:" + ACCENT + "; border-radius:5px; }"
        )
        root.addWidget(self.progress)

        # --- log ---
        grp_log = QGroupBox("Log")
        grp_log.setStyleSheet(GROUP_STYLE)
        lv = QVBoxLayout(grp_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet(
            f"background:#111; color:#aaffaa; font-family:monospace;"
            "font-size:11px; border:none;")
        lv.addWidget(self.txt_log)
        root.addWidget(grp_log, 1)

    def _browse_video(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Video", str(ROOT),
            "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if p:
            self.txt_video.setText(p)

    def _browse_calib(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Calibration JSON", str(ROOT / "config"),
            "JSON Files (*.json)")
        if p:
            self.txt_calib.setText(p)

    def _browse_output(self):
        p = QFileDialog.getExistingDirectory(self, "Select Output Dir", str(ROOT))
        if p:
            self.txt_output.setText(p)

    def _run(self):
        video  = self.txt_video.text()
        calib  = self.txt_calib.text()
        output = self.txt_output.text()

        if not video or video == "—":
            QMessageBox.warning(self, "Missing", "Select a video file first.")
            return
        if not calib or calib == "—":
            QMessageBox.warning(self, "Missing", "Select a calibration JSON first.")
            return

        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.txt_log.clear()

        self._worker = PipelineWorker(
            video_path=video,
            calib_path=calib,
            output_dir=output,
            roi_points=self._roi_points,
            write_video=self.chk_video.isChecked(),
            write_map=self.chk_map.isChecked(),
        )
        self._worker.progress.connect(self.progress.setValue)
        self._worker.log.connect(lambda s: self.txt_log.append(s))
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, out_dir: str):
        self.btn_run.setEnabled(True)
        self.txt_log.append(f"\n✅  Done!  Outputs in: {out_dir}")
        self.analysis_done.emit(out_dir, self.txt_video.text())
        QMessageBox.information(
            self, "Analysis Complete",
            f"Pipeline finished.\nOutputs saved to:\n{out_dir}\n\n"
            "Switch to 'Trajectories' tab to explore results.")

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.txt_log.append(f"\n❌  ERROR:\n{msg}")
        QMessageBox.critical(self, "Pipeline Error", msg[:400])


# ---------------------------------------------------------------------------
# Tab 3 — Trajectory Viewer
# ---------------------------------------------------------------------------
class TrajectoryTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._summary: list[dict] = []
        self._gps_rows: list[dict] = []
        self._output_dir = ""
        self._video_path = ""
        self._background_path = ""
        self._build_ui()

    def load_output_dir(self, out_dir: str, video_path: str = ""):
        """Auto-called after analysis finishes."""
        self._output_dir = out_dir
        if video_path and video_path != "—":
            self._video_path = video_path
        # find gps_trails CSV and summary CSV
        d = Path(out_dir)
        gps_files = list(d.glob("*_gps_trails.csv"))
        sum_files  = list(d.glob("*_summary.csv"))
        if gps_files:
            self._load_gps_csv(str(gps_files[0]))
        if sum_files:
            self._load_summary_csv(str(sum_files[0]))

    def _build_ui(self):
        self.setStyleSheet(f"background:{DARK_BG};")
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ---- LEFT PANEL ----
        left = QWidget()
        left.setFixedWidth(260)
        left.setStyleSheet(f"background:{PANEL_BG}; border-radius:8px;")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(12, 12, 12, 12)
        lv.setSpacing(10)

        lv.addWidget(self._header("📂  Load Results"))

        self.btn_load_gps = QPushButton("Load GPS Trails CSV")
        self.btn_load_gps.setStyleSheet(BTN_STYLE)
        self.btn_load_gps.clicked.connect(self._browse_gps_csv)

        self.btn_load_sum = QPushButton("Load Summary CSV")
        self.btn_load_sum.setStyleSheet(BTN_STYLE)
        self.btn_load_sum.clicked.connect(self._browse_summary_csv)

        lv.addWidget(self.btn_load_gps)
        lv.addWidget(self.btn_load_sum)
        lv.addWidget(self._divider())

        lv.addWidget(self._header("🔍  Filter"))

        lv.addWidget(QLabel("Vehicle Class:"))
        self.cmb_class = QComboBox()
        self.cmb_class.setStyleSheet(COMBO_STYLE)
        self.cmb_class.addItem("All Classes")
        self.cmb_class.currentTextChanged.connect(self._refresh_track_list)
        lv.addWidget(self.cmb_class)

        lv.addWidget(QLabel("Track ID:"))
        self.cmb_track = QComboBox()
        self.cmb_track.setStyleSheet(COMBO_STYLE)
        self.cmb_track.addItem("— select —")
        self.cmb_track.currentTextChanged.connect(self._on_track_selected)
        lv.addWidget(self.cmb_track)

        lv.addWidget(self._divider())
        lv.addWidget(self._header("📊  Track Info"))

        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setStyleSheet(
            f"background:#1a1a2e; color:{TEXT_WHITE}; font-size:12px;"
            "border:1px solid #444; border-radius:4px;")
        self.info_box.setFixedHeight(180)
        lv.addWidget(self.info_box)

        lv.addWidget(self._divider())

        self.btn_show_all = QPushButton("🗺️  Show All Tracks")
        self.btn_show_all.setStyleSheet(BTN_STYLE)
        self.btn_show_all.clicked.connect(self._show_all_tracks)
        lv.addWidget(self.btn_show_all)

        self.btn_load_video = QPushButton("📹  Load Road Video")
        self.btn_load_video.setStyleSheet(BTN_STYLE)
        self.btn_load_video.clicked.connect(self._browse_road_video)
        lv.addWidget(self.btn_load_video)

        self.btn_road_view = QPushButton("🛣️  Show on Road Image")
        self.btn_road_view.setStyleSheet(BTN_STYLE)
        self.btn_road_view.clicked.connect(self._show_on_road_image)
        lv.addWidget(self.btn_road_view)

        lv.addStretch()

        # coordinate display
        lv.addWidget(self._header("📍  GPS Coordinates"))
        self.txt_gps = QTextEdit()
        self.txt_gps.setReadOnly(True)
        self.txt_gps.setStyleSheet(
            f"background:#1a1a2e; color:#aaffaa; font-size:10px;"
            "border:1px solid #444; border-radius:4px; font-family:monospace;")
        lv.addWidget(self.txt_gps)

        root.addWidget(left)

        # ---- RIGHT PANEL: map ----
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        if HAS_WEBENGINE and HAS_FOLIUM:
            self.map_view = QWebEngineView()
            self.map_view.setStyleSheet("border-radius:8px;")
            rv.addWidget(self.map_view)
            self._show_placeholder_map()
        else:
            # Fallback: matplotlib canvas
            self._build_mpl_fallback(rv)

        root.addWidget(right, 1)

    def _header(self, text: str) -> QLabel:
        l = QLabel(text)
        l.setStyleSheet(HEADER_STYLE)
        return l

    def _divider(self) -> QFrame:
        d = QFrame()
        d.setFrameShape(QFrame.HLine)
        d.setStyleSheet("color:#555;")
        return d

    # --- data loading ------------------------------------------------------

    def _browse_gps_csv(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select GPS Trails CSV", self._output_dir or str(ROOT / "outputs"),
            "CSV Files (*_gps_trails.csv *.csv)")
        if p:
            self._load_gps_csv(p)

    def _browse_summary_csv(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Summary CSV", self._output_dir or str(ROOT / "outputs"),
            "CSV Files (*_summary.csv *.csv)")
        if p:
            self._load_summary_csv(p)

    def _browse_road_video(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Source Video", str(ROOT),
            "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if p:
            self._video_path = p
            self._background_path = ""
            self.info_box.setText(f"Road video loaded:\n{Path(p).name}")

    def _load_gps_csv(self, path: str):
        import csv
        rows = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        self._gps_rows = rows
        self._populate_dropdowns()
        self.info_box.setText(f"Loaded {len(rows)} GPS points from:\n{Path(path).name}")

    def _load_summary_csv(self, path: str):
        import csv
        rows = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        self._summary = rows
        self._populate_dropdowns()

    def _populate_dropdowns(self):
        # Collect classes and track IDs from gps_rows
        classes = sorted(set(r.get("class_name", "unknown")
                             for r in self._gps_rows if r.get("class_name")))
        self.cmb_class.blockSignals(True)
        self.cmb_class.clear()
        self.cmb_class.addItem("All Classes")
        for c in classes:
            self.cmb_class.addItem(c)
        self.cmb_class.blockSignals(False)
        self._refresh_track_list()

    def _refresh_track_list(self):
        sel_class = self.cmb_class.currentText()
        if sel_class == "All Classes":
            ids = sorted(set(r["track_id"] for r in self._gps_rows if r.get("track_id")),
                         key=lambda x: int(x) if str(x).isdigit() else 0)
        else:
            ids = sorted(set(r["track_id"] for r in self._gps_rows
                             if r.get("class_name") == sel_class),
                         key=lambda x: int(x) if str(x).isdigit() else 0)
        self.cmb_track.blockSignals(True)
        self.cmb_track.clear()
        self.cmb_track.addItem("— select —")
        for tid in ids:
            self.cmb_track.addItem(str(tid))
        self.cmb_track.blockSignals(False)

    # --- track selection ---------------------------------------------------

    def _on_track_selected(self, val: str):
        if val == "— select —" or not val:
            return
        try:
            tid = int(val)
        except ValueError:
            return
        if self._video_path and self._has_pixel_tracks():
            self._display_track(tid, render_map=False)
            self._show_on_road_image(show_warnings=False)
        else:
            self._display_track(tid)

    def _display_track(self, track_id: int, render_map: bool = True):
        # Get GPS points for this track
        pts = [(float(r["latitude"]), float(r["longitude"]))
               for r in self._gps_rows
               if str(r.get("track_id")) == str(track_id)
               and r.get("latitude") and r.get("longitude")
               and r["latitude"] != "" and r["longitude"] != ""]

        if not pts:
            self.info_box.setText(f"Track {track_id}: No GPS data available.")
            return

        # Get summary info
        summary_row = next(
            (r for r in self._summary if str(r.get("track_id")) == str(track_id)),
            None)

        cls_name = next(
            (r.get("class_name", "unknown") for r in self._gps_rows
             if str(r.get("track_id")) == str(track_id)), "unknown")

        # Info card
        info = [f"Track ID : {track_id}", f"Class    : {cls_name}",
                f"Points   : {len(pts)}",
                f"Start    : {pts[0][0]:.7f}, {pts[0][1]:.7f}",
                f"End      : {pts[-1][0]:.7f}, {pts[-1][1]:.7f}"]
        if summary_row:
            info += [
                f"Duration : {summary_row.get('duration_sec','—')} s",
                f"Distance : {summary_row.get('world_distance_m','—')} m",
                f"Speed    : {summary_row.get('world_velocity_kmph','—')} km/h",
            ]
        self.info_box.setText("\n".join(info))

        # GPS coordinate list
        gps_lines = "\n".join(
            f"F{r.get('frame','?'):>4}  {float(r['latitude']):.7f}, {float(r['longitude']):.7f}"
            for r in self._gps_rows
            if str(r.get("track_id")) == str(track_id)
            and r.get("latitude") and r.get("longitude")
            and r["latitude"] != ""
        )
        self.txt_gps.setText(gps_lines)

        if not render_map:
            return

        # Draw map
        color = CLASS_COLORS.get(cls_name, "#FFFFFF")
        if HAS_WEBENGINE and HAS_FOLIUM:
            self._render_folium_single(pts, track_id, cls_name, color)
        else:
            self._render_mpl_single(pts, track_id, cls_name, color)

    def _show_all_tracks(self):
        if not self._gps_rows:
            QMessageBox.information(self, "No data", "Load a GPS trails CSV first.")
            return

        if self._video_path and self._has_pixel_tracks():
            self.cmb_track.blockSignals(True)
            self.cmb_track.setCurrentIndex(0)
            self.cmb_track.blockSignals(False)
            self._show_on_road_image()
            return

        # Group by track_id
        tracks: dict[str, list] = {}
        for r in self._gps_rows:
            tid = str(r.get("track_id", ""))
            if not tid:
                continue
            if r.get("latitude") and r.get("longitude") \
               and r["latitude"] != "" and r["longitude"] != "":
                tracks.setdefault(tid, []).append(
                    (float(r["latitude"]), float(r["longitude"])))

        if HAS_WEBENGINE and HAS_FOLIUM:
            self._render_folium_all(tracks)
        else:
            self._render_mpl_all(tracks)
        self.info_box.setText(f"Showing all {len(tracks)} tracks.")

    def _show_on_road_image(self, show_warnings: bool = True):
        if not self._gps_rows:
            if show_warnings:
                QMessageBox.information(self, "No data", "Load a GPS trails CSV first.")
            return
        if not self._video_path:
            if show_warnings:
                QMessageBox.warning(
                    self, "No Video",
                    "Load the source road video first, or run analysis from this UI.")
            return

        tracks, cls_map, _ = self._pixel_tracks()
        if not tracks:
            if show_warnings:
                QMessageBox.warning(
                    self, "No Pixel Data",
                    "The loaded GPS trails CSV must contain cx and cy columns.")
            return

        selected = self.cmb_track.currentText()
        if selected not in tracks:
            selected = ""

        try:
            frame = self._get_road_background()
        except RuntimeError as exc:
            if show_warnings:
                QMessageBox.warning(self, "Background Error", str(exc))
            return

        draw_ids = [selected] if selected else list(tracks.keys())
        for tid in draw_ids:
            pts = tracks.get(tid, [])
            if len(pts) < 2:
                continue
            cls_name = cls_map.get(tid, "unknown")
            color = (255, 0, 255) if selected else CLASS_COLORS_BGR.get(
                cls_name, CLASS_COLORS_BGR["unknown"])
            thickness = 4 if selected else 2
            arr = np.array(pts, np.int32)
            cv2.polylines(frame, [arr], False, color, thickness)
            for i, pt in enumerate(arr):
                if i % 5 == 0:
                    cv2.circle(frame, tuple(pt), 3, color, -1)
            cv2.circle(frame, tuple(arr[0]), 7, (0, 255, 0), -1)
            cv2.circle(frame, tuple(arr[-1]), 7, (0, 0, 255), -1)
            mid = arr[len(arr) // 2]
            cv2.putText(
                frame, f"ID{tid} {cls_name}", tuple(mid),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        out_dir = Path(self._output_dir) if self._output_dir else ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_track_{selected}" if selected else "_all_tracks"
        out_path = out_dir / f"road_trajectory_overlay{suffix}.png"
        cv2.imwrite(str(out_path), frame)

        self._display_road_overlay(frame, out_path)
        track_msg = (
            f"Track shown: ID{selected}"
            if selected else f"Tracks shown: {len(draw_ids)}"
        )
        self.info_box.setText(
            f"Road overlay shown.\nSaved to:\n{out_path}\n"
            f"Background:\n{self._background_path}\n"
            "Green=start  Red=end  Magenta=selected track\n"
            f"{track_msg}")

    def _has_pixel_tracks(self) -> bool:
        return any(
            r.get("track_id") and r.get("cx") and r.get("cy")
            for r in self._gps_rows
        )

    def _get_road_background(self) -> np.ndarray:
        out_dir = Path(self._output_dir) if self._output_dir else ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        bg_path = self._cached_background_path(out_dir)
        if bg_path.exists():
            bg = cv2.imread(str(bg_path))
            if bg is not None:
                self._background_path = str(bg_path)
                return bg

        self.info_box.setText(
            "Extracting clean road background from video...\n"
            "This runs once per video, then reuses the saved image.")
        QApplication.processEvents()

        bg = self._extract_median_background(self._video_path, sample_count=45)
        cv2.imwrite(str(bg_path), bg)
        self._background_path = str(bg_path)
        return bg.copy()

    def _cached_background_path(self, out_dir: Path) -> Path:
        video = Path(self._video_path)
        try:
            stat = video.stat()
            key = f"{video.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        except OSError:
            key = str(video)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        return out_dir / f"{video.stem}_median_background_{digest}.jpg"

    def _extract_median_background(self, video_path: str, sample_count: int = 45) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video:\n{video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            cap.release()
            raise RuntimeError("Could not read frame count from the video.")

        indices = np.linspace(0, max(0, total - 1), min(sample_count, total), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()

        if not frames:
            raise RuntimeError("Could not read frames from the video.")
        if len(frames) == 1:
            return frames[0].copy()

        return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)

    def _pixel_tracks(self) -> tuple[dict[str, list], dict[str, str], dict[str, list]]:
        tracks: dict[str, list] = {}
        cls_map: dict[str, str] = {}
        frame_map: dict[str, list] = {}

        def as_int(value):
            return int(float(value))

        sorted_rows = sorted(
            self._gps_rows,
            key=lambda r: (
                int(float(r.get("track_id", 0))) if str(r.get("track_id", "")).isdigit() else 0,
                int(float(r.get("frame", 0))) if str(r.get("frame", "")).replace(".", "", 1).isdigit() else 0,
            ),
        )

        for r in sorted_rows:
            tid = str(r.get("track_id", "")).strip()
            if not tid or not r.get("cx") or not r.get("cy"):
                continue
            try:
                point = [as_int(r["cx"]), as_int(r["cy"])]
                frame = as_int(r.get("frame", 0) or 0)
            except (TypeError, ValueError):
                continue
            tracks.setdefault(tid, []).append(point)
            frame_map.setdefault(tid, []).append(frame)
            cls_map[tid] = str(r.get("class_name", "unknown") or "unknown")
        return tracks, cls_map, frame_map

    def _display_road_overlay(self, frame: np.ndarray, out_path: Path):
        if HAS_WEBENGINE and HAS_FOLIUM:
            from PyQt5.QtCore import QUrl
            html_path = out_path.with_suffix(".html")
            image_url = QUrl.fromLocalFile(str(out_path)).toString()
            html = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<style>"
                "html,body{margin:0;width:100%;height:100%;background:#111;overflow:hidden;}"
                "body{display:flex;align-items:center;justify-content:center;}"
                "img{max-width:100vw;max-height:100vh;width:auto;height:auto;object-fit:contain;}"
                "</style></head><body>"
                f"<img src='{image_url}' alt='Road trajectory overlay'>"
                "</body></html>"
            )
            html_path.write_text(html, encoding="utf-8")
            self.map_view.load(QUrl.fromLocalFile(str(html_path)))
        elif getattr(self, "_has_mpl", False):
            self._ax.clear()
            self._ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self._ax.axis("off")
            self._ax.set_title("Road Trajectory Overlay", color="white")
            self._mpl_canvas.draw()

    # --- Folium rendering --------------------------------------------------

    def _show_placeholder_map(self):
        html = (
            f"<html><body style='background:{DARK_BG};display:flex;"
            "align-items:center;justify-content:center;height:100vh;'>"
            f"<p style='color:{TEXT_GRAY};font-family:Arial;font-size:16px;'>"
            "Load GPS trail data and select a Track ID to view trajectory</p>"
            "</body></html>"
        )
        self.map_view.setHtml(html)

    def _render_folium_single(self, pts, track_id, cls_name, color):
        center = [sum(p[0] for p in pts)/len(pts),
                  sum(p[1] for p in pts)/len(pts)]
        m = folium.Map(location=center, zoom_start=18,
                       tiles="CartoDB dark_matter")

        folium.PolyLine(
            locations=pts,
            color=color, weight=3, opacity=0.9,
            tooltip=f"Track {track_id} ({cls_name})"
        ).add_to(m)

        # Start marker
        folium.CircleMarker(
            pts[0], radius=8, color=color, fill=True,
            fill_color="#00ff00",
            popup=folium.Popup(f"<b>Start</b><br>Track {track_id}<br>"
                               f"{pts[0][0]:.7f}, {pts[0][1]:.7f}", max_width=200)
        ).add_to(m)

        # End marker
        folium.Marker(
            pts[-1],
            icon=folium.Icon(color="red", icon="flag"),
            popup=folium.Popup(f"<b>End</b><br>Track {track_id}<br>"
                               f"{pts[-1][0]:.7f}, {pts[-1][1]:.7f}", max_width=200)
        ).add_to(m)

        # Intermediate dots every 10 points
        for i in range(0, len(pts), max(1, len(pts)//10)):
            folium.CircleMarker(
                pts[i], radius=3, color=color, fill=True,
                popup=f"Frame point {i}"
            ).add_to(m)

        self._set_folium_map(m)

    def _render_folium_all(self, tracks: dict[str, list]):
        # Find center
        all_pts = [p for pts in tracks.values() for p in pts]
        if not all_pts:
            return
        clat = sum(p[0] for p in all_pts) / len(all_pts)
        clon = sum(p[1] for p in all_pts) / len(all_pts)
        m = folium.Map(location=[clat, clon], zoom_start=17,
                       tiles="CartoDB dark_matter")

        # Get class per track_id
        cls_map = {str(r.get("track_id")): r.get("class_name", "unknown")
                   for r in self._gps_rows}

        for tid, pts in tracks.items():
            if len(pts) < 2:
                continue
            cls_name = cls_map.get(str(tid), "unknown")
            color = CLASS_COLORS.get(cls_name, "#FFFFFF")
            folium.PolyLine(
                locations=pts, color=color, weight=2,
                opacity=0.8, tooltip=f"ID {tid} ({cls_name})"
            ).add_to(m)
            folium.CircleMarker(
                pts[0], radius=4, color=color, fill=True
            ).add_to(m)

        self._set_folium_map(m)

    def _set_folium_map(self, m):
        with tempfile.NamedTemporaryFile(
                suffix=".html", delete=False, mode="w") as f:
            m.save(f.name)
            self.map_view.load(
                __import__("PyQt5.QtCore", fromlist=["QUrl"]).QUrl.fromLocalFile(f.name))

    # --- Matplotlib fallback (no WebEngine) --------------------------------

    def _build_mpl_fallback(self, layout):
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self._fig = Figure(facecolor=DARK_BG)
            self._ax  = self._fig.add_subplot(111)
            self._ax.set_facecolor(DARK_BG)
            self._mpl_canvas = FigureCanvasQTAgg(self._fig)
            layout.addWidget(self._mpl_canvas)
            self._has_mpl = True
        except ImportError:
            lbl = QLabel("Install PyQtWebEngine + folium\nor matplotlib for map view")
            lbl.setStyleSheet(f"color:{TEXT_GRAY}; font-size:14px;")
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl)
            self._has_mpl = False

    def _render_mpl_single(self, pts, track_id, cls_name, color):
        if not getattr(self, "_has_mpl", False):
            return
        self._ax.clear()
        self._ax.set_facecolor(DARK_BG)
        lats = [p[0] for p in pts]
        lons = [p[1] for p in pts]
        mpl_color = CLASS_COLORS.get(cls_name, "white")
        self._ax.plot(lons, lats, color=mpl_color, linewidth=2)
        self._ax.scatter(lons[0],  lats[0],  color="lime",  s=60, zorder=5, label="Start")
        self._ax.scatter(lons[-1], lats[-1], color="red",   s=60, marker="^",
                         zorder=5, label="End")
        self._ax.set_title(f"Track {track_id} ({cls_name})", color="white")
        self._ax.tick_params(colors="gray", labelsize=7)
        self._ax.set_xlabel("Longitude", color="gray", fontsize=8)
        self._ax.set_ylabel("Latitude",  color="gray", fontsize=8)
        self._ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")
        self._mpl_canvas.draw()

    def _render_mpl_all(self, tracks: dict[str, list]):
        if not getattr(self, "_has_mpl", False):
            return
        cls_map = {str(r.get("track_id")): r.get("class_name","unknown")
                   for r in self._gps_rows}
        self._ax.clear()
        self._ax.set_facecolor(DARK_BG)
        for tid, pts in tracks.items():
            if len(pts) < 2:
                continue
            cls_name = cls_map.get(str(tid), "unknown")
            lats = [p[0] for p in pts]
            lons = [p[1] for p in pts]
            self._ax.plot(lons, lats,
                          color=CLASS_COLORS.get(cls_name, "white"),
                          linewidth=1.2, alpha=0.8)
        self._ax.set_title("All Tracks", color="white")
        self._ax.tick_params(colors="gray", labelsize=7)
        self._mpl_canvas.draw()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aerial Traffic Analytics")
        self.setMinimumSize(1200, 750)
        self.setStyleSheet(f"background:{DARK_BG};")

        tabs = QTabWidget()
        tab_style = (
            "QTabWidget::pane { border:none; background:" + DARK_BG + "; }"
            "QTabBar::tab { background:" + PANEL_BG + "; color:" + TEXT_GRAY + ";"
            "  padding:10px 22px; font-size:13px; border-radius:4px; margin:2px; }"
            "QTabBar::tab:selected { background:" + ACCENT + "; color:white; font-weight:bold; }"
        )
        tabs.setStyleSheet(tab_style)

        self.tab_roi      = ROITab()
        self.tab_analysis = AnalysisTab()
        self.tab_traj     = TrajectoryTab()

        tabs.addTab(self.tab_roi,      "  ROI Setup  ")
        tabs.addTab(self.tab_analysis, "  Run Analysis  ")
        tabs.addTab(self.tab_traj,     "  Trajectories  ")

        # Wire signals
        self.tab_roi.roi_saved.connect(
            lambda pts, vid: self.tab_analysis.set_roi(pts, vid))
        self.tab_analysis.analysis_done.connect(
            lambda out, vid: (self.tab_traj.load_output_dir(out, vid), tabs.setCurrentIndex(2)))

        self.setCentralWidget(tabs)

        # Status bar
        self.statusBar().setStyleSheet(f"color:{TEXT_GRAY}; background:{PANEL_BG};")
        self.statusBar().showMessage(
            "Ready  |  Tab 1: Draw ROI → Tab 2: Run Analysis → Tab 3: View Trajectories")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
