#!/usr/bin/env python
"""
interfaces/calibration_generator_ui.py
======================================
PyQt5 utility to generate calibration JSON configurations.
Users can input:
  - Camera ID and Video Path
  - Origin GPS coordinates
  - Calibration points (pixel coords and GPS coords)
The tool calculates real-world world [wx, wy] meter coordinates relative to origin_gps
and generates the calibration JSON config layout dynamically.

Run:
  python interfaces/calibration_generator_ui.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QTextEdit, QHeaderView, QMessageBox, QFileDialog, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

# ---------------------------------------------------------------------------
# Styling Configurations (Matching Rest of the Application)
# ---------------------------------------------------------------------------
DARK_BG      = "#1e1e2e"
PANEL_BG     = "#2a2a3e"
ACCENT       = "#7c6af7"
TEXT_WHITE   = "#e0e0e0"
TEXT_GRAY    = "#888888"
INPUT_BG     = "#1b1b2a"
BORDER_COLOR = "#44445c"

BTN_STYLE = (
    "QPushButton {"
    "  background:" + ACCENT + "; color:white; border-radius:6px;"
    "  padding:8px 16px; font-weight:bold; font-size:13px; border: none;"
    "}"
    "QPushButton:hover { background:#9d8df8; }"
    "QPushButton:pressed { background:#5a49d8; }"
    "QPushButton:disabled { background:#444; color:#777; }"
)

DANGER_BTN_STYLE = (
    "QPushButton {"
    "  background:#d9534f; color:white; border-radius:6px;"
    "  padding:8px 16px; font-weight:bold; font-size:13px; border: none;"
    "}"
    "QPushButton:hover { background:#e27c79; }"
    "QPushButton:pressed { background:#c9302c; }"
)

EDIT_STYLE = (
    "QLineEdit {"
    "  background:" + INPUT_BG + "; color:" + TEXT_WHITE + ";"
    "  border: 1px solid " + BORDER_COLOR + "; border-radius:4px;"
    "  padding: 6px; font-size: 13px;"
    "}"
    "QLineEdit:focus { border: 1px solid " + ACCENT + "; }"
)

TABLE_STYLE = (
    "QTableWidget {"
    "  background:" + PANEL_BG + "; color:" + TEXT_WHITE + ";"
    "  gridline-color: " + BORDER_COLOR + "; border: 1px solid " + BORDER_COLOR + ";"
    "  border-radius: 4px;"
    "}"
    "QTableWidget::item:selected { background:" + ACCENT + "; color: white; }"
    "QHeaderView::section {"
    "  background:#1b1b2a; color:" + TEXT_WHITE + "; padding: 5px;"
    "  border: 1px solid " + BORDER_COLOR + ";"
    "}"
)

TEXT_EDIT_STYLE = (
    "QTextEdit {"
    "  background:" + INPUT_BG + "; color:#00ff66; font-family:'Consolas', 'Courier New', monospace;"
    "  border: 1px solid " + BORDER_COLOR + "; border-radius: 4px; padding: 10px; font-size: 13px;"
    "}"
)

GROUP_STYLE = (
    "QGroupBox { color:" + TEXT_WHITE + "; font-weight:bold; font-size:13px;"
    "  border:1px solid " + BORDER_COLOR + "; border-radius:6px; margin-top:12px; padding-top:16px; }"
    "QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }"
)

LABEL_STYLE = "color: " + TEXT_WHITE + "; font-size: 13px;"
HEADER_STYLE = "color: " + TEXT_WHITE + "; font-size: 16px; font-weight: bold;"


# ---------------------------------------------------------------------------
# Geodetic Calculations (Standard Equations)
# ---------------------------------------------------------------------------
def get_meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """
    Standard geodetic series expansion formulas to calculate
    meters per degree of latitude and longitude at a given latitude.
    """
    lat = math.radians(lat_deg)
    m_per_deg_lat = (
        111132.92
        - 559.82 * math.cos(2 * lat)
        + 1.175 * math.cos(4 * lat)
        - 0.0023 * math.cos(6 * lat)
    )
    m_per_deg_lon = (
        111412.84 * math.cos(lat)
        - 93.5 * math.cos(3 * lat)
        + 0.118 * math.cos(5 * lat)
    )
    return m_per_deg_lat, m_per_deg_lon


def safe_float(val: str, default=None) -> float | None:
    try:
        return float(val.strip())
    except (ValueError, TypeError):
        return default


def safe_int(val: str, default=0) -> int:
    try:
        return int(val.strip())
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Main PyQt5 Interface Window
# ---------------------------------------------------------------------------
class CalibrationGeneratorWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Calibration JSON Generator")
        self.setMinimumSize(1200, 750)
        self.setStyleSheet(f"background:{DARK_BG};")

        # Main Widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # -------------------------------------------------------------------
        # Left Panel (Inputs and Points Editor)
        # -------------------------------------------------------------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # 1. Metadata Group Box
        meta_group = QGroupBox("Camera & Video Metadata")
        meta_group.setStyleSheet(GROUP_STYLE)
        meta_layout = QGridLayout(meta_group)
        meta_layout.setSpacing(8)

        lbl_cam = QLabel("Camera ID:")
        lbl_cam.setStyleSheet(LABEL_STYLE)
        self.txt_cam_id = QLineEdit("cam1")
        self.txt_cam_id.setStyleSheet(EDIT_STYLE)
        self.txt_cam_id.textChanged.connect(self.recalculate_and_update)

        lbl_video = QLabel("Video Path:")
        lbl_video.setStyleSheet(LABEL_STYLE)
        self.txt_video_path = QLineEdit("assets/sample_video/drone_3.mp4")
        self.txt_video_path.setStyleSheet(EDIT_STYLE)
        self.txt_video_path.textChanged.connect(self.recalculate_and_update)

        lbl_org_lon = QLabel("Origin Longitude:")
        lbl_org_lon.setStyleSheet(LABEL_STYLE)
        self.txt_origin_lon = QLineEdit("80.5058750")
        self.txt_origin_lon.setStyleSheet(EDIT_STYLE)
        self.txt_origin_lon.textChanged.connect(self.recalculate_and_update)

        lbl_org_lat = QLabel("Origin Latitude:")
        lbl_org_lat.setStyleSheet(LABEL_STYLE)
        self.txt_origin_lat = QLineEdit("16.3683500")
        self.txt_origin_lat.setStyleSheet(EDIT_STYLE)
        self.txt_origin_lat.textChanged.connect(self.recalculate_and_update)

        meta_layout.addWidget(lbl_cam, 0, 0)
        meta_layout.addWidget(self.txt_cam_id, 0, 1)
        meta_layout.addWidget(lbl_video, 0, 2)
        meta_layout.addWidget(self.txt_video_path, 0, 3)
        meta_layout.addWidget(lbl_org_lon, 1, 0)
        meta_layout.addWidget(self.txt_origin_lon, 1, 1)
        meta_layout.addWidget(lbl_org_lat, 1, 2)
        meta_layout.addWidget(self.txt_origin_lat, 1, 3)

        left_layout.addWidget(meta_group)

        # 2. Points Editor Group Box
        points_group = QGroupBox("Calibration Points")
        points_group.setStyleSheet(GROUP_STYLE)
        points_layout = QVBoxLayout(points_group)
        points_layout.setContentsMargins(10, 15, 10, 10)
        points_layout.setSpacing(10)

        # Table Widget
        self.table = QTableWidget(0, 7)
        self.table.setStyleSheet(TABLE_STYLE)
        self.table.setHorizontalHeaderLabels([
            "Name", "Pixel X", "Pixel Y", "GPS Longitude", "GPS Latitude", "World X (m)", "World Y (m)"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.itemChanged.connect(self.recalculate_and_update)
        points_layout.addWidget(self.table)

        # Buttons Panel (under Table)
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.btn_add = QPushButton("+ Add Point")
        self.btn_add.setStyleSheet(BTN_STYLE)
        self.btn_add.clicked.connect(self.add_point)

        self.btn_remove = QPushButton("- Remove Point")
        self.btn_remove.setStyleSheet(DANGER_BTN_STYLE)
        self.btn_remove.clicked.connect(self.remove_selected_point)

        self.btn_mark_origin = QPushButton("Set Selected as Origin")
        self.btn_mark_origin.setStyleSheet(BTN_STYLE)
        self.btn_mark_origin.clicked.connect(self.set_selected_as_origin)

        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.setStyleSheet(DANGER_BTN_STYLE)
        self.btn_clear.clicked.connect(self.clear_all_points)

        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_mark_origin)
        btn_layout.addWidget(self.btn_clear)
        points_layout.addLayout(btn_layout)

        left_layout.addWidget(points_group)
        main_layout.addWidget(left_widget, stretch=3)

        # -------------------------------------------------------------------
        # Right Panel (Live JSON Output and Copy/Load functions)
        # -------------------------------------------------------------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        lbl_json_header = QLabel("Generated Calibration JSON")
        lbl_json_header.setStyleSheet(HEADER_STYLE)
        right_layout.addWidget(lbl_json_header)

        self.txt_json = QTextEdit()
        self.txt_json.setStyleSheet(TEXT_EDIT_STYLE)
        self.txt_json.setReadOnly(True)
        right_layout.addWidget(self.txt_json)

        right_btn_layout = QHBoxLayout()
        right_btn_layout.setSpacing(8)

        self.btn_copy = QPushButton("📋 Copy JSON")
        self.btn_copy.setStyleSheet(BTN_STYLE)
        self.btn_copy.clicked.connect(self.copy_to_clipboard)

        self.btn_load = QPushButton("📂 Load Existing JSON...")
        self.btn_load.setStyleSheet(BTN_STYLE)
        self.btn_load.clicked.connect(self.load_existing_json)

        self.btn_save = QPushButton("💾 Save JSON File...")
        self.btn_save.setStyleSheet(BTN_STYLE)
        self.btn_save.clicked.connect(self.save_json_file)

        right_btn_layout.addWidget(self.btn_copy)
        right_btn_layout.addWidget(self.btn_load)
        right_btn_layout.addWidget(self.btn_save)
        right_layout.addLayout(right_btn_layout)

        main_layout.addWidget(right_widget, stretch=2)

        # Populate with some default points to showcase standard inputs
        self.add_default_sample_points()

        # Initial calculation
        self.recalculate_and_update()

    # -----------------------------------------------------------------------
    # Table Operations
    # -----------------------------------------------------------------------
    def add_point(self, name="", px=0, py=0, lon=0.0, lat=0.0):
        row = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(row)

        # Determine default point name if empty
        if not name:
            name = f"C{row + 1}"

        # Populate columns
        self.table.setItem(row, 0, QTableWidgetItem(str(name)))
        self.table.setItem(row, 1, QTableWidgetItem(str(px)))
        self.table.setItem(row, 2, QTableWidgetItem(str(py)))
        self.table.setItem(row, 3, QTableWidgetItem(str(lon)))
        self.table.setItem(row, 4, QTableWidgetItem(str(lat)))

        # World coordinates (Calculated dynamically, marked Read-Only)
        wx_item = QTableWidgetItem("0.00")
        wx_item.setFlags(wx_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(row, 5, wx_item)

        wy_item = QTableWidgetItem("0.00")
        wy_item.setFlags(wy_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(row, 6, wy_item)

        self.table.blockSignals(False)
        self.recalculate_and_update()

    def remove_selected_point(self):
        cur_row = self.table.currentRow()
        if cur_row < 0:
            QMessageBox.warning(self, "Selection Required", "Please click on a row to remove it.")
            return
        self.table.removeRow(cur_row)
        self.recalculate_and_update()

    def clear_all_points(self):
        self.table.setRowCount(0)
        self.recalculate_and_update()

    def set_selected_as_origin(self):
        cur_row = self.table.currentRow()
        if cur_row < 0:
            QMessageBox.warning(self, "Selection Required", "Please click on a row to set as origin.")
            return

        lon_item = self.table.item(cur_row, 3)
        lat_item = self.table.item(cur_row, 4)

        if lon_item and lat_item:
            self.txt_origin_lon.setText(lon_item.text())
            self.txt_origin_lat.setText(lat_item.text())
            self.recalculate_and_update()

    def add_default_sample_points(self):
        # Sample points matching config/calibration_cam1.json to populate cleanly on launch
        self.add_point("C8", 943, 257, 80.5058750, 16.3683500)
        self.add_point("C5", 1504, 265, 80.5061616, 16.3685303)
        self.add_point("C9", 945, 630, 80.5060056, 16.3681639)
        self.add_point("C4", 1521, 638, 80.5062840, 16.3683418)

    # -----------------------------------------------------------------------
    # Calculations & Live Updates
    # -----------------------------------------------------------------------
    def recalculate_and_update(self):
        origin_lon = safe_float(self.txt_origin_lon.text())
        origin_lat = safe_float(self.txt_origin_lat.text())

        gps_enabled = (origin_lon is not None and origin_lat is not None)
        if gps_enabled:
            m_per_deg_lat, m_per_deg_lon = get_meters_per_degree(origin_lat)
        else:
            m_per_deg_lat = m_per_deg_lon = 0.0

        points_list = []
        self.table.blockSignals(True)

        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            name = name_item.text().strip() if name_item else f"P{row+1}"

            px_item = self.table.item(row, 1)
            px = safe_int(px_item.text() if px_item else "0")

            py_item = self.table.item(row, 2)
            py = safe_int(py_item.text() if py_item else "0")

            lon_item = self.table.item(row, 3)
            lon = safe_float(lon_item.text() if lon_item else "0.0") or 0.0

            lat_item = self.table.item(row, 4)
            lat = safe_float(lat_item.text() if lat_item else "0.0") or 0.0

            if gps_enabled:
                wx = (lon - origin_lon) * m_per_deg_lon
                wy = (lat - origin_lat) * m_per_deg_lat
            else:
                wx = 0.0
                wy = 0.0

            # Set calculated world values in table columns (Read-Only)
            self.table.setItem(row, 5, QTableWidgetItem(f"{wx:.2f}"))
            self.table.item(row, 5).setFlags(self.table.item(row, 5).flags() & ~Qt.ItemIsEditable)

            self.table.setItem(row, 6, QTableWidgetItem(f"{wy:.2f}"))
            self.table.item(row, 6).setFlags(self.table.item(row, 6).flags() & ~Qt.ItemIsEditable)

            points_list.append({
                "name": name,
                "pixel": [px, py],
                "world": [round(wx, 2), round(wy, 2)],
                "gps": [lon, lat]
            })

        self.table.blockSignals(False)

        # Generate JSON representation
        calib_data = {
            "camera_id": self.txt_cam_id.text().strip(),
            "video": self.txt_video_path.text().strip(),
            "origin_gps": [origin_lon, origin_lat] if gps_enabled else None,
            "calibration_points": points_list
        }

        self.txt_json.setText(json.dumps(calib_data, indent=2))

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------
    def copy_to_clipboard(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.txt_json.toPlainText())
        # Display a small, clean success message in the status bar
        self.statusBar().showMessage("Copied JSON configuration to clipboard!", 3000)

    def load_existing_json(self):
        options = QFileDialog.Options()
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration JSON", "", "JSON Files (*.json);;All Files (*)", options=options
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.txt_cam_id.setText(data.get("camera_id", "cam1"))
            self.txt_video_path.setText(data.get("video", ""))

            origin = data.get("origin_gps")
            if origin and len(origin) == 2:
                self.txt_origin_lon.setText(str(origin[0]))
                self.txt_origin_lat.setText(str(origin[1]))
            else:
                self.txt_origin_lon.clear()
                self.txt_origin_lat.clear()

            self.table.blockSignals(True)
            self.table.setRowCount(0)

            pts = data.get("calibration_points", [])
            for p in pts:
                row = self.table.rowCount()
                self.table.insertRow(row)

                self.table.setItem(row, 0, QTableWidgetItem(str(p.get("name", f"P{row+1}"))))

                pixel = p.get("pixel", [0, 0])
                self.table.setItem(row, 1, QTableWidgetItem(str(pixel[0])))
                self.table.setItem(row, 2, QTableWidgetItem(str(pixel[1])))

                gps = p.get("gps", [0.0, 0.0])
                self.table.setItem(row, 3, QTableWidgetItem(str(gps[0])))
                self.table.setItem(row, 4, QTableWidgetItem(str(gps[1])))

                # Placeholder values for calculated World coordinates
                self.table.setItem(row, 5, QTableWidgetItem("0.00"))
                self.table.setItem(row, 6, QTableWidgetItem("0.00"))

            self.table.blockSignals(False)
            self.recalculate_and_update()
            self.statusBar().showMessage("Loaded calibration JSON configuration successfully!", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", f"Could not load/parse calibration JSON: {e}")

    def save_json_file(self):
        options = QFileDialog.Options()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Calibration JSON", "calibration.json", "JSON Files (*.json);;All Files (*)", options=options
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.txt_json.toPlainText())
            self.statusBar().showMessage(f"Saved JSON configuration to: {Path(path).name}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Error Saving File", f"Could not save JSON output: {e}")


# ---------------------------------------------------------------------------
# Main Script Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CalibrationGeneratorWindow()
    window.show()
    sys.exit(app.exec_())
