"""
config/settings.py
──────────────────
Single source of truth for all paths and hyperparameters.
Override any value via environment variable or .env file.
"""
 
from pathlib import Path
import os
 
# ── Project root (two levels up from this file: config/ → root) ─────────────
ROOT = Path(__file__).resolve().parent.parent
 
# ─────────────────────────────────────────────────────────────────────────────
# File paths
# ─────────────────────────────────────────────────────────────────────────────
 
MODEL_PATH     = Path(os.getenv("MODEL_PATH",     str(ROOT / "assets" / "models" / "epoch40.pt")))
BYTETRACK_YAML = Path(os.getenv("BYTETRACK_YAML", str(ROOT / "config" / "bytetrack_drone.yaml")))
OUTPUT_DIR     = Path(os.getenv("OUTPUT_DIR",     str(ROOT / "outputs")))
 
# ─────────────────────────────────────────────────────────────────────────────
# Detection hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
 
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE", "0.15"))
IOU_THRESHOLD        = float(os.getenv("IOU",        "0.45"))
IMG_SIZE             = int(os.getenv("IMG_SIZE",     "1280"))
 
# ─────────────────────────────────────────────────────────────────────────────
# Class mapping  (raw model class_id → merged label)
# Edit this if your model outputs different class indices
# ─────────────────────────────────────────────────────────────────────────────
 
CLASS_MAP: dict[int, str] = {
    0: "rikshaw",
    1: "motorcycle",
    2: "HMV",
    3: "car",
    4: "rikshaw",
    5: "rikshaw",
    6: "motorcycle",
    7: "pedestrian",
    8: "HMV",
    9: "HMV",
}
 
# ─────────────────────────────────────────────────────────────────────────────
# Box colours per merged label  (BGR for OpenCV)
# ─────────────────────────────────────────────────────────────────────────────
 
COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "rikshaw":    (0,   255, 255),
    "motorcycle": (255, 255,   0),
    "HMV":        (0,     0, 255),
    "car":        (0,   255,   0),
    "pedestrian": (255,   0, 255),
}
DEFAULT_COLOR: tuple[int, int, int] = (255, 255, 255)
 
# ─────────────────────────────────────────────────────────────────────────────
# API server
# ─────────────────────────────────────────────────────────────────────────────
 
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
 
# ─────────────────────────────────────────────────────────────────────────────
# Multi-camera fusion
# ─────────────────────────────────────────────────────────────────────────────
 
FUSION_DISTANCE_THRESHOLD_M = float(os.getenv("FUSION_DIST", "5.0"))   # metres
FUSION_TIME_WINDOW_FRAMES   = int(os.getenv("FUSION_TIME",   "5"))     # frames