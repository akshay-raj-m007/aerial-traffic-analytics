"""
core/detector.py
────────────────
YOLOv8 + ByteTrack detector for aerial traffic footage.
Wraps the Ultralytics model and returns structured detections per frame.
No I/O here — video reading / writing lives in the caller.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
 
import cv2
import numpy as np
from ultralytics import YOLO
 
from config.settings import (
    BYTETRACK_YAML,
    CLASS_MAP,
    COLOR_MAP,
    CONFIDENCE_THRESHOLD,
    DEFAULT_COLOR,
    IMG_SIZE,
    IOU_THRESHOLD,
    MODEL_PATH,
)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Detection dataclass  — one instance per detected vehicle per frame
# ─────────────────────────────────────────────────────────────────────────────
 
@dataclass
class Detection:
    frame:      int
    track_id:   Optional[int]
    class_id:   int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
 
    # world-coordinate fields — filled later by homography transform
    wx: Optional[float] = field(default=None)
    wy: Optional[float] = field(default=None)
 
    def to_dict(self) -> dict:
        return {
            "frame":      self.frame,
            "track_id":   self.track_id,
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "x1":         self.x1,
            "y1":         self.y1,
            "x2":         self.x2,
            "y2":         self.y2,
            "cx":         self.cx,
            "cy":         self.cy,
            "wx":         self.wx,
            "wy":         self.wy,
        }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Detector class
# ─────────────────────────────────────────────────────────────────────────────
 
class Detector:
    """
    Loads the YOLOv8 model once and exposes a per-frame `detect()` method.
 
    Usage
    -----
    detector = Detector()
    detections = detector.detect(frame, frame_idx=0)
    """
 
    def __init__(
        self,
        model_path:  str | Path = MODEL_PATH,
        bytetrack_yaml: str | Path = BYTETRACK_YAML,
        conf:        float = CONFIDENCE_THRESHOLD,
        iou:         float = IOU_THRESHOLD,
        imgsz:       int   = IMG_SIZE,
    ) -> None:
        self.model_path     = Path(model_path)
        self.bytetrack_yaml = str(bytetrack_yaml)
        self.conf           = conf
        self.iou            = iou
        self.imgsz          = imgsz
 
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Place epoch40.pt in assets/models/ "
                "or update MODEL_PATH in config/settings.py"
            )
 
        print(f"[Detector] Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print(f"[Detector] Model classes: {self.model.names}")
 
    # ── public API ───────────────────────────────────────────────────────────
 
    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[Detection]:
        """
        Run ByteTrack on a single BGR frame.
        Returns a list of Detection objects (may be empty).
        """
        results = self.model.track(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            tracker=self.bytetrack_yaml,
            persist=True,
            verbose=False,
        )
 
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
 
        track_ids = (
            boxes.id.int().cpu().tolist()
            if boxes.id is not None
            else [None] * len(boxes)
        )
 
        detections: list[Detection] = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
            conf       = round(float(boxes.conf[i]), 4)
            cls        = int(boxes.cls[i])
            track_id   = track_ids[i]
            class_name = CLASS_MAP.get(cls, "unknown")
            cx, cy     = (x1 + x2) // 2, (y1 + y2) // 2
 
            detections.append(Detection(
                frame=frame_idx,
                track_id=track_id,
                class_id=cls,
                class_name=class_name,
                confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
                cx=cx,  cy=cy,
            ))
 
        return detections
 
    # ── drawing helpers ──────────────────────────────────────────────────────
 
    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        detections: list[Detection],
        trajectories: dict[int, list[tuple[int, int]]],
    ) -> np.ndarray:
        """
        Draw bounding boxes, labels, and trajectory trails on a copy of the frame.
        Returns the annotated copy; original is untouched.
        """
        out = frame.copy()
 
        for det in detections:
            color = COLOR_MAP.get(det.class_name, DEFAULT_COLOR)
            label = f"ID{det.track_id} {det.class_name} {det.confidence:.2f}"
 
            # trajectory trail
            if det.track_id is not None:
                pts = trajectories.get(det.track_id, [])
                for j in range(1, len(pts)):
                    cv2.line(out, pts[j - 1], pts[j], color, 2)
 
            # bounding box
            cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)
 
            # label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                out,
                (det.x1, det.y1 - th - 8),
                (det.x1 + tw + 5, det.y1),
                color, -1,
            )
 
            # label text
            cv2.putText(
                out, label,
                (det.x1 + 2, det.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
            )
 
        return out
 