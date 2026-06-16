"""
core/detector.py
YOLOv8 + ByteTrack detector for aerial traffic footage.

Trail rendering exactly mirrors the v2 Colab script:
- All trails in track_paths are drawn every frame (persistent after vehicle gone)
- track_classes.get(tid, "car") fallback — never skips a trail, never draws white
- Single shared overlay blended once at 0.75 (no per-segment compounding)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from ultralytics import YOLO
from config.settings import (
    BYTETRACK_YAML, CLASS_MAP, COLOR_MAP,
    CONFIDENCE_THRESHOLD, DEFAULT_COLOR,
    IMG_SIZE, IOU_THRESHOLD, MODEL_PATH,
    MIN_TRAIL_POINTS,
)


@dataclass
class Detection:
    frame:      int
    track_id:   Optional[int]
    class_id:   int
    class_name: str
    confidence: float
    x1: int; y1: int; x2: int; y2: int
    cx: int; cy: int
    wx: Optional[float] = field(default=None)
    wy: Optional[float] = field(default=None)
    lat: Optional[float] = field(default=None)
    lon: Optional[float] = field(default=None)

    def to_dict(self) -> dict:
        return {
            "frame":      self.frame,
            "track_id":   self.track_id,
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
            "cx": self.cx, "cy": self.cy,
            "wx": self.wx, "wy": self.wy,
        }


class Detector:
    def __init__(
        self,
        model_path:     str | Path = MODEL_PATH,
        bytetrack_yaml: str | Path = BYTETRACK_YAML,
        conf:           float = CONFIDENCE_THRESHOLD,
        iou:            float = IOU_THRESHOLD,
        imgsz:          int   = IMG_SIZE,
        agnostic_nms:   bool  = True,
        augment:        bool  = True,
    ) -> None:
        self.model_path     = Path(model_path)
        self.bytetrack_yaml = str(bytetrack_yaml)
        self.conf         = conf
        self.iou          = iou
        self.imgsz        = imgsz
        self.agnostic_nms = agnostic_nms
        self.augment      = augment

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        print(f"[Detector] Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print(f"[Detector] Model classes: {self.model.names}")

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[Detection]:
        results = self.model.track(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            agnostic_nms=self.agnostic_nms,
            augment=self.augment,
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

        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
            conf       = round(float(boxes.conf[i]), 4)
            cls        = int(boxes.cls[i])
            track_id   = track_ids[i]
            class_name = CLASS_MAP.get(cls, "unknown")
            cx, cy     = (x1 + x2) // 2, (y1 + y2) // 2
            detections.append(Detection(
                frame=frame_idx, track_id=track_id,
                class_id=cls, class_name=class_name, confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2, cx=cx, cy=cy,
            ))
        return detections

    @staticmethod
    def annotate_frame(
        frame:         np.ndarray,
        detections:    list[Detection],
        trajectories:  dict[int, list[tuple[int, int]]],
        track_classes: dict | None = None,
    ) -> np.ndarray:
        """
        Args:
            trajectories:  full path history {track_id: [(cx,cy), ...]}
                           from tracker.get_pixel_trails()
            track_classes: {track_id: class_name} from tracker.get_track_classes()
                           Used to color trails for vehicles no longer visible.
                           Falls back to "car" color if a tid is missing — matching
                           the Colab script's track_classes.get(tid, "car") pattern.
        """
        out = frame.copy()
        tc  = track_classes or {}

        # ── STEP 1: Draw ALL trails (including gone vehicles) onto shared overlay ──
        trail_overlay = out.copy()

        for tid, pts in trajectories.items():
            if len(pts) < MIN_TRAIL_POINTS:
                continue
            # Exact Colab pattern: track_classes.get(tid, "car") — always a valid color
            label_name = tc.get(tid, "car")
            color      = COLOR_MAP.get(label_name, DEFAULT_COLOR)

            for i in range(1, len(pts)):
                alpha     = i / len(pts)
                thickness = max(1, int(4 * alpha))
                pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                pt2 = (int(pts[i][0]),     int(pts[i][1]))
                cv2.line(trail_overlay, pt1, pt2, color, thickness)

        # Single blend for ALL trails at once
        cv2.addWeighted(trail_overlay, 0.75, out, 0.25, 0, out)

        # ── STEP 2: Draw boxes, labels, center dots on top ──
        for det in detections:
            color = COLOR_MAP.get(det.class_name, DEFAULT_COLOR)
            label = f"ID:{det.track_id} {det.class_name} {det.confidence:.2f}"

            cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(out, (det.x1, det.y1 - th - 8), (det.x1 + tw + 5, det.y1), color, -1)
            cv2.putText(out, label, (det.x1 + 2, det.y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            cv2.circle(out, (det.cx, det.cy), 4, color, -1)

        return out