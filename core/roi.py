"""
core/roi.py
===========
Region-of-Interest filter using OpenCV pointPolygonTest.

Usage:
    from core.roi import ROIFilter

    roi = ROIFilter([(100,200), (400,200), (400,500), (100,500)])
    filtered = roi.filter(detections)   # keeps only detections inside polygon
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.detector import Detection


class ROIFilter:
    """
    Wraps a polygon defined in pixel coordinates.
    Call .filter(detections) to keep only vehicles whose centroid
    falls inside or on the polygon boundary.
    """

    def __init__(self, points: list[tuple[int, int]]) -> None:
        """
        Args:
            points: List of (x, y) pixel coordinates defining the polygon.
                    Minimum 3 points required.
        """
        if len(points) < 3:
            raise ValueError("ROI polygon requires at least 3 points.")
        self._poly = np.array(points, dtype=np.float32)

    @classmethod
    def from_json(cls, data: dict) -> "ROIFilter":
        """Load from a dict like {"roi": [[x,y], ...]}"""
        pts = [(int(p[0]), int(p[1])) for p in data["roi"]]
        return cls(pts)

    def contains(self, cx: float, cy: float) -> bool:
        """
        Returns True if point (cx, cy) is inside or on the polygon.
        Uses cv2.pointPolygonTest: returns >= 0 for inside/on boundary.
        """
        result = cv2.pointPolygonTest(self._poly, (float(cx), float(cy)), False)
        return result >= 0

    def filter(self, detections: list) -> list:
        """
        Filter a list of Detection objects, keeping only those whose
        centroid (cx, cy) is inside the ROI polygon.
        """
        return [d for d in detections if self.contains(d.cx, d.cy)]

    def draw(self, frame: np.ndarray,
             color: tuple = (124, 106, 247),
             thickness: int = 2,
             fill_alpha: float = 0.15) -> np.ndarray:
        """
        Draw the ROI polygon on a frame (in-place copy).
        Semi-transparent fill + solid border.
        """
        out = frame.copy()
        pts = self._poly.astype(np.int32).reshape((-1, 1, 2))

        # Semi-transparent fill
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, fill_alpha, out, 1 - fill_alpha, 0, out)

        # Border
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)

        # Vertex labels
        for i, (x, y) in enumerate(self._poly.astype(int)):
            cv2.circle(out, (x, y), 5, (255, 255, 0), -1)
            cv2.putText(out, str(i+1), (x+6, y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        return out

    @property
    def points(self) -> list[tuple[int, int]]:
        return [(int(x), int(y)) for x, y in self._poly]

    def __repr__(self) -> str:
        return f"ROIFilter(points={self.points})"