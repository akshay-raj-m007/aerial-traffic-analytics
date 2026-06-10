"""
core/homography.py
Pixel (x,y) → real-world ground-plane (wx, wy) in metres.
Always instantiate via Homography.from_json().

Calibration (cam1):
  C8 pixel=(943,257)   world=(0.00,  0.00)  GPS=16.3683500N 80.5058750E
  C5 pixel=(1504,265)  world=(30.61,+20.07) GPS=16.3685303N 80.5061616E
  C9 pixel=(945,630)   world=(13.95,-20.72) GPS=16.3681639N 80.5060056E
  C4 pixel=(1521,638)  world=(43.69, -0.91) GPS=16.3683418N 80.5062840E
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Optional
import cv2
import numpy as np


class Homography:
    def __init__(self, pixel_points: np.ndarray, world_points: np.ndarray,
                 origin_gps: Optional[tuple[float, float]] = None) -> None:
        """
        Args:
            pixel_points: Nx2 array of pixel (x, y) coords.
            world_points: Nx2 array of world (wx, wy) coords in metres.
            origin_gps:   (longitude, latitude) of the world origin (0, 0).
        """
        if len(pixel_points) < 4 or len(world_points) < 4:
            raise ValueError("At least 4 calibration points required.")
        if len(pixel_points) != len(world_points):
            raise ValueError("pixel_points and world_points must match in length.")

        self.pixel_points = np.float32(pixel_points)
        self.world_points = np.float32(world_points)

        # Store GPS origin for world → GPS conversion
        # origin_gps is (longitude, latitude) — same order as calibration JSON
        self._origin_gps: Optional[tuple[float, float]] = origin_gps  # (lon, lat)

        self.H, self.mask = cv2.findHomography(
            self.pixel_points, self.world_points,
            method=cv2.RANSAC, ransacReprojThreshold=3.0,
        )
        if self.H is None:
            raise RuntimeError("findHomography failed — check points are not collinear.")
        self.H_inv = np.linalg.inv(self.H)
        self._reprojection_error()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, json_path: str | Path) -> "Homography":
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        pts = data["calibration_points"]
        origin_gps = tuple(data["origin_gps"]) if "origin_gps" in data else None
        return cls(
            pixel_points=np.float32([p["pixel"] for p in pts]),
            world_points=np.float32([p["world"] for p in pts]),
            origin_gps=origin_gps,
        )

    @classmethod
    def from_gps(
        cls,
        pixel_points: list[tuple[float, float]],
        gps_points:   list[tuple[float, float]],
        origin_gps:   Optional[tuple[float, float]] = None,
    ) -> "Homography":
        if origin_gps is None:
            origin_gps = gps_points[0]
        origin_lon, origin_lat = origin_gps
        lat_rad = np.radians(origin_lat)
        M_PER_DEG_LAT = 111_320.0
        M_PER_DEG_LON = 111_320.0 * np.cos(lat_rad)
        world_pts = [
            [(lon - origin_lon) * M_PER_DEG_LON, (lat - origin_lat) * M_PER_DEG_LAT]
            for lon, lat in gps_points
        ]
        return cls(np.float32(pixel_points), np.float32(world_pts), origin_gps=origin_gps)

    # ------------------------------------------------------------------
    # Core transforms
    # ------------------------------------------------------------------

    def transform(self, cx: float, cy: float) -> tuple[float, float]:
        """Pixel → world metres (wx, wy)."""
        pt = np.float32([[[cx, cy]]])
        result = cv2.perspectiveTransform(pt, self.H)
        return float(result[0][0][0]), float(result[0][0][1])

    def transform_batch(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Batch pixel → world metres."""
        if not points:
            return []
        pts = np.float32([[[x, y] for x, y in points]])
        result = cv2.perspectiveTransform(pts, self.H)
        return [(float(r[0]), float(r[1])) for r in result[0]]

    def pixel_to_world(self, cx, cy):
        return self.transform(cx, cy)

    def world_to_pixel(self, wx, wy):
        pt = np.float32([[[wx, wy]]])
        result = cv2.perspectiveTransform(pt, self.H_inv)
        return float(result[0][0][0]), float(result[0][0][1])

    # ------------------------------------------------------------------
    # GPS conversion
    # ------------------------------------------------------------------

    def world_to_gps(self, wx: float, wy: float) -> tuple[float, float]:
        """
        Convert world coordinates (metres, East/North) to (latitude, longitude).

        Uses the equirectangular approximation centred on origin_gps.
        Accurate to <0.1 m over distances <1 km.

        Args:
            wx: metres East  of origin
            wy: metres North of origin

        Returns:
            (latitude, longitude) in decimal degrees
        """
        if self._origin_gps is None:
            raise RuntimeError(
                "origin_gps not set. Load calibration JSON with 'origin_gps' key, "
                "or pass origin_gps to the constructor."
            )
        origin_lon, origin_lat = self._origin_gps
        M_PER_DEG_LAT = 111_320.0
        M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(origin_lat))

        lat = origin_lat + wy / M_PER_DEG_LAT
        lon = origin_lon + wx / M_PER_DEG_LON
        return lat, lon

    def gps_to_world(self, lat: float, lon: float) -> tuple[float, float]:
        """
        Convert (latitude, longitude) → world metres (wx, wy).
        Inverse of world_to_gps.
        """
        if self._origin_gps is None:
            raise RuntimeError("origin_gps not set.")
        origin_lon, origin_lat = self._origin_gps
        M_PER_DEG_LAT = 111_320.0
        M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(origin_lat))

        wy = (lat - origin_lat) * M_PER_DEG_LAT
        wx = (lon - origin_lon) * M_PER_DEG_LON
        return wx, wy

    def pixel_to_gps(self, cx: float, cy: float) -> tuple[float, float]:
        """Convenience: pixel → GPS in one call. Returns (lat, lon)."""
        wx, wy = self.transform(cx, cy)
        return self.world_to_gps(wx, wy)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _reprojection_error(self) -> None:
        errors = []
        print("\n[Homography] Reprojection errors:")
        for i, (px_pt, w_pt) in enumerate(zip(self.pixel_points, self.world_points)):
            wx_c, wy_c = self.transform(px_pt[0], px_pt[1])
            err = np.hypot(wx_c - w_pt[0], wy_c - w_pt[1])
            errors.append(err)
            print(f"  Point {i+1}: pixel({px_pt[0]:.0f},{px_pt[1]:.0f}) "
                  f"expected({w_pt[0]:.2f},{w_pt[1]:.2f}) "
                  f"got({wx_c:.2f},{wy_c:.2f}) err={err:.4f}m")
        mean_err = float(np.mean(errors))
        print(f"  Mean error: {mean_err:.4f}m  "
              f"{'OK' if max(errors) < 2.0 else 'WARNING: re-pick points'}\n")

    @property
    def matrix(self) -> np.ndarray:
        return self.H

    @property
    def origin_gps(self) -> Optional[tuple[float, float]]:
        """Returns (longitude, latitude) of world origin."""
        return self._origin_gps