"""
core/tracker.py
───────────────
Maintains per-vehicle trajectory state across frames.
Completely separate from the model — receives Detection objects
from detector.py and updates internal state.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass, field
from typing import Optional
 
import math
import numpy as np
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Per-track state
# ─────────────────────────────────────────────────────────────────────────────
 
@dataclass
class TrackState:
    track_id:   int
    class_name: str
 
    # pixel trajectory — list of (cx, cy) one entry per frame seen
    pixel_trail: list[tuple[int, int]] = field(default_factory=list)
 
    # world-coordinate trajectory — list of (wx, wy) if homography is applied
    world_trail: list[tuple[float, float]] = field(default_factory=list)
 
    first_frame: int = 0
    last_frame:  int = 0
 
    @property
    def pixel_distance(self) -> float:
        """Total Euclidean distance travelled in pixels."""
        if len(self.pixel_trail) < 2:
            return 0.0
        return sum(
            math.hypot(
                self.pixel_trail[i][0] - self.pixel_trail[i - 1][0],
                self.pixel_trail[i][1] - self.pixel_trail[i - 1][1],
            )
            for i in range(1, len(self.pixel_trail))
        )
 
    @property
    def world_distance(self) -> Optional[float]:
        """Total distance in metres (only if world trail populated)."""
        if len(self.world_trail) < 2:
            return None
        return sum(
            math.hypot(
                self.world_trail[i][0] - self.world_trail[i - 1][0],
                self.world_trail[i][1] - self.world_trail[i - 1][1],
            )
            for i in range(1, len(self.world_trail))
        )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TrajectoryTracker
# ─────────────────────────────────────────────────────────────────────────────
 
class TrajectoryTracker:
    """
    Receives Detection objects frame-by-frame and maintains trajectory state.
 
    Usage
    -----
    tracker = TrajectoryTracker(fps=30.0)
    tracker.update(detections, frame_idx=0)
    trails = tracker.get_pixel_trails()   # pass to annotate_frame
    summary = tracker.summary()           # for the output CSV
    """
 
    def __init__(self, fps: float = 30.0) -> None:
        self.fps    = fps
        self._tracks: dict[int, TrackState] = {}
 
    # ── public API ───────────────────────────────────────────────────────────
 
    def update(
        self,
        detections,                    # list[Detection] from detector.py
        frame_idx: int,
    ) -> None:
        """Register detections for the current frame."""
        for det in detections:
            if det.track_id is None:
                continue
 
            tid = det.track_id
            if tid not in self._tracks:
                self._tracks[tid] = TrackState(
                    track_id=tid,
                    class_name=det.class_name,
                    first_frame=frame_idx,
                )
 
            track = self._tracks[tid]
            track.pixel_trail.append((det.cx, det.cy))
            track.last_frame = frame_idx
 
            # world coordinates (populated after homography — optional)
            if det.wx is not None and det.wy is not None:
                track.world_trail.append((det.wx, det.wy))
 
    def get_pixel_trails(self) -> dict[int, list[tuple[int, int]]]:
        """
        Returns {track_id: [(cx, cy), ...]} — ready to pass into
        Detector.annotate_frame().
        """
        return {tid: t.pixel_trail for tid, t in self._tracks.items()}
 
    def summary(self) -> list[dict]:
        """
        One row per track — suitable for the output summary CSV.
        This is a per-track aggregate, not the per-frame detection CSV.
        """
        rows = []
        for tid, t in self._tracks.items():
            n_frames = (t.last_frame - t.first_frame) + 1
            duration_sec = n_frames / self.fps if self.fps > 0 else 0.0
 
            # pixel velocity (px/s)
            px_vel = (
                t.pixel_distance / duration_sec
                if duration_sec > 0 else 0.0
            )
 
            # world velocity (m/s) — None if no homography applied
            w_dist = t.world_distance
            w_vel  = (w_dist / duration_sec) if (w_dist and duration_sec > 0) else None
 
            rows.append({
                "track_id":         tid,
                "class_name":       t.class_name,
                "first_frame":      t.first_frame,
                "last_frame":       t.last_frame,
                "duration_frames":  n_frames,
                "duration_sec":     round(duration_sec, 3),
                "pixel_distance":   round(t.pixel_distance, 2),
                "pixel_velocity_px_s": round(px_vel, 2),
                "world_distance_m": round(w_dist, 3) if w_dist is not None else None,
                "world_velocity_m_s": round(w_vel, 3) if w_vel is not None else None,
                "start_cx":         t.pixel_trail[0][0]  if t.pixel_trail else None,
                "start_cy":         t.pixel_trail[0][1]  if t.pixel_trail else None,
                "end_cx":           t.pixel_trail[-1][0] if t.pixel_trail else None,
                "end_cy":           t.pixel_trail[-1][1] if t.pixel_trail else None,
                "pixel_trail":      t.pixel_trail,
                "world_trail":      t.world_trail if t.world_trail else None,
            })
 
        return rows
 
    @property
    def total_tracks(self) -> int:
        return len(self._tracks)
 
    def reset(self) -> None:
        """Clear all state — call before processing a new video."""
        self._tracks.clear()