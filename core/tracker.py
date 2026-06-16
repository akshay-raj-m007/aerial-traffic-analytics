"""
core/tracker.py
Maintains per-vehicle trajectory state across frames.

pixel_trail is an unbounded list (full history) — every position since
first detection is kept so persistent trails render after vehicle is gone,
matching the Colab script's defaultdict(list) pattern.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class TrackState:
    track_id:    int
    class_name:  str
    pixel_trail: list[tuple[int, int]]      = field(default_factory=list)
    world_trail: list[tuple[float, float]]  = field(default_factory=list)
    gps_trail:   list[tuple[float, float]]  = field(default_factory=list)
    frame_trail: list[int]                  = field(default_factory=list)
    first_frame: int   = 0
    last_frame:  int   = 0
    _fps:        float = 30.0

    @property
    def pixel_distance(self) -> float:
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
        if len(self.world_trail) < 2:
            return None
        return sum(
            math.hypot(
                self.world_trail[i][0] - self.world_trail[i - 1][0],
                self.world_trail[i][1] - self.world_trail[i - 1][1],
            )
            for i in range(1, len(self.world_trail))
        )

    @property
    def live_speed_kmph(self) -> Optional[float]:
        """Instantaneous speed using 5-frame rolling window on world trail."""
        if len(self.world_trail) < 2:
            return None
        window = self.world_trail[-5:]
        dist_m = sum(
            math.hypot(
                window[i][0] - window[i - 1][0],
                window[i][1] - window[i - 1][1],
            )
            for i in range(1, len(window))
        )
        duration_sec = (len(window) - 1) / self._fps if self._fps > 0 else 0
        if duration_sec == 0:
            return None
        return (dist_m / duration_sec) * 3.6


class TrajectoryTracker:
    def __init__(self, fps: float = 30.0) -> None:
        self.fps = fps
        self._tracks: dict[int, TrackState] = {}

    def update(self, detections, frame_idx: int) -> None:
        """
        Call BEFORE annotate_frame so the current frame's position is already
        in the trail when trails are drawn (matches Colab Step 1 → Step 2 order).
        """
        for det in detections:
            if det.track_id is None:
                continue
            tid = det.track_id
            if tid not in self._tracks:
                self._tracks[tid] = TrackState(
                    track_id=tid, class_name=det.class_name,
                    first_frame=frame_idx, _fps=self.fps,
                )
            track = self._tracks[tid]
            track.pixel_trail.append((det.cx, det.cy))
            track.frame_trail.append(frame_idx)
            track.last_frame = frame_idx

            if det.wx is not None and det.wy is not None:
                track.world_trail.append((det.wx, det.wy))

            if hasattr(det, "lat") and det.lat is not None and det.lon is not None:
                track.gps_trail.append((det.lat, det.lon))

    def get_pixel_trails(self) -> dict[int, list[tuple[int, int]]]:
        """Full unbounded pixel trail history for every track ever seen."""
        return {tid: t.pixel_trail for tid, t in self._tracks.items()}

    def get_track_classes(self) -> dict[int, str]:
        """Returns {track_id: class_name} for every known track."""
        return {tid: t.class_name for tid, t in self._tracks.items()}

    def get_live_speeds(self) -> dict[int, Optional[float]]:
        return {tid: t.live_speed_kmph for tid, t in self._tracks.items()}

    def summary(self) -> list[dict]:
        rows = []
        for tid, t in self._tracks.items():
            n_frames     = (t.last_frame - t.first_frame) + 1
            duration_sec = n_frames / self.fps if self.fps > 0 else 0.0
            px_vel = t.pixel_distance / duration_sec if duration_sec > 0 else 0.0
            w_dist = t.world_distance
            w_vel  = (w_dist / duration_sec) if (w_dist and duration_sec > 0) else None

            start_gps = t.gps_trail[0]  if t.gps_trail else None
            end_gps   = t.gps_trail[-1] if t.gps_trail else None

            rows.append({
                "track_id":            tid,
                "class_name":          t.class_name,
                "first_frame":         t.first_frame,
                "last_frame":          t.last_frame,
                "duration_frames":     n_frames,
                "duration_sec":        round(duration_sec, 3),
                "pixel_distance":      round(t.pixel_distance, 2),
                "pixel_velocity_px_s": round(px_vel, 2),
                "world_distance_m":    round(w_dist, 3) if w_dist is not None else None,
                "world_velocity_m_s":  round(w_vel, 3)  if w_vel  is not None else None,
                "world_velocity_kmph": round(w_vel * 3.6, 2) if w_vel is not None else None,
                "start_cx": t.pixel_trail[0][0]  if t.pixel_trail else None,
                "start_cy": t.pixel_trail[0][1]  if t.pixel_trail else None,
                "end_cx":   t.pixel_trail[-1][0] if t.pixel_trail else None,
                "end_cy":   t.pixel_trail[-1][1] if t.pixel_trail else None,
                "start_lat": round(start_gps[0], 8) if start_gps else None,
                "start_lon": round(start_gps[1], 8) if start_gps else None,
                "end_lat":   round(end_gps[0], 8)   if end_gps   else None,
                "end_lon":   round(end_gps[1], 8)   if end_gps   else None,
                "pixel_trail": t.pixel_trail,
                "world_trail": t.world_trail if t.world_trail else None,
                "gps_trail":   t.gps_trail   if t.gps_trail   else None,
            })
        return rows

    def gps_trail_rows(self) -> list[dict]:
        rows = []
        for tid, t in self._tracks.items():
            for i, (px, py) in enumerate(t.pixel_trail):
                frame = t.frame_trail[i] if i < len(t.frame_trail) else None
                wx  = t.world_trail[i][0] if i < len(t.world_trail) else None
                wy  = t.world_trail[i][1] if i < len(t.world_trail) else None
                lat = round(t.gps_trail[i][0], 8) if i < len(t.gps_trail) else None
                lon = round(t.gps_trail[i][1], 8) if i < len(t.gps_trail) else None
                rows.append({
                    "track_id":   tid,
                    "class_name": t.class_name,
                    "frame":      frame,
                    "cx":         px,
                    "cy":         py,
                    "wx_m":       round(wx, 4) if wx is not None else None,
                    "wy_m":       round(wy, 4) if wy is not None else None,
                    "latitude":   lat,
                    "longitude":  lon,
                })
        return rows

    @property
    def total_tracks(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()