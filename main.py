"""
main.py — entry point with WorldStitcher integration.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run_pipeline(
    video_path: Path,
    output_dir: Path,
    write_video: bool,
    calib_path: Path,
    write_map: bool = True,
):
    import cv2
    import pandas as pd
    import numpy as np

    from core.detector import Detector
    from core.tracker import TrajectoryTracker
    from core.homography import Homography
    from core.exporter import export_all
    from core.stitcher import WorldStitcher
    from config.settings import COLOR_MAP, DEFAULT_COLOR

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    if not calib_path or not calib_path.exists():
        raise FileNotFoundError(
            f"Calibration JSON not found: {calib_path}\n"
            "Provide one with: --calibration config/calibration_cam1.json"
        )

    print(f"[Homography] Loading: {calib_path}")
    hom = Homography.from_json(calib_path)

    stitcher = WorldStitcher(
        homography=hom,
        meters_per_pixel=0.10,
        canvas_width_m=1000,
        canvas_height_m=1000,
    )

    if hom.origin_gps is None:
        print(
            "[WARNING] calibration JSON has no 'origin_gps' key.\n"
            "GPS trails and map exports will be skipped."
        )
        gps_enabled = False
    else:
        gps_enabled = True
        print(f"[Homography] GPS origin: lon={hom.origin_gps[0]}, lat={hom.origin_gps[1]}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Video] {width}x{height}  {fps:.1f}fps")

    writer = None
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_dir / f"{stem}_annotated.mp4"),
            fourcc,
            fps,
            (width, height),
        )

    detector = Detector()
    tracker = TrajectoryTracker(fps=fps)

    all_detections = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        print(f"\rProcessing frame {frame_idx}", end="", flush=True)

        detections = detector.detect(frame, frame_idx=frame_idx)

        for det in detections:
            det.wx, det.wy = hom.transform(det.cx, det.cy)

            if gps_enabled:
                det.lat, det.lon = hom.world_to_gps(det.wx, det.wy)
            else:
                det.lat = None
                det.lon = None

        tracker.update(detections, frame_idx=frame_idx)

        if frame_idx % 10 == 0:
            stitcher.add_frame(frame)

        if write_video:
            annotated = detector.annotate_frame(
                frame,
                detections,
                tracker.get_pixel_trails(),
                tracker.get_track_classes(),
            )

            live_speeds = tracker.get_live_speeds()

            for det in detections:
                if det.track_id is None:
                    continue

                if det.class_name == "pedestrian":
                    pts = tracker.get_pixel_trails().get(det.track_id, [])
                    if len(pts) < 20:
                        continue

                if det.wx is not None:
                    cv2.putText(
                        annotated,
                        f"({det.wx:.1f}m, {det.wy:.1f}m)",
                        (det.x1, det.y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (200, 200, 200),
                        1,
                    )

                if gps_enabled and det.lat is not None:
                    cv2.putText(
                        annotated,
                        f"({det.lat:.6f}, {det.lon:.6f})",
                        (det.x1, det.y2 + 27),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (180, 220, 255),
                        1,
                    )

                speed_kmph = live_speeds.get(det.track_id)

                if speed_kmph is not None:
                    lbl = f"{speed_kmph:.1f} km/h"
                    cv2.putText(
                        annotated,
                        lbl,
                        (det.x1, det.y2 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                    )

            writer.write(annotated)

        for det in detections:
            row = det.to_dict()

            if gps_enabled:
                row["latitude"] = round(det.lat, 8) if det.lat is not None else None
                row["longitude"] = round(det.lon, 8) if det.lon is not None else None

            all_detections.append(row)

        frame_idx += 1

    print()

    cap.release()

    if writer:
        writer.release()

    # Filter out short-lived pedestrian tracks
    MIN_PEDESTRIAN_FRAMES = 20
    summary_rows = tracker.summary()
    
    # 1. Filter summary_rows
    summary_rows = [
        row for row in summary_rows 
        if not (row["class_name"] == "pedestrian" and row["duration_frames"] < MIN_PEDESTRIAN_FRAMES)
    ]
    
    # Get the valid track IDs (long-lived peds + all other classes)
    valid_track_ids = {row["track_id"] for row in summary_rows}
    
    # 2. Filter all_detections
    all_detections = [
        row for row in all_detections 
        if row["track_id"] in valid_track_ids
    ]

    pd.DataFrame(all_detections).to_csv(
        output_dir / f"{stem}_detections.csv",
        index=False,
    )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / f"{stem}_summary.csv",
        index=False,
    )

    for row in summary_rows:
        trail = row.get("world_trail")

        if not trail or len(trail) < 2:
            continue

        pts = np.array(
            [stitcher.world_to_canvas(wx, wy) for wx, wy in trail],
            dtype=np.int32,
        )

        color = COLOR_MAP.get(row["class_name"], DEFAULT_COLOR)

        cv2.polylines(
            stitcher.canvas,
            [pts],
            False,
            color,
            2,
        )

    mosaic_path = output_dir / f"{stem}_orthomosaic.png"
    stitcher.save(mosaic_path)

    print(f"[Stitcher] Saved: {mosaic_path}")
    print(f"\nFrames: {frame_idx}  |  Tracks: {len(summary_rows)}")

    if gps_enabled:
        gps_rows = tracker.gps_trail_rows()
        # 3. Filter gps_rows
        gps_rows = [
            row for row in gps_rows 
            if row["track_id"] in valid_track_ids
        ]

        export_results = export_all(
            summary=summary_rows,
            gps_trail_rows=gps_rows,
            output_dir=output_dir,
            stem=stem,
            skip_map=(not write_map),
        )

        print("\n--- Exports ---")
        for fmt, path in export_results.items():
            print(f"  {fmt}: {path}")

    print(f"\nAll outputs in: {output_dir}/")


def run_server():
    import uvicorn
    from api.server import app
    from config.settings import API_HOST, API_PORT

    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-map", action="store_true")
    parser.add_argument("--serve", action="store_true")

    args = parser.parse_args()

    if args.serve:
        run_server()
    elif args.video:
        run_pipeline(
            video_path=args.video,
            output_dir=args.output,
            write_video=not args.no_video,
            calib_path=args.calibration,
            write_map=not args.no_map,
        )
