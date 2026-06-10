"""
main.py — entry point.

python main.py --video assets/sample_video/clip.mp4 --calibration config/calibration_cam1.json
python main.py --video ... --no-video
python main.py --video ... --no-map          # skip static PNG map
python main.py --video ... --no-video --no-map
python main.py --serve
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run_pipeline(
    video_path:  Path,
    output_dir:  Path,
    write_video: bool,
    calib_path:  Path,
    write_map:   bool = True,
):
    import cv2
    import pandas as pd
    from core.detector   import Detector
    from core.tracker    import TrajectoryTracker
    from core.homography import Homography
    from core.exporter   import export_all

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # ------------------------------------------------------------------
    # Homography / calibration
    # ------------------------------------------------------------------
    if not calib_path or not calib_path.exists():
        raise FileNotFoundError(
            f"Calibration JSON not found: {calib_path}\n"
            "Provide one with: --calibration config/calibration_cam1.json"
        )
    print(f"[Homography] Loading: {calib_path}")
    hom = Homography.from_json(calib_path)

    # Validate that origin_gps is present — required for GPS export
    if hom.origin_gps is None:
        print(
            "[WARNING] calibration JSON has no 'origin_gps' key.\n"
            "GPS trails and map exports will be skipped.\n"
            "Add: \"origin_gps\": [longitude, latitude] to your calibration file."
        )
        gps_enabled = False
    else:
        gps_enabled = True
        print(f"[Homography] GPS origin: lon={hom.origin_gps[0]}, lat={hom.origin_gps[1]}")

    # ------------------------------------------------------------------
    # Video setup
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Video] {width}x{height}  {fps:.1f}fps")

    writer = None
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_dir / f"{stem}_annotated.mp4"), fourcc, fps, (width, height)
        )

    # ------------------------------------------------------------------
    # Detection / tracking loop
    # ------------------------------------------------------------------
    detector = Detector()
    tracker  = TrajectoryTracker(fps=fps)

    all_detections = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print(f"\rProcessing frame {frame_idx}", end="", flush=True)

        detections = detector.detect(frame, frame_idx=frame_idx)

        for det in detections:
            # World metres
            det.wx, det.wy = hom.transform(det.cx, det.cy)
            # GPS coordinates (lat, lon)
            if gps_enabled:
                det.lat, det.lon = hom.world_to_gps(det.wx, det.wy)
            else:
                det.lat = det.lon = None

        tracker.update(detections, frame_idx=frame_idx)

        # ------------------------------------------------------------------
        # Annotated video frame
        # ------------------------------------------------------------------
        if write_video:
            annotated   = detector.annotate_frame(frame, detections, tracker.get_pixel_trails())
            live_speeds = tracker.get_live_speeds()

            for det in detections:
                if det.track_id is None:
                    continue

                # World coordinates below box
                if det.wx is not None:
                    cv2.putText(
                        annotated,
                        f"({det.wx:.1f}m, {det.wy:.1f}m)",
                        (det.x1, det.y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
                    )

                # GPS below world coords
                if gps_enabled and det.lat is not None:
                    cv2.putText(
                        annotated,
                        f"({det.lat:.6f}, {det.lon:.6f})",
                        (det.x1, det.y2 + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 220, 255), 1,
                    )

                # Speed in km/h
                speed_kmph = live_speeds.get(det.track_id)
                if speed_kmph is not None:
                    spd_color = (
                        (0, 255, 0)   if speed_kmph < 30 else
                        (0, 165, 255) if speed_kmph < 60 else
                        (0, 0, 255)
                    )
                    lbl = f"{speed_kmph:.1f} km/h"
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(
                        annotated,
                        (det.x1, det.y2 + 31),
                        (det.x1 + tw + 4, det.y2 + 31 + th + 4),
                        (30, 30, 30), -1,
                    )
                    cv2.putText(
                        annotated, lbl,
                        (det.x1 + 2, det.y2 + 31 + th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, spd_color, 1,
                    )

            writer.write(annotated)

        # Accumulate per-frame detections
        for det in detections:
            row = det.to_dict()
            if gps_enabled:
                row["latitude"]  = round(det.lat, 8) if det.lat is not None else None
                row["longitude"] = round(det.lon, 8) if det.lon is not None else None
            all_detections.append(row)

        frame_idx += 1

    # ------------------------------------------------------------------
    # Release resources
    # ------------------------------------------------------------------
    print()
    cap.release()
    if writer:
        writer.release()

    # ------------------------------------------------------------------
    # Save core CSVs
    # ------------------------------------------------------------------
    pd.DataFrame(all_detections).to_csv(
        output_dir / f"{stem}_detections.csv", index=False
    )
    summary_rows = tracker.summary()
    pd.DataFrame(summary_rows).to_csv(
        output_dir / f"{stem}_summary.csv", index=False
    )

    print(f"\nFrames: {frame_idx}  |  Tracks: {tracker.total_tracks}")

    # ------------------------------------------------------------------
    # GPS / map exports
    # ------------------------------------------------------------------
    if gps_enabled:
        gps_rows = tracker.gps_trail_rows()
        export_results = export_all(
            summary=summary_rows,
            gps_trail_rows=gps_rows,
            output_dir=output_dir,
            stem=stem,
            skip_map=(not write_map),
        )

        print("\n--- Exports ---")
        for fmt, path in export_results.items():
            if path:
                print(f"  {fmt:15s}: {path}")
            else:
                print(f"  {fmt:15s}: skipped")
    else:
        print("[GPS exports skipped — no origin_gps in calibration file]")

    print(f"\nAll outputs in: {output_dir}/")


# ---------------------------------------------------------------------------

def run_server():
    import uvicorn
    from api.server import app
    from config.settings import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aerial Traffic Analytics — detection, tracking, GPS export"
    )
    parser.add_argument("--video",       type=Path,
                        help="Path to input video file")
    parser.add_argument("--output",      type=Path, default=Path("outputs"),
                        help="Output directory (default: outputs/)")
    parser.add_argument("--calibration", type=Path,
                        help="Path to calibration JSON")
    parser.add_argument("--no-video",    action="store_true",
                        help="Skip annotated video output")
    parser.add_argument("--no-map",      action="store_true",
                        help="Skip static PNG map (saves time if contextily not installed)")
    parser.add_argument("--serve",       action="store_true",
                        help="Start FastAPI server")
    args = parser.parse_args()

    if args.serve:
        run_server()
    elif args.video:
        if not args.calibration:
            parser.error("--calibration is required when running the pipeline.")
        run_pipeline(
            video_path=args.video,
            output_dir=args.output,
            write_video=not args.no_video,
            calib_path=args.calibration,
            write_map=not args.no_map,
        )
    else:
        parser.print_help()
        sys.exit(1)