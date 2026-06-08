"""
main.py
───────
Entry point for the aerial traffic analytics pipeline.
 
Run (single video, no API):
    python main.py --video assets/sample_video/test.mp4
 
Run API server (for frontend / multi-camera):
    python main.py --serve
 
Flags
-----
--video   PATH     path to input video  (required unless --serve)
--output  PATH     output directory     (default: outputs/)
--no-video         skip writing annotated video (faster, CSV only)
--serve            start FastAPI server instead of processing a video
"""
 
from __future__ import annotations
 
import argparse
import sys
from pathlib import Path
 
# ── make project root importable regardless of cwd ───────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
 
 
def run_pipeline(video_path: Path, output_dir: Path, write_video: bool) -> None:
    """Process one video end-to-end and write outputs to output_dir."""
    import cv2
    import pandas as pd
 
    from core.detector import Detector
    from core.tracker  import TrajectoryTracker
    from config.settings import OUTPUT_DIR
 
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
 
    # ── open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
 
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    # ── outputs ───────────────────────────────────────────────────────────────
    output_video_path = output_dir / f"{stem}_annotated.mp4"
    output_csv_path   = output_dir / f"{stem}_detections.csv"
    output_summary_path = output_dir / f"{stem}_summary.csv"
 
    writer = None
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
 
    # ── model + tracker ───────────────────────────────────────────────────────
    detector = Detector()
    tracker  = TrajectoryTracker(fps=fps)
 
    # ── frame loop ────────────────────────────────────────────────────────────
    all_detections = []
    frame_idx = 0
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        print(f"\rProcessing frame {frame_idx}", end="", flush=True)
 
        detections = detector.detect(frame, frame_idx=frame_idx)
        tracker.update(detections, frame_idx=frame_idx)
 
        if write_video:
            annotated = detector.annotate_frame(
                frame, detections, tracker.get_pixel_trails()
            )
            writer.write(annotated)
 
        all_detections.extend(d.to_dict() for d in detections)
        frame_idx += 1
 
    print()  # newline after progress
 
    # ── release ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
 
    # ── save CSVs ─────────────────────────────────────────────────────────────
    pd.DataFrame(all_detections).to_csv(output_csv_path, index=False)
    pd.DataFrame(tracker.summary()).to_csv(output_summary_path, index=False)
 
    # ── report ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Frames processed   : {frame_idx}")
    print(f"Unique tracks      : {tracker.total_tracks}")
    if write_video:
        print(f"Annotated video    : {output_video_path}")
    print(f"Per-frame CSV      : {output_csv_path}")
    print(f"Per-track summary  : {output_summary_path}")
    print(f"{'─'*50}")
 
 
def run_server() -> None:
    """Start the FastAPI server (implemented in api/server.py)."""
    import uvicorn
    from api.server import app
    from config.settings import API_HOST, API_PORT
 
    print(f"[Server] Starting at http://{API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
 
 
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerial traffic analytics")
    parser.add_argument("--video",    type=Path, help="Path to input video")
    parser.add_argument("--output",   type=Path, default=Path("outputs"),
                        help="Output directory (default: outputs/)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip writing annotated video (CSV only)")
    parser.add_argument("--serve",    action="store_true",
                        help="Start FastAPI server")
 
    args = parser.parse_args()
 
    if args.serve:
        run_server()
    elif args.video:
        run_pipeline(
            video_path=args.video,
            output_dir=args.output,
            write_video=not args.no_video,
        )
    else:
        parser.print_help()
        sys.exit(1)
 