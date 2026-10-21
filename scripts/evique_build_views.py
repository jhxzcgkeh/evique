"""Thin command-line interface to the native EVIQUE evidence-view builder."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-window-seconds", type=int, default=120)
    parser.add_argument("--track-gap-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new independent directory")
    from evique.builder import build_evique_from_segments
    build_evique_from_segments(video_segments_path=args.segments,
        output_dir=args.output, event_window_seconds=args.event_window_seconds,
        track_gap_seconds=args.track_gap_seconds)

if __name__ == "__main__":
    main()
