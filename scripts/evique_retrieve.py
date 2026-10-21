"""Retrieve an EVIQUE evidence package from a user-provided native index."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--dataset", default="user_dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-evidence", type=int, default=18)
    parser.add_argument("--token-budget", type=int, default=12000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    from evique.retriever import EvidenceRetriever
    model = EvidenceRetriever(args.index, max_evidence=args.max_evidence,
                              token_budget=args.token_budget)
    result = model.retrieve(args.query_file.read_text(encoding="utf-8").strip(),
                           {"dataset": args.dataset, "video_id": args.video_id})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
