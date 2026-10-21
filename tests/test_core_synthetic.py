"""In-memory tests only; network, models, subprocesses and writes are blocked."""
import dataclasses
import json
from pathlib import Path
import sys
import os
import unittest
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "baselines"))
FORBIDDEN = {"torch", "transformers", "cv2", "openai", "ultralytics",
             "faster_whisper", "numpy", "PIL", "requests", "httpx", "tiktoken"}
def guard(event, args):
    if event.startswith(("socket.", "subprocess.")) or event in {"os.system", "os.exec", "os.posix_spawn", "ctypes.dlopen"}:
        raise RuntimeError("External execution is forbidden in synthetic tests")
    if event == "import" and args and str(args[0]).split(".")[0] in FORBIDDEN:
        raise RuntimeError("Model or external client import is forbidden")
    if event == "open":
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(x in mode for x in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            raise RuntimeError("File writes are forbidden in synthetic tests")
sys.addaudithook(guard)
import evique
from evique.planner import QueryPlanner
from evique.evidence_packer import EvidencePacker
from temporal_adapter import normalize_native_result

class CoreTests(unittest.TestCase):
    def test_public_name(self):
        self.assertEqual(evique.MODEL_NAME, "EVIQUE")
    def test_source_import(self):
        self.assertTrue(Path(evique.__file__).resolve().is_relative_to(ROOT / "src"))
    def test_planner_is_deterministic(self):
        query = "What happens before the synthetic object stops?"
        self.assertEqual(QueryPlanner().plan(query), QueryPlanner().plan(query))
    def test_planner_result_serializable(self):
        result = QueryPlanner().plan("Where is the synthetic object?")
        if dataclasses.is_dataclass(result): result = dataclasses.asdict(result)
        self.assertTrue(json.dumps(result))
    def test_empty_packing_is_deterministic(self):
        self.assertEqual(EvidencePacker().pack("synthetic query", []), EvidencePacker().pack("synthetic query", []))
    def test_nonempty_packing_preserves_input(self):
        nodes = [{"id": "synthetic:scope:0", "evidence_id": "synthetic:scope:0", "video_id": "synthetic_video", "view": "scope", "view_type": "scope", "start": 0.0, "end": 2.0, "text": "A synthetic object moves.", "content": "A synthetic object moves.", "score": 0.8, "source_ids": ["synthetic_source"]}]
        before = json.dumps(nodes, sort_keys=True)
        first = EvidencePacker().pack("synthetic movement", nodes)
        self.assertEqual(first, EvidencePacker().pack("synthetic movement", nodes))
        self.assertEqual(before, json.dumps(nodes, sort_keys=True))
    def test_adapter_unknown_method_fails_closed(self):
        with self.assertRaises((ValueError, KeyError, RuntimeError)):
            normalize_native_result("UnsupportedSyntheticMethod", {}, {})

if __name__ == "__main__":
    unittest.main(verbosity=2)
