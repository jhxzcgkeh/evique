"""Synthetic checks for the EVIQUE-only answer entry point."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "evique_answer.py"
spec = importlib.util.spec_from_file_location("evique_answer", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class _Completions:
    def __init__(self, result=None, error=None):
        self.calls = 0
        self.result = result
        self.error = error
    def create(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result

class _Client:
    def __init__(self, completions):
        self.chat = type("Chat", (), {"completions": completions})()

class _Message:
    content = "synthetic answer"
class _Choice:
    message = _Message()
class _Result:
    choices = [_Choice()]

class AnswerEntryPointTests(unittest.TestCase):
    def test_open_question_messages(self):
        messages = module.build_answer_messages("What happens?", "system")
        self.assertEqual(messages, [{"role":"system","content":"system"},
                                    {"role":"user","content":"What happens?"}])
    def test_closed_question_messages(self):
        messages = module.build_answer_messages("What happens?", "system", ["one", "two"])
        self.assertIn("A. one\nB. two", messages[1]["content"])
        self.assertIn("select exactly one", messages[0]["content"])
    def test_one_physical_attempt_on_failure(self):
        calls = _Completions(error=RuntimeError("synthetic failure"))
        with self.assertRaises(RuntimeError):
            module.request_answer(_Client(calls), model="synthetic", temperature=0.0,
                                  max_tokens=8, messages=[])
        self.assertEqual(calls.calls, 1)
    def test_response_content(self):
        calls = _Completions(result=_Result())
        answer = module.request_answer(_Client(calls), model="synthetic", temperature=0.0,
                                       max_tokens=8, messages=[])
        self.assertEqual(answer, "synthetic answer")
        self.assertEqual(calls.calls, 1)

if __name__ == "__main__":
    unittest.main(verbosity=2)
