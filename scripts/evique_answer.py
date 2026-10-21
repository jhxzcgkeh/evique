"""Run EVIQUE retrieval and optional answer generation on user-provided inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_MC_POLICY = """
This is a closed-set multiple-choice question. This policy overrides the
general open-ended answer flow.

- You must select exactly one of the supplied choices.
- Use the EVIQUE evidence to choose the best-supported choice.
- Do not refuse to answer. If evidence is incomplete, select the
  best-supported choice.
- Reproduce the selected choice text exactly.
- Keep the grounding explanation short and cite a timestamp or segment id
  when available.

Required output format:
Answer: <letter>. <exact choice text>

Evidence:
<short grounding explanation>
""".strip()

def build_answer_messages(question: str, system_prompt: str,
                          choices: Sequence[str] | None = None) -> list[dict[str, str]]:
    """Build the answer messages used by the EVIQUE answer path."""
    clean = [str(x).strip() for x in choices or []]
    if not clean or any(not x for x in clean) or len(clean) > 26:
        return [{"role": "system", "content": system_prompt},
                {"role": "user", "content": question}]
    choice_lines = "\n".join(f"{chr(ord('A') + i)}. {value}"
                             for i, value in enumerate(clean))
    return [
        {"role": "system", "content": f"{system_prompt.rstrip()}\n\n{_MC_POLICY}"},
        {"role": "user", "content": f"{question}\n\nChoices:\n{choice_lines}"},
    ]

def request_answer(client: Any, *, model: str, temperature: float,
                   max_tokens: int, messages: list[dict[str, str]]) -> str:
    """Send one logical request with exactly one SDK call and no runner retry."""
    response = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=messages,
    )
    return str(response.choices[0].message.content or "").strip()

def _load_choices(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError("choices file must contain a JSON array of strings")
    return value

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--dataset", default="user_dataset")
    parser.add_argument("--choices-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    parser.add_argument("--max-evidence", type=int, default=18)
    parser.add_argument("--token-budget", type=int, default=12000)
    parser.add_argument("--response-type", default="concise evidence-grounded answer")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if not args.model:
        parser.error("--model or OPENAI_MODEL is required")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        parser.error("OPENAI_API_KEY is required for answer generation")

    from evique.retriever import EvidenceRetriever, build_prompt_package
    retriever = EvidenceRetriever(args.index, max_evidence=args.max_evidence,
                                  token_budget=args.token_budget)
    question = args.query_file.read_text(encoding="utf-8").strip()
    package = retriever.retrieve(
        question, {"dataset": args.dataset, "video_id": args.video_id}
    )
    context = retriever.format_package(package)
    system_prompt = build_prompt_package(
        package, context, response_type=args.response_type
    )
    messages = build_answer_messages(
        question, system_prompt, _load_choices(args.choices_file)
    )

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=args.base_url, max_retries=0)
    answer = request_answer(
        client, model=args.model, temperature=args.temperature,
        max_tokens=args.max_answer_tokens, messages=messages,
    )
    result = {
        "method": "EVIQUE",
        "query": question,
        "dataset": args.dataset,
        "video_id": args.video_id,
        "retrieval": package,
        "context": context,
        "answer": answer,
        "request_policy": {"sdk_max_retries": 0, "runner_retries": 0,
                           "maximum_physical_attempts": 1},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

if __name__ == "__main__":
    main()
