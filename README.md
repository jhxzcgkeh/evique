# EVIQUE

EVIQUE organizes video evidence into Scope, Target, Track, and Event views.
Its query planner and cost-based planner select retrieval operations; the
retriever assembles candidates and the evidence packer selects and formats
evidence for a downstream answer model. Native temporal and source identifiers
remain attached to evidence records.

## Included scope

This repository contains the EVIQUE core, planning, retrieval, evidence-view
construction, packing, prompt construction, local preprocessing interfaces,
focused command-line entry points, and synthetic tests. It contains no videos,
captions, transcripts, benchmark queries, ground truth, predictions, metrics,
indexes, model weights, credentials, or experiment logs.

Complete NaiveRAG, TextVideoRAG, VideoRAG, LightRAG, and GraphRAG source trees
are not distributed. Obtain any desired baseline from a lawful source and obey
its license. `baselines/temporal_adapter.py` is an interface for converting a
method's own ranked segments or chunks into timestamped predictions; it is not
a baseline implementation and never supplies candidates or scores.

## Environment

Use Python 3.10 or later. The planner, packer, schema utilities, and synthetic
tests use the standard library. `requirements.txt` lists optional execution
dependencies for answer generation and local media processing; it is not a
historical lockfile. Native media processing also requires ffmpeg/ffprobe.

Obtain compatible Caption, ASR, and detection weights independently. Models and
their licenses are user-provided. No dependency or model is downloaded by the
synthetic checks. For local-only operation, set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
python -B tests/test_core_synthetic.py
python -B tests/test_answer_entrypoint_synthetic.py
python -B scripts/evique_build_views.py --help
python -B scripts/evique_retrieve.py --help
python -B scripts/evique_answer.py --help
```

## Build evidence views

The view builder consumes an externally supplied mapping
`{video_id: {segment_id: segment_record}}`. Segment records retain the timestamp,
caption, and transcript fields used by the native builder. Do not substitute
answer text for video evidence.

```bash
python -B scripts/evique_build_views.py \
  --segments user_data/video_segments.json --output runs/user_index
```

The wrapper invokes the native builder and refuses to overwrite an existing
output directory. Optional visual inputs remain available through the native
builder interface; omitting them does not reproduce a full visual index. The
standalone preprocessing interface can be inspected with:

```bash
python -B -m evique.cli.build_standalone_base --help
```

This preprocessing path may invoke local video models or a configured ASR
provider. It was not executed during release packaging.

## Retrieve and pack evidence

```bash
python -B scripts/evique_retrieve.py \
  --index runs/user_index --query-file user_data/query.txt \
  --video-id user_video --dataset user_dataset --output runs/evidence.json
```

This wrapper calls `EvidenceRetriever.retrieve` directly and emits the native
package, including planner decisions, packed evidence, metadata, and provenance.
Its evidence token budget retains the implementation's selector semantics; it
is not a hard cap on a downstream model's complete prompt.

## Generate an answer

```bash
# Set OPENAI_API_KEY in your shell or secret manager.
export OPENAI_MODEL='<provider model id>'
export OPENAI_BASE_URL='<compatible endpoint>'
python -B scripts/evique_answer.py \
  --index runs/user_index --query-file user_data/query.txt \
  --video-id user_video --dataset user_dataset --output runs/answer.json
```

The answer entry point uses the same native retriever, evidence formatter, and
prompt builder. An optional `--choices-file` accepts a JSON array of choices.
The SDK is configured with `max_retries=0`, and this wrapper performs one call
with no runner retry. Configure credentials only through the environment. Real
model execution and paid API use were not run during packaging.

## Baseline adapters

The temporal adapter accepts only a baseline's own native ranked hits with
video identity, timestamps, rank, and available score provenance. It rejects
unsupported methods and must not receive EVIQUE candidates or ground-truth
windows. Users must install baseline implementations separately and write a
small integration layer that produces this documented native-hit structure.

## Validation boundary

Synthetic tests cover source-only imports, deterministic planning, evidence
packing, adapter fail-closed behavior, message construction, and the one-call
answer request boundary. Real model execution, video preprocessing, API calls,
and third-party baseline integration are `NOT_RUN` in this release audit.

The production packer is not an experimentally verified explicit
connected-component constraint. Relation-truth evaluation requires external
relation annotations. Dataset ground truth belongs only in a separate evaluator.

## License

The authorized EVIQUE source code in this repository is licensed under the MIT
License. See `LICENSE` for the complete terms. External datasets, pretrained
models, API services, and third-party software remain subject to their own
licenses and terms of use; see `THIRD_PARTY_NOTICES.md`.
