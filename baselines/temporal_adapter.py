from __future__ import annotations

import re
from typing import Any, Iterable


def _parse_time(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        start, end = float(value[0]), float(value[1])
    else:
        match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*", str(value or ""))
        if not match:
            raise ValueError(f"invalid segment time: {value!r}")
        start, end = float(match.group(1)), float(match.group(2))
    if start < 0 or end <= start:
        raise ValueError(f"invalid segment interval: {value!r}")
    return start, end


def _segment_record(segment_id: str, store: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for video_id, segments in store.items():
        prefix = f"{video_id}_"
        if not str(segment_id).startswith(prefix) or not isinstance(segments, dict):
            continue
        index = str(segment_id)[len(prefix):]
        record = segments.get(index)
        if isinstance(record, dict):
            candidates.append((len(str(video_id)), str(video_id), record))
    if not candidates:
        raise KeyError(f"segment not found in native store: {segment_id}")
    _, video_id, record = max(candidates)
    return video_id, record


def _prediction(
    *, segment_id: str, source_id: str, native_rank: int,
    native_score: float | None, score_source: str,
    store: dict[str, Any], ranking_stream: str = "primary",
) -> dict[str, Any]:
    video_id, record = _segment_record(segment_id, store)
    start, end = _parse_time(record.get("time") or record.get("timestamp") or record.get("time_range"))
    result = {
        "video_id": video_id,
        "start": start,
        "end": end,
        "native_rank": int(native_rank),
        "native_score": float(native_score) if native_score is not None else None,
        "score_source": score_source,
        "source_id": str(source_id),
        "source_segment_id": str(segment_id),
    }
    if ranking_stream != "primary":
        result["ranking_stream"] = ranking_stream
    return result


def _direct_prediction(
    *, source: dict[str, Any], source_id: str, native_rank: int,
    native_score: float | None, score_source: str,
) -> dict[str, Any]:
    video_id = str(source.get("video_id") or source.get("vid") or "")
    if not video_id:
        raise ValueError("native source is missing video identity")
    start, end = _parse_time([
        source.get("start_time", source.get("start")),
        source.get("end_time", source.get("end")),
    ])
    return {
        "video_id": video_id, "start": start, "end": end,
        "native_rank": int(native_rank),
        "native_score": float(native_score) if native_score is not None else None,
        "score_source": score_source, "source_id": str(source_id),
        "source_segment_id": str(source.get("segment_id") or source_id),
    }


def _source_from_content(source: dict[str, Any]) -> dict[str, Any] | None:
    text = str(source.get("content") or source.get("text") or "")
    match = re.search(
        r"\[video_id=(?P<video>.+?)\s+segment_id=(?P<segment>\S+)\s+time=(?P<start>-?\d+(?:\.\d+)?)-(?P<end>-?\d+(?:\.\d+)?)\]",
        text,
    )
    if not match:
        return None
    return {
        **source, "video_id": match.group("video"),
        "segment_id": match.group("segment"),
        "start_time": float(match.group("start")), "end_time": float(match.group("end")),
    }


def _chunk_segments(chunk_id: str, chunks: dict[str, Any]) -> list[str]:
    record = chunks.get(str(chunk_id))
    if not isinstance(record, dict):
        raise KeyError(f"chunk not found in native store: {chunk_id}")
    values = record.get("video_segment_id") or record.get("segment_ids") or []
    if isinstance(values, str):
        values = [values]
    return [str(value) for value in values]


def _dedupe_predictions(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float, str]] = set()
    for row in rows:
        key = (str(row["video_id"]), float(row["start"]), float(row["end"]), str(row["source_id"]))
        if key not in seen:
            result.append(row)
            seen.add(key)
    return result


def _ranked_chunk_predictions(
    hits: list[dict[str, Any]], segment_store: dict[str, Any], chunk_store: dict[str, Any],
    *, score_source: str, ranking_stream: str = "primary",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fallback_rank, hit in enumerate(hits, 1):
        chunk_id = str(hit.get("item_id") or hit.get("id") or "")
        if not chunk_id:
            raise ValueError("native chunk hit is missing item_id")
        rank = int(hit.get("rank") or fallback_rank)
        raw_score = hit.get("score")
        score = float(raw_score) if raw_score is not None else None
        for segment_id in _chunk_segments(chunk_id, chunk_store):
            rows.append(_prediction(
                segment_id=segment_id, source_id=chunk_id, native_rank=rank,
                native_score=score, score_source=score_source, store=segment_store,
                ranking_stream=ranking_stream,
            ))
    return _dedupe_predictions(rows)


def normalize_native_result(
    method: str,
    result: dict[str, Any],
    segment_store: dict[str, Any],
    chunk_store: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert only a method's own ranked units into clip-local time windows.

    The adapter never consults GT and never creates a numeric score when the
    production method exposes only an order.
    """
    method = method.lower()
    retrieval = result.get("retrieval") or {}
    predictions: list[dict[str, Any]] = []
    score_contract = "NATIVE_RANK_ONLY"
    ranking_basis = "native_result_order"
    if method == "videorag":
        segment_ids = retrieval.get("used_segments") or retrieval.get("retrieved_segments") or []
        for rank, segment_id in enumerate(segment_ids, 1):
            predictions.append(_prediction(
                segment_id=str(segment_id), source_id=str(segment_id), native_rank=rank,
                native_score=None, score_source="production_post_union_order",
                store=segment_store,
            ))
        ranking_basis = "production_post_union_segment_order"
    elif method == "naiverag":
        predictions = _ranked_chunk_predictions(
            list(retrieval), segment_store, chunk_store or {},
            score_source="chunk_cosine_similarity",
        )
        score_contract = "NATIVE_NUMERIC"
        ranking_basis = "native_chunk_cosine_rank"
    elif method == "textvideorag":
        predictions = _ranked_chunk_predictions(
            list(retrieval.get("text_first_reranked_chunks") or []),
            segment_store, chunk_store or {}, score_source="text_visual_fused_score",
            ranking_stream="text_first_fused_chunks",
        )
        for fallback_rank, hit in enumerate(retrieval.get("auxiliary_visual_segments") or [], 1):
            segment_id = str(hit.get("item_id") or hit.get("id") or "")
            if not segment_id:
                raise ValueError("native visual hit is missing item_id")
            predictions.append(_prediction(
                segment_id=segment_id, source_id=segment_id,
                native_rank=int(hit.get("rank") or fallback_rank),
                native_score=float(hit["score"]) if hit.get("score") is not None else None,
                score_source="auxiliary_visual_cosine_similarity", store=segment_store,
                ranking_stream="auxiliary_visual_segments",
            ))
        score_contract = "NATIVE_NUMERIC"
        ranking_basis = "two_native_streams_no_cross_stream_rank_fabrication"
    elif method == "lightrag":
        data = retrieval.get("raw_data") or result.get("context", {}).get("raw_data") or {}
        chunks = (data.get("data") or data).get("chunks") or [] if isinstance(data, dict) else []
        for rank, chunk in enumerate(chunks, 1):
            if not isinstance(chunk, dict):
                continue
            segment_ids = chunk.get("segment_ids") or chunk.get("video_segment_id") or []
            if isinstance(segment_ids, str):
                segment_ids = [segment_ids]
            score = chunk.get("score")
            direct = _source_from_content(chunk)
            if not segment_ids and direct:
                predictions.append(_direct_prediction(
                    source=direct, source_id=str(chunk.get("id") or f"chunk-{rank}"),
                    native_rank=rank, native_score=float(score) if score is not None else None,
                    score_source="structured_context_score" if score is not None else "structured_context_order",
                ))
            for segment_id in segment_ids:
                predictions.append(_prediction(
                    segment_id=str(segment_id), source_id=str(chunk.get("id") or segment_id),
                    native_rank=rank, native_score=float(score) if score is not None else None,
                    score_source="structured_context_score" if score is not None else "structured_context_order",
                    store=segment_store,
                ))
        score_contract = "NATIVE_NUMERIC" if any(row["native_score"] is not None for row in predictions) else "NATIVE_RANK_ONLY"
        ranking_basis = "native_structured_context_order"
    elif method in {"graphrag-l", "graphrag-g"}:
        source_records = retrieval.get("source_records") or []
        for rank, source in enumerate(source_records, 1):
            if not isinstance(source, dict):
                continue
            segment_ids = source.get("segment_ids") or source.get("video_segment_id") or []
            if isinstance(segment_ids, str):
                segment_ids = [segment_ids]
            score = source.get("score")
            direct = source if source.get("video_id") is not None else _source_from_content(source)
            if not segment_ids and direct:
                predictions.append(_direct_prediction(
                    source=direct, source_id=str(source.get("id") or f"source-{rank}"),
                    native_rank=rank, native_score=float(score) if score is not None else None,
                    score_source="graphrag_source_score" if score is not None else "citation_order",
                ))
            for segment_id in segment_ids:
                predictions.append(_prediction(
                    segment_id=str(segment_id), source_id=str(source.get("id") or segment_id),
                    native_rank=rank, native_score=float(score) if score is not None else None,
                    score_source="graphrag_source_score" if score is not None else "citation_order",
                    store=segment_store,
                ))
        score_contract = "NATIVE_NUMERIC" if any(row["native_score"] is not None for row in predictions) else "NATIVE_RANK_ONLY"
        ranking_basis = "native_source_citation_order"
    else:
        raise ValueError(f"unsupported method: {method}")
    return {
        "method": method,
        "query": result.get("query") or {},
        "predictions": _dedupe_predictions(predictions),
        "score_contract": score_contract,
        "ranking_basis": ranking_basis,
        "timestamp_basis": "clip_local_seconds_origin_0",
        "gt_used_for_prediction": False,
    }
