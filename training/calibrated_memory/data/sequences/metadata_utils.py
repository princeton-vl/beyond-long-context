from __future__ import annotations

from typing import Any


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_entropy_value(payload: Any, sequence_key: str | None = None) -> float | None:
    if isinstance(payload, (int, float)):
        return float(payload)
    if not isinstance(payload, dict):
        return None
    if sequence_key and sequence_key in payload:
        return resolve_entropy_value(payload.get(sequence_key), None)
    for key in ("analytic_bits", "empirical_bits"):
        value = payload.get(key)
        if value is not None:
            resolved = resolve_entropy_value(value, sequence_key)
            if resolved is not None:
                return resolved
    return None


def build_video_metadata(
    record_meta: dict[str, Any] | None,
    *,
    stream_length: int | None = None,
) -> dict[str, Any] | None:
    if record_meta is None and stream_length is None:
        return None
    payload: dict[str, Any] = {}
    sequence_keys: list[str] = []
    primary_key = None
    entropy_payload: Any = None
    if record_meta:
        video_index = record_meta.get("video_index")
        if video_index is not None:
            payload["video_index"] = video_index
        bucket_id = record_meta.get("bucket_id")
        if bucket_id is not None:
            payload["bucket_id"] = bucket_id
        bucket_from = record_meta.get("bucket_from")
        if bucket_from is not None:
            payload["bucket_from"] = bucket_from
        entropy_payload = record_meta.get("entropy_overall")
        sequence_keys = record_meta.get("sequence_keys") or []
        primary_key = sequence_keys[0] if sequence_keys else record_meta.get("sequence_key")
    length_value = None
    if record_meta:
        length_value = record_meta.get("length_value") or record_meta.get("stream_length")
    if length_value is None and stream_length is not None:
        length_value = stream_length
    length_value = _coerce_float(length_value)
    if length_value is not None:
        payload["length_value"] = length_value
    entropy_value = None
    if record_meta:
        entropy_value = record_meta.get("entropy_value")
        if entropy_value is None:
            entropy_value = resolve_entropy_value(
                entropy_payload,
                primary_key,
            )
    entropy_value = _coerce_float(entropy_value)
    if entropy_value is not None:
        payload["entropy_value"] = entropy_value
    if primary_key is not None and entropy_payload is not None:
        if isinstance(entropy_payload, dict):
            empirical_source = entropy_payload.get("empirical_bits")
            analytic_source = entropy_payload.get("analytic_bits")
        else:
            empirical_source = entropy_payload
            analytic_source = None
        empirical = resolve_entropy_value(empirical_source, primary_key)
        analytic = resolve_entropy_value(analytic_source, primary_key)
        if empirical is not None:
            payload["entropy_empirical_bits"] = empirical
        if analytic is not None:
            payload["entropy_analytic_bits"] = analytic
    return payload or None
