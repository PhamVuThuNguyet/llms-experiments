"""Response parsing and semantic normalization for MLLM evaluation."""

from __future__ import annotations

import json
import re
from typing import Any


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except (json.JSONDecodeError, TypeError):
        match = _JSON_OBJECT_RE.search(cleaned)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None


def normalize_json_value(value: Any) -> Any:
    """Canonical value for semantic comparison (order-insensitive)."""
    if isinstance(value, dict):
        return {
            str(key): normalize_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return text


def normalize_response_text(response_text: str | None) -> str | None:
    """Return canonical JSON string, or None if unparseable."""
    parsed = _extract_json_object(response_text or "")
    if parsed is None:
        return None
    return json.dumps(normalize_json_value(parsed), sort_keys=True, separators=(",", ":"))


def parse_binary_field(response_text: str | None, field: str) -> int | None:
    parsed = _extract_json_object(response_text or "")
    if parsed is None or field not in parsed:
        return None
    value = normalize_json_value(parsed[field])
    if value in (0, 1):
        return int(value)
    if value in ("0", "1"):
        return int(value)
    return None


TRIAGE_RISK_MAPPING = {
    "high risk": "High Risk",
    "low risk": "Low Risk",
    "unclear": "Unclear",
}


def parse_triage_risk(response_text: str | None) -> str | None:
    parsed = _extract_json_object(response_text or "")
    if parsed is None or "risk" not in parsed:
        return None
    raw = str(parsed["risk"]).strip().lower().replace("_", " ").replace("-", " ")
    raw = re.sub(r"\s+", " ", raw)
    return TRIAGE_RISK_MAPPING.get(raw)


def responses_semantically_equal(a: str | None, b: str | None) -> bool:
    left = normalize_response_text(a)
    right = normalize_response_text(b)
    if left is None or right is None:
        return False
    return left == right
