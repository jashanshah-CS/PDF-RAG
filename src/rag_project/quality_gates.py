"""Validate a retrieval benchmark report against versioned quality gates."""

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Load a JSON object with a useful error for malformed inputs."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exception:
        raise ValueError(f"{label} file does not exist: {path}") from exception
    except json.JSONDecodeError as exception:
        raise ValueError(f"Invalid {label} JSON: {exception.msg}.") from exception
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return payload


def _value_at(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def _rules(policy: Mapping[str, Any], name: str) -> dict[str, Any]:
    rules = policy.get(name)
    if not isinstance(rules, dict) or not rules:
        raise ValueError(f"Quality-gate policy needs a non-empty {name} object.")
    if not all(isinstance(path, str) and path.strip() for path in rules):
        raise ValueError(f"Every {name} metric path must be a non-empty string.")
    return rules


def evaluate_quality_gates(
    report: Mapping[str, Any], policy: Mapping[str, Any]
) -> list[str]:
    """Return every failed gate; an empty list means the report is accepted."""
    if policy.get("schema_version") != 1:
        raise ValueError("Quality-gate policy schema_version must be 1.")

    failures: list[str] = []
    for path, expected in _rules(policy, "required_values").items():
        try:
            actual = _value_at(report, path)
        except KeyError:
            failures.append(f"{path}: missing (expected {expected!r})")
            continue
        if actual != expected:
            failures.append(f"{path}: {actual!r} != required {expected!r}")

    for path, minimum in _rules(policy, "minimum_values").items():
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
            raise ValueError(f"Minimum for {path} must be numeric.")
        try:
            actual = _value_at(report, path)
        except KeyError:
            failures.append(f"{path}: missing (minimum {minimum})")
            continue
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            failures.append(f"{path}: {actual!r} is not numeric")
        elif actual < minimum:
            failures.append(f"{path}: {actual} < minimum {minimum}")
    return failures
