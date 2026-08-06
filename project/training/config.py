"""Small, strict helpers shared by the command-line entry points.

The training machine deliberately has two isolated environments.  Keeping the
configuration loader dependency-light lets repository checks and unit tests run
without importing torch/transformers.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


class ConfigError(ValueError):
    """Raised when a user configuration is malformed."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ConfigError("YAML configuration requires PyYAML; use JSON or install pyyaml") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"top-level config must be an object, got {type(value).__name__}")
    return value


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a JSON or YAML object from *path*."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {config_path}")
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ConfigError(f"top-level config must be an object, got {type(value).__name__}")
        return value
    if suffix in {".yaml", ".yml"}:
        return _load_yaml(config_path)
    raise ConfigError(f"unsupported config extension {suffix!r}; expected .json/.yaml/.yml")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input.

    Lists are intentionally replaced rather than concatenated.  This makes a
    target-module or seed override predictable from the command line.
    """

    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_scalar(raw: str) -> Any:
    """Parse one command-line override value using YAML-like JSON scalars."""

    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_overrides(config: Mapping[str, Any], assignments: Sequence[str]) -> dict[str, Any]:
    """Apply dotted ``section.key=value`` assignments to a copy of *config*."""

    result: dict[str, Any] = copy.deepcopy(dict(config))
    for assignment in assignments:
        if "=" not in assignment:
            raise ConfigError(f"override must be KEY=VALUE, got {assignment!r}")
        dotted_key, raw_value = assignment.split("=", 1)
        parts = [part.strip() for part in dotted_key.split(".") if part.strip()]
        if not parts:
            raise ConfigError(f"empty override key in {assignment!r}")
        cursor: MutableMapping[str, Any] = result
        for part in parts[:-1]:
            current = cursor.get(part)
            if current is None:
                current = {}
                cursor[part] = current
            if not isinstance(current, MutableMapping):
                raise ConfigError(f"cannot set {dotted_key!r}: {part!r} is not an object")
            cursor = current
        cursor[parts[-1]] = parse_scalar(raw_value)
    return result


def require_keys(config: Mapping[str, Any], paths: Sequence[str]) -> None:
    """Validate that all dotted paths exist and are non-empty."""

    missing: list[str] = []
    for path in paths:
        value: Any = config
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if value is None or value == "":
            missing.append(path)
    if missing:
        raise ConfigError("missing required configuration: " + ", ".join(missing))


def resolve_path(value: str | os.PathLike[str], *, relative_to: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a configured path, optionally relative to the config directory."""

    path = Path(os.path.expandvars(os.path.expanduser(os.fspath(value))))
    if not path.is_absolute() and relative_to is not None:
        path = Path(relative_to) / path
    return path.resolve()


def redact_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a log-safe copy with likely credentials redacted."""

    sensitive = {"api_key", "token", "password", "secret", "authorization"}

    def visit(value: Any, key: str = "") -> Any:
        if any(marker in key.lower() for marker in sensitive):
            return "***REDACTED***"
        if isinstance(value, Mapping):
            return {str(k): visit(v, str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return copy.deepcopy(value)

    return visit(config)
