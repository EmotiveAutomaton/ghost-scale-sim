"""Configuration loader.

Loads ``config/default.yaml`` into a lightweight attribute-accessible object.
Supports a ``--quick`` smoke mode (deep-merges the ``quick:`` block over the
defaults) and arbitrary dotted-key overrides used by experiment sweeps
(e.g. ``{"signal_model.kappa": 0.1}``).

Every parameter in Spec §3 is expected to be present in the YAML, never hardcoded
in model code. Experiments read from a Config and record the resolved values into
their output CSVs for reproducibility.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


class Config:
    """Attribute- and dotted-access wrapper around the config dict.

    ``cfg.signal_model.kappa`` and ``cfg.get("signal_model.kappa")`` are equivalent.
    ``cfg.raw`` returns the underlying nested dict; ``cfg.flat()`` returns a flat
    dict of dotted-key -> scalar for CSV metadata.
    """

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            val = self._data[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        return Config(val) if isinstance(val, dict) else val

    def __getitem__(self, name: str) -> Any:
        val = self._data[name]
        return Config(val) if isinstance(val, dict) else val

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    def set(self, dotted: str, value: Any) -> None:
        """In-place set of a dotted key (used by sweeps operating on a copy)."""
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def copy(self) -> "Config":
        return Config(copy.deepcopy(self._data))

    @property
    def raw(self) -> dict:
        return self._data

    def flat(self, _prefix: str = "") -> dict:
        out: dict = {}
        for key, val in self._data.items():
            dotted = f"{_prefix}{key}"
            if isinstance(val, dict):
                out.update(Config(val).flat(f"{dotted}."))
            else:
                out[dotted] = val
        return out


def load_config(path: str | Path | None = None, quick: bool = False,
                overrides: dict[str, Any] | None = None) -> Config:
    """Load configuration.

    Args:
        path: YAML path; defaults to ``config/default.yaml``.
        quick: if True, deep-merge the ``quick:`` block over the defaults for fast dev runs.
        overrides: dotted-key -> value overrides applied last (used by sweeps).
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    quick_block = data.pop("quick", {})
    if quick:
        data = _deep_merge(data, quick_block)

    cfg = Config(data)
    if overrides:
        for dotted, value in overrides.items():
            cfg.set(dotted, value)
    return cfg
