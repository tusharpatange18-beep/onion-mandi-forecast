"""Load the project configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read configs/config.yaml (or the given path) and return it as a dict."""
    cfg_path = path or ROOT / "configs" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
