"""Load the project configuration.

The config file describes onion. Other crops (tomato, potato) are listed under `crops:` in the same file; each one
only lists the settings that differ from onion. Choose the crop with the environment variable MANDI_CROP, e.g.

    MANDI_CROP=tomato python -m mandi.run_phase1

Without MANDI_CROP (or with MANDI_CROP=onion) the onion settings are used exactly as written.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CROP = "onion"


def _merge(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `base` with `changes` applied.

    A setting that is itself a group of settings (a dict) is merged key by key; anything else (a number, a text,
    a list) replaces the old value completely.
    """
    result = copy.deepcopy(base)
    for key, new_value in changes.items():
        old_value = result.get(key)
        if isinstance(old_value, dict) and isinstance(new_value, dict):
            # Both are groups of settings: merge them one level deeper.
            result[key] = _merge(old_value, new_value)
        else:
            # Plain value or list: the crop's value wins.
            result[key] = copy.deepcopy(new_value)
    return result


def load_config(path: Path | None = None, crop: str | None = None) -> dict[str, Any]:
    """Read configs/config.yaml (or the given path) and return the settings for one crop as a dict.

    `crop` defaults to the MANDI_CROP environment variable, and to onion if that is not set.
    The returned dict has no `crops` section. For onion it is exactly the file's onion settings (nothing added, so
    onion runs keep the same config fingerprint as before); other crops get their name under `crop`.
    Use `crop_of(cfg)` to read the crop name.
    """
    cfg_path = path or ROOT / "configs" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Take the per-crop settings out of the config; what is left are the onion settings.
    crop_settings = cfg.pop("crops", {})

    # Which crop? Argument first, then the environment variable, then onion.
    if crop is None:
        crop = os.environ.get("MANDI_CROP", DEFAULT_CROP)
    crop = crop.strip().lower()

    # Onion: return the settings unchanged.
    if crop == DEFAULT_CROP:
        return cfg

    if crop not in crop_settings:
        known = [DEFAULT_CROP] + list(crop_settings)
        raise ValueError(f"Unknown crop '{crop}'. Known crops: {known}")

    # Apply the crop's own settings on top of the onion settings.
    cfg = _merge(cfg, crop_settings[crop])
    cfg["crop"] = crop
    return cfg


def crop_of(cfg: dict[str, Any]) -> str:
    """The crop these settings are for ("onion" if the config does not say)."""
    return cfg.get("crop", DEFAULT_CROP)


def results_dir(cfg: dict[str, Any]) -> Path:
    """Folder for this crop's result tables (results/ for onion, results/<crop>/ for the others)."""
    return ROOT / cfg["paths"]["results"]
