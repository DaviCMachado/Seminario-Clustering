"""Leitura e composição de configurações YAML do experimento."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


def _merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict:
    """Carrega `inherits` relativo ao YAML e aplica seus campos por cima."""
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("inherits", None)
    if parent:
        return _merge(load_config(path.parent / parent), config)
    return config
