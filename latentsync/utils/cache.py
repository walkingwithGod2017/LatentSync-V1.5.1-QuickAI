import hashlib
import os
from pathlib import Path

import numpy as np
import torch


def make_file_cache_key(path: str, extra=None) -> str:
    stat = os.stat(path)
    parts = [
        os.path.abspath(path),
        str(stat.st_size),
        str(stat.st_mtime_ns),
    ]
    if extra:
        for key in sorted(extra):
            parts.append(f"{key}={extra[key]}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _to_safe_cache_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value)).cpu()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_safe_cache_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_safe_cache_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported cache value type: {type(value).__name__}")


def load_cache(path: str):
    if not os.path.isfile(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"Cache load failed, removing stale cache: {type(exc).__name__} - {exc} - {path}")
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def save_cache(path: str, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(_to_safe_cache_value(obj), path)
