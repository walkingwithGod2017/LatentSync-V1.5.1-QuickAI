import hashlib
import os
from pathlib import Path

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
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def load_cache(path: str):
    if not os.path.isfile(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Cache load failed, removing stale cache: {type(exc).__name__} - {exc} - {path}")
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def save_cache(path: str, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
