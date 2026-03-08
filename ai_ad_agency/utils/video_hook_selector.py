"""
VideoHookSelector — picks a downloaded video hook clip from the local manifest.

Usage:
    selector = VideoHookSelector()
    path = selector.pick()          # random clip
    path = selector.pick(seed=42)   # deterministic pick
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from .logging_utils import get_module_logger

logger = get_module_logger("video_hook_selector")

_DEFAULT_MANIFEST = Path("ai_ad_agency/data/inputs/video_hooks/manifest.json")


class VideoHookSelector:
    def __init__(self, manifest_path: str | Path = _DEFAULT_MANIFEST):
        self._manifest_path = Path(manifest_path)
        self._available: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self._manifest_path.exists():
            logger.warning("Video hook manifest not found: %s", self._manifest_path)
            return
        try:
            data: dict = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            for entry in data.values():
                p = entry.get("local_file")
                if p and Path(p).exists():
                    self._available.append(p)
            logger.info("VideoHookSelector: %d clips available", len(self._available))
        except Exception as exc:
            logger.warning("Failed to load video hook manifest: %s", exc)

    def pick(self, seed: Optional[int] = None) -> Optional[str]:
        """Return a random video hook file path, or None if none are available."""
        if not self._available:
            return None
        rng = random.Random(seed)
        return rng.choice(self._available)

    def pick_by_index(self, index: int) -> Optional[str]:
        """Pick clip by index (wraps around). Useful for deterministic batch variation."""
        if not self._available:
            return None
        return self._available[index % len(self._available)]

    @property
    def count(self) -> int:
        return len(self._available)
