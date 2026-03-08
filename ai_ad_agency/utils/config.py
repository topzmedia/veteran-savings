"""
Configuration loader and validator.
Loads app config, offer config, provider config, avatar config, render config, variant config, and QA config.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from ..models.enums import (
    AvatarProvider,
    ImageProvider,
    LLMProvider,
    VideoProvider,
    VoiceProvider,
)
from ..models.schemas import OfferConfig


# ---------------------------------------------------------------------------
# Provider Config
# ---------------------------------------------------------------------------

class LLMProviderConfig(BaseModel):
    provider: LLMProvider = LLMProvider.OPENAI
    api_key: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.9
    max_tokens: int = 2048
    requests_per_minute: int = 60
    base_url: Optional[str] = None


class AvatarProviderConfig(BaseModel):
    provider: AvatarProvider = AvatarProvider.HEYGEN
    api_key: str = ""
    base_url: Optional[str] = None
    requests_per_minute: int = 10
    poll_interval_sec: int = 10
    max_poll_attempts: int = 60  # 10 min at 10s intervals
    render_width: int = 1080
    render_height: int = 1920


class ImageProviderConfig(BaseModel):
    provider: ImageProvider = ImageProvider.OPENAI_DALLE
    api_key: str = ""
    model: str = "dall-e-3"
    requests_per_minute: int = 15
    base_url: Optional[str] = None
    default_size: str = "1024x1024"


class VideoProviderConfig(BaseModel):
    provider: VideoProvider = VideoProvider.MOCK
    api_key: str = ""
    base_url: Optional[str] = None
    requests_per_minute: int = 5
    poll_interval_sec: int = 15
    max_poll_attempts: int = 40


class VoiceProviderConfig(BaseModel):
    provider: VoiceProvider = VoiceProvider.ELEVENLABS
    api_key: str = ""
    base_url: Optional[str] = None
    requests_per_minute: int = 20
    model_id: str = "eleven_monolingual_v1"


class ProviderConfig(BaseModel):
    llm: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    avatar: AvatarProviderConfig = Field(default_factory=AvatarProviderConfig)
    image: ImageProviderConfig = Field(default_factory=ImageProviderConfig)
    video: VideoProviderConfig = Field(default_factory=VideoProviderConfig)
    voice: VoiceProviderConfig = Field(default_factory=VoiceProviderConfig)


# ---------------------------------------------------------------------------
# Avatar Selection Config
# ---------------------------------------------------------------------------

class AvatarSelectionConfig(BaseModel):
    """Quota rules for avatar diversity."""
    total_avatars: int = 50
    quotas: Dict[str, int] = Field(
        default_factory=lambda: {
            "young_adult_masculine": 5,
            "young_adult_feminine": 5,
            "middle_aged_masculine": 10,
            "middle_aged_feminine": 10,
            "older_adult_masculine": 10,
            "older_adult_feminine": 10,
        }
    )
    max_reuse_per_batch: int = 3  # How many times one avatar can appear in a single run
    prefer_realism_score_above: float = 7.0
    catalog_path: str = "ai_ad_agency/data/metadata/avatar_catalog.json"
    sync_on_startup: bool = False


# ---------------------------------------------------------------------------
# Render Config
# ---------------------------------------------------------------------------

class RenderConfig(BaseModel):
    video_formats: List[str] = Field(default_factory=lambda: ["1080x1920", "1080x1080"])
    image_sizes: List[str] = Field(default_factory=lambda: ["1080x1080", "1080x1350", "1200x628"])
    video_fps: int = 30
    video_crf: int = 23         # FFmpeg quality (lower=better, 23 is default)
    video_preset: str = "medium"
    audio_bitrate: str = "128k"
    ffmpeg_threads: int = 4
    output_video_codec: str = "libx264"
    output_audio_codec: str = "aac"
    # Watermark / overlay font
    overlay_font: str = "DejaVuSans-Bold"
    overlay_font_size: int = 52
    hook_card_duration_sec: float = 2.5
    cta_card_duration_sec: float = 3.0
    # B-roll insert config
    broll_insert_position: str = "after_hook"   # after_hook, after_script, random
    music_volume: float = 0.08                  # Background music volume (0-1)


# ---------------------------------------------------------------------------
# Variant Engine Config
# ---------------------------------------------------------------------------

class VariantConfig(BaseModel):
    max_variants: int = 500
    hooks_per_run: int = 200
    rotations_per_hook: int = 4
    scripts_per_hook: int = 3
    script_variants_per_script: int = 2
    images_per_run: int = 200
    broll_clips_per_run: int = 40
    talking_actor_jobs_per_run: int = 100
    # Sampling weights
    use_rotated_hooks_weight: float = 0.6    # 60% chance to use rotated variant
    use_broll_weight: float = 0.7
    use_captions_weight: float = 0.9
    # Anti-duplicate
    max_same_hook_in_batch: int = 10
    max_same_avatar_in_batch: int = 8
    max_same_script_in_batch: int = 5
    dedupe_similarity_threshold: float = 0.85


# ---------------------------------------------------------------------------
# QA Config
# ---------------------------------------------------------------------------

class QAConfig(BaseModel):
    min_video_duration_sec: float = 5.0
    max_video_duration_sec: float = 120.0
    min_file_size_bytes: int = 50_000        # 50 KB
    required_audio: bool = True
    check_dimensions: bool = True
    expected_video_formats: List[str] = Field(
        default_factory=lambda: ["1080x1920", "1080x1080"]
    )
    reject_duplicate_hashes: bool = True
    reject_near_duplicate_metadata: bool = True
    near_duplicate_fields: List[str] = Field(
        default_factory=lambda: ["hook_id", "avatar_id", "script_id"]
    )


# ---------------------------------------------------------------------------
# App Config (top-level)
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    project_name: str = "AI Ad Agency"
    version: str = "1.0.0"
    base_output_dir: str = "ai_ad_agency/outputs"
    base_data_dir: str = "ai_ad_agency/data"
    base_log_dir: str = "ai_ad_agency/data/logs"
    log_level: str = "INFO"
    log_to_file: bool = True
    max_concurrent_renders: int = 5
    max_retries: int = 4
    retry_base_delay_sec: float = 2.0
    cache_enabled: bool = True
    cache_dir: str = "ai_ad_agency/data/cache"
    db_path: str = "ai_ad_agency/data/metadata/runs.db"
    providers: ProviderConfig = Field(default_factory=ProviderConfig)
    avatar_selection: AvatarSelectionConfig = Field(default_factory=AvatarSelectionConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    variants: VariantConfig = Field(default_factory=VariantConfig)
    qa: QAConfig = Field(default_factory=QAConfig)

    @model_validator(mode="after")
    def inject_env_api_keys(self) -> "AppConfig":
        """Pull API keys from environment if not set in config."""
        env_map = {
            "OPENAI_API_KEY": [("providers", "llm", "api_key"), ("providers", "image", "api_key")],
            "HEYGEN_API_KEY": [("providers", "avatar", "api_key")],
            "TAVUS_API_KEY": [("providers", "avatar", "api_key")],
            "STABILITY_API_KEY": [("providers", "image", "api_key")],
            "RUNWAY_API_KEY": [("providers", "video", "api_key")],
            "ELEVENLABS_API_KEY": [("providers", "voice", "api_key")],
        }
        for env_var, paths in env_map.items():
            val = os.environ.get(env_var, "")
            if not val:
                continue
            for path in paths:
                if path[0] == "providers":
                    sub = getattr(self.providers, path[1])
                    if not getattr(sub, path[2]):
                        setattr(sub, path[2], val)
        return self


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_offer_config(path: str | Path) -> OfferConfig:
    data = load_json(path)
    return OfferConfig(**data)


def load_app_config(path: str | Path | None = None) -> AppConfig:
    """Load app config from JSON file or environment defaults."""
    if path and Path(path).exists():
        data = load_json(path)
        return AppConfig(**data)
    return AppConfig()


def load_provider_config(path: str | Path | None = None) -> ProviderConfig:
    if path and Path(path).exists():
        data = load_json(path)
        return ProviderConfig(**data)
    return ProviderConfig()


def save_config(obj: BaseModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(obj.model_dump_json(indent=2))


def ensure_dirs(app_config: AppConfig) -> None:
    """Create all required output/data directories."""
    dirs = [
        app_config.base_output_dir,
        app_config.base_data_dir,
        app_config.base_log_dir,
        app_config.cache_dir,
        f"{app_config.base_output_dir}/hooks",
        f"{app_config.base_output_dir}/scripts",
        f"{app_config.base_output_dir}/images",
        f"{app_config.base_output_dir}/voices",
        f"{app_config.base_output_dir}/avatars",
        f"{app_config.base_output_dir}/broll",
        f"{app_config.base_output_dir}/videos",
        f"{app_config.base_output_dir}/exports",
        f"{app_config.base_data_dir}/inputs",
        f"{app_config.base_data_dir}/intermediate",
        f"{app_config.base_data_dir}/metadata",
        f"{app_config.base_data_dir}/cache",
        f"{app_config.base_data_dir}/logs",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
