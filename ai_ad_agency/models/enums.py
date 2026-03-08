"""
Enumerations for the AI Ad Agency platform.
All shared enums live here to avoid circular imports.
"""
from enum import Enum


class HookCategory(str, Enum):
    CURIOSITY = "curiosity"
    WARNING = "warning"
    DISCOVERY = "discovery"
    URGENCY = "urgency"
    AUTHORITY = "authority"


class ScriptStyle(str, Enum):
    TESTIMONIAL = "testimonial"
    AUTHORITY = "authority"
    STORY = "story"
    DIRECT_RESPONSE = "direct_response"
    COMPARISON = "comparison"
    ALMOST_MISSED = "almost_missed"
    NEWS_UPDATE = "news_update"
    UGC = "ugc"


class VideoLength(str, Enum):
    SHORT = "short"      # 15-20 sec
    MEDIUM = "medium"    # 30-45 sec
    LONG = "long"        # 45-60 sec


class ImageStyle(str, Enum):
    LIFESTYLE = "lifestyle"
    TESTIMONIAL_STILL = "testimonial_still"
    HEADLINE = "headline"
    QUOTE_CARD = "quote_card"
    INFOGRAPHIC = "infographic"
    BEFORE_AFTER = "before_after"
    UGC_SCREENSHOT = "ugc_screenshot"


class AssetStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    ACCEPTED = "accepted"
    CACHED = "cached"


class AgeGroup(str, Enum):
    YOUNG_ADULT = "young_adult"       # 18-30
    MIDDLE_AGED = "middle_aged"       # 31-50
    OLDER_ADULT = "older_adult"       # 51+


class GenderPresentation(str, Enum):
    MASCULINE = "masculine"
    FEMININE = "feminine"
    NEUTRAL = "neutral"


class AppearanceTag(str, Enum):
    # Broad appearance categories for catalog diversity tracking
    LIGHT = "light"
    MEDIUM = "medium"
    MEDIUM_DARK = "medium_dark"
    DARK = "dark"
    EAST_ASIAN = "east_asian"
    SOUTH_ASIAN = "south_asian"
    LATIN = "latin"
    MIDDLE_EASTERN = "middle_eastern"
    MIXED = "mixed"


class VoiceTone(str, Enum):
    CALM = "calm"
    AUTHORITATIVE = "authoritative"
    CONVERSATIONAL = "conversational"
    TESTIMONIAL = "testimonial"
    ENERGETIC = "energetic"
    WARM = "warm"


class CreativeType(str, Enum):
    STATIC_IMAGE = "static_image"
    TALKING_HEAD = "talking_head"
    MIXED_VIDEO = "mixed_video"
    BROLL_CLIP = "broll_clip"
    VOICE_ONLY = "voice_only"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"
    GROQ = "groq"


class AvatarProvider(str, Enum):
    HEYGEN = "heygen"
    TAVUS = "tavus"
    DID = "did"
    SYNTHESIA = "synthesia"
    MOCK = "mock"


class ImageProvider(str, Enum):
    OPENAI_DALLE = "openai_dalle"
    STABILITY = "stability"
    FLUX = "flux"
    MOCK = "mock"


class VideoProvider(str, Enum):
    RUNWAY = "runway"
    PIKA = "pika"
    KLING = "kling"
    LUMA = "luma"
    MOCK = "mock"


class VoiceProvider(str, Enum):
    ELEVENLABS = "elevenlabs"
    OPENAI_TTS = "openai_tts"
    HEYGEN_NATIVE = "heygen_native"
    MOCK = "mock"


class WardrobeStyle(str, Enum):
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    BUSINESS_CASUAL = "business_casual"
    ACTIVEWEAR = "activewear"
    EVERYDAY = "everyday"


class RenderStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
