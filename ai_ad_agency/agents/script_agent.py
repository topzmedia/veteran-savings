"""
Script Agent — generates video ad scripts from hooks, with multiple styles and lengths.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..models.enums import AssetStatus, ScriptStyle, VideoLength
from ..models.schemas import Hook, OfferConfig, Script, ScriptSection
from ..providers.llm_provider import BaseLLMProvider
from ..utils.io import models_to_csv, write_models_json
from ..utils.logging_utils import get_module_logger
from ..utils.prompt_templates import (
    SCRIPT_LENGTH_DESCRIPTIONS,
    SCRIPT_STYLE_DESCRIPTIONS,
    SCRIPT_SYSTEM_PROMPT,
    SCRIPT_USER_PROMPT,
)
from ..utils.retries import TransientError, with_retries

logger = get_module_logger("script_agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUTPUT_DIR = Path("ai_ad_agency/outputs/scripts")

# Duration estimates per VideoLength (seconds)
_DURATION_MAP: Dict[str, int] = {
    VideoLength.SHORT: 18,
    VideoLength.MEDIUM: 37,
    VideoLength.LONG: 52,
}

# Default style+length combos to generate per hook (2-4 variants)
_DEFAULT_COMBOS: List[Tuple[ScriptStyle, VideoLength]] = [
    (ScriptStyle.DIRECT_RESPONSE, VideoLength.SHORT),
    (ScriptStyle.STORY, VideoLength.MEDIUM),
    (ScriptStyle.TESTIMONIAL, VideoLength.MEDIUM),
    (ScriptStyle.AUTHORITY, VideoLength.LONG),
]


# ---------------------------------------------------------------------------
# Voice-safe text generation
# ---------------------------------------------------------------------------

def _make_voice_safe(text: str) -> str:
    """
    Strip special characters and normalize text for TTS voice rendering.

    Steps:
      - Remove URLs (http/https/www)
      - Strip markdown formatting (**, __, ##, *, _)
      - Normalize smart quotes and curly quotes to straight quotes
      - Remove asterisks
      - Normalize multiple whitespace to single space
      - Strip leading/trailing whitespace
    """
    # Resolve markdown links before removing URLs: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)

    # Strip markdown bold/italic markers
    text = re.sub(r"\*{1,3}", "", text)
    text = re.sub(r"_{1,3}", "", text)

    # Strip markdown headers
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Normalize curly/smart quotes to straight quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2032", "'").replace("\u2033", '"')

    # Remove backticks (inline code markers)
    text = text.replace("`", "")

    # Remove remaining square/angle brackets
    text = re.sub(r"[<>\[\]]", "", text)

    # Collapse multiple spaces/newlines to single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _fetch_script(
    llm: BaseLLMProvider,
    hook: Hook,
    offer: OfferConfig,
    style: ScriptStyle,
    length: VideoLength,
) -> dict:
    """
    Call the LLM and return the raw parsed JSON dict for a single script.
    Raises TransientError if the response is malformed.
    """
    style_desc = SCRIPT_STYLE_DESCRIPTIONS.get(style.value, style.value)
    length_desc = SCRIPT_LENGTH_DESCRIPTIONS.get(length.value, length.value)

    system_prompt = SCRIPT_SYSTEM_PROMPT
    user_prompt = SCRIPT_USER_PROMPT.format(
        style=style.value,
        hook_text=hook.text,
        offer_name=offer.offer_name,
        offer_description=offer.offer_description,
        target_audience=offer.target_audience,
        pain_points=", ".join(offer.pain_points),
        benefits=", ".join(offer.benefits),
        cta=offer.cta,
        tone=", ".join(offer.tone) if offer.tone else "professional, empathetic",
        length=length.value,
        length_description=length_desc,
        style_description=style_desc,
    )

    raw = llm.complete_json(
        system_prompt,
        user_prompt,
        temperature=0.85,
        max_tokens=2048,
    )

    if not isinstance(raw, dict):
        raise TransientError(
            f"Expected dict from LLM for script, got {type(raw).__name__}"
        )

    # Validate required keys
    required_keys = {"hook", "problem", "discovery", "benefit", "cta", "full_text"}
    missing = required_keys - set(raw.keys())
    if missing:
        raise TransientError(
            f"Script response missing required keys: {missing}"
        )

    return raw


# ---------------------------------------------------------------------------
# Script builder
# ---------------------------------------------------------------------------

def _build_script(
    raw: dict,
    hook: Hook,
    style: ScriptStyle,
    length: VideoLength,
    offer: OfferConfig,
) -> Script:
    """
    Convert a raw LLM response dict into a Script model instance.
    """
    sections = ScriptSection(
        hook=str(raw.get("hook", "")).strip(),
        problem=str(raw.get("problem", "")).strip(),
        discovery=str(raw.get("discovery", "")).strip(),
        benefit=str(raw.get("benefit", "")).strip(),
        cta=str(raw.get("cta", "")).strip(),
    )

    full_text = str(raw.get("full_text", "")).strip()
    if not full_text:
        # Reconstruct from sections if full_text is missing
        full_text = " ".join([
            sections.hook,
            sections.problem,
            sections.discovery,
            sections.benefit,
            sections.cta,
        ]).strip()

    voice_safe = _make_voice_safe(full_text)

    # Use LLM-provided duration estimate if reasonable, otherwise use our map
    llm_duration = raw.get("estimated_duration_sec")
    if isinstance(llm_duration, (int, float)) and 5 <= int(llm_duration) <= 120:
        estimated_duration = int(llm_duration)
    else:
        estimated_duration = _DURATION_MAP.get(length, 37)

    tags: List[str] = raw.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tags = [str(t) for t in tags if t]

    return Script(
        hook_id=hook.hook_id,
        hook_text=hook.text,
        style=style,
        length=length,
        sections=sections,
        full_text=full_text,
        voice_safe_text=voice_safe,
        estimated_duration_sec=estimated_duration,
        tags=tags,
        offer_name=offer.offer_name,
        status=AssetStatus.COMPLETED,
    )


# ---------------------------------------------------------------------------
# Combo selection
# ---------------------------------------------------------------------------

def _select_combos(
    offer: OfferConfig,
    scripts_per_hook: int,
) -> List[Tuple[ScriptStyle, VideoLength]]:
    """
    Build a list of (style, length) combos based on the offer's configured
    styles and lengths. Returns `scripts_per_hook` combos (2-4).
    """
    scripts_per_hook = max(2, min(4, scripts_per_hook))

    styles = offer.script_styles or list(ScriptStyle)
    lengths = offer.video_lengths or list(VideoLength)

    # Build all combos and cycle through them
    all_combos: List[Tuple[ScriptStyle, VideoLength]] = []
    for s in styles:
        for ln in lengths:
            all_combos.append((s, ln))

    if not all_combos:
        all_combos = _DEFAULT_COMBOS[:]

    # Take a spread across the full combo list
    step = max(1, len(all_combos) // scripts_per_hook)
    selected: List[Tuple[ScriptStyle, VideoLength]] = []
    for i in range(scripts_per_hook):
        idx = (i * step) % len(all_combos)
        selected.append(all_combos[idx])

    # Deduplicate while preserving order
    seen: set = set()
    unique: List[Tuple[ScriptStyle, VideoLength]] = []
    for combo in selected:
        if combo not in seen:
            seen.add(combo)
            unique.append(combo)

    # If deduplication reduced us below target, pad with remaining combos
    for combo in all_combos:
        if len(unique) >= scripts_per_hook:
            break
        if combo not in seen:
            seen.add(combo)
            unique.append(combo)

    return unique[:scripts_per_hook]


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------

def run_script_agent(
    llm: BaseLLMProvider,
    hooks: List[Hook],
    offer: OfferConfig,
    scripts_per_hook: int = 3,
    output_dir: str | Path | None = None,
) -> List[Script]:
    """
    Generate 2-4 script variants per hook using different styles and lengths.

    Steps:
      1. For each hook, select a set of (style, length) combos.
      2. Call the LLM per combo (with retries).
      3. Parse the JSON response into a Script model.
      4. Generate voice_safe_text by stripping special chars.
      5. Save JSON + CSV to outputs/scripts/.
      6. Return List[Script].

    Args:
        llm: LLM provider instance.
        hooks: List of Hook objects to generate scripts from.
        offer: OfferConfig with offer details.
        scripts_per_hook: Number of script variants per hook (clamped 2-4).
        output_dir: Override output directory (default: ai_ad_agency/outputs/scripts).

    Returns:
        List[Script] — all generated scripts.
    """
    out_dir = Path(output_dir) if output_dir else _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    scripts_per_hook = max(2, min(4, scripts_per_hook))
    combos = _select_combos(offer, scripts_per_hook)

    logger.info(
        "ScriptAgent: generating %d scripts per hook for %d hooks "
        "(%d total expected) with combos: %s",
        scripts_per_hook,
        len(hooks),
        len(hooks) * scripts_per_hook,
        [(s.value, ln.value) for s, ln in combos],
    )

    all_scripts: List[Script] = []
    failed_count = 0

    for hook_idx, hook in enumerate(hooks):
        logger.debug(
            "[Hook %d/%d] hook_id=%s text=%s...",
            hook_idx + 1,
            len(hooks),
            hook.hook_id,
            hook.text[:60],
        )

        for style, length in combos:
            try:
                raw = with_retries(
                    _fetch_script,
                    llm,
                    hook,
                    offer,
                    style,
                    length,
                    max_attempts=4,
                    base_delay=2.0,
                    max_delay=30.0,
                    reraise=True,
                    retryable_exceptions=(
                        TransientError,
                        ValueError,  # JSON parse failures
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to generate script for hook_id=%s style=%s length=%s: %s",
                    hook.hook_id,
                    style.value,
                    length.value,
                    exc,
                )
                failed_count += 1
                continue

            try:
                script = _build_script(raw, hook, style, length, offer)
            except Exception as exc:
                logger.warning(
                    "Failed to build Script model for hook_id=%s style=%s length=%s: %s",
                    hook.hook_id,
                    style.value,
                    length.value,
                    exc,
                )
                failed_count += 1
                continue

            all_scripts.append(script)
            logger.debug(
                "Script created: script_id=%s style=%s length=%s words=%d",
                script.script_id,
                style.value,
                length.value,
                script.word_count,
            )

        if (hook_idx + 1) % 10 == 0:
            logger.info(
                "[%d/%d] hooks processed — %d scripts generated so far",
                hook_idx + 1,
                len(hooks),
                len(all_scripts),
            )

    logger.info(
        "ScriptAgent complete: %d scripts generated | %d failures",
        len(all_scripts),
        failed_count,
    )

    # Persist
    json_path = out_dir / "scripts.json"
    csv_path = out_dir / "scripts.csv"
    write_models_json(all_scripts, json_path)
    models_to_csv(all_scripts, csv_path)
    logger.info("Saved %d scripts → %s", len(all_scripts), out_dir)

    return all_scripts


class ScriptAgent:
    def __init__(self, config: object, llm_provider: BaseLLMProvider):
        self.config = config
        self.llm = llm_provider

    def generate(
        self,
        offer: OfferConfig,
        hooks: List[Hook],
        scripts_per_hook: int = 3,
        output_dir: str | Path | None = None,
    ) -> List[Script]:
        return run_script_agent(
            llm=self.llm,
            hooks=hooks,
            offer=offer,
            scripts_per_hook=scripts_per_hook,
            output_dir=output_dir,
        )
