"""
Script Variant Agent — creates additional variants of existing scripts by cycling
through variation aspects (intro, cta, tone, framing, benefit_emphasis).
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from ..models.enums import AssetStatus
from ..models.schemas import Script, ScriptSection, ScriptVariant
from ..providers.llm_provider import BaseLLMProvider
from ..utils.io import models_to_csv, write_models_json
from ..utils.logging_utils import get_module_logger
from ..utils.prompt_templates import (
    SCRIPT_SYSTEM_PROMPT,
    SCRIPT_VARIANT_USER,
    VARIATION_ASPECTS,
)
from ..utils.retries import TransientError, with_retries

logger = get_module_logger("script_variant_agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUTPUT_DIR = Path("ai_ad_agency/outputs/scripts")


# ---------------------------------------------------------------------------
# Voice-safe text generation (mirrors script_agent logic)
# ---------------------------------------------------------------------------

def _make_voice_safe(text: str) -> str:
    """
    Strip special characters and normalize text for TTS voice rendering.
    Mirrors the implementation in script_agent for consistency.
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

    # Remove backticks
    text = text.replace("`", "")

    # Remove remaining square/angle brackets
    text = re.sub(r"[<>\[\]]", "", text)

    # Collapse multiple spaces/newlines to single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Aspect cycler
# ---------------------------------------------------------------------------

def _make_aspect_cycler() -> Iterator[Dict]:
    """
    Returns an infinite iterator that cycles through VARIATION_ASPECTS.
    Each call to next() yields the next aspect dict.
    """
    return itertools.cycle(VARIATION_ASPECTS)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _fetch_variant(
    llm: BaseLLMProvider,
    script: Script,
    variation_aspect: str,
    variation_instruction: str,
) -> dict:
    """
    Call the LLM to produce one variant of the given script.
    Raises TransientError if the response is malformed.
    """
    user_prompt = SCRIPT_VARIANT_USER.format(
        variation_aspect=variation_aspect,
        original_full_text=script.full_text,
        variation_instruction=variation_instruction,
    )

    raw = llm.complete_json(
        SCRIPT_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.85,
        max_tokens=2048,
    )

    if not isinstance(raw, dict):
        raise TransientError(
            f"Expected dict from LLM for script variant, got {type(raw).__name__}"
        )

    # Validate required keys
    required_keys = {"hook", "problem", "discovery", "benefit", "cta", "full_text"}
    missing = required_keys - set(raw.keys())
    if missing:
        raise TransientError(
            f"Script variant response missing required keys: {missing}"
        )

    return raw


# ---------------------------------------------------------------------------
# Variant builder
# ---------------------------------------------------------------------------

def _build_variant(
    raw: dict,
    parent: Script,
    variation_note: str,
) -> ScriptVariant:
    """
    Convert a raw LLM response dict into a ScriptVariant model.
    Inherits style, length, and duration from the parent script.
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
        full_text = " ".join([
            sections.hook,
            sections.problem,
            sections.discovery,
            sections.benefit,
            sections.cta,
        ]).strip()

    voice_safe = _make_voice_safe(full_text)

    # Use LLM-provided duration if reasonable, else inherit from parent
    llm_duration = raw.get("estimated_duration_sec")
    if isinstance(llm_duration, (int, float)) and 5 <= int(llm_duration) <= 120:
        estimated_duration = int(llm_duration)
    else:
        estimated_duration = parent.estimated_duration_sec

    return ScriptVariant(
        parent_script_id=parent.script_id,
        hook_id=parent.hook_id,
        style=parent.style,
        length=parent.length,
        sections=sections,
        full_text=full_text,
        voice_safe_text=voice_safe,
        estimated_duration_sec=estimated_duration,
        variation_note=variation_note,
        status=AssetStatus.COMPLETED,
    )


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------

def run_script_variant_agent(
    llm: BaseLLMProvider,
    scripts: List[Script],
    variants_per_script: int = 2,
    output_dir: str | Path | None = None,
) -> List[ScriptVariant]:
    """
    Create 1-2 additional variants per script by cycling through variation aspects.

    Steps:
      1. Maintain a global aspect cycler (intro → cta → tone → framing → benefit_emphasis → ...).
      2. For each script, pick the next `variants_per_script` aspects from the cycler.
      3. Call LLM per aspect (with retries), handle failures gracefully (skip & continue).
      4. Build ScriptVariant models linking to parent via parent_script_id and hook_id.
      5. Save to outputs/scripts/script_variants.json and script_variants.csv.
      6. Return List[ScriptVariant].

    Args:
        llm: LLM provider instance.
        scripts: List of Script objects to create variants from.
        variants_per_script: Number of variants per script (clamped to 1-2).
        output_dir: Override output directory.

    Returns:
        List[ScriptVariant] — all generated variants.
    """
    out_dir = Path(output_dir) if output_dir else _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    variants_per_script = max(1, min(2, variants_per_script))
    aspect_cycler = _make_aspect_cycler()

    logger.info(
        "ScriptVariantAgent: generating %d variant(s) per script for %d scripts "
        "(%d total expected)",
        variants_per_script,
        len(scripts),
        len(scripts) * variants_per_script,
    )

    all_variants: List[ScriptVariant] = []
    failed_count = 0

    for script_idx, script in enumerate(scripts):
        logger.debug(
            "[Script %d/%d] script_id=%s hook_id=%s style=%s length=%s",
            script_idx + 1,
            len(scripts),
            script.script_id,
            script.hook_id,
            script.style.value,
            script.length.value,
        )

        for _ in range(variants_per_script):
            aspect_dict = next(aspect_cycler)
            variation_aspect: str = aspect_dict["aspect"]
            variation_instruction: str = aspect_dict["instruction"]
            variation_note = f"alternate {variation_aspect}"

            try:
                raw = with_retries(
                    _fetch_variant,
                    llm,
                    script,
                    variation_aspect,
                    variation_instruction,
                    max_attempts=3,
                    base_delay=2.0,
                    max_delay=20.0,
                    reraise=True,
                    retryable_exceptions=(
                        TransientError,
                        ValueError,
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to generate variant for script_id=%s aspect=%s: %s",
                    script.script_id,
                    variation_aspect,
                    exc,
                )
                failed_count += 1
                continue

            try:
                variant = _build_variant(raw, script, variation_note)
            except Exception as exc:
                logger.warning(
                    "Failed to build ScriptVariant for script_id=%s aspect=%s: %s",
                    script.script_id,
                    variation_aspect,
                    exc,
                )
                failed_count += 1
                continue

            all_variants.append(variant)
            logger.debug(
                "ScriptVariant created: variant_id=%s parent_script_id=%s note=%s",
                variant.variant_id,
                variant.parent_script_id,
                variant.variation_note,
            )

        if (script_idx + 1) % 10 == 0:
            logger.info(
                "[%d/%d] scripts processed — %d variants generated so far",
                script_idx + 1,
                len(scripts),
                len(all_variants),
            )

    logger.info(
        "ScriptVariantAgent complete: %d variants generated | %d failures",
        len(all_variants),
        failed_count,
    )

    # Persist
    json_path = out_dir / "script_variants.json"
    csv_path = out_dir / "script_variants.csv"
    write_models_json(all_variants, json_path)
    models_to_csv(all_variants, csv_path)
    logger.info("Saved %d script variants → %s", len(all_variants), out_dir)

    return all_variants


class ScriptVariantAgent:
    def __init__(self, config: object, llm_provider: BaseLLMProvider):
        self.config = config
        self.llm = llm_provider

    def generate(
        self,
        scripts: List[Script],
        variants_per_script: int = 2,
        output_dir: str | Path | None = None,
    ) -> List[ScriptVariant]:
        return run_script_variant_agent(
            llm=self.llm,
            scripts=scripts,
            variants_per_script=variants_per_script,
            output_dir=output_dir,
        )
