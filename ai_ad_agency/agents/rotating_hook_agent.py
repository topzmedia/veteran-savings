"""
Rotating Hook Agent — generates rephrased variants of existing hooks.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

from ..models.schemas import Hook, RotatedHook
from ..providers.llm_provider import BaseLLMProvider
from ..utils.dedupe import TextDedupe, string_similarity
from ..utils.io import models_to_csv, write_models_json
from ..utils.logging_utils import get_module_logger
from ..utils.prompt_templates import ROTATING_HOOK_SYSTEM, ROTATING_HOOK_USER
from ..utils.retries import TransientError, with_retries

logger = get_module_logger("rotating_hook_agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_HOOK_CHARS = 150
_PARENT_SIMILARITY_THRESHOLD = 0.85  # Reject variants too similar to parent
_OUTPUT_DIR = Path("ai_ad_agency/outputs/hooks")


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _fetch_variants_for_hook(
    llm: BaseLLMProvider,
    hook: Hook,
    count: int,
    offer_description: str = "",
) -> List[str]:
    """
    Call the LLM to generate `count` variants of a single hook.
    Raises TransientError if the response shape is unexpected.
    """
    user_prompt = ROTATING_HOOK_USER.format(
        count=count,
        hook_text=hook.text,
        offer_description=offer_description,
        category=hook.category.value,
        max_chars=_MAX_HOOK_CHARS,
    )

    raw = llm.complete_json(
        ROTATING_HOOK_SYSTEM,
        user_prompt,
        temperature=0.95,
        max_tokens=2048,
    )

    if isinstance(raw, list):
        return [str(item).strip() for item in raw if item and str(item).strip()]

    if isinstance(raw, dict):
        for key in ("variants", "hooks", "results", "data"):
            if isinstance(raw.get(key), list):
                return [str(item).strip() for item in raw[key] if item]

    raise TransientError(
        f"Unexpected LLM response shape for hook_id={hook.hook_id}: {type(raw)}"
    )


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------

def run_rotating_hook_agent(
    llm: BaseLLMProvider,
    hooks: List[Hook],
    variants_per_hook: int = 4,
    offer_description: str = "",
    output_dir: str | Path | None = None,
    dedupe_threshold: float = 0.85,
) -> List[RotatedHook]:
    """
    For each input hook, generate 3–5 rotated variants.

    Steps:
      1. For each hook, call LLM with retry logic.
      2. Reject variants that are too similar to the parent (>85% similarity).
      3. Deduplicate across all variants globally.
      4. Save JSON + CSV.
      5. Return List[RotatedHook].
    """
    # Clamp variants_per_hook to the required range
    variants_per_hook = max(3, min(5, variants_per_hook))
    # Request a few more to allow for similarity rejections
    request_count = variants_per_hook + 2

    out_dir = Path(output_dir) if output_dir else _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "RotatingHookAgent: rotating %d hooks, requesting %d variants each",
        len(hooks),
        request_count,
    )

    global_dedupe = TextDedupe(similarity_threshold=dedupe_threshold)
    all_rotated: List[RotatedHook] = []
    total_processed = 0
    total_rejected_similarity = 0
    total_rejected_dedupe = 0

    for idx, hook in enumerate(hooks):
        # Progress log every 10 hooks
        if idx > 0 and idx % 10 == 0:
            logger.info(
                "[%d/%d] hooks processed — %d variants accepted so far",
                idx,
                len(hooks),
                len(all_rotated),
            )

        try:
            raw_variants = with_retries(
                _fetch_variants_for_hook,
                llm,
                hook,
                request_count,
                offer_description,
                max_attempts=4,
                base_delay=2.0,
                max_delay=30.0,
                reraise=True,
            )
        except Exception as exc:
            logger.warning(
                "Skipping hook_id=%s (%s) after LLM failure: %s",
                hook.hook_id,
                hook.text[:50],
                exc,
            )
            total_processed += 1
            continue

        accepted_for_hook = 0
        for variant_text in raw_variants:
            variant_text = variant_text.strip()
            if not variant_text:
                continue

            # Trim to max length
            if len(variant_text) > _MAX_HOOK_CHARS:
                variant_text = variant_text[: _MAX_HOOK_CHARS].rsplit(" ", 1)[0]

            # Check parent similarity — reject if too close
            sim = string_similarity(variant_text, hook.text)
            if sim >= _PARENT_SIMILARITY_THRESHOLD:
                logger.debug(
                    "Variant too similar to parent (sim=%.2f), rejecting: %s",
                    sim,
                    variant_text[:60],
                )
                total_rejected_similarity += 1
                continue

            # Global deduplication
            if not global_dedupe.add(variant_text):
                total_rejected_dedupe += 1
                continue

            rotated = RotatedHook(
                parent_hook_id=hook.hook_id,
                text=variant_text,
                similarity_score=round(sim, 4),
            )
            all_rotated.append(rotated)
            accepted_for_hook += 1

        logger.debug(
            "hook_id=%s: accepted %d / %d variants",
            hook.hook_id,
            accepted_for_hook,
            len(raw_variants),
        )
        total_processed += 1

    dedupe_stats = global_dedupe.stats()
    logger.info(
        "RotatingHookAgent complete: %d rotated hooks | "
        "parent_similarity_rejected=%d | dedupe_rejected=%d | "
        "exact_dupes=%d | near_dupes=%d",
        len(all_rotated),
        total_rejected_similarity,
        total_rejected_dedupe,
        dedupe_stats["exact_duplicates_rejected"],
        dedupe_stats["near_duplicates_rejected"],
    )

    # Persist
    json_path = out_dir / "rotated_hooks.json"
    csv_path = out_dir / "rotated_hooks.csv"
    write_models_json(all_rotated, json_path)
    models_to_csv(all_rotated, csv_path)
    logger.info("Saved %d rotated hooks → %s", len(all_rotated), out_dir)

    return all_rotated


class RotatingHookAgent:
    def __init__(self, config: object, llm_provider: BaseLLMProvider):
        self.config = config
        self.llm = llm_provider

    def generate_variants(
        self,
        hooks: List[Hook],
        variants_per_hook: int = 4,
        output_dir: str | Path | None = None,
    ) -> List[RotatedHook]:
        return run_rotating_hook_agent(
            llm=self.llm,
            hooks=hooks,
            variants_per_hook=variants_per_hook,
            output_dir=output_dir,
        )
