"""
Hook Agent — generates ad hooks for all categories, deduplicates, scores, and persists.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import List

from ..models.enums import HookCategory
from ..models.schemas import Hook, OfferConfig
from ..providers.llm_provider import BaseLLMProvider
from ..utils.dedupe import TextDedupe
from ..utils.io import models_to_csv, write_models_json
from ..utils.logging_utils import get_module_logger
from ..utils.prompt_templates import (
    HOOK_CATEGORY_DESCRIPTIONS,
    HOOK_SYSTEM_PROMPT,
    HOOK_USER_PROMPT,
)
from ..utils.retries import TransientError, with_retries

logger = get_module_logger("hook_agent")

# ---------------------------------------------------------------------------
# Pre-scraped hook bank (TransitionalHooks.com)
# ---------------------------------------------------------------------------

_HOOKS_BANK_PATH = Path("ai_ad_agency/data/inputs/transitional_hooks.json")


def _load_hooks_bank() -> List[str]:
    """Load pre-scraped hooks from the static bank file, if it exists."""
    if not _HOOKS_BANK_PATH.exists():
        return []
    try:
        data = json.loads(_HOOKS_BANK_PATH.read_text(encoding="utf-8"))
        texts = []
        for item in data:
            if isinstance(item, dict):
                h = item.get("hook", "") or item.get("text", "")
            elif isinstance(item, str):
                h = item
            else:
                continue
            h = h.strip()
            if h and len(h) > 5:
                texts.append(h)
        logger.info("Loaded %d hooks from bank (%s)", len(texts), _HOOKS_BANK_PATH)
        return texts
    except Exception as exc:
        logger.warning("Could not load hooks bank: %s", exc)
        return []

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_POWER_WORDS: List[str] = [
    "secret",
    "warning",
    "mistake",
    "never",
    "shocking",
    "revealed",
    "truth",
    "hidden",
    "urgent",
    "miss",
    "discover",
]

_MAX_HOOK_CHARS = 150
_MAX_HOOK_WORDS = 25
_OUTPUT_DIR = Path("ai_ad_agency/outputs/hooks")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_hook(text: str) -> float:
    """
    Heuristic strength score for a hook (0–10).

    Scoring rules:
      +0.5  if text contains a question mark
      +0.3  if text contains "you" or "your" (case-insensitive)
      +0.4  if text contains any digit
      +0.2  if word count < 10
      +0.15 per power word present (capped at +1.0 total from power words)
      Max final score capped at 10.0
    """
    score = 5.0  # baseline
    lower = text.lower()

    if "?" in text:
        score += 0.5

    if re.search(r"\byou\b|\byour\b", lower):
        score += 0.3

    if re.search(r"\d", text):
        score += 0.4

    words = text.split()
    if len(words) < 10:
        score += 0.2

    power_bonus = 0.0
    for pw in _POWER_WORDS:
        if pw in lower:
            power_bonus += 0.15
        if power_bonus >= 1.0:
            power_bonus = 1.0
            break
    score += power_bonus

    return min(round(score, 2), 10.0)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _fetch_hooks_for_category(
    llm: BaseLLMProvider,
    offer: OfferConfig,
    category: HookCategory,
    count: int,
) -> List[str]:
    """
    Call the LLM and return a list of raw hook strings for a single category.
    Raises TransientError on unexpected response shapes to trigger retries.
    """
    system_prompt = HOOK_SYSTEM_PROMPT.format(
        max_chars=_MAX_HOOK_CHARS,
        max_words=_MAX_HOOK_WORDS,
    )
    user_prompt = HOOK_USER_PROMPT.format(
        count=count,
        offer_name=offer.offer_name,
        offer_description=offer.offer_description,
        target_audience=offer.target_audience,
        pain_points=", ".join(offer.pain_points),
        benefits=", ".join(offer.benefits),
        tone=", ".join(offer.tone) if offer.tone else "professional, empathetic",
        category=category.value,
        category_description=HOOK_CATEGORY_DESCRIPTIONS.get(category.value, ""),
        max_chars=_MAX_HOOK_CHARS,
    )

    raw = llm.complete_json(system_prompt, user_prompt, temperature=0.95, max_tokens=4096)

    if isinstance(raw, list):
        return [str(item).strip() for item in raw if item and str(item).strip()]

    # Sometimes the LLM wraps in a dict
    if isinstance(raw, dict):
        for key in ("hooks", "results", "data"):
            if isinstance(raw.get(key), list):
                return [str(item).strip() for item in raw[key] if item]

    raise TransientError(
        f"Unexpected LLM response shape for category={category.value}: {type(raw)}"
    )


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------

_FAST_PATH_THRESHOLD = 50  # use single API call when requesting ≤ this many hooks


def _fetch_hooks_single_call(
    llm: BaseLLMProvider,
    offer: OfferConfig,
    count: int,
    categories: list,
) -> List[str]:
    """Single API call that requests all hooks at once across all categories."""
    cat_list = ", ".join(c.value for c in categories)
    system_prompt = HOOK_SYSTEM_PROMPT.format(
        max_chars=_MAX_HOOK_CHARS,
        max_words=_MAX_HOOK_WORDS,
    )
    user_prompt = (
        f"Generate exactly {count} unique hooks for the following offer.\n\n"
        f"OFFER: {offer.offer_name}\n"
        f"DESCRIPTION: {offer.offer_description}\n"
        f"TARGET AUDIENCE: {offer.target_audience}\n"
        f"PAIN POINTS: {', '.join(offer.pain_points)}\n"
        f"BENEFITS: {', '.join(offer.benefits)}\n"
        f"TONE: {', '.join(offer.tone) if offer.tone else 'professional, empathetic'}\n\n"
        f"Mix across these styles: {cat_list}\n\n"
        "Rules:\n"
        f"- Each hook must be under {_MAX_HOOK_CHARS} characters\n"
        "- Each hook must be fresh, specific, and punchy\n"
        "- Do NOT repeat the same concept\n"
        "- Vary sentence structure — mix questions, statements, partial sentences\n\n"
        "Return ONLY a JSON array of strings. No explanation. No numbering.\n"
        'Example: ["Hook one here", "Hook two here", ...]'
    )
    raw = llm.complete_json(system_prompt, user_prompt, temperature=0.95, max_tokens=4096)
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if item and str(item).strip()]
    if isinstance(raw, dict):
        for key in ("hooks", "results", "data"):
            if isinstance(raw.get(key), list):
                return [str(item).strip() for item in raw[key] if item]
    raise TransientError(f"Unexpected LLM response shape: {type(raw)}")


def run_hook_agent(
    llm: BaseLLMProvider,
    offer: OfferConfig,
    total_hooks: int = 200,
    output_dir: str | Path | None = None,
    dedupe_threshold: float = 0.85,
) -> List[Hook]:
    """
    Generate `total_hooks` hooks across all categories.

    For small counts (≤ _FAST_PATH_THRESHOLD) uses a single API call.
    For large counts divides across categories with one call each.
    """
    out_dir = Path(output_dir) if output_dir else _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = offer.hook_categories or list(HookCategory)
    n_cats = len(categories)

    # ── Fast path: single API call ──────────────────────────────────────────
    if total_hooks <= _FAST_PATH_THRESHOLD:
        logger.info("HookAgent fast-path: requesting %d hooks in one call", total_hooks)
        dedupe = TextDedupe(similarity_threshold=dedupe_threshold)
        all_hooks: List[Hook] = []
        try:
            raw_texts = with_retries(
                _fetch_hooks_single_call,
                llm, offer, total_hooks, categories,
                max_attempts=3, base_delay=1.0, max_delay=10.0, reraise=True,
            )
            for text in raw_texts:
                text = text.strip()
                if not text:
                    continue
                if len(text) > _MAX_HOOK_CHARS:
                    text = text[:_MAX_HOOK_CHARS].rsplit(" ", 1)[0]
                if not dedupe.add(text):
                    continue
                score = _score_hook(text)
                # assign category based on position round-robin
                cat = categories[len(all_hooks) % n_cats]
                all_hooks.append(Hook(
                    text=text, category=cat,
                    strength_score=score, offer_name=offer.offer_name,
                ))
        except Exception as exc:
            logger.error("Fast-path hook generation failed: %s", exc)

        # Blend in pre-scraped bank hooks
        bank_texts = _load_hooks_bank()
        if bank_texts:
            added_from_bank = 0
            for text in bank_texts:
                text = text.strip()
                if not text or len(text) > _MAX_HOOK_CHARS:
                    continue
                if not dedupe.add(text):
                    continue
                score = _score_hook(text)
                cat = categories[len(all_hooks) % n_cats]
                all_hooks.append(Hook(
                    text=text,
                    category=cat,
                    strength_score=score,
                    offer_name=offer.offer_name,
                ))
                added_from_bank += 1
            logger.info("Added %d hooks from pre-scraped bank", added_from_bank)

        all_hooks.sort(key=lambda h: h.strength_score, reverse=True)
        json_path = out_dir / "hooks.json"
        csv_path = out_dir / "hooks.csv"
        write_models_json(all_hooks, json_path)
        models_to_csv(all_hooks, csv_path)
        logger.info("Saved %d hooks → %s", len(all_hooks), out_dir)
        return all_hooks

    # ── Slow path: one call per category (for large counts) ─────────────────
    per_cat = math.ceil(total_hooks / n_cats)

    # Request ~20% more to allow for deduplication losses
    request_per_cat = math.ceil(per_cat * 1.2)

    logger.info(
        "HookAgent: generating %d hooks across %d categories (%d per cat, requesting %d)",
        total_hooks,
        n_cats,
        per_cat,
        request_per_cat,
    )

    dedupe = TextDedupe(similarity_threshold=dedupe_threshold)
    all_hooks: List[Hook] = []

    for idx, category in enumerate(categories):
        logger.info(
            "[%d/%d] Generating hooks for category: %s",
            idx + 1,
            n_cats,
            category.value,
        )

        try:
            raw_texts = with_retries(
                _fetch_hooks_for_category,
                llm,
                offer,
                category,
                request_per_cat,
                max_attempts=4,
                base_delay=2.0,
                max_delay=30.0,
                reraise=True,
            )
        except Exception as exc:
            logger.error(
                "Failed to generate hooks for category=%s after retries: %s",
                category.value,
                exc,
            )
            continue

        logger.debug(
            "Category=%s: LLM returned %d raw hooks",
            category.value,
            len(raw_texts),
        )

        accepted_cat = 0
        for text in raw_texts:
            text = text.strip()
            if not text:
                continue
            # Length guards
            if len(text) > _MAX_HOOK_CHARS:
                logger.debug("Hook too long (%d chars), trimming: %s...", len(text), text[:60])
                text = text[: _MAX_HOOK_CHARS].rsplit(" ", 1)[0]

            if not dedupe.add(text):
                continue

            score = _score_hook(text)
            hook = Hook(
                text=text,
                category=category,
                strength_score=score,
                offer_name=offer.offer_name,
            )
            all_hooks.append(hook)
            accepted_cat += 1

        logger.info(
            "Category=%s: accepted %d hooks (dedupe stats: %s)",
            category.value,
            accepted_cat,
            dedupe.stats(),
        )

    # If we came up short, do a top-up pass on the first category
    shortfall = total_hooks - len(all_hooks)
    if shortfall > 0 and categories:
        logger.info("Shortfall of %d hooks — running top-up pass", shortfall)
        top_up_cat = categories[0]
        try:
            extra_texts = with_retries(
                _fetch_hooks_for_category,
                llm,
                offer,
                top_up_cat,
                math.ceil(shortfall * 1.5),
                max_attempts=3,
                base_delay=2.0,
                max_delay=20.0,
                reraise=False,
            )
            if extra_texts:
                for text in extra_texts:
                    text = text.strip()
                    if not text:
                        continue
                    if len(text) > _MAX_HOOK_CHARS:
                        text = text[: _MAX_HOOK_CHARS].rsplit(" ", 1)[0]
                    if not dedupe.add(text):
                        continue
                    score = _score_hook(text)
                    hook = Hook(
                        text=text,
                        category=top_up_cat,
                        strength_score=score,
                        offer_name=offer.offer_name,
                    )
                    all_hooks.append(hook)
                    if len(all_hooks) >= total_hooks:
                        break
        except Exception as exc:
            logger.warning("Top-up pass failed: %s", exc)

    # ── Blend in pre-scraped bank hooks ────────────────────────────────────
    bank_texts = _load_hooks_bank()
    if bank_texts:
        added_from_bank = 0
        for text in bank_texts:
            text = text.strip()
            if not text or len(text) > _MAX_HOOK_CHARS:
                continue
            if not dedupe.add(text):
                continue
            score = _score_hook(text)
            cat = categories[len(all_hooks) % n_cats]
            all_hooks.append(Hook(
                text=text,
                category=cat,
                strength_score=score,
                offer_name=offer.offer_name,
            ))
            added_from_bank += 1
        logger.info("Added %d hooks from pre-scraped bank", added_from_bank)

    # Sort by strength score descending
    all_hooks.sort(key=lambda h: h.strength_score, reverse=True)

    final_stats = dedupe.stats()
    logger.info(
        "HookAgent complete: %d hooks generated | exact_dupes=%d | near_dupes=%d",
        len(all_hooks),
        final_stats["exact_duplicates_rejected"],
        final_stats["near_duplicates_rejected"],
    )

    # Persist
    json_path = out_dir / "hooks.json"
    csv_path = out_dir / "hooks.csv"
    write_models_json(all_hooks, json_path)
    models_to_csv(all_hooks, csv_path)
    logger.info("Saved %d hooks → %s", len(all_hooks), out_dir)

    return all_hooks


# ---------------------------------------------------------------------------
# Class wrapper (for pipeline compatibility)
# ---------------------------------------------------------------------------

class HookAgent:
    def __init__(self, config: object, llm_provider: "BaseLLMProvider"):
        self.config = config
        self.llm = llm_provider

    def generate(
        self,
        offer: "OfferConfig",
        total_count: int = 200,
        output_dir: str | Path | None = None,
    ) -> List[Hook]:
        return run_hook_agent(
            llm=self.llm,
            offer=offer,
            total_hooks=total_count,
            output_dir=output_dir,
        )
