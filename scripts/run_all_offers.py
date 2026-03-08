#!/usr/bin/env python3
"""
Run the full ad generation pipeline for every offer config.

Usage (from aiagency project root):
    python scripts/run_all_offers.py

Requirements:
    OPENAI_API_KEY set in environment or .env file

Each offer produces:
    ai_ad_agency/outputs/<offer_slug>/
        hooks/      — generated hook texts
        scripts/    — generated ad scripts
        images/     — DALL-E images
        videos/     — assembled MP4s (video hook + b-roll + CTA)
        exports/    — final export package
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# Load .env if present
_env_file = Path(".env")
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

OFFER_CONFIGS = [
    "configs/offer_medicare.json",
    "configs/offer_auto_insurance.json",
    "configs/offer_health_insurance.json",
    "configs/offer_home_insurance.json",
    "configs/offer_debt_relief.json",
    "configs/offer_home_services.json",
    "configs/offer_ecom.json",
]

# Variants to generate per offer. Lower = faster + cheaper.
VARIANTS_PER_OFFER = 50


def check_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        print("Add it to a .env file in this directory or export it:")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)


def run_offer(config_path: str, count: int) -> bool:
    """Run autopilot for a single offer. Returns True on success."""
    offer_name = Path(config_path).stem.replace("offer_", "").replace("_", " ").title()
    print(f"\n{'='*60}")
    print(f"  Running: {offer_name}")
    print(f"  Config:  {config_path}")
    print(f"  Count:   {count} variants")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "main.py",
        "autopilot",
        "--config", config_path,
        "--count", str(count),
    ]

    result = subprocess.run(cmd, cwd=Path("ai_ad_agency"))
    if result.returncode != 0:
        print(f"\n[FAILED] {offer_name} exited with code {result.returncode}")
        return False

    print(f"\n[DONE] {offer_name}")
    return True


def main() -> None:
    check_api_key()

    total = len(OFFER_CONFIGS)
    passed = 0
    failed = []

    print(f"Running {total} offers × {VARIANTS_PER_OFFER} variants each")
    print(f"Estimated OpenAI cost: ~${total * VARIANTS_PER_OFFER * 0.002:.2f}\n")

    for i, config in enumerate(OFFER_CONFIGS, 1):
        print(f"\n[{i}/{total}] Starting {config}")
        ok = run_offer(config, VARIANTS_PER_OFFER)
        if ok:
            passed += 1
        else:
            failed.append(config)
        # Brief pause between offers to avoid rate limit bursts
        if i < total:
            time.sleep(3)

    print(f"\n{'='*60}")
    print(f"All offers complete: {passed}/{total} succeeded")
    if failed:
        print("Failed offers:")
        for f in failed:
            print(f"  - {f}")
    print(f"\nOutputs in: ai_ad_agency/outputs/")


if __name__ == "__main__":
    main()
