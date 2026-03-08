#!/usr/bin/env python3
"""
Run the full ad generation pipeline for every offer config.

Usage (from the ai_ad_agency directory):
    python run_all_offers.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# Load .env from this directory or parent
for _env in [Path(".env"), Path("../.env")]:
    if _env.exists():
        for line in _env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
        break

OFFER_CONFIGS = [
    "configs/offer_medicare.json",
    "configs/offer_auto_insurance.json",
    "configs/offer_health_insurance.json",
    "configs/offer_home_insurance.json",
    "configs/offer_debt_relief.json",
    "configs/offer_home_services.json",
    "configs/offer_ecom.json",
]

VARIANTS_PER_OFFER = 50


def check_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        print("Create a .env file here with: OPENAI_API_KEY=sk-...")
        sys.exit(1)


def run_offer(config_path: str, count: int) -> bool:
    offer_name = Path(config_path).stem.replace("offer_", "").replace("_", " ").title()
    print(f"\n{'='*60}\n  Running: {offer_name}\n{'='*60}")
    result = subprocess.run(
        [sys.executable, "main.py", "autopilot", "--config", config_path, "--count", str(count)],
    )
    ok = result.returncode == 0
    print(f"\n[{'DONE' if ok else 'FAILED'}] {offer_name}")
    return ok


def main() -> None:
    check_api_key()
    total = len(OFFER_CONFIGS)
    passed, failed = 0, []
    print(f"Running {total} offers × {VARIANTS_PER_OFFER} variants each\n")
    for i, config in enumerate(OFFER_CONFIGS, 1):
        print(f"[{i}/{total}] {config}")
        if run_offer(config, VARIANTS_PER_OFFER):
            passed += 1
        else:
            failed.append(config)
        if i < total:
            time.sleep(2)
    print(f"\n{'='*60}")
    print(f"Complete: {passed}/{total} succeeded")
    if failed:
        for f in failed:
            print(f"  FAILED: {f}")
    print(f"Outputs in: outputs/")


if __name__ == "__main__":
    main()
