"""
QA Agent — quality assurance and validation for rendered creative assets.

Checks video and image files for existence, size, duration, audio, dimensions,
and duplicate content. Updates CreativeVariant.status and CreativeVariant.qa_notes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..models.enums import AssetStatus
from ..models.schemas import CreativeVariant, ImageCreative, QAResult
from ..utils.config import AppConfig
from ..utils.dedupe import FileDedupe
from ..utils.hashing import safe_hash_file
from ..utils.io import ensure_dir, write_models_json
from ..utils.logging_utils import get_module_logger
from ..utils.validators import probe_media, validate_image_file, validate_video_file

logger = get_module_logger("qa_agent")


class QAAgent:
    """Validates rendered creative assets and flags QA issues."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Video QA
    # ------------------------------------------------------------------

    def check_video(self, creative: CreativeVariant) -> QAResult:
        """
        Perform QA checks on a video creative.

        Checks (in order):
          1. File exists
          2. File size >= config.qa.min_file_size_bytes
          3. Duration within [min, max] from config.qa
          4. Has audio stream (if config.qa.required_audio)
          5. Dimensions match an expected format (if config.qa.check_dimensions)
          6. Hash file for duplicate detection

        Sets passed=True only if all enabled checks pass.
        """
        qa_cfg = self.config.qa
        file_path = creative.file_path or ""
        asset_id = creative.creative_id
        issues: List[str] = []

        result = QAResult(
            asset_id=asset_id,
            asset_type="video",
            file_path=file_path,
            passed=False,
        )

        # Check 1: File exists
        if not file_path or not Path(file_path).exists():
            issues.append(f"File not found: {file_path!r}")
            result.issues = issues
            return result

        result.file_exists = True
        result.file_size_bytes = Path(file_path).stat().st_size

        # Check 2: File size
        if result.file_size_bytes < qa_cfg.min_file_size_bytes:
            issues.append(
                f"File too small: {result.file_size_bytes} bytes "
                f"(min {qa_cfg.min_file_size_bytes})"
            )

        # Check 3–5: Probe media for duration, audio, dimensions
        media_info = probe_media(file_path)
        if media_info is None:
            issues.append("ffprobe failed or unavailable — cannot inspect media")
        else:
            duration = media_info.get("duration", 0.0)
            result.duration_sec = duration
            result.has_audio = media_info.get("has_audio", False)
            result.width = media_info.get("width", 0)
            result.height = media_info.get("height", 0)

            # Duration bounds
            if duration < qa_cfg.min_video_duration_sec:
                issues.append(
                    f"Duration too short: {duration:.2f}s (min {qa_cfg.min_video_duration_sec}s)"
                )
            if duration > qa_cfg.max_video_duration_sec:
                issues.append(
                    f"Duration too long: {duration:.2f}s (max {qa_cfg.max_video_duration_sec}s)"
                )

            # Audio check
            if qa_cfg.required_audio and not result.has_audio:
                issues.append("No audio stream found (required_audio=True)")

            # Dimensions check
            if qa_cfg.check_dimensions and qa_cfg.expected_video_formats:
                dim_str = f"{result.width}x{result.height}"
                if dim_str not in qa_cfg.expected_video_formats:
                    issues.append(
                        f"Unexpected dimensions: {dim_str} "
                        f"(expected one of {qa_cfg.expected_video_formats})"
                    )

        # Check 6: Content hash
        content_hash = safe_hash_file(file_path)
        result.content_hash = content_hash

        # Final pass/fail
        result.passed = len(issues) == 0
        result.issues = issues
        return result

    # ------------------------------------------------------------------
    # Image QA
    # ------------------------------------------------------------------

    def check_image(self, creative: ImageCreative) -> QAResult:
        """
        Perform QA checks on an image creative.

        Checks:
          1. File exists and size > 0
          2. Image file is valid (PIL verify)
          3. Hash for duplicate detection
        """
        file_path = creative.file_path
        asset_id = creative.image_id
        issues: List[str] = []

        result = QAResult(
            asset_id=asset_id,
            asset_type="image",
            file_path=file_path,
            passed=False,
        )

        # Check 1: File exists + size
        if not file_path or not Path(file_path).exists():
            issues.append(f"File not found: {file_path!r}")
            result.issues = issues
            return result

        result.file_exists = True
        result.file_size_bytes = Path(file_path).stat().st_size

        if result.file_size_bytes == 0:
            issues.append("File is empty (0 bytes)")
            result.issues = issues
            return result

        # Check 2: Image validity
        ok, img_issues = validate_image_file(file_path, min_size_bytes=1)
        if not ok:
            issues.extend(img_issues)

        # Check 3: Hash
        content_hash = safe_hash_file(file_path)
        result.content_hash = content_hash

        result.passed = len(issues) == 0
        result.issues = issues
        return result

    # ------------------------------------------------------------------
    # Batch QA
    # ------------------------------------------------------------------

    def run_batch(
        self,
        creatives: List[CreativeVariant],
        images: Optional[List[ImageCreative]] = None,
    ) -> Tuple[List[QAResult], List[QAResult]]:
        """
        Run QA on all video creatives and (optionally) all images.

        Updates:
          - creative.qa_passed
          - creative.qa_notes
          - creative.status → AssetStatus.ACCEPTED or REJECTED

        Uses FileDedupe to flag duplicate file content (hash collisions).

        Returns:
            (passed_results, failed_results) — combined from videos + images.
        """
        passed: List[QAResult] = []
        failed: List[QAResult] = []

        file_dedupe = FileDedupe()

        # -- Video creatives --
        for creative in creatives:
            # STATIC_IMAGE variants have no video file; accept them without video QA
            if creative.creative_type and creative.creative_type.value == "static_image":
                creative.qa_passed = True
                creative.qa_notes = []
                creative.status = AssetStatus.ACCEPTED
                passed.append(QAResult(
                    asset_id=creative.creative_id,
                    asset_type="image",
                    file_path=creative.file_path or "",
                    passed=True,
                    file_exists=bool(creative.file_path),
                ))
                continue

            result = self.check_video(creative)

            # Duplicate content check
            if result.content_hash:
                is_unique = file_dedupe.check_and_add(result.content_hash)
                if not is_unique:
                    result.is_duplicate = True
                    if self.config.qa.reject_duplicate_hashes:
                        result.passed = False
                        result.issues.append("Duplicate file content hash detected")

            # Update the creative in-place
            creative.qa_passed = result.passed
            creative.qa_notes = list(result.issues)
            if result.content_hash:
                creative.content_hash = result.content_hash

            if result.passed:
                creative.status = AssetStatus.ACCEPTED
                passed.append(result)
            else:
                creative.status = AssetStatus.REJECTED
                failed.append(result)

            logger.debug(
                "QA video creative=%s passed=%s issues=%s",
                creative.creative_id,
                result.passed,
                result.issues,
            )

        # -- Image creatives --
        if images:
            for image in images:
                result = self.check_image(image)

                if result.content_hash:
                    is_unique = file_dedupe.check_and_add(result.content_hash)
                    if not is_unique:
                        result.is_duplicate = True
                        if self.config.qa.reject_duplicate_hashes:
                            result.passed = False
                            result.issues.append("Duplicate file content hash detected")

                if result.passed:
                    image.status = AssetStatus.ACCEPTED
                    passed.append(result)
                else:
                    image.status = AssetStatus.REJECTED
                    failed.append(result)

                logger.debug(
                    "QA image=%s passed=%s issues=%s",
                    image.image_id,
                    result.passed,
                    result.issues,
                )

        logger.info(
            "QAAgent batch complete: %d passed, %d failed | duplicates_rejected=%d",
            len(passed),
            len(failed),
            file_dedupe.rejected,
        )
        return passed, failed

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def save_results(self, results: List[QAResult], output_dir: str) -> None:
        """Persist QA results to qa_results.json in the given directory."""
        if not results:
            return
        out_dir = Path(output_dir)
        ensure_dir(out_dir)
        out_path = out_dir / "qa_results.json"
        try:
            write_models_json(results, out_path)
            logger.info("QAAgent: saved %d QA results → %s", len(results), out_path)
        except Exception as exc:
            logger.error("QAAgent: failed to save QA results: %s", exc)
