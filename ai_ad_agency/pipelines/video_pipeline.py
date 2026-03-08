"""
Video assembly pipeline.
Assembles final videos from: hook card → talking actor → b-roll → CTA card → subtitles.
Uses FFmpeg exclusively for media operations.
"""
from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..models.enums import AssetStatus, CreativeType
from ..models.schemas import (
    AvatarMetadata,
    BRollClip,
    CaptionFile,
    CreativeVariant,
    Overlay,
    Script,
    ScriptVariant,
    TalkingActorJob,
)
from ..utils.config import AppConfig
from ..utils.ffmpeg_utils import (
    add_subtitles,
    check_ffmpeg,
    concatenate_videos,
    create_text_card,
    get_duration,
    get_dimensions,
    scale_video,
    trim_video,
)
from ..utils.io import get_file_size, write_models_json
from ..utils.logging_utils import get_module_logger
from ..utils.validators import probe_media

logger = get_module_logger("video_pipeline")


class VideoPipeline:
    """
    Assembles final video creatives from component assets.

    Assembly order:
    1. Hook card (text overlay on solid color) — 2.5s
    2. Talking actor clip
    3. Optional: B-roll insert (after avatar)
    4. CTA end card — 3.0s
    5. Burn subtitles (optional)
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._ffmpeg_ok = check_ffmpeg()
        if not self._ffmpeg_ok:
            logger.warning("FFmpeg not found in PATH. Video assembly will be skipped.")

    def assemble_video(
        self,
        variant: CreativeVariant,
        hook_text: str,
        talking_actor_path: Optional[str],
        broll_paths: List[str],
        cta_text: str,
        caption_file: Optional[CaptionFile],
        output_path: str,
        width: int = 1080,
        height: int = 1920,
        tmp_dir: Optional[str] = None,
        hook_video_path: Optional[str] = None,
    ) -> bool:
        """
        Assemble one final video. Returns True on success.

        All intermediate files are created in a temporary directory and cleaned up.
        If hook_video_path is provided it is used as the opening hook segment;
        otherwise a text card is generated from hook_text.
        """
        if not self._ffmpeg_ok:
            logger.warning("FFmpeg unavailable — skipping assembly for %s", output_path)
            return False

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        use_tmp = tmp_dir is None
        if use_tmp:
            tmp = tempfile.mkdtemp(prefix="ai_agency_")
            tmp_dir = tmp
        else:
            tmp = None
            Path(tmp_dir).mkdir(parents=True, exist_ok=True)

        try:
            return self._assemble_inner(
                variant=variant,
                hook_text=hook_text,
                talking_actor_path=talking_actor_path,
                broll_paths=broll_paths,
                cta_text=cta_text,
                caption_file=caption_file,
                output_path=output_path,
                width=width,
                height=height,
                tmp_dir=tmp_dir,
                hook_video_path=hook_video_path,
            )
        finally:
            if use_tmp and tmp:
                import shutil
                try:
                    shutil.rmtree(tmp, ignore_errors=True)
                except Exception:
                    pass

    def _assemble_inner(
        self,
        variant: CreativeVariant,
        hook_text: str,
        talking_actor_path: Optional[str],
        broll_paths: List[str],
        cta_text: str,
        caption_file: Optional[CaptionFile],
        output_path: str,
        width: int,
        height: int,
        tmp_dir: str,
        hook_video_path: Optional[str] = None,
    ) -> bool:
        segments: List[str] = []

        # ---- 1. Hook segment ----
        if hook_video_path and Path(hook_video_path).exists():
            # Use downloaded video hook clip, scaled to output dimensions
            hook_seg = str(Path(tmp_dir) / "seg_01_hook_video.mp4")
            dims = get_dimensions(hook_video_path)
            if dims and dims != (width, height):
                ok = scale_video(hook_video_path, hook_seg, width, height, pad=True)
                segments.append(hook_seg if ok else hook_video_path)
            else:
                segments.append(hook_video_path)
            logger.info("Using video hook clip: %s", Path(hook_video_path).name)
        else:
            # Fallback: generate text card
            hook_card = str(Path(tmp_dir) / "seg_01_hook_card.mp4")
            hook_duration = self.config.render.hook_card_duration_sec
            ok = create_text_card(
                text=hook_text,
                output_path=hook_card,
                width=width,
                height=height,
                duration_sec=hook_duration,
                bg_color="#000000",
                font_color="white",
                font_size=min(60, 1080 * 60 // width),
            )
            if ok:
                segments.append(hook_card)
            else:
                logger.warning("Hook card failed for variant %s", variant.creative_id[:8])

        # ---- 2. Talking actor ----
        if talking_actor_path and Path(talking_actor_path).exists():
            actor_scaled = str(Path(tmp_dir) / "seg_02_actor.mp4")
            dims = get_dimensions(talking_actor_path)
            if dims and dims != (width, height):
                ok2 = scale_video(talking_actor_path, actor_scaled, width, height, pad=True)
                if ok2:
                    segments.append(actor_scaled)
                else:
                    segments.append(talking_actor_path)
            else:
                segments.append(talking_actor_path)
        else:
            logger.debug("No talking actor for variant %s", variant.creative_id[:8])
            # Create placeholder silent segment
            placeholder = str(Path(tmp_dir) / "seg_02_placeholder.mp4")
            create_text_card(
                text="[No actor clip]",
                output_path=placeholder,
                width=width,
                height=height,
                duration_sec=5.0,
                bg_color="#1a1a2e",
            )
            if Path(placeholder).exists():
                segments.append(placeholder)

        # ---- 3. B-roll inserts ----
        for i, broll_path in enumerate(broll_paths[:2]):  # Max 2 b-roll clips
            if not Path(broll_path).exists():
                continue
            br_scaled = str(Path(tmp_dir) / f"seg_03_broll_{i}.mp4")
            dims = get_dimensions(broll_path)
            if dims and dims != (width, height):
                ok3 = scale_video(broll_path, br_scaled, width, height, pad=True)
                if ok3:
                    segments.append(br_scaled)
                else:
                    segments.append(broll_path)
            else:
                segments.append(broll_path)

        # ---- 4. CTA card ----
        cta_card = str(Path(tmp_dir) / "seg_04_cta.mp4")
        cta_duration = self.config.render.cta_card_duration_sec
        ok4 = create_text_card(
            text=cta_text,
            output_path=cta_card,
            width=width,
            height=height,
            duration_sec=cta_duration,
            bg_color="#1B4F72",
            font_color="white",
            font_size=52,
        )
        if ok4:
            segments.append(cta_card)

        if not segments:
            logger.error("No valid segments for variant %s", variant.creative_id[:8])
            return False

        # ---- 5. Concatenate ----
        if len(segments) == 1:
            # Just copy the single segment
            import shutil
            shutil.copy2(segments[0], output_path)
            concatenated = output_path
        else:
            concat_out = str(Path(tmp_dir) / "concatenated.mp4")
            ok5 = concatenate_videos(
                input_paths=segments,
                output_path=concat_out,
                video_codec=self.config.render.output_video_codec,
                audio_codec=self.config.render.output_audio_codec,
                crf=self.config.render.video_crf,
                preset=self.config.render.video_preset,
                audio_bitrate=self.config.render.audio_bitrate,
                threads=self.config.render.ffmpeg_threads,
            )
            if not ok5:
                logger.error("Concatenation failed for variant %s", variant.creative_id[:8])
                return False
            concatenated = concat_out

        # ---- 6. Burn subtitles ----
        if caption_file and Path(caption_file.srt_path).exists():
            subtitled = str(Path(tmp_dir) / "subtitled.mp4")
            ok6 = add_subtitles(
                video_path=concatenated,
                srt_path=caption_file.srt_path,
                output_path=subtitled,
                font_size=28,
            )
            if ok6:
                final_source = subtitled
            else:
                logger.warning("Subtitle burn failed — using unstyled version")
                final_source = concatenated
        else:
            final_source = concatenated

        # ---- 7. Copy to output ----
        import shutil
        shutil.copy2(final_source, output_path)
        logger.info(
            "Assembled video: %s (segments=%d)",
            Path(output_path).name,
            len(segments),
        )
        return True

    def assemble_batch(
        self,
        variants: List[CreativeVariant],
        component_lookup: Dict[str, Any],
        output_dir: str,
        run_id: str,
    ) -> List[CreativeVariant]:
        """
        Assemble all video variants in a batch.

        component_lookup must contain:
        - "hooks": {hook_id: Hook}
        - "scripts": {script_id: Script or ScriptVariant}
        - "actor_jobs": {job_id: TalkingActorJob} keyed by (avatar_id, script_id) or actor job_id
        - "broll": {broll_id: BRollClip}
        - "captions": {caption_id: CaptionFile}
        """
        if not self._ffmpeg_ok:
            logger.error("FFmpeg unavailable — cannot assemble videos")
            return variants

        from ..utils.video_hook_selector import VideoHookSelector
        hook_selector = VideoHookSelector()
        if hook_selector.count:
            logger.info("VideoHookSelector: %d clips available for batch", hook_selector.count)
        else:
            logger.info("VideoHookSelector: no clips found, will use text cards")

        total = len(variants)
        completed = 0
        failed = 0

        for i, variant in enumerate(variants):
            if variant.creative_type == CreativeType.STATIC_IMAGE:
                continue  # Skip images — handled by image agent
            if variant.status == AssetStatus.ACCEPTED and variant.file_path and Path(variant.file_path).exists():
                logger.debug("Skipping already-assembled variant %s", variant.creative_id[:8])
                completed += 1
                continue

            # Build output path
            out_path = str(
                Path(output_dir) / f"video_{variant.creative_id[:12]}.mp4"
            )

            # Resolve components
            hook_text = variant.hook_text or "Watch this."

            # Pick a video hook clip — rotate through available clips across variants
            hook_video = hook_selector.pick_by_index(i)

            # Get talking actor path
            actor_path = self._resolve_actor_path(variant, component_lookup)

            # Get b-roll paths
            broll_paths = self._resolve_broll_paths(variant, component_lookup)

            # Get CTA text
            scripts = component_lookup.get("scripts", {})
            script = scripts.get(variant.script_id) or scripts.get(variant.script_variant_id)
            cta_text = "Learn More → Click Below"
            if script and hasattr(script, "sections"):
                cta_text = script.sections.cta

            # Get caption file
            captions = component_lookup.get("captions", {})
            caption_file = captions.get(variant.caption_id) if variant.caption_id else None

            logger.info(
                "[VIDEO ASSEMBLY %d/%d] Variant %s | hook_clip=%s",
                i + 1,
                total,
                variant.creative_id[:8],
                Path(hook_video).name if hook_video else "text_card",
            )

            ok = self.assemble_video(
                variant=variant,
                hook_text=hook_text,
                talking_actor_path=actor_path,
                broll_paths=broll_paths,
                cta_text=cta_text,
                caption_file=caption_file,
                output_path=out_path,
                width=variant.width or 1080,
                height=variant.height or 1920,
                hook_video_path=hook_video,
            )

            if ok and Path(out_path).exists():
                variant.file_path = out_path
                variant.file_size_bytes = get_file_size(out_path)
                variant.status = AssetStatus.COMPLETED

                # Get duration and dimensions
                dur = get_duration(out_path)
                if dur:
                    variant.duration_sec = dur
                dims = get_dimensions(out_path)
                if dims:
                    variant.width, variant.height = dims

                completed += 1
            else:
                variant.status = AssetStatus.FAILED
                failed += 1
                logger.error("Assembly failed for variant %s", variant.creative_id[:8])

        logger.info(
            "[VIDEO PIPELINE] Completed: %d  Failed: %d  Total: %d",
            completed,
            failed,
            total,
        )
        return variants

    def _resolve_actor_path(
        self, variant: CreativeVariant, lookup: Dict[str, Any]
    ) -> Optional[str]:
        """Find the talking actor video file for this variant."""
        actor_jobs: Dict = lookup.get("actor_jobs", {})

        # Try to find by avatar_id + script_id combo
        key = f"{variant.avatar_id}:{variant.script_id}"
        if key in actor_jobs:
            job = actor_jobs[key]
            if job.file_path and Path(job.file_path).exists():
                return job.file_path

        # Try all jobs with matching avatar_id
        for job_key, job in actor_jobs.items():
            if hasattr(job, "avatar_id") and job.avatar_id == variant.avatar_id:
                if job.file_path and Path(job.file_path).exists():
                    return job.file_path

        return None

    def _resolve_broll_paths(
        self, variant: CreativeVariant, lookup: Dict[str, Any]
    ) -> List[str]:
        """Get valid b-roll file paths for this variant."""
        broll_map: Dict = lookup.get("broll", {})
        paths = []
        for bid in (variant.broll_ids or []):
            clip = broll_map.get(bid)
            if clip and clip.file_path and Path(clip.file_path).exists():
                paths.append(clip.file_path)
        return paths

    def render_multi_format(
        self,
        source_path: str,
        output_dir: str,
        base_name: str,
        formats: List[str],
    ) -> Dict[str, str]:
        """
        Render a video in multiple output formats (dimensions).
        Returns {format_str: output_path}.
        """
        results: Dict[str, str] = {}
        for fmt in formats:
            try:
                w, h = [int(x) for x in fmt.split("x")]
            except ValueError:
                logger.warning("Invalid format string: %s", fmt)
                continue

            out_path = str(Path(output_dir) / f"{base_name}_{fmt}.mp4")
            ok = scale_video(source_path, out_path, w, h, pad=True)
            if ok:
                results[fmt] = out_path
            else:
                logger.warning("Failed to scale %s to %s", source_path, fmt)

        return results
