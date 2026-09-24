from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from app.services.asset_probe_service import calculate_sha256
from app.utils.utils import storage_dir

COMPOSITION_PREVIEW_POLICY_VERSION: str = "composition-preview-v1"


class RenderContext(BaseModel):
    """
    Rendering configuration for derived composition previews.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    target_width: int = 1080
    target_height: int = 1920
    aspect_ratio: str = "9:16"
    fit_mode: str = "contain"  # "contain" or "cover"
    safe_margin_percent: float = 0.10

    def compute_fingerprint(self) -> str:
        """Computes deterministic canonical SHA-256 fingerprint of render options."""
        payload = {
            "aspect_ratio": self.aspect_ratio,
            "fit_mode": self.fit_mode,
            "safe_margin_percent": round(self.safe_margin_percent, 4),
            "target_height": self.target_height,
            "target_width": self.target_width,
        }
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class CompositionPreview(BaseModel):
    """
    Derived canvas preview artifact for evaluating composition suitability.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    composition_preview_id: str = Field(default_factory=lambda: str(uuid4()))
    shot_asset_version_id: str
    render_context_fingerprint: str
    preview_path: str
    file_hash: str
    width: int
    height: int
    preview_policy_version: str = COMPOSITION_PREVIEW_POLICY_VERSION
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CompositionPreviewService:
    """
    Service responsible for generating and caching derived composition preview canvases.
    """

    @classmethod
    def generate_or_get_preview(
        cls,
        shot_asset_version_id: str,
        source_image_path: str | Path,
        render_context: RenderContext | None = None,
        output_dir: str | None = None,
    ) -> CompositionPreview:
        """
        Generates or reuses a derived composition preview canvas image.

        1. Creates canvas matching target dimensions.
        2. Applies fit/crop centering.
        3. Overlays safe margin guide box for text/action safety.
        4. Caches deterministically by (shot_asset_version_id, render_context_fingerprint).
        """
        ctx = render_context or RenderContext()
        ctx_fingerprint = ctx.compute_fingerprint()

        previews_dir = output_dir or storage_dir("evaluation_cache/previews", create=True)
        os.makedirs(previews_dir, exist_ok=True)

        preview_filename = f"{shot_asset_version_id}_{ctx_fingerprint}.png"
        preview_path = os.path.join(previews_dir, preview_filename)

        if os.path.exists(preview_path) and os.path.getsize(preview_path) > 0:
            file_hash = calculate_sha256(preview_path)
            return CompositionPreview(
                shot_asset_version_id=shot_asset_version_id,
                render_context_fingerprint=ctx_fingerprint,
                preview_path=preview_path,
                file_hash=file_hash,
                width=ctx.target_width,
                height=ctx.target_height,
                preview_policy_version=COMPOSITION_PREVIEW_POLICY_VERSION,
            )

        # Open source image
        src_path = Path(source_image_path).resolve()
        if not src_path.is_file():
            raise FileNotFoundError(f"Source image for preview not found: {src_path}")

        try:
            with Image.open(src_path) as src_img:
                src_rgb = src_img.convert("RGBA")
                src_w, src_h = src_rgb.size

                # Create canvas
                canvas = Image.new("RGBA", (ctx.target_width, ctx.target_height), (24, 24, 28, 255))

                if ctx.fit_mode == "cover":
                    scale = max(ctx.target_width / src_w, ctx.target_height / src_h)
                    new_w = int(src_w * scale)
                    new_h = int(src_h * scale)
                    resized = src_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    # Center crop
                    left = (new_w - ctx.target_width) // 2
                    top = (new_h - ctx.target_height) // 2
                    cropped = resized.crop((left, top, left + ctx.target_width, top + ctx.target_height))
                    canvas.paste(cropped, (0, 0))
                else:  # default "contain"
                    scale = min(ctx.target_width / src_w, ctx.target_height / src_h)
                    new_w = max(1, int(src_w * scale))
                    new_h = max(1, int(src_h * scale))
                    resized = src_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    offset_x = (ctx.target_width - new_w) // 2
                    offset_y = (ctx.target_height - new_h) // 2
                    canvas.paste(resized, (offset_x, offset_y))

                # Draw safe margin guidelines
                draw = ImageDraw.Draw(canvas)
                margin_x = int(ctx.target_width * ctx.safe_margin_percent)
                margin_y = int(ctx.target_height * ctx.safe_margin_percent)
                safe_box = [
                    margin_x,
                    margin_y,
                    ctx.target_width - margin_x,
                    ctx.target_height - margin_y,
                ]
                draw.rectangle(safe_box, outline=(255, 255, 255, 100), width=2)

                # Save as RGB PNG
                final_img = canvas.convert("RGB")
                final_img.save(preview_path, format="PNG")

        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Failed to render composition preview from {src_path}: {exc}") from exc

        file_hash = calculate_sha256(preview_path)
        return CompositionPreview(
            shot_asset_version_id=shot_asset_version_id,
            render_context_fingerprint=ctx_fingerprint,
            preview_path=preview_path,
            file_hash=file_hash,
            width=ctx.target_width,
            height=ctx.target_height,
            preview_policy_version=COMPOSITION_PREVIEW_POLICY_VERSION,
        )
