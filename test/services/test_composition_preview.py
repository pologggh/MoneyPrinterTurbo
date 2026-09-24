from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from app.services.evaluation.composition_preview import (
    CompositionPreview,
    CompositionPreviewService,
    RenderContext,
)


def test_render_context_fingerprint_deterministic():
    ctx1 = RenderContext(target_width=1080, target_height=1920, aspect_ratio="9:16", fit_mode="contain")
    ctx2 = RenderContext(target_width=1080, target_height=1920, aspect_ratio="9:16", fit_mode="contain")
    assert ctx1.compute_fingerprint() == ctx2.compute_fingerprint()
    assert len(ctx1.compute_fingerprint()) == 64

    # Changed aspect ratio changes fingerprint
    ctx3 = RenderContext(target_width=1920, target_height=1080, aspect_ratio="16:9", fit_mode="contain")
    assert ctx1.compute_fingerprint() != ctx3.compute_fingerprint()

    # Changed fit mode changes fingerprint
    ctx4 = RenderContext(target_width=1080, target_height=1920, aspect_ratio="9:16", fit_mode="cover")
    assert ctx1.compute_fingerprint() != ctx4.compute_fingerprint()


def test_generate_and_cache_preview(tmp_path: Path):
    src_img_path = tmp_path / "source.png"
    img = Image.new("RGB", (800, 600), color="red")
    img.save(str(src_img_path))

    previews_dir = tmp_path / "previews"
    ctx = RenderContext(target_width=300, target_height=600, aspect_ratio="9:18", fit_mode="contain")

    preview1 = CompositionPreviewService.generate_or_get_preview(
        shot_asset_version_id="asset_ver_1",
        source_image_path=src_img_path,
        render_context=ctx,
        output_dir=str(previews_dir),
    )

    assert isinstance(preview1, CompositionPreview)
    assert preview1.width == 300
    assert preview1.height == 600
    assert os.path.exists(preview1.preview_path)
    assert len(preview1.file_hash) == 64

    # Verify generated preview dimensions with PIL
    with Image.open(preview1.preview_path) as p_img:
        assert p_img.size == (300, 600)

    # Calling again should reuse the existing cached file
    preview2 = CompositionPreviewService.generate_or_get_preview(
        shot_asset_version_id="asset_ver_1",
        source_image_path=src_img_path,
        render_context=ctx,
        output_dir=str(previews_dir),
    )

    assert preview2.preview_path == preview1.preview_path
    assert preview2.file_hash == preview1.file_hash


def test_preview_cover_mode(tmp_path: Path):
    src_img_path = tmp_path / "source2.png"
    img = Image.new("RGB", (1200, 400), color="yellow")
    img.save(str(src_img_path))

    previews_dir = tmp_path / "previews"
    ctx = RenderContext(target_width=400, target_height=400, aspect_ratio="1:1", fit_mode="cover")

    preview = CompositionPreviewService.generate_or_get_preview(
        shot_asset_version_id="asset_ver_cover",
        source_image_path=src_img_path,
        render_context=ctx,
        output_dir=str(previews_dir),
    )

    assert preview.width == 400
    assert preview.height == 400
    with Image.open(preview.preview_path) as p_img:
        assert p_img.size == (400, 400)
