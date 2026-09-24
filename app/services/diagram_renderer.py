from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.domain.asset_router import AssetRoutingRequest

_CANVAS_SIZES = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
}


def _font_path(*, bold: bool) -> Path:
    root = Path(__file__).resolve().parents[2]
    filename = "MicrosoftYaHeiBold.ttc" if bold else "MicrosoftYaHeiNormal.ttc"
    return root / "resource" / "fonts" / filename


def _load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(_font_path(bold=bold)), size=size)


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return []

    lines: list[str] = []
    current = ""
    for character in normalized:
        candidate = current + character
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current.rstrip())
            current = character.lstrip()
            if len(lines) == max_lines:
                break
        else:
            current = candidate

    if len(lines) < max_lines and current:
        lines.append(current.rstrip())

    consumed = "".join(lines)
    if len(consumed.replace(" ", "")) < len(normalized.replace(" ", "")) and lines:
        last = lines[-1].rstrip("… ")
        while last and draw.textlength(f"{last}…", font=font) > max_width:
            last = last[:-1]
        lines[-1] = f"{last}…"
    return lines


def _content_blocks(request: AssetRoutingRequest) -> list[str]:
    source = request.scene_description.strip() or request.generation_prompt.strip()
    parts = [
        part.strip(" ,，。；;:-")
        for part in re.split(r"[。！？；;\n]+|(?<=[,，])\s*", source)
        if part.strip(" ,，。；;:-")
    ]
    if len(parts) < 2 and request.generation_prompt.strip():
        prompt_parts = [
            part.strip(" ,，。；;:-")
            for part in re.split(r"\s*(?:->|→|=>)\s*", request.generation_prompt)
            if part.strip(" ,，。；;:-")
        ]
        for part in prompt_parts:
            if part not in parts:
                parts.append(part)

    if not parts:
        parts = [request.visual_goal.strip() or "知识要点"]
    return parts[:3]


def _draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    box: tuple[int, int, int, int],
    fill: str,
    line_gap: int,
) -> None:
    left, top, right, bottom = box
    line_height = font.size + line_gap
    total_height = len(lines) * line_height - line_gap
    y = top + max(0, (bottom - top - total_height) // 2)
    for line in lines:
        line_width = draw.textlength(line, font=font)
        x = left + max(0, int((right - left - line_width) / 2))
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height


def render_diagram_card(request: AssetRoutingRequest, target_dir: Path) -> Path:
    """Render a deterministic local knowledge diagram as a PNG image."""
    target_dir.mkdir(parents=True, exist_ok=True)
    width, height = _CANVAS_SIZES.get(request.aspect_ratio or "16:9", (1920, 1080))
    image = Image.new("RGB", (width, height), "#08111F")
    draw = ImageDraw.Draw(image)

    scale = min(width / 1920, height / 1080)
    title_font = _load_font(max(42, int(64 * scale)), bold=True)
    body_font = _load_font(max(30, int(40 * scale)))
    badge_font = _load_font(max(22, int(28 * scale)), bold=True)

    margin = max(54, int(width * 0.055))
    draw.rounded_rectangle(
        (margin, margin, margin + int(190 * scale), margin + int(54 * scale)),
        radius=max(12, int(18 * scale)),
        fill="#1677FF",
    )
    draw.text(
        (margin + int(24 * scale), margin + int(9 * scale)),
        "知识图解",
        font=badge_font,
        fill="#FFFFFF",
    )

    title_lines = _wrap_text(
        draw,
        request.visual_goal or "知识图解",
        title_font,
        width - margin * 2,
        2,
    )
    title_y = margin + int(95 * scale)
    for line in title_lines:
        draw.text((margin, title_y), line, font=title_font, fill="#F5F9FF")
        title_y += title_font.size + int(12 * scale)

    blocks = _content_blocks(request)
    landscape = width >= height
    gap = max(32, int((70 if landscape else 42) * scale))
    if landscape:
        card_top = max(title_y + int(70 * scale), int(height * 0.35))
        card_bottom = height - margin
        card_width = int((width - margin * 2 - gap * (len(blocks) - 1)) / len(blocks))
        cards = [
            (
                margin + index * (card_width + gap),
                card_top,
                margin + index * (card_width + gap) + card_width,
                card_bottom,
            )
            for index in range(len(blocks))
        ]
    else:
        card_top = max(title_y + int(55 * scale), int(height * 0.25))
        card_height = int((height - card_top - margin - gap * (len(blocks) - 1)) / len(blocks))
        cards = [
            (
                margin,
                card_top + index * (card_height + gap),
                width - margin,
                card_top + index * (card_height + gap) + card_height,
            )
            for index in range(len(blocks))
        ]

    for index, (text, card) in enumerate(zip(blocks, cards, strict=True), start=1):
        left, top, right, bottom = card
        draw.rounded_rectangle(
            card,
            radius=max(20, int(28 * scale)),
            fill="#12233D",
            outline="#2F80ED",
            width=max(3, int(4 * scale)),
        )
        marker_radius = max(23, int(28 * scale))
        marker_x = left + marker_radius + int(22 * scale)
        marker_y = top + marker_radius + int(22 * scale)
        draw.ellipse(
            (
                marker_x - marker_radius,
                marker_y - marker_radius,
                marker_x + marker_radius,
                marker_y + marker_radius,
            ),
            fill="#1677FF",
        )
        number = str(index)
        number_width = draw.textlength(number, font=badge_font)
        draw.text(
            (marker_x - number_width / 2, marker_y - badge_font.size / 2 - 2),
            number,
            font=badge_font,
            fill="#FFFFFF",
        )

        text_box = (
            left + int(38 * scale),
            top + int(95 * scale),
            right - int(38 * scale),
            bottom - int(34 * scale),
        )
        lines = _wrap_text(
            draw,
            text,
            body_font,
            text_box[2] - text_box[0],
            5 if landscape else 3,
        )
        _draw_centered_lines(
            draw,
            lines,
            body_font,
            text_box,
            "#DCEAFF",
            max(8, int(12 * scale)),
        )

        if index < len(cards):
            if landscape:
                start = (right + int(10 * scale), (top + bottom) // 2)
                end = (cards[index][0] - int(10 * scale), (top + bottom) // 2)
            else:
                start = ((left + right) // 2, bottom + int(10 * scale))
                end = ((left + right) // 2, cards[index][1] - int(10 * scale))
            draw.line((start, end), fill="#55A7FF", width=max(5, int(7 * scale)))

    safe_shot_id = re.sub(r"[^A-Za-z0-9_-]+", "_", request.shot_id).strip("_")
    output_path = target_dir / f"diagram_{safe_shot_id or 'shot'}.png"
    image.save(output_path, format="PNG", optimize=True)
    return output_path
