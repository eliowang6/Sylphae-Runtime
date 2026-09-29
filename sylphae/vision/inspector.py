"""UI image inspector for screenshots, mocks, and layout-bug images.

Two kinds of input are combined:

- the caller's own wording, mapped to defect categories and search keywords;
- the image's pixels, via two checks -- activity at the right edge (content spilling
  out of its container) and an overall dynamic range that is too narrow to read.

Scope, stated plainly.  There is no learned model and no layout parser here.
``torchvision`` used to be named in this docstring and was never imported, and
``confidence`` used to be a constant, which together made the module sound like
something it is not.  What it genuinely does is turn a screenshot into spatial and
component keywords the exploration stage can search with, plus a coarse defect guess.
A defect the two pixel checks cannot see -- a glyph drawn off-centre inside its box,
say -- will not be reported, and the caller should not read silence as absence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

__all__ = [
    "DefectType",
    "VisionReport",
    "inspect_ui_image",
]


class DefectType:
    TEXT_OVERFLOW = "text_overflow"
    LOW_CONTRAST = "low_contrast"
    MISALIGNMENT = "misalignment"
    CRAMPED_SPACING = "cramped_spacing"
    CLIPPED_ELEMENT = "clipped_element"
    NONE = "none"


def _defect_confidence(has_defect: bool, from_prompt: bool, from_pixels: bool) -> float:
    """How well supported the verdict is, given where the evidence came from.

    The pixel checks are observations about the image.  The keyword branch is a
    restatement of what the caller already said, which is useful for routing but is not
    evidence the caller did not supply -- so the two cannot be worth the same, and a
    verdict both agree on is worth more than either alone.
    """
    if not has_defect:
        # The image was analysed and nothing was flagged.  That is a real finding, and a
        # weaker one than a confirmed defect.
        return 0.55
    score = 0.45
    if from_prompt:
        score += 0.15
    if from_pixels:
        score += 0.30
    return min(0.95, score)


@dataclass
class VisionReport:
    image_path: str
    width: int
    height: int
    has_defect: bool
    defect_type: str
    spatial_region: str = "unknown"
    extracted_keywords: list[str] = field(default_factory=list)
    confidence: float = 0.85
    description: str = ""


def inspect_ui_image(
    image_path: Path | str,
    prompt_hint: str = "",
) -> VisionReport:
    """Analyze a UI screenshot image for layout defects, overflow, and spatial anchors."""
    path = Path(image_path)
    if not path.exists():
        return VisionReport(
            image_path=str(path),
            width=0,
            height=0,
            has_defect=False,
            defect_type=DefectType.NONE,
            confidence=0.0,
            description=f"Image file not found: {path}",
        )

    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:
        return VisionReport(
            image_path=str(path),
            width=0,
            height=0,
            has_defect=False,
            defect_type=DefectType.NONE,
            confidence=0.0,
            description=f"Cannot open image: {exc}",
        )

    w, h = img.size
    keywords: list[str] = []
    spatial_region = "unknown"
    defect_type = DefectType.NONE
    has_defect = False
    #: Which source actually supports the verdict.  Tracked so ``confidence`` can be
    #: derived instead of asserted -- see ``_defect_confidence``.
    prompt_named = False
    pixel_found = False

    lower_prompt = prompt_hint.lower()

    # 1. Defect extraction from natural language hint
    if any(w in prompt_hint for w in ("溢出", "截断", "放不下", "超", "overflow", "clipped text", "ellipsis")):
        has_defect = True
        prompt_named = True
        defect_type = DefectType.TEXT_OVERFLOW
        keywords.extend(["overflow", "text-overflow", "white-space", "ellipsis", "truncate"])
    elif any(w in prompt_hint for w in ("看不清", "对比度", "太暗", "太亮", "contrast", "readability", "unreadable")):
        has_defect = True
        prompt_named = True
        defect_type = DefectType.LOW_CONTRAST
        keywords.extend(["color", "text-white", "contrast", "background"])
    elif any(w in prompt_hint for w in ("不对齐", "歪", "没对齐", "misalign", "align", "alignment", "center")):
        has_defect = True
        prompt_named = True
        defect_type = DefectType.MISALIGNMENT
        keywords.extend(["align", "items-center", "justify-center", "flex"])
    elif any(w in prompt_hint for w in ("太挤", "拥挤", "间距", "重叠", "cramped", "spacing", "overlap")):
        has_defect = True
        prompt_named = True
        defect_type = DefectType.CRAMPED_SPACING
        keywords.extend(["gap", "padding", "margin", "spacing"])

    # 2. Spatial extraction
    if any(w in prompt_hint or w in lower_prompt for w in ("右上角", "top-right", "upper right")):
        spatial_region = "top-right"
        keywords.extend(["top-right", "header", "nav", "btn", "action"])
    elif any(w in prompt_hint or w in lower_prompt for w in ("左上角", "top-left", "brand")):
        spatial_region = "top-left"
        keywords.extend(["top-left", "logo", "brand", "header"])
    elif any(w in prompt_hint or w in lower_prompt for w in ("居中", "中间", "center")):
        spatial_region = "center"
        keywords.extend(["center", "main", "content", "hero"])
    elif any(w in prompt_hint or w in lower_prompt for w in ("底部", "footer", "bottom")):
        spatial_region = "bottom"
        keywords.extend(["footer", "bottom", "bottom-bar"])

    # Component hints
    if any(w in prompt_hint or w in lower_prompt for w in ("按钮", "button", "btn")):
        keywords.extend(["button", "btn", "action"])
    if any(w in prompt_hint or w in lower_prompt for w in ("卡片", "card")):
        keywords.extend(["card", "panel", "box"])
    if any(w in prompt_hint or w in lower_prompt for w in ("导航", "nav", "navbar")):
        keywords.extend(["nav", "navbar", "header"])

    # 3. Visual image pixel analysis
    gray = img.convert("L")
    pixels = gray.load()

    # Right edge activity check (text spilling outside)
    right_edge_activity = 0
    sample_step = max(1, h // 40)
    for y in range(0, h, sample_step):
        row_diff = abs(pixels[w - 2, y] - pixels[w - 10, y])
        if row_diff > 30:
            right_edge_activity += 1

    if right_edge_activity > (h // (sample_step * 3)):
        has_defect = True
        pixel_found = True
        if defect_type == DefectType.NONE:
            defect_type = DefectType.TEXT_OVERFLOW

    # Low contrast check (average difference between text regions and background)
    # If image is overall very dark or very light with narrow dynamic range
    min_val, max_val = gray.getextrema()
    # The pixel verdict is recorded even when the prompt already named a defect.  Gating
    # the check on ``not has_defect`` made the two sources mutually exclusive, so a
    # request that said "contrast is bad" *and* came with an image that confirmed it
    # scored no better than the wording alone -- the confirmation was never seen.
    if max_val - min_val < 45:
        pixel_found = True
        if not has_defect:
            has_defect = True
            defect_type = DefectType.LOW_CONTRAST
            keywords.extend(["contrast", "color", "text-white"])

    if not keywords:
        keywords = ["button", "btn", "navbar", "card", "container"]

    # Deduplicate preserving order
    seen: set[str] = set()
    dedup: list[str] = []
    for kw in keywords:
        k = kw.lower().strip()
        if k and k not in seen:
            seen.add(k)
            dedup.append(kw)

    return VisionReport(
        image_path=str(path),
        width=w,
        height=h,
        has_defect=has_defect,
        defect_type=defect_type,
        spatial_region=spatial_region,
        extracted_keywords=dedup,
        confidence=_defect_confidence(has_defect, prompt_named, pixel_found),
        description=f"UI Image {w}x{h}, defect={defect_type}, region={spatial_region}",
    )
