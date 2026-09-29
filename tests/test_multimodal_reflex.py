import asyncio
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from sylphae.agent.effort import LowEffort
from sylphae.agent.reflex import ReflexAgent
from sylphae.vision.defect_mapper import map_vision_defect_to_patch
from sylphae.vision.inspector import DefectType, inspect_ui_image


def _create_mock_overflow_image(path: Path) -> Path:
    """Create a small synthetic mockup with text touching the right boundary."""
    img = Image.new("RGB", (200, 60), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw container box
    draw.rectangle([10, 10, 190, 50], outline=(0, 0, 0), width=2)
    # Draw text extending to the edge
    draw.text((15, 20), "SuperLongUserIdentifierExtendingBeyondBounds", fill=(0, 0, 0))
    img.save(path)
    return path


def test_inspect_ui_image():
    with tempfile.TemporaryDirectory() as td:
        img_path = Path(td) / "test_ui.png"
        _create_mock_overflow_image(img_path)

        rep = inspect_ui_image(img_path, prompt_hint="图中文字溢出控件了，请修复")
        assert rep.has_defect is True
        assert rep.defect_type == DefectType.TEXT_OVERFLOW
        assert "overflow" in rep.extracted_keywords or "text-overflow" in rep.extracted_keywords


def test_map_vision_defect_to_patch_css():
    with tempfile.TemporaryDirectory() as td:
        img_path = Path(td) / "test_ui.png"
        _create_mock_overflow_image(img_path)
        rep = inspect_ui_image(img_path, prompt_hint="图中文字溢出控件了，请修复")

        css = """
        .user-card {
            width: 150px;
        }
        """
        new_src, spec = map_vision_defect_to_patch(rep, "styles/card.css", css, hint_selector="user-card")
        assert spec is not None
        assert "overflow: hidden;" in new_src
        assert "text-overflow: ellipsis;" in new_src


def test_multimodal_reflex_agent_e2e():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "src").mkdir()
        css_file = root / "src" / "card.css"
        css_file.write_text("""
        .user-card {
            width: 180px;
        }
        """)

        img_path = root / "overflow_screenshot.png"
        _create_mock_overflow_image(img_path)

        agent = ReflexAgent(root=root, effort=LowEffort)
        res = asyncio.run(agent.run(
            prompt="图中文字溢出控件了，请修复",
            image_path=img_path,
            dry_run=True,
        ))

        assert res.immediate_patch_ok is True
        assert "card.css" in res.target_path
        assert res.new_source is not None
        assert "overflow: hidden" in res.new_source
        assert res.vision_report is not None
        assert res.vision_report.has_defect is True
