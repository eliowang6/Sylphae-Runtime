"""Maps detected multimodal visual defects to concrete code patches and CSS rules.

Generates targeted fixes for:
- Text overflow (ellipsis truncation, flex wrapping, container widening)
- Low contrast (color boost, high-contrast text)
- Layout misalignment (flex centering, items-center)
- Cramped spacing (gap expansion, padding)
"""

from __future__ import annotations

import re

from sylphae.ops.polyglot import add_css_properties
from sylphae.ops.spec import OpSpec
from sylphae.vision.inspector import DefectType, VisionReport

__all__ = [
    "map_vision_defect_to_patch",
]


def map_vision_defect_to_patch(
    report: VisionReport,
    target_path: str,
    source_content: str,
    hint_selector: str | None = None,
) -> tuple[str, OpSpec | None]:
    """Given a vision defect report and candidate source file, construct an appropriate layout fix."""
    if not report.has_defect or report.defect_type == DefectType.NONE:
        return source_content, None

    ext = target_path.lower()
    is_css = ext.endswith((".css", ".scss", ".less"))
    is_jsx = ext.endswith((".tsx", ".jsx", ".vue", ".html"))

    lines = source_content.splitlines(keepends=True)

    # 1. Text Overflow Defect Fix
    if report.defect_type == DefectType.TEXT_OVERFLOW:
        if is_css:
            selector = hint_selector or ("button" if "btn" in target_path or "button" in target_path else ".overflow-fix")
            props = {
                "overflow": "hidden",
                "text-overflow": "ellipsis",
                "white-space": "nowrap",
            }
            new_src, spec = add_css_properties(source_content, selector, props, path=target_path)
            return new_src, spec

        elif is_jsx:
            for i, line in enumerate(lines, 1):
                if re.search(r'<(?:button|span|p|div|h[1-6]|a)\b', line):
                    # Check for className or class
                    m_cls = re.search(r'(class(?:Name)?=["\'])([^"\']*?)(["\'])', line)
                    if m_cls:
                        classes = m_cls.group(2).strip()
                        if "truncate" not in classes:
                            new_classes = f"{classes} truncate".strip()
                            new_line = line[:m_cls.start(2)] + new_classes + line[m_cls.end(2):]
                            new_lines = list(lines)
                            new_lines[i - 1] = new_line
                            spec = OpSpec(
                                kind="replace_literal",
                                path=target_path,
                                span=(i, i),
                                props={
                                    "old_text": line,
                                    "new_text": new_line,
                                    "old": classes,
                                    "new": new_classes,
                                },
                            )
                            return "".join(new_lines), spec

                    # Check for cn(...) or clsx(...)
                    m_cn = re.search(r'((?:cn|clsx)\s*\([^)]*["\'])([^"\']*?)(["\'])', line)
                    if m_cn:
                        inside = m_cn.group(2).strip()
                        if "truncate" not in inside:
                            new_inside = f"{inside} truncate".strip()
                            new_line = line[:m_cn.start(2)] + new_inside + line[m_cn.end(2):]
                            new_lines = list(lines)
                            new_lines[i - 1] = new_line
                            spec = OpSpec(
                                kind="replace_literal",
                                path=target_path,
                                span=(i, i),
                                props={
                                    "old_text": line,
                                    "new_text": new_line,
                                    "old": inside,
                                    "new": new_inside,
                                },
                            )
                            return "".join(new_lines), spec

    # 2. Low Contrast Defect Fix
    elif report.defect_type == DefectType.LOW_CONTRAST:
        if is_css:
            selector = hint_selector or ".high-contrast"
            props = {"color": "#ffffff", "text-shadow": "0 1px 2px rgba(0,0,0,0.6)"}
            new_src, spec = add_css_properties(source_content, selector, props, path=target_path)
            return new_src, spec

        elif is_jsx:
            for i, line in enumerate(lines, 1):
                if re.search(r'<(?:button|span|p|div|h[1-6]|a)\b', line):
                    m_cls = re.search(r'(class(?:Name)?=["\'])([^"\']*?)(["\'])', line)
                    if m_cls:
                        classes = m_cls.group(2).strip()
                        new_classes = f"{classes} text-white drop-shadow".strip()
                        new_line = line[:m_cls.start(2)] + new_classes + line[m_cls.end(2):]
                        new_lines = list(lines)
                        new_lines[i - 1] = new_line
                        spec = OpSpec(
                            kind="replace_literal",
                            path=target_path,
                            span=(i, i),
                            props={"old_text": line, "new_text": new_line, "old": classes, "new": new_classes},
                        )
                        return "".join(new_lines), spec

    # 3. Misalignment Defect Fix
    elif report.defect_type == DefectType.MISALIGNMENT:
        if is_css:
            selector = hint_selector or ".flex-aligned"
            props = {"display": "flex", "align-items": "center", "justify-content": "center"}
            new_src, spec = add_css_properties(source_content, selector, props, path=target_path)
            return new_src, spec

        elif is_jsx:
            for i, line in enumerate(lines, 1):
                if re.search(r'<(?:div|header|nav|section)\b', line):
                    m_cls = re.search(r'(class(?:Name)?=["\'])([^"\']*?)(["\'])', line)
                    if m_cls:
                        classes = m_cls.group(2).strip()
                        new_classes = f"{classes} flex items-center justify-center".strip()
                        new_line = line[:m_cls.start(2)] + new_classes + line[m_cls.end(2):]
                        new_lines = list(lines)
                        new_lines[i - 1] = new_line
                        spec = OpSpec(
                            kind="replace_literal",
                            path=target_path,
                            span=(i, i),
                            props={"old_text": line, "new_text": new_line, "old": classes, "new": new_classes},
                        )
                        return "".join(new_lines), spec

    # 4. Cramped Spacing Defect Fix
    elif report.defect_type == DefectType.CRAMPED_SPACING:
        if is_css:
            selector = hint_selector or ".spacious-box"
            props = {"gap": "1rem", "padding": "1rem"}
            new_src, spec = add_css_properties(source_content, selector, props, path=target_path)
            return new_src, spec

        elif is_jsx:
            for i, line in enumerate(lines, 1):
                if re.search(r'<(?:div|header|nav|section|ul)\b', line):
                    m_cls = re.search(r'(class(?:Name)?=["\'])([^"\']*?)(["\'])', line)
                    if m_cls:
                        classes = m_cls.group(2).strip()
                        new_classes = f"{classes} gap-4 p-4".strip()
                        new_line = line[:m_cls.start(2)] + new_classes + line[m_cls.end(2):]
                        new_lines = list(lines)
                        new_lines[i - 1] = new_line
                        spec = OpSpec(
                            kind="replace_literal",
                            path=target_path,
                            span=(i, i),
                            props={"old_text": line, "new_text": new_line, "old": classes, "new": new_classes},
                        )
                        return "".join(new_lines), spec

    return source_content, None
