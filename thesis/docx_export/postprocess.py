from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from .openxml import (
    CONTENT_TYPES_NS,
    M_NS,
    PKG_REL_NS,
    R_NS,
    W_NS,
    DocxPackage,
    empty_relationships,
    qn,
    relationship_elements,
)


DOCUMENT = "word/document.xml"
DOCUMENT_RELS = "word/_rels/document.xml.rels"
CONTENT_TYPES = "[Content_Types].xml"
STYLES = "word/styles.xml"
IMAGE_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
CHAPTER_HEADER_STYLE_NAME = "大标题"
BIBLIOGRAPHY_MARKER = "DOCX_EXPORT_BIBLIOGRAPHY_MARKER"
XML_NS = "http://www.w3.org/XML/1998/namespace"


@dataclass(frozen=True)
class ThesisMetadata:
    values: dict[str, str]
    cabstract: str = ""
    eabstract: str = ""

    def get(self, key: str, default: str = "") -> str:
        return self.values.get(key, default)


@dataclass(frozen=True)
class EquationInfo:
    labels: dict[str, str]
    numbers: tuple[str, ...]


@dataclass(frozen=True)
class CaptionInfo:
    figures: tuple[str, ...]
    tables: tuple[str, ...]


def merge_with_template(
    template_docx: str | Path,
    intermediate_docx: str | Path,
    output_docx: str | Path,
    source_root: str | Path | None = None,
) -> None:
    template = DocxPackage.read(template_docx)
    intermediate = DocxPackage.read(intermediate_docx)
    source_path = Path(source_root) if source_root is not None else None
    metadata = _load_thesis_metadata(source_path)
    equations = _load_equation_info(source_path)
    captions = _load_caption_info(source_path)

    template_document = template.xml_part(DOCUMENT)
    intermediate_document = intermediate.xml_part(DOCUMENT)
    template_body = template_document.find(qn(W_NS, "body"))
    intermediate_body = intermediate_document.find(qn(W_NS, "body"))
    if template_body is None or intermediate_body is None:
        raise ValueError("Both template and intermediate DOCX must contain word/document.xml bodies")

    relationship_map = _copy_document_relationships(template, intermediate, intermediate_document)
    _rewrite_relationship_ids(intermediate_document, relationship_map)
    _merge_numbering_definitions(template, intermediate, intermediate_document)

    template_section = _extract_body_section_properties(template_body) or _extract_section_properties(template_body)
    front_matter = _front_matter_from_template(template_body, metadata)
    template_body.clear()
    if front_matter:
        for child in front_matter:
            template_body.append(child)
    else:
        _insert_toc_placeholder(template_body)
    for child in _content_children_from_intermediate(intermediate_body):
        if child.tag == qn(W_NS, "sectPr"):
            continue
        copied = copy.deepcopy(child)
        _style_tree(copied)
        template_body.append(copied)
    if template_section is not None:
        template_body.append(template_section)
    _relocate_bibliography_to_marker(template_body)
    _normalize_numeric_citation_ranges(template_body)
    _normalize_captions(template_body, captions)
    _normalize_equations_and_references(template_body, equations)
    _link_author_year_citations(template_body)

    template.set_xml_part(DOCUMENT, template_document)
    _normalize_template_styles(template)
    _update_template_headers(template, metadata)
    _ensure_media_content_types(template)
    template.write(output_docx)


def normalize_docx_styles(docx_path: str | Path) -> None:
    package = DocxPackage.read(docx_path)
    changed = _normalize_template_styles(package)
    changed = _normalize_caption_paragraph_overrides(package) or changed
    if changed:
        package.write(docx_path)


def _load_thesis_metadata(source_root: Path | None) -> ThesisMetadata:
    if source_root is None:
        return ThesisMetadata({})
    cover = source_root / "front" / "cover.tex"
    if not cover.exists():
        return ThesisMetadata({})
    text = cover.read_text(encoding="utf-8")
    return ThesisMetadata(
        values=_parse_heusetup_values(text),
        cabstract=_extract_latex_environment(text, "cabstract"),
        eabstract=_extract_latex_environment(text, "eabstract"),
    )


def _load_equation_info(source_root: Path | None) -> EquationInfo:
    if source_root is None:
        return EquationInfo({}, ())
    aux_files = sorted((source_root / "body").glob("*.aux")) if (source_root / "body").exists() else []
    if (source_root / "thesis.aux").exists():
        aux_files.append(source_root / "thesis.aux")

    labels: dict[str, str] = {}
    numbers: list[str] = []
    for aux_file in aux_files:
        text = aux_file.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"\\newlabel\{(eq:[^{}]+)\}\{\{([^{}]+)\}", text):
            labels[match.group(1)] = match.group(2)
        for match in re.finditer(r"\\contentsline\s*\{equation\}\s*\{\\numberline\s*\{([^{}]+)\}", text):
            numbers.append(match.group(1))
    return EquationInfo(labels, tuple(numbers))


def _load_caption_info(source_root: Path | None) -> CaptionInfo:
    if source_root is None:
        return CaptionInfo((), ())
    aux_files = sorted((source_root / "body").glob("*.aux")) if (source_root / "body").exists() else []
    figures: list[str] = []
    tables: list[str] = []
    for aux_file in aux_files:
        text = aux_file.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"\\newlabel\{(fig|tab):[^{}]+\}\{\{([^{}]+)\}", text):
            kind, number = match.groups()
            if kind == "fig":
                figures.append(number)
            else:
                tables.append(number)
    return CaptionInfo(tuple(figures), tuple(tables))


def _parse_heusetup_values(text: str) -> dict[str, str]:
    setup_body = _extract_braced_command(text, r"\heusetup")
    if not setup_body:
        return {}

    values: dict[str, str] = {}
    index = 0
    while index < len(setup_body):
        while index < len(setup_body) and setup_body[index] in " \t\r\n,":
            index += 1
        key_start = index
        while index < len(setup_body) and (setup_body[index].isalpha() or setup_body[index] == "_"):
            index += 1
        key = setup_body[key_start:index]
        while index < len(setup_body) and setup_body[index].isspace():
            index += 1
        if not key or index >= len(setup_body) or setup_body[index] != "=":
            index += 1
            continue
        index += 1
        while index < len(setup_body) and setup_body[index].isspace():
            index += 1
        if index >= len(setup_body) or setup_body[index] != "{":
            continue
        value, index = _read_braced_value(setup_body, index)
        values[key] = _clean_latex_text(value)
    return values


def _extract_braced_command(text: str, command: str) -> str:
    start = text.find(command)
    if start < 0:
        return ""
    brace = text.find("{", start + len(command))
    if brace < 0:
        return ""
    value, _ = _read_braced_value(text, brace)
    return value


def _read_braced_value(text: str, open_index: int) -> tuple[str, int]:
    depth = 0
    chars: list[str] = []
    index = open_index
    while index < len(text):
        char = text[index]
        escaped = index > 0 and text[index - 1] == "\\"
        if char == "{" and not escaped:
            depth += 1
            if depth > 1:
                chars.append(char)
        elif char == "}" and not escaped:
            depth -= 1
            if depth == 0:
                return "".join(chars), index + 1
            chars.append(char)
        else:
            chars.append(char)
        index += 1
    return "".join(chars), index


def _extract_latex_environment(text: str, name: str) -> str:
    match = re.search(
        rf"\\begin\{{{re.escape(name)}\}}(?P<body>.*?)\\end\{{{re.escape(name)}\}}",
        text,
        re.DOTALL,
    )
    if not match:
        return ""
    return _clean_latex_text(match.group("body"))


def _clean_latex_text(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = text.replace(r"\\", "\n")
    text = text.replace(r"\ ", " ")
    text = text.replace("---", "-")
    text = text.replace("--", "-")
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", lambda match: match.group(1) or "", text)
    text = text.replace("{", "").replace("}", "")
    lines = [line.strip() for line in text.splitlines()]
    paragraphs = [" ".join(part for part in line.split()) for line in lines if line]
    return "\n".join(paragraphs)


def _front_matter_from_template(template_body: ET.Element, metadata: ThesisMetadata) -> list[ET.Element]:
    children = list(template_body)
    first_body = _find_template_body_start(children)
    if first_body is None:
        return []
    front = [copy.deepcopy(child) for child in children[:first_body]]
    front = _drop_duplicate_simple_cover(front)
    _replace_cover_placeholders(front, metadata)
    front = _replace_abstract_blocks(front, metadata)
    front = _replace_toc_block(front)
    return front


def _find_template_body_start(children: list[ET.Element]) -> int | None:
    for index, child in enumerate(children):
        if child.tag != qn(W_NS, "p"):
            continue
        text = _paragraph_text(child).strip()
        style = _paragraph_style(child)
        if style == "aff8" and (text == "绪论" or text.startswith("第")):
            return index
    return None


def _drop_duplicate_simple_cover(children: list[ET.Element]) -> list[ET.Element]:
    cover_starts = [
        index
        for index, child in enumerate(children)
        if child.tag == qn(W_NS, "p") and _paragraph_text(child).strip().startswith("分类号：")
    ]
    if len(cover_starts) < 2:
        return children
    return children[cover_starts[1] :]


def _replace_cover_placeholders(children: list[ET.Element], metadata: ThesisMetadata) -> None:
    ctitle = metadata.get("ctitle", "")
    ctitlecover = metadata.get("ctitlecover", ctitle)
    etitle = metadata.get("etitle", "")
    esubtitle = metadata.get("esubtitle", "")
    if esubtitle:
        etitle = f"{etitle}\n{esubtitle}" if etitle else esubtitle
    cdegree = f"{metadata.get('cxueke', '工学')}硕士学位论文"
    english_degree = f"A Dissertation for the Degree of {metadata.get('estudenttype', 'Master')}"
    replacements = {
        "理学硕士学位论文": cdegree,
        "理(工)学硕士学位论文": cdegree,
        "论文题目（25字以内）": ctitlecover,
        "论文题目": ctitlecover,
        "20xx年xx月": metadata.get("csubmitdate", "2026年3月"),
        "Classified Index:": f"Classified Index: {metadata.get('natclassifiedindex')}",
        "U.D.C:": f"U.D.C: {metadata.get('intclassifiedindex')}",
        "A Dissertation for the Degree of M. Sci": english_degree,
        "Title of the Dissertation": etitle,
    }
    chinese_detail_anchor: int | None = None
    english_detail_anchor: int | None = None
    has_chinese_detail_table = any(_table_has_label(child, "学位级别") for child in children if child.tag == qn(W_NS, "tbl"))
    has_english_detail_table = any(_table_has_label(child, "Academic Degree Applied for:") for child in children if child.tag == qn(W_NS, "tbl"))
    for index, child in enumerate(children):
        if child.tag == qn(W_NS, "tbl"):
            _replace_cover_table(child, metadata)
            continue
        if child.tag != qn(W_NS, "p"):
            continue
        text = _paragraph_text(child).strip()
        if _replace_chinese_cover_index_line(child, metadata):
            continue
        if text == "论文题目":
            chinese_detail_anchor = index
        elif text == "Title of the Dissertation":
            english_detail_anchor = index
        if text in replacements and replacements[text]:
            _replace_paragraph_text(child, replacements[text])
    if english_detail_anchor is not None and etitle and has_english_detail_table:
        _drop_following_blank_paragraphs(children, english_detail_anchor, 3)
    if chinese_detail_anchor is not None and not has_chinese_detail_table:
        _fill_following_blank_paragraphs(children, chinese_detail_anchor, _chinese_cover_detail_lines(metadata))
    if english_detail_anchor is not None and not has_english_detail_table:
        _fill_following_blank_paragraphs(children, english_detail_anchor, _english_cover_detail_lines(metadata))


def _update_template_headers(package: DocxPackage, metadata: ThesisMetadata) -> None:
    title = metadata.get("ctitle", "") or metadata.get("ctitlecover", "")
    for part_name in sorted(name for name in package.parts if re.fullmatch(r"word/header\d+\.xml", name)):
        root = package.xml_part(part_name)
        changed = False
        for paragraph in root.findall(qn(W_NS, "p")):
            text = _paragraph_text(paragraph).strip()
            if _has_styleref_field(paragraph):
                continue
            if text in {"论文题目", "论文题目（25字以内）"} and title:
                _replace_paragraph_text(paragraph, title)
                changed = True
            elif _is_chapter_header_text(text):
                _replace_paragraph_with_styleref_field(paragraph, CHAPTER_HEADER_STYLE_NAME, _normalize_header_text(text))
                changed = True
        if changed:
            package.set_xml_part(part_name, root)


def _normalize_template_styles(package: DocxPackage) -> bool:
    if STYLES not in package.parts:
        return False
    styles = package.xml_part(STYLES)
    changed = False
    for style in styles.findall(qn(W_NS, "style")):
        name = style.find(qn(W_NS, "name"))
        style_name = name.attrib.get(qn(W_NS, "val")) if name is not None else ""
        style_id = style.attrib.get(qn(W_NS, "styleId"), "")
        if style.attrib.get(qn(W_NS, "type")) != "paragraph":
            continue
        if style_id == "a0" or style_name == "Normal":
            _set_style_paragraph_format(style, line="440", line_rule="exact", jc="both")
            _set_style_run_format(style, size="24", east_asia="宋体", ascii_font="Times New Roman")
            changed = True
        elif style_name == "毕设图题":
            _set_style_paragraph_format(style, after="50", line="440", line_rule="exact", jc="center", first_line="0")
            _set_style_run_format(style, size="21", east_asia="宋体", ascii_font="Times New Roman")
            changed = True
        elif style_name == "图表":
            _set_style_paragraph_format(style, line="392", line_rule="atLeast", jc="center", first_line="0")
            _set_style_run_format(style, size="21", east_asia="宋体", ascii_font="Times New Roman")
            changed = True
        elif style_name in {"毕设论文公式", "公式新标准", "必须标准公式"}:
            _set_style_paragraph_format(style, before="156", after="40", line="240", line_rule="auto", jc="center", first_line="0")
            _set_style_run_format(style, size="24", east_asia="宋体", ascii_font="Times New Roman")
            changed = True
        elif style_id in {"2", "3"} or style_name in {"heading 2", "heading 3"}:
            _set_style_paragraph_format(style, before="260", after="260", line="416", line_rule="atLeast", first_line="0")
            _set_style_run_format(style, size="32", east_asia="黑体", ascii_font="Times New Roman", bold=False)
            changed = True
    if changed:
        package.set_xml_part(STYLES, styles)
    return changed


def _normalize_caption_paragraph_overrides(package: DocxPackage) -> bool:
    if DOCUMENT not in package.parts:
        return False
    document = package.xml_part(DOCUMENT)
    changed = False
    for paragraph in document.iter(qn(W_NS, "p")):
        if _paragraph_style(paragraph) not in {"aa", "affa", "a2", "af9"}:
            continue
        ppr = paragraph.find(qn(W_NS, "pPr"))
        if ppr is None:
            continue
        spacing = ppr.find(qn(W_NS, "spacing"))
        if spacing is not None:
            ppr.remove(spacing)
            changed = True
    if changed:
        package.set_xml_part(DOCUMENT, document)
    return changed


def _set_style_paragraph_format(
    style: ET.Element,
    *,
    before: str | None = None,
    after: str | None = None,
    line: str | None = None,
    line_rule: str | None = None,
    jc: str | None = None,
    first_line: str | None = None,
) -> None:
    ppr = style.find(qn(W_NS, "pPr"))
    if ppr is None:
        ppr = ET.Element(qn(W_NS, "pPr"))
        style.append(ppr)
    if any(value is not None for value in (before, after, line, line_rule)):
        spacing = ppr.find(qn(W_NS, "spacing"))
        if spacing is None:
            spacing = ET.SubElement(ppr, qn(W_NS, "spacing"))
        for attr_name, value in {
            "before": before,
            "after": after,
            "line": line,
            "lineRule": line_rule,
        }.items():
            if value is not None:
                spacing.set(qn(W_NS, attr_name), value)
    if jc is not None:
        justification = ppr.find(qn(W_NS, "jc"))
        if justification is None:
            justification = ET.SubElement(ppr, qn(W_NS, "jc"))
        justification.set(qn(W_NS, "val"), jc)
    if first_line is not None:
        indent = ppr.find(qn(W_NS, "ind"))
        if indent is None:
            indent = ET.SubElement(ppr, qn(W_NS, "ind"))
        if first_line == "0":
            for attr_name in ("left", "leftChars", "right", "rightChars", "hanging", "hangingChars"):
                indent.attrib.pop(qn(W_NS, attr_name), None)
        indent.set(qn(W_NS, "firstLine"), first_line)
        indent.set(qn(W_NS, "firstLineChars"), "0" if first_line == "0" else "200")


def _set_style_run_format(style: ET.Element, *, size: str, east_asia: str, ascii_font: str, bold: bool | None = None) -> None:
    rpr = style.find(qn(W_NS, "rPr"))
    if rpr is None:
        rpr = ET.Element(qn(W_NS, "rPr"))
        style.append(rpr)
    fonts = rpr.find(qn(W_NS, "rFonts"))
    if fonts is None:
        fonts = ET.Element(qn(W_NS, "rFonts"))
        rpr.insert(0, fonts)
    fonts.set(qn(W_NS, "ascii"), ascii_font)
    fonts.set(qn(W_NS, "hAnsi"), ascii_font)
    fonts.set(qn(W_NS, "eastAsia"), east_asia)
    fonts.set(qn(W_NS, "cs"), ascii_font)
    for theme_attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "csTheme"):
        fonts.attrib.pop(qn(W_NS, theme_attr), None)
    _set_run_size(rpr, size)
    if bold is not None:
        _set_run_bold(rpr, bold)


def _has_styleref_field(paragraph: ET.Element) -> bool:
    return any("STYLEREF" in (instr.text or "") for instr in paragraph.findall(f".//{qn(W_NS, 'instrText')}"))


def _is_chapter_header_text(text: str) -> bool:
    compact = _normalize_header_text(text)
    return bool(re.match(r"^第\s*\d+\s*章\b", compact))


def _normalize_header_text(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split())


def _replace_paragraph_with_styleref_field(paragraph: ET.Element, style_name: str, fallback_text: str) -> None:
    ppr = paragraph.find(qn(W_NS, "pPr"))
    first_rpr = paragraph.find(f".//{qn(W_NS, 'rPr')}")
    for child in list(paragraph):
        if child is ppr:
            continue
        paragraph.remove(child)
    _append_field_char_run(paragraph, "begin", first_rpr)
    instr_run = _append_run_with_properties(paragraph, first_rpr)
    instr = ET.SubElement(instr_run, qn(W_NS, "instrText"))
    instr.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    instr.text = f' STYLEREF "{style_name}" \\* MERGEFORMAT '
    _append_field_char_run(paragraph, "separate", first_rpr)
    _append_text_run(paragraph, fallback_text, first_rpr)
    _append_field_char_run(paragraph, "end", first_rpr)


def _append_field_char_run(paragraph: ET.Element, field_type: str, rpr: ET.Element | None) -> None:
    run = _append_run_with_properties(paragraph, rpr)
    ET.SubElement(run, qn(W_NS, "fldChar"), {qn(W_NS, "fldCharType"): field_type})


def _append_run_with_properties(paragraph: ET.Element, rpr: ET.Element | None) -> ET.Element:
    run = ET.SubElement(paragraph, qn(W_NS, "r"))
    if rpr is not None:
        run.append(copy.deepcopy(rpr))
    return run


def _relocate_bibliography_to_marker(body: ET.Element) -> None:
    marker = _find_direct_paragraph(body, BIBLIOGRAPHY_MARKER)
    if marker is None:
        return
    bibliography = [child for child in list(body) if child.tag == qn(W_NS, "p") and _is_numbered_bibliography_entry(child)]
    for paragraph in bibliography:
        body.remove(paragraph)
    insert_at = list(body).index(marker)
    body.remove(marker)
    for offset, paragraph in enumerate(bibliography):
        body.insert(insert_at + offset, paragraph)


def _find_direct_paragraph(body: ET.Element, text: str) -> ET.Element | None:
    for child in list(body):
        if child.tag == qn(W_NS, "p") and _paragraph_text(child).strip() == text:
            return child
    return None


def _is_numbered_bibliography_entry(paragraph: ET.Element) -> bool:
    style = _paragraph_style(paragraph)
    text = _paragraph_text(paragraph).strip()
    return style in {"a", "Bibliography", "Reference", "references"} and bool(re.match(r"^\[\d+\]", text))


def _normalize_numeric_citation_ranges(root: ET.Element) -> None:
    for run in root.iter(qn(W_NS, "r")):
        if not _is_superscript_run(run):
            continue
        for text in run.findall(qn(W_NS, "t")):
            if text.text and re.fullmatch(r"\[[0-9, \-–]+\]", text.text):
                text.text = text.text.replace("–", "-")


def _normalize_captions(body: ET.Element, captions: CaptionInfo) -> None:
    figure_index = 0
    table_index = 0
    for child in list(body):
        if child.tag != qn(W_NS, "p"):
            continue
        style = _paragraph_style(child)
        text = _paragraph_text(child).strip()
        if not text:
            continue
        if style in {"affa", "aa", "ImageCaption"}:
            if figure_index < len(captions.figures):
                _ensure_caption_label(child, "图", captions.figures[figure_index])
            _strip_caption_terminal_punctuation(child)
            _format_figure_caption_paragraph(child)
            figure_index += 1
        elif style in {"af9", "a2", "TableCaption"}:
            if table_index < len(captions.tables):
                _ensure_caption_label(child, "表", captions.tables[table_index])
            _strip_caption_terminal_punctuation(child)
            _format_table_caption_paragraph(child)
            table_index += 1


def _ensure_caption_label(paragraph: ET.Element, kind: str, number: str) -> None:
    text = _paragraph_text(paragraph).strip()
    if re.match(rf"^{kind}\s*\d+(?:\.\d+)*\s+", text):
        return
    prefix_run = ET.Element(qn(W_NS, "r"))
    rpr = _ensure_rpr(prefix_run)
    _set_run_size(rpr, "21")
    fonts = ET.SubElement(rpr, qn(W_NS, "rFonts"))
    fonts.set(qn(W_NS, "ascii"), "Times New Roman")
    fonts.set(qn(W_NS, "hAnsi"), "Times New Roman")
    fonts.set(qn(W_NS, "eastAsia"), "宋体")
    fonts.set(qn(W_NS, "cs"), "Times New Roman")
    text_node = ET.SubElement(prefix_run, qn(W_NS, "t"))
    text_node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_node.text = f"{kind}{number} "
    insert_at = 1 if paragraph.find(qn(W_NS, "pPr")) is not None else 0
    paragraph.insert(insert_at, prefix_run)


def _strip_caption_terminal_punctuation(paragraph: ET.Element) -> None:
    text_nodes = [node for node in paragraph.iter(qn(W_NS, "t")) if node.text]
    if not text_nodes:
        return
    last = text_nodes[-1]
    last.text = re.sub(r"[。．.；;：:，,、\s]+$", "", last.text)


def _normalize_equations_and_references(body: ET.Element, equations: EquationInfo) -> None:
    if not equations.labels and not equations.numbers:
        return
    if equations.labels:
        _replace_equation_reference_labels(body, equations.labels)
    if equations.numbers:
        _number_display_equation_paragraphs(body, equations.numbers)


def _replace_equation_reference_labels(root: ET.Element, labels: dict[str, str]) -> None:
    label_pattern = re.compile(r"\[(eq:[A-Za-z0-9_.:-]+)\]|\\eqref\{(eq:[A-Za-z0-9_.:-]+)\}")
    for paragraph in root.iter(qn(W_NS, "p")):
        for text in paragraph.iter(qn(W_NS, "t")):
            if not text.text:
                continue

            def replace(match: re.Match[str]) -> str:
                label = match.group(1) or match.group(2)
                number = labels.get(label)
                return f"（{number}）" if number else match.group(0)

            normalized = label_pattern.sub(replace, text.text)
            normalized = re.sub(r"(式|公式)\s+（", r"\1（", normalized)
            normalized = re.sub(r"（([A-Z]?\d+(?:-\d+)?)）\s+(?=[\u4e00-\u9fff，。、；：])", r"（\1）", normalized)
            text.text = normalized
        _normalize_equation_reference_spacing(paragraph)


def _normalize_equation_reference_spacing(paragraph: ET.Element) -> None:
    text_nodes = list(paragraph.iter(qn(W_NS, "t")))
    for index in range(1, len(text_nodes) - 1):
        previous = text_nodes[index - 1]
        current = text_nodes[index]
        following = text_nodes[index + 1]
        if current.text is None or not current.text.isspace() or not previous.text or not following.text:
            continue
        if re.search(r"(式|公式)$", previous.text) and re.match(r"^（[A-Z]?\d+(?:-\d+)?）", following.text):
            current.text = ""
        elif re.search(r"（[A-Z]?\d+(?:-\d+)?）$", previous.text) and re.match(r"^[\u4e00-\u9fff，。、；：]", following.text):
            current.text = ""
    for index in range(len(text_nodes) - 1):
        current = text_nodes[index]
        following = text_nodes[index + 1]
        if not current.text or not following.text:
            continue
        if re.search(r"(式|公式)\s+$", current.text) and re.match(r"^（[A-Z]?\d+(?:-\d+)?）", following.text):
            current.text = re.sub(r"(式|公式)\s+$", r"\1", current.text)
        if re.search(r"（[A-Z]?\d+(?:-\d+)?）\s*$", current.text) and re.match(r"^\s*[\u4e00-\u9fff，。、；：]", following.text):
            current.text = re.sub(r"(（[A-Z]?\d+(?:-\d+)?）)\s*$", r"\1", current.text)
            following.text = re.sub(r"^\s+(?=[\u4e00-\u9fff，。、；：])", "", following.text)


def _number_display_equation_paragraphs(body: ET.Element, numbers: tuple[str, ...]) -> None:
    number_index = 0
    for child in list(body):
        if child.tag != qn(W_NS, "p") or not _is_display_equation_paragraph(child):
            continue
        if _paragraph_has_equation_number(child):
            continue
        if number_index >= len(numbers):
            return
        _format_display_equation_paragraph(child, numbers[number_index])
        number_index += 1


def _is_display_equation_paragraph(paragraph: ET.Element) -> bool:
    if paragraph.find(f".//{qn(M_NS, 'oMathPara')}") is not None:
        return True
    text = _paragraph_text(paragraph).strip()
    return not text and paragraph.find(f".//{qn(M_NS, 'oMath')}") is not None


def _paragraph_has_equation_number(paragraph: ET.Element) -> bool:
    return bool(re.search(r"[（(][A-Z]?\d+(?:-\d+)?[）)]", _paragraph_text(paragraph)))


def _format_display_equation_paragraph(paragraph: ET.Element, number: str) -> None:
    _set_paragraph_style(paragraph, "affe")
    _set_first_line_indent(paragraph, "0")
    _set_centered(paragraph)
    _set_paragraph_spacing(paragraph, before="156", after="40", line="240", line_rule="auto")
    _ensure_formula_tab_stops(paragraph)
    _ensure_formula_leading_tab(paragraph)
    run = ET.SubElement(paragraph, qn(W_NS, "r"))
    rpr = _ensure_rpr(run)
    fonts = ET.SubElement(rpr, qn(W_NS, "rFonts"))
    fonts.set(qn(W_NS, "ascii"), "Times New Roman")
    fonts.set(qn(W_NS, "hAnsi"), "Times New Roman")
    fonts.set(qn(W_NS, "eastAsia"), "宋体")
    fonts.set(qn(W_NS, "cs"), "Times New Roman")
    _set_run_size(rpr, "24")
    ET.SubElement(run, qn(W_NS, "tab"))
    text = ET.SubElement(run, qn(W_NS, "t"))
    text.text = f"（{number}）"


def _ensure_formula_tab_stops(paragraph: ET.Element) -> None:
    ppr = _ensure_ppr(paragraph)
    tabs = ppr.find(qn(W_NS, "tabs"))
    if tabs is None:
        tabs = ET.Element(qn(W_NS, "tabs"))
        insert_at = 0
        for index, child in enumerate(list(ppr)):
            if child.tag == qn(W_NS, "pStyle"):
                insert_at = index + 1
                break
        ppr.insert(insert_at, tabs)
    wanted = {"center": "4536", "right": "9070"}
    existing = {tab.attrib.get(qn(W_NS, "val")) for tab in tabs.findall(qn(W_NS, "tab"))}
    for value, position in wanted.items():
        if value not in existing:
            ET.SubElement(tabs, qn(W_NS, "tab"), {qn(W_NS, "val"): value, qn(W_NS, "pos"): position})


def _ensure_formula_leading_tab(paragraph: ET.Element) -> None:
    children = list(paragraph)
    first_content_index = 1 if children and children[0].tag == qn(W_NS, "pPr") else 0
    if first_content_index < len(children):
        first_content = children[first_content_index]
        if first_content.tag == qn(W_NS, "r") and first_content.find(qn(W_NS, "tab")) is not None:
            return
    run = ET.Element(qn(W_NS, "r"))
    ET.SubElement(run, qn(W_NS, "tab"))
    paragraph.insert(first_content_index, run)


def _is_superscript_run(run: ET.Element) -> bool:
    vertical = run.find(f"{qn(W_NS, 'rPr')}/{qn(W_NS, 'vertAlign')}")
    return vertical is not None and vertical.attrib.get(qn(W_NS, "val")) == "superscript"


def _table_has_label(table: ET.Element, label: str) -> bool:
    return any(_paragraph_text(cell).strip() == label for cell in table.findall(f".//{qn(W_NS, 'tc')}"))


def _replace_cover_table(table: ET.Element, metadata: ThesisMetadata) -> None:
    chinese_values = {
        "硕士研究生": f"：{metadata.get('cauthor')}",
        "指导教师": f"：{metadata.get('csupervisor')}",
        "副导师": f"：{metadata.get('cassosupervisor')}",
        "副 导 师": f"：{metadata.get('cassosupervisor')}",
        "学位级别": f"：{metadata.get('cxueke', '工学')}硕士",
        "学科、专业": f"：{metadata.get('csubject')}",
        "所在单位": f"：{metadata.get('caffil')}",
        "论文提交日期": f"：{metadata.get('csubmitdate')}",
        "论文答辩日期": f"：{metadata.get('cdefensedate')}",
        "学位授予单位": "：哈尔滨工程大学",
    }
    english_values = {
        "Candidate:": metadata.get("eauthor"),
        "Supervisor:": metadata.get("esupervisor"),
        "Academic Degree Applied for:": metadata.get("estudenttype"),
        "Specialty:": metadata.get("esubject"),
        "Date of Submission:": metadata.get("esubmitdate"),
        "Date of Oral Examination:": metadata.get("edefensedate"),
        "University:": "Harbin Engineering University",
    }
    for label, value in {**chinese_values, **english_values}.items():
        _replace_table_value(table, label, value)


def _replace_table_value(table: ET.Element, label: str, value: str) -> None:
    for row in table.findall(qn(W_NS, "tr")):
        cells = row.findall(qn(W_NS, "tc"))
        if len(cells) < 2:
            continue
        if _paragraph_text(cells[0]).strip() != label:
            continue
        paragraph = cells[1].find(qn(W_NS, "p"))
        if paragraph is not None:
            _replace_paragraph_text(paragraph, value)
        return


def _replace_chinese_cover_index_line(paragraph: ET.Element, metadata: ThesisMetadata) -> bool:
    text = _paragraph_text(paragraph).strip()
    if text.startswith("分类号：") and "密级：" in text:
        return _fill_underlined_cover_fields(
            paragraph,
            (metadata.get("natclassifiedindex"), metadata.get("statesecrets")),
            _classification_line(metadata),
        )
    if text.startswith("U D C") and "编号：" in text:
        return _fill_underlined_cover_fields(
            paragraph,
            (metadata.get("intclassifiedindex"), metadata.get("cnumber")),
            _udc_line(metadata),
        )
    return False


def _fill_underlined_cover_fields(paragraph: ET.Element, values: tuple[str, str], fallback_text: str) -> bool:
    underlined_runs = [run for run in paragraph.findall(qn(W_NS, "r")) if _is_underlined_run(run)]
    if len(underlined_runs) < len(values):
        _replace_paragraph_text(paragraph, fallback_text)
        return True
    for run, value in zip(underlined_runs, values):
        _fill_underlined_placeholder(run, value)
    return True


def _is_underlined_run(run: ET.Element) -> bool:
    return run.find(f"{qn(W_NS, 'rPr')}/{qn(W_NS, 'u')}") is not None


def _fill_underlined_placeholder(run: ET.Element, value: str) -> None:
    original = _run_text(run)
    if value:
        text = value + (" " * max(0, len(original) - len(value)))
    else:
        text = original
    _replace_run_text(run, text)


def _run_text(run: ET.Element) -> str:
    return "".join(text.text or "" for text in run.findall(qn(W_NS, "t")))


def _replace_run_text(run: ET.Element, text: str) -> None:
    rpr = run.find(qn(W_NS, "rPr"))
    for child in list(run):
        if child is rpr:
            continue
        run.remove(child)
    text_node = ET.SubElement(run, qn(W_NS, "t"))
    text_node.text = text
    if text != text.strip():
        text_node.set(f"{{{XML_NS}}}space", "preserve")


def _classification_line(metadata: ThesisMetadata) -> str:
    index = metadata.get("natclassifiedindex")
    secrecy = metadata.get("statesecrets")
    return f"分类号：{index}                             密级：{secrecy}"


def _udc_line(metadata: ThesisMetadata) -> str:
    udc = metadata.get("intclassifiedindex")
    number = metadata.get("cnumber")
    return f"U D C ：{udc}                             编号：{number}"


def _chinese_cover_detail_lines(metadata: ThesisMetadata) -> list[str]:
    lines = [
        f"硕士研究生：{metadata.get('cauthor')}",
        f"导        师：{metadata.get('csupervisor')}",
    ]
    assosupervisor = metadata.get("cassosupervisor")
    if assosupervisor:
        lines.append(f"副  导  师：{assosupervisor}")
    lines.extend(
        [
            f"学科专业：{metadata.get('csubject')}",
            f"所在单位：{metadata.get('caffil')}",
            f"论文提交日期：{metadata.get('csubmitdate')}",
        ]
    )
    return [line for line in lines if line.split("：", 1)[-1].strip()]


def _english_cover_detail_lines(metadata: ThesisMetadata) -> list[str]:
    return [
        line
        for line in [
            f"Candidate: {metadata.get('eauthor')}",
            f"Supervisor: {metadata.get('esupervisor')}",
            f"Specialty: {metadata.get('esubject')}",
            f"Affiliation: {metadata.get('eaffil')}",
            f"Date: {metadata.get('esubmitdate')}",
        ]
        if line.split(":", 1)[-1].strip()
    ]


def _fill_following_blank_paragraphs(children: list[ET.Element], anchor: int, lines: list[str]) -> None:
    cursor = anchor + 1
    for line in lines:
        while cursor < len(children):
            child = children[cursor]
            if child.tag == qn(W_NS, "p") and not _paragraph_text(child).strip():
                _replace_paragraph_text(child, line)
                cursor += 1
                break
            cursor += 1


def _drop_following_blank_paragraphs(children: list[ET.Element], anchor: int, count: int) -> None:
    cursor = anchor + 1
    removed = 0
    while cursor < len(children) and removed < count:
        child = children[cursor]
        if child.tag == qn(W_NS, "p") and not _paragraph_text(child).strip():
            if child.find(f"{qn(W_NS, 'pPr')}/{qn(W_NS, 'sectPr')}") is not None:
                break
            del children[cursor]
            removed += 1
            continue
        if child.tag in {qn(W_NS, "p"), qn(W_NS, "tbl")}:
            break
        cursor += 1


def _replace_abstract_blocks(children: list[ET.Element], metadata: ThesisMetadata) -> list[ET.Element]:
    chinese_start = _find_paragraph_index(children, "摘    要")
    english_start = _find_paragraph_index(children, "Abstract")
    toc_start = _find_paragraph_index(children, "目    录")
    if chinese_start is None or english_start is None or toc_start is None:
        return children

    result = children[: chinese_start + 1]
    result.extend(_abstract_content_block(children, chinese_start, english_start, metadata.cabstract, _keywords_line("关键词：", metadata.get("ckeywords"))))
    result.append(children[english_start])
    result.extend(_abstract_content_block(children, english_start, toc_start, metadata.eabstract, _keywords_line("Keywords: ", metadata.get("ekeywords"))))
    result.extend(children[toc_start:])
    return result


def _abstract_content_block(
    children: list[ET.Element],
    start: int,
    end: int,
    abstract_text: str,
    keywords: str,
) -> list[ET.Element]:
    section_break = _last_section_paragraph(children, start + 1, end)
    body_template = _first_non_section_paragraph(children, start + 1, end) or children[start]
    keyword_template = _paragraph_with_prefix(children, start + 1, end, "关键词") or _paragraph_with_prefix(children, start + 1, end, "Keywords") or body_template
    block = [
        _paragraph_from_template(body_template, abstract_text),
        _paragraph_from_template(keyword_template, keywords),
    ]
    if section_break is not None:
        blank = copy.deepcopy(section_break)
        _replace_paragraph_text(blank, "")
        block.append(blank)
    return block


def _keywords_line(prefix: str, value: str) -> str:
    keywords = [item.strip() for item in re.split(r"[,，;；]", value) if item.strip()]
    separator = "; " if prefix.startswith("Keywords") else "；"
    return f"{prefix}{separator.join(keywords)}"


def _last_section_paragraph(children: list[ET.Element], start: int, end: int) -> ET.Element | None:
    for child in reversed(children[start:end]):
        if child.tag == qn(W_NS, "p") and child.find(f"{qn(W_NS, 'pPr')}/{qn(W_NS, 'sectPr')}") is not None:
            return child
    return None


def _first_non_section_paragraph(children: list[ET.Element], start: int, end: int) -> ET.Element | None:
    for child in children[start:end]:
        if child.tag == qn(W_NS, "p") and child.find(f"{qn(W_NS, 'pPr')}/{qn(W_NS, 'sectPr')}") is None:
            if _paragraph_text(child).strip():
                return child
    return None


def _paragraph_with_prefix(children: list[ET.Element], start: int, end: int, prefix: str) -> ET.Element | None:
    for child in children[start:end]:
        if child.tag == qn(W_NS, "p") and _paragraph_text(child).strip().startswith(prefix):
            return child
    return None


def _replace_toc_block(children: list[ET.Element]) -> list[ET.Element]:
    toc_start = _find_paragraph_index(children, "目    录")
    if toc_start is None:
        return children
    section_break = _last_section_paragraph(children, toc_start + 1, len(children))
    title = copy.deepcopy(children[toc_start])
    _replace_paragraph_text(title, "目    录")
    result = children[:toc_start]
    result.append(title)
    result.append(_toc_field_paragraph())
    if section_break is not None:
        blank = copy.deepcopy(section_break)
        _replace_paragraph_text(blank, "")
        result.append(blank)
    return result


def _find_paragraph_index(children: list[ET.Element], text: str) -> int | None:
    for index, child in enumerate(children):
        if child.tag == qn(W_NS, "p") and _paragraph_text(child).strip() == text:
            return index
    return None


def _content_children_from_intermediate(intermediate_body: ET.Element) -> list[ET.Element]:
    children = list(intermediate_body)
    for index, child in enumerate(children):
        if child.tag == qn(W_NS, "p") and _paragraph_style(child) in {"Heading1", "heading 1", "1"}:
            return children[index:]
    return children


def _extract_section_properties(body: ET.Element) -> ET.Element | None:
    for child in list(body):
        if child.tag == qn(W_NS, "sectPr"):
            return copy.deepcopy(child)
    return None


def _extract_body_section_properties(body: ET.Element) -> ET.Element | None:
    children = list(body)
    body_start = _find_template_body_start(children)
    if body_start is None:
        return None
    for child in children[body_start + 1 :]:
        if child.tag == qn(W_NS, "sectPr"):
            return copy.deepcopy(child)
        if child.tag == qn(W_NS, "p"):
            section = child.find(f"{qn(W_NS, 'pPr')}/{qn(W_NS, 'sectPr')}")
            if section is not None:
                return copy.deepcopy(section)
    return None


def _copy_document_relationships(
    template: DocxPackage,
    intermediate: DocxPackage,
    intermediate_document: ET.Element,
) -> dict[str, str]:
    if DOCUMENT_RELS in template.parts:
        template_rels = ET.fromstring(template.parts[DOCUMENT_RELS])
    else:
        template_rels = empty_relationships()
    if DOCUMENT_RELS not in intermediate.parts:
        template.set_xml_part(DOCUMENT_RELS, template_rels)
        return {}

    source_rels = ET.fromstring(intermediate.parts[DOCUMENT_RELS])
    used_relationships = _used_relationship_ids(intermediate_document)
    mapping: dict[str, str] = {}
    next_index = 1
    for relationship in relationship_elements(source_rels):
        target = relationship.attrib.get("Target", "")
        rel_id = relationship.attrib.get("Id", "")
        rel_type = relationship.attrib.get("Type", IMAGE_REL_TYPE)
        if not rel_id or rel_id not in used_relationships:
            continue
        new_target = target
        if target.startswith("media/"):
            source_part = f"word/{target}"
            if source_part not in intermediate.parts:
                continue
            suffix = Path(target).suffix
            while f"word/media/exported_{next_index}{suffix}" in template.parts:
                next_index += 1
            new_target = f"media/exported_{next_index}{suffix}"
            next_index += 1
            template.parts[f"word/{new_target}"] = intermediate.parts[source_part]
        new_id = _next_relationship_id(template_rels)
        attrs = {
            "Id": new_id,
            "Type": rel_type,
            "Target": new_target,
        }
        if "TargetMode" in relationship.attrib:
            attrs["TargetMode"] = relationship.attrib["TargetMode"]
        ET.SubElement(
            template_rels,
            qn(PKG_REL_NS, "Relationship"),
            attrs,
        )
        mapping[rel_id] = new_id

    template.set_xml_part(DOCUMENT_RELS, template_rels)
    return mapping


def _used_relationship_ids(document: ET.Element) -> set[str]:
    relationship_attrs = {qn(R_NS, "embed"), qn(R_NS, "link"), qn(R_NS, "id")}
    used: set[str] = set()
    for element in document.iter():
        for attr in relationship_attrs:
            value = element.attrib.get(attr)
            if value:
                used.add(value)
    return used


def _next_relationship_id(root: ET.Element) -> str:
    ids = []
    for relationship in relationship_elements(root):
        rel_id = relationship.attrib.get("Id", "")
        match = re.fullmatch(r"rId(\d+)", rel_id)
        if match:
            ids.append(int(match.group(1)))
    return f"rId{max(ids, default=0) + 1}"


def _rewrite_relationship_ids(document: ET.Element, mapping: dict[str, str]) -> None:
    if not mapping:
        return
    rel_attrs = {qn(R_NS, "embed"), qn(R_NS, "link"), qn(R_NS, "id")}
    for element in document.iter():
        for attr in rel_attrs:
            value = element.attrib.get(attr)
            if value in mapping:
                element.set(attr, mapping[value])


def _style_tree(element: ET.Element) -> None:
    if element.tag == qn(W_NS, "p"):
        _style_paragraph(element)
    elif element.tag == qn(W_NS, "tbl"):
        _style_table(element)
    for table in element.findall(f".//{qn(W_NS, 'tbl')}"):
        _style_table(table)
    for paragraph in element.findall(f".//{qn(W_NS, 'p')}"):
        _style_paragraph(paragraph)


def _style_paragraph(paragraph: ET.Element) -> None:
    text = _paragraph_text(paragraph)
    current_style = _paragraph_style(paragraph)
    compact_text = text.replace("\xa0", " ").strip()
    if _is_image_paragraph(paragraph):
        _format_image_paragraph(paragraph)
    elif current_style in {"Heading1", "heading 1", "1"} or _is_chapter_title(compact_text):
        _normalize_chapter_heading(paragraph)
        _set_paragraph_style(paragraph, "aff8")
        _set_centered(paragraph)
    elif current_style in {"2", "Heading2", "heading 2"} or re.match(r"^\d+\.\d+\s", compact_text):
        _normalize_numbered_heading(paragraph)
        _set_paragraph_style(paragraph, "2")
        _format_numbered_heading(paragraph)
    elif current_style in {"3", "Heading3", "heading 3"} or re.match(r"^\d+\.\d+\.\d+\s", compact_text):
        _normalize_numbered_heading(paragraph)
        _set_paragraph_style(paragraph, "3")
        _format_numbered_heading(paragraph)
    elif current_style in {"4", "Heading4", "heading 4"}:
        _set_paragraph_style(paragraph, "4")
    elif _is_figure_caption(current_style, compact_text):
        _format_figure_caption_paragraph(paragraph)
    elif _is_table_caption(current_style, compact_text):
        _format_table_caption_paragraph(paragraph)
    elif current_style in {"Bibliography", "references", "Reference"}:
        _set_paragraph_style(paragraph, "a")
    elif current_style not in {"1", "11", "23", "33", "aff8", "2", "3", "4"}:
        _set_paragraph_style(paragraph, "a0")
        _format_body_paragraph(paragraph)


def _style_table(table: ET.Element) -> None:
    tblpr = table.find(qn(W_NS, "tblPr"))
    if tblpr is None:
        tblpr = ET.Element(qn(W_NS, "tblPr"))
        table.insert(0, tblpr)
    borders = tblpr.find(qn(W_NS, "tblBorders"))
    if borders is None:
        borders = ET.Element(qn(W_NS, "tblBorders"))
        tblpr.append(borders)
    for side in ("left", "right"):
        border = borders.find(qn(W_NS, side))
        if border is not None:
            borders.remove(border)
    _set_table_border(borders, "top", "12")
    _set_table_border(borders, "bottom", "12")
    _set_table_border(borders, "insideH", "6")
    _set_table_border(borders, "insideV", "6")


def _format_body_paragraph(paragraph: ET.Element) -> None:
    _set_paragraph_spacing(paragraph, line="440", line_rule="exact")
    first_line = "0" if _paragraph_text(paragraph).strip().startswith(("式中", "其中", "这里")) else "480"
    _set_first_line_indent(paragraph, first_line)
    _set_justification(paragraph, "both")
    _format_direct_runs(paragraph, size="24", east_asia="宋体", ascii_font="Times New Roman")


def _format_figure_caption_paragraph(paragraph: ET.Element) -> None:
    _set_paragraph_style(paragraph, "affa")
    _set_centered(paragraph)
    _set_first_line_indent(paragraph, "0")
    _set_paragraph_spacing(paragraph, after="50", line="440", line_rule="exact")
    _format_direct_runs(paragraph, size="21", east_asia="宋体", ascii_font="Times New Roman")


def _format_table_caption_paragraph(paragraph: ET.Element) -> None:
    _set_paragraph_style(paragraph, "af9")
    _set_centered(paragraph)
    _set_first_line_indent(paragraph, "0")
    _set_paragraph_spacing(paragraph, line="392", line_rule="atLeast")
    _format_direct_runs(paragraph, size="21", east_asia="宋体", ascii_font="Times New Roman")


def _format_direct_runs(
    paragraph: ET.Element,
    *,
    size: str,
    east_asia: str,
    ascii_font: str,
    bold: bool | None = None,
) -> None:
    for run in paragraph.findall(qn(W_NS, "r")):
        rpr = _ensure_rpr(run)
        fonts = rpr.find(qn(W_NS, "rFonts"))
        if fonts is None:
            fonts = ET.Element(qn(W_NS, "rFonts"))
            rpr.insert(0, fonts)
        fonts.set(qn(W_NS, "ascii"), ascii_font)
        fonts.set(qn(W_NS, "hAnsi"), ascii_font)
        fonts.set(qn(W_NS, "eastAsia"), east_asia)
        fonts.set(qn(W_NS, "cs"), ascii_font)
        _set_run_size(rpr, size)
        if bold is not None:
            _set_run_bold(rpr, bold)


def _ensure_rpr(run: ET.Element) -> ET.Element:
    rpr = run.find(qn(W_NS, "rPr"))
    if rpr is None:
        rpr = ET.Element(qn(W_NS, "rPr"))
        run.insert(0, rpr)
    return rpr


def _set_run_size(rpr: ET.Element, size: str) -> None:
    sz = rpr.find(qn(W_NS, "sz"))
    if sz is None:
        sz = ET.SubElement(rpr, qn(W_NS, "sz"))
    sz.set(qn(W_NS, "val"), size)
    sz_cs = rpr.find(qn(W_NS, "szCs"))
    if sz_cs is None:
        sz_cs = ET.SubElement(rpr, qn(W_NS, "szCs"))
    sz_cs.set(qn(W_NS, "val"), size)


def _set_run_bold(rpr: ET.Element, bold: bool) -> None:
    value = "1" if bold else "0"
    for tag in ("b", "bCs"):
        element = rpr.find(qn(W_NS, tag))
        if element is None:
            element = ET.SubElement(rpr, qn(W_NS, tag))
        element.set(qn(W_NS, "val"), value)


def _set_table_border(borders: ET.Element, name: str, size: str) -> None:
    border = borders.find(qn(W_NS, name))
    if border is None:
        border = ET.SubElement(borders, qn(W_NS, name))
    border.set(qn(W_NS, "val"), "single")
    border.set(qn(W_NS, "sz"), size)
    border.set(qn(W_NS, "space"), "0")
    border.set(qn(W_NS, "color"), "auto")


def _is_image_paragraph(paragraph: ET.Element) -> bool:
    return paragraph.find(f".//{qn(W_NS, 'drawing')}") is not None


def _format_image_paragraph(paragraph: ET.Element) -> None:
    _set_paragraph_style(paragraph, "aff6")
    _set_centered(paragraph)
    _set_paragraph_spacing(paragraph, before="50", after=None, line="240", line_rule="auto")
    _set_first_line_indent(paragraph, "0")


def _paragraph_text(paragraph: ET.Element) -> str:
    return "".join(text.text or "" for text in paragraph.findall(f".//{qn(W_NS, 't')}"))


def _paragraph_style(paragraph: ET.Element) -> str | None:
    style = paragraph.find(f"{qn(W_NS, 'pPr')}/{qn(W_NS, 'pStyle')}")
    return style.attrib.get(qn(W_NS, "val")) if style is not None else None


def _ensure_ppr(paragraph: ET.Element) -> ET.Element:
    ppr = paragraph.find(qn(W_NS, "pPr"))
    if ppr is None:
        ppr = ET.Element(qn(W_NS, "pPr"))
        paragraph.insert(0, ppr)
    return ppr


def _set_paragraph_style(paragraph: ET.Element, style_id: str) -> None:
    ppr = _ensure_ppr(paragraph)
    style = ppr.find(qn(W_NS, "pStyle"))
    if style is None:
        style = ET.Element(qn(W_NS, "pStyle"))
        ppr.insert(0, style)
    style.set(qn(W_NS, "val"), style_id)


def _set_centered(paragraph: ET.Element) -> None:
    _set_justification(paragraph, "center")


def _set_justification(paragraph: ET.Element, value: str) -> None:
    ppr = _ensure_ppr(paragraph)
    jc = ppr.find(qn(W_NS, "jc"))
    if jc is None:
        jc = ET.SubElement(ppr, qn(W_NS, "jc"))
    jc.set(qn(W_NS, "val"), value)


def _set_paragraph_spacing(
    paragraph: ET.Element,
    *,
    before: str | None = None,
    after: str | None = None,
    line: str | None = None,
    line_rule: str | None = None,
) -> None:
    ppr = _ensure_ppr(paragraph)
    spacing = ppr.find(qn(W_NS, "spacing"))
    if spacing is None:
        spacing = ET.SubElement(ppr, qn(W_NS, "spacing"))
    values = {
        "before": before,
        "after": after,
        "line": line,
        "lineRule": line_rule,
    }
    for name, value in values.items():
        attr = qn(W_NS, name)
        if value is None:
            spacing.attrib.pop(attr, None)
        else:
            spacing.set(attr, value)


def _set_first_line_indent(paragraph: ET.Element, value: str) -> None:
    ppr = _ensure_ppr(paragraph)
    indent = ppr.find(qn(W_NS, "ind"))
    if indent is None:
        indent = ET.SubElement(ppr, qn(W_NS, "ind"))
    if value == "0":
        for attr_name in ("left", "leftChars", "right", "rightChars", "hanging", "hangingChars"):
            indent.attrib.pop(qn(W_NS, attr_name), None)
    indent.set(qn(W_NS, "firstLine"), value)
    indent.set(qn(W_NS, "firstLineChars"), "0" if value == "0" else "200")


def _is_chapter_title(text: str) -> bool:
    return bool(re.match(r"^第\s*\d+\s*章", text)) or text in {"绪论", "结  论", "结论", "参考文献", "致  谢", "致谢"}


def _normalize_chapter_heading(paragraph: ET.Element) -> None:
    text = _paragraph_text(paragraph).strip()
    match = re.match(r"^(\d+)([^.\d].*)$", text)
    if match:
        _replace_paragraph_text(paragraph, f"第{match.group(1)}章 {match.group(2).strip()}")


def _normalize_numbered_heading(paragraph: ET.Element) -> None:
    text = _paragraph_text(paragraph).strip()
    match = re.match(r"^(\d+(?:\.\d+)+)(\S.*)$", text)
    if match:
        _replace_paragraph_text(paragraph, f"{match.group(1)} {match.group(2).strip()}")


def _format_numbered_heading(paragraph: ET.Element) -> None:
    _set_first_line_indent(paragraph, "0")
    _set_paragraph_spacing(paragraph, before="260", after="260", line="416", line_rule="atLeast")
    _format_direct_runs(paragraph, size="32", east_asia="黑体", ascii_font="Times New Roman", bold=False)


def _is_figure_caption(style: str | None, text: str) -> bool:
    return bool(text) and style in {"ImageCaption", "CaptionedFigure", "affa", "aa"}


def _is_table_caption(style: str | None, text: str) -> bool:
    return bool(text) and style in {"TableCaption", "af9", "a2"}


def _insert_toc_placeholder(body: ET.Element) -> None:
    title = ET.Element(qn(W_NS, "p"))
    _set_paragraph_style(title, "1")
    _set_centered(title)
    run = ET.SubElement(title, qn(W_NS, "r"))
    text = ET.SubElement(run, qn(W_NS, "t"))
    text.text = "目    录"
    body.append(title)
    body.append(_toc_field_paragraph())

def _toc_field_paragraph() -> ET.Element:
    field = ET.Element(qn(W_NS, "p"))
    begin_run = ET.SubElement(field, qn(W_NS, "r"))
    ET.SubElement(begin_run, qn(W_NS, "fldChar"), {qn(W_NS, "fldCharType"): "begin"})
    instr_run = ET.SubElement(field, qn(W_NS, "r"))
    instr = ET.SubElement(instr_run, qn(W_NS, "instrText"))
    instr.text = 'TOC \\o "1-3" \\u'
    sep_run = ET.SubElement(field, qn(W_NS, "r"))
    ET.SubElement(sep_run, qn(W_NS, "fldChar"), {qn(W_NS, "fldCharType"): "separate"})
    text_run = ET.SubElement(field, qn(W_NS, "r"))
    placeholder = ET.SubElement(text_run, qn(W_NS, "t"))
    placeholder.text = "右键更新域以生成目录。"
    end_run = ET.SubElement(field, qn(W_NS, "r"))
    ET.SubElement(end_run, qn(W_NS, "fldChar"), {qn(W_NS, "fldCharType"): "end"})
    return field


def _paragraph_from_template(template: ET.Element, text: str) -> ET.Element:
    paragraph = copy.deepcopy(template)
    _replace_paragraph_text(paragraph, text)
    return paragraph


def _replace_paragraph_text(paragraph: ET.Element, text: str) -> None:
    ppr = paragraph.find(qn(W_NS, "pPr"))
    first_rpr = paragraph.find(f".//{qn(W_NS, 'rPr')}")
    for child in list(paragraph):
        if child is ppr:
            continue
        paragraph.remove(child)
    run = ET.SubElement(paragraph, qn(W_NS, "r"))
    if first_rpr is not None:
        run.append(copy.deepcopy(first_rpr))
    parts = text.split("\n")
    for index, part in enumerate(parts):
        if index:
            ET.SubElement(run, qn(W_NS, "br"))
        text_node = ET.SubElement(run, qn(W_NS, "t"))
        text_node.text = part


@dataclass(frozen=True)
class ReferenceRecord:
    anchor: str
    labels: tuple[str, ...]
    paragraph: ET.Element


def _link_author_year_citations(body: ET.Element) -> None:
    paragraphs = [child for child in list(body) if child.tag == qn(W_NS, "p")]
    records = _reference_records(paragraphs)
    if not records:
        return
    next_id = _next_bookmark_id(body)
    reference_paragraphs = {id(record.paragraph) for record in records}
    for record in records:
        _add_bookmark(record.paragraph, record.anchor, next_id)
        next_id += 1

    label_to_anchor: dict[str, str] = {}
    for record in records:
        for label in record.labels:
            label_to_anchor.setdefault(label, record.anchor)
    labels = sorted(label_to_anchor, key=len, reverse=True)
    for paragraph in paragraphs:
        if id(paragraph) in reference_paragraphs:
            continue
        if paragraph.find(f".//{qn(W_NS, 'drawing')}") is not None:
            continue
        text = _paragraph_text(paragraph)
        matches = _citation_matches(text, labels, label_to_anchor)
        if not matches:
            continue
        if paragraph.find(f".//{qn(M_NS, 'oMath')}") is not None:
            _link_text_runs_in_paragraph(paragraph, labels, label_to_anchor)
        else:
            _replace_paragraph_with_hyperlinks(paragraph, text, matches)


def _reference_records(paragraphs: list[ET.Element]) -> list[ReferenceRecord]:
    records: list[ReferenceRecord] = []
    previous_authors = ""
    for index, paragraph in enumerate(paragraphs):
        style = _paragraph_style(paragraph)
        if style not in {"a", "Bibliography", "Reference", "references"}:
            continue
        text = _paragraph_text(paragraph).strip()
        match = re.search(r"\b(?:18|19|20)\d{2}[a-z]?\b", text)
        if not match:
            continue
        authors = text[: match.start()].strip(" .")
        if authors.startswith("———") or authors.startswith("---"):
            authors = previous_authors
        elif authors:
            previous_authors = authors
        labels = _citation_labels(authors, match.group(0))
        if labels:
            records.append(ReferenceRecord(anchor=f"bibref_{index + 1}", labels=tuple(labels), paragraph=paragraph))
    return records


def _citation_labels(authors: str, year: str) -> list[str]:
    authors = " ".join(authors.replace("，", ",").split())
    if not authors:
        return []
    labels = {f"{authors} {year}"}
    surnames = _author_surnames(authors)
    if surnames:
        labels.add(f"{surnames[0]} {year}")
        labels.add(f"{surnames[0]} et al. {year}")
    if len(surnames) == 2:
        labels.add(f"{surnames[0]} and {surnames[1]} {year}")
    elif len(surnames) >= 3:
        labels.add(f"{surnames[0]}, {surnames[1]}, and {surnames[2]} {year}")
        labels.add(f"{surnames[0]}, {surnames[1]}, and al. {year}")
    return sorted(labels, key=len, reverse=True)


def _author_surnames(authors: str) -> list[str]:
    if re.search(r"[\u4e00-\u9fff]", authors):
        first = re.split(r"[,、;；]\s*", authors, maxsplit=1)[0].strip()
        return [first] if first else []
    first = authors.split(",", 1)[0].strip()
    if not first:
        return []
    parts = re.split(r",\s+and\s+|\s+and\s+", authors)
    last_author = parts[-1].strip(" .") if len(parts) > 1 else ""
    last = _last_word(last_author)
    middle_surnames = re.findall(r",\s+([A-Z][A-Za-z'’-]+)(?=,|\s+and\s+)", authors)
    surnames = [first]
    for surname in middle_surnames:
        if surname not in surnames:
            surnames.append(surname)
    if last and last not in surnames and last != first:
        surnames.append(last)
    return surnames


def _last_word(text: str) -> str:
    words = re.findall(r"[A-Z][A-Za-z'’-]+", text)
    return words[-1] if words else ""


def _next_bookmark_id(root: ET.Element) -> int:
    ids = []
    for bookmark in root.iter(qn(W_NS, "bookmarkStart")):
        value = bookmark.attrib.get(qn(W_NS, "id"))
        if value and value.isdigit():
            ids.append(int(value))
    return max(ids, default=0) + 1


def _add_bookmark(paragraph: ET.Element, name: str, bookmark_id: int) -> None:
    start = ET.Element(qn(W_NS, "bookmarkStart"), {qn(W_NS, "id"): str(bookmark_id), qn(W_NS, "name"): name})
    end = ET.Element(qn(W_NS, "bookmarkEnd"), {qn(W_NS, "id"): str(bookmark_id)})
    insert_at = 1 if paragraph.find(qn(W_NS, "pPr")) is not None else 0
    paragraph.insert(insert_at, start)
    paragraph.append(end)


def _citation_matches(text: str, labels: list[str], label_to_anchor: dict[str, str]) -> list[tuple[int, int, str]]:
    occupied: list[tuple[int, int, str]] = []
    for label in labels:
        pattern = re.escape(label)
        for match in re.finditer(pattern, text):
            start, end = match.span()
            if any(not (end <= old_start or start >= old_end) for old_start, old_end, _ in occupied):
                continue
            occupied.append((start, end, label_to_anchor[label]))
    return sorted(occupied)


def _replace_paragraph_with_hyperlinks(paragraph: ET.Element, text: str, matches: list[tuple[int, int, str]]) -> None:
    ppr = paragraph.find(qn(W_NS, "pPr"))
    first_rpr = paragraph.find(f".//{qn(W_NS, 'rPr')}")
    for child in list(paragraph):
        if child is ppr:
            continue
        paragraph.remove(child)
    cursor = 0
    for start, end, anchor in matches:
        if start > cursor:
            _append_text_run(paragraph, text[cursor:start], first_rpr)
        hyperlink = ET.SubElement(paragraph, qn(W_NS, "hyperlink"), {qn(W_NS, "anchor"): anchor})
        _append_text_run(hyperlink, text[start:end], first_rpr)
        cursor = end
    if cursor < len(text):
        _append_text_run(paragraph, text[cursor:], first_rpr)


def _link_text_runs_in_paragraph(paragraph: ET.Element, labels: list[str], label_to_anchor: dict[str, str]) -> None:
    rebuilt: list[ET.Element] = []
    changed = False
    for child in list(paragraph):
        if child.tag != qn(W_NS, "r"):
            rebuilt.append(child)
            continue
        text_node = child.find(qn(W_NS, "t"))
        if text_node is None or not text_node.text:
            rebuilt.append(child)
            continue
        matches = _citation_matches(text_node.text, labels, label_to_anchor)
        if not matches:
            rebuilt.append(child)
            continue
        rpr = child.find(qn(W_NS, "rPr"))
        cursor = 0
        for start, end, anchor in matches:
            if start > cursor:
                rebuilt.append(_text_run_element(text_node.text[cursor:start], rpr))
            hyperlink = ET.Element(qn(W_NS, "hyperlink"), {qn(W_NS, "anchor"): anchor})
            hyperlink.append(_text_run_element(text_node.text[start:end], rpr))
            rebuilt.append(hyperlink)
            cursor = end
        if cursor < len(text_node.text):
            rebuilt.append(_text_run_element(text_node.text[cursor:], rpr))
        changed = True
    if not changed:
        return
    for child in list(paragraph):
        paragraph.remove(child)
    for child in rebuilt:
        paragraph.append(child)


def _text_run_element(text: str, rpr: ET.Element | None = None) -> ET.Element:
    run = ET.Element(qn(W_NS, "r"))
    if rpr is not None:
        run.append(copy.deepcopy(rpr))
    text_node = ET.SubElement(run, qn(W_NS, "t"))
    if text[:1].isspace() or text[-1:].isspace():
        text_node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_node.text = text
    return run


def _append_text_run(parent: ET.Element, text: str, rpr: ET.Element | None = None) -> None:
    if not text:
        return
    run = ET.SubElement(parent, qn(W_NS, "r"))
    if rpr is not None:
        run.append(copy.deepcopy(rpr))
    text_node = ET.SubElement(run, qn(W_NS, "t"))
    if text[:1].isspace() or text[-1:].isspace():
        text_node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_node.text = text


def _ensure_media_content_types(package: DocxPackage) -> None:
    if CONTENT_TYPES not in package.parts:
        return
    root = ET.fromstring(package.parts[CONTENT_TYPES])
    existing = {item.attrib.get("Extension", "").lower() for item in root.findall(qn(CONTENT_TYPES_NS, "Default"))}
    media_types = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "wmf": "image/x-wmf",
    }
    needed = {
        Path(name).suffix.lower().lstrip(".")
        for name in package.parts
        if name.startswith("word/media/")
    }
    for extension in sorted(needed):
        if extension in media_types and extension not in existing:
            ET.SubElement(
                root,
                qn(CONTENT_TYPES_NS, "Default"),
                {"Extension": extension, "ContentType": media_types[extension]},
            )
    package.set_xml_part(CONTENT_TYPES, root)


def _merge_numbering_definitions(
    template: DocxPackage,
    intermediate: DocxPackage,
    intermediate_document: ET.Element,
) -> None:
    numbering_part = "word/numbering.xml"
    if numbering_part not in template.parts or numbering_part not in intermediate.parts:
        return
    used_num_ids = {
        element.attrib.get(qn(W_NS, "val"))
        for element in intermediate_document.iter(qn(W_NS, "numId"))
        if element.attrib.get(qn(W_NS, "val"))
    }
    if not used_num_ids:
        return

    template_numbering = template.xml_part(numbering_part)
    source_numbering = intermediate.xml_part(numbering_part)
    template_num_ids = {
        element.attrib.get(qn(W_NS, "numId"))
        for element in template_numbering.iter(qn(W_NS, "num"))
    }
    template_abstract_ids = {
        element.attrib.get(qn(W_NS, "abstractNumId"))
        for element in template_numbering.iter(qn(W_NS, "abstractNum"))
    }
    source_nums = {
        element.attrib.get(qn(W_NS, "numId")): element
        for element in source_numbering.iter(qn(W_NS, "num"))
    }
    source_abstracts = {
        element.attrib.get(qn(W_NS, "abstractNumId")): element
        for element in source_numbering.iter(qn(W_NS, "abstractNum"))
    }

    changed = False
    for num_id in sorted(used_num_ids, key=_numeric_sort_key):
        if num_id in template_num_ids:
            continue
        source_num = source_nums.get(num_id)
        if source_num is None:
            continue
        abstract_ref = source_num.find(qn(W_NS, "abstractNumId"))
        abstract_id = abstract_ref.attrib.get(qn(W_NS, "val")) if abstract_ref is not None else None
        if abstract_id and abstract_id not in template_abstract_ids:
            source_abstract = source_abstracts.get(abstract_id)
            if source_abstract is not None:
                template_numbering.append(copy.deepcopy(source_abstract))
                template_abstract_ids.add(abstract_id)
                changed = True
        template_numbering.append(copy.deepcopy(source_num))
        template_num_ids.add(num_id)
        changed = True

    if changed:
        template.set_xml_part(numbering_part, template_numbering)


def _numeric_sort_key(value: str | None) -> tuple[int, str]:
    if value and value.isdigit():
        return int(value), value
    return 10**9, value or ""
