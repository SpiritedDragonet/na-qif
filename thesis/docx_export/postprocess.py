from __future__ import annotations

import copy
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from .openxml import (
    CONTENT_TYPES_NS,
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
IMAGE_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"


def merge_with_template(template_docx: str | Path, intermediate_docx: str | Path, output_docx: str | Path) -> None:
    template = DocxPackage.read(template_docx)
    intermediate = DocxPackage.read(intermediate_docx)

    template_document = template.xml_part(DOCUMENT)
    intermediate_document = intermediate.xml_part(DOCUMENT)
    template_body = template_document.find(qn(W_NS, "body"))
    intermediate_body = intermediate_document.find(qn(W_NS, "body"))
    if template_body is None or intermediate_body is None:
        raise ValueError("Both template and intermediate DOCX must contain word/document.xml bodies")

    relationship_map = _copy_media_relationships(template, intermediate)
    _rewrite_relationship_ids(intermediate_document, relationship_map)

    template_section = _extract_section_properties(template_body)
    template_body.clear()
    _insert_toc_placeholder(template_body)
    for child in list(intermediate_body):
        if child.tag == qn(W_NS, "sectPr"):
            continue
        copied = copy.deepcopy(child)
        _style_tree(copied)
        template_body.append(copied)
    if template_section is not None:
        template_body.append(template_section)

    template.set_xml_part(DOCUMENT, template_document)
    _ensure_media_content_types(template)
    template.write(output_docx)


def _extract_section_properties(body: ET.Element) -> ET.Element | None:
    for child in list(body):
        if child.tag == qn(W_NS, "sectPr"):
            return copy.deepcopy(child)
    return None


def _copy_media_relationships(template: DocxPackage, intermediate: DocxPackage) -> dict[str, str]:
    if DOCUMENT_RELS in template.parts:
        template_rels = ET.fromstring(template.parts[DOCUMENT_RELS])
    else:
        template_rels = empty_relationships()
    if DOCUMENT_RELS not in intermediate.parts:
        template.set_xml_part(DOCUMENT_RELS, template_rels)
        return {}

    source_rels = ET.fromstring(intermediate.parts[DOCUMENT_RELS])
    mapping: dict[str, str] = {}
    next_index = 1
    for relationship in relationship_elements(source_rels):
        target = relationship.attrib.get("Target", "")
        rel_id = relationship.attrib.get("Id", "")
        rel_type = relationship.attrib.get("Type", IMAGE_REL_TYPE)
        if not rel_id or not target.startswith("media/"):
            continue
        source_part = f"word/{target}"
        if source_part not in intermediate.parts:
            continue
        suffix = Path(target).suffix
        while f"word/media/exported_{next_index}{suffix}" in template.parts:
            next_index += 1
        new_target = f"media/exported_{next_index}{suffix}"
        next_index += 1
        new_id = _next_relationship_id(template_rels)
        template.parts[f"word/{new_target}"] = intermediate.parts[source_part]
        ET.SubElement(
            template_rels,
            qn(PKG_REL_NS, "Relationship"),
            {"Id": new_id, "Type": rel_type, "Target": new_target},
        )
        mapping[rel_id] = new_id

    template.set_xml_part(DOCUMENT_RELS, template_rels)
    return mapping


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
    for paragraph in element.findall(f".//{qn(W_NS, 'p')}"):
        _style_paragraph(paragraph)


def _style_paragraph(paragraph: ET.Element) -> None:
    text = _paragraph_text(paragraph)
    current_style = _paragraph_style(paragraph)
    compact_text = text.replace("\xa0", " ").strip()
    if _is_chapter_title(compact_text):
        _set_paragraph_style(paragraph, "aff8")
        _set_centered(paragraph)
    elif current_style in {"2", "Heading2", "heading 2"} or re.match(r"^\d+\.\d+\s", compact_text):
        _set_paragraph_style(paragraph, "2")
    elif current_style in {"3", "Heading3", "heading 3"} or re.match(r"^\d+\.\d+\.\d+\s", compact_text):
        _set_paragraph_style(paragraph, "3")
    elif current_style in {"4", "Heading4", "heading 4"}:
        _set_paragraph_style(paragraph, "4")
    elif _is_figure_caption(current_style, compact_text):
        _set_paragraph_style(paragraph, "affa")
        _set_centered(paragraph)
    elif _is_table_caption(current_style, compact_text):
        _set_paragraph_style(paragraph, "af9")
        _set_centered(paragraph)
    elif current_style in {"Bibliography", "references", "Reference"}:
        _set_paragraph_style(paragraph, "a")
    elif current_style not in {"1", "11", "23", "33", "aff8", "2", "3", "4"}:
        _set_paragraph_style(paragraph, "a0")


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
    ppr = _ensure_ppr(paragraph)
    jc = ppr.find(qn(W_NS, "jc"))
    if jc is None:
        jc = ET.SubElement(ppr, qn(W_NS, "jc"))
    jc.set(qn(W_NS, "val"), "center")


def _is_chapter_title(text: str) -> bool:
    return bool(re.match(r"^第\s*\d+\s*章", text)) or text in {"绪论", "结  论", "结论", "参考文献", "致  谢", "致谢"}


def _is_figure_caption(style: str | None, text: str) -> bool:
    return style in {"ImageCaption", "CaptionedFigure"} or bool(re.match(r"^图\s*\d", text))


def _is_table_caption(style: str | None, text: str) -> bool:
    return style == "TableCaption" or bool(re.match(r"^表\s*\d", text))


def _insert_toc_placeholder(body: ET.Element) -> None:
    title = ET.Element(qn(W_NS, "p"))
    _set_paragraph_style(title, "1")
    _set_centered(title)
    run = ET.SubElement(title, qn(W_NS, "r"))
    text = ET.SubElement(run, qn(W_NS, "t"))
    text.text = "目    录"
    body.append(title)

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
    body.append(field)


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

