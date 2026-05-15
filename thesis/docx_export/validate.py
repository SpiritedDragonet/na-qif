from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

from .openxml import M_NS, PKG_REL_NS, R_NS, W_NS, qn


LATEX_RESIDUE = (r"\rm", r"\Bigl", r"\Bigr", r"\allowbreak", r"\includegraphics")
UNSUPPORTED_MEDIA = {".pdf", ".eps"}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        lines: list[str] = []
        lines.extend(f"ERROR: {item}" for item in self.errors)
        lines.extend(f"WARNING: {item}" for item in self.warnings)
        if not lines:
            return "DOCX validation passed."
        return "\n".join(lines)


def _decode_xml(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def validate_docx(path: str | Path) -> ValidationReport:
    report = ValidationReport()
    docx_path = Path(path)
    if not docx_path.exists():
        report.errors.append(f"Missing DOCX: {docx_path}")
        return report

    try:
        with zipfile.ZipFile(docx_path) as archive:
            names = set(archive.namelist())
            if "word/document.xml" not in names:
                report.errors.append("Missing word/document.xml")
                return report

            xml_texts = {
                name: _decode_xml(archive.read(name))
                for name in names
                if name.endswith(".xml") and not name.startswith("docProps/")
            }
            combined_xml = "\n".join(xml_texts.values())
            for residue in LATEX_RESIDUE:
                if residue in combined_xml:
                    report.errors.append(f"LaTeX residue found: {residue}")
            _validate_equation_reference_residue(combined_xml, report)

            media_parts = [name for name in names if name.startswith("word/media/")]
            for media in media_parts:
                if Path(media).suffix.lower() in UNSUPPORTED_MEDIA:
                    report.errors.append(f"Unsupported media in DOCX: {media}")

            _validate_media_relationships(archive, names, report)
            _validate_used_relationships(archive, names, report)
            _validate_numbering(archive, names, report)
            _validate_author_year_citation_links(archive, names, report)
            _validate_mc_ignorable_prefixes(archive, names, report)

            document_xml = xml_texts.get("word/document.xml", "")
            if "TOC " not in document_xml:
                report.warnings.append("Missing Word TOC field")
            if qn(M_NS, "oMath") not in document_xml and "<m:oMath" not in document_xml:
                report.warnings.append("No editable OMML formulas found")
            if media_parts and ("图" not in document_xml and "Figure" not in document_xml):
                report.warnings.append("Media exists but no figure captions were detected")
    except zipfile.BadZipFile:
        report.errors.append(f"Invalid DOCX ZIP package: {docx_path}")

    return report


def _validate_media_relationships(archive: zipfile.ZipFile, names: set[str], report: ValidationReport) -> None:
    rels_name = "word/_rels/document.xml.rels"
    if rels_name not in names:
        report.warnings.append("Missing document relationships part")
        return
    try:
        root = ET.fromstring(archive.read(rels_name))
    except ET.ParseError:
        report.errors.append("Invalid document relationships XML")
        return
    for relationship in root.findall(qn(PKG_REL_NS, "Relationship")):
        target = relationship.attrib.get("Target", "")
        mode = relationship.attrib.get("TargetMode", "")
        if mode == "External" or not target.startswith("media/"):
            continue
        part_name = f"word/{target}"
        if part_name not in names:
            report.errors.append(f"Missing related media part: {part_name}")


def _validate_used_relationships(archive: zipfile.ZipFile, names: set[str], report: ValidationReport) -> None:
    rels_name = "word/_rels/document.xml.rels"
    if rels_name not in names or "word/document.xml" not in names:
        return
    try:
        document = ET.fromstring(archive.read("word/document.xml"))
        rels = ET.fromstring(archive.read(rels_name))
    except ET.ParseError:
        return
    rel_ids = {
        relationship.attrib.get("Id")
        for relationship in rels.findall(qn(PKG_REL_NS, "Relationship"))
    }
    relationship_attrs = {qn(R_NS, "id"), qn(R_NS, "embed"), qn(R_NS, "link")}
    missing: set[str] = set()
    for element in document.iter():
        for attr in relationship_attrs:
            value = element.attrib.get(attr)
            if value and value not in rel_ids:
                missing.add(value)
    for rel_id in sorted(missing, key=_relationship_sort_key):
        report.errors.append(f"Missing document relationship for {rel_id}")


def _relationship_sort_key(rel_id: str) -> tuple[int, str]:
    if rel_id.startswith("rId") and rel_id[3:].isdigit():
        return int(rel_id[3:]), rel_id
    return 10**9, rel_id


def _validate_numbering(archive: zipfile.ZipFile, names: set[str], report: ValidationReport) -> None:
    if "word/document.xml" not in names or "word/numbering.xml" not in names:
        return
    try:
        document = ET.fromstring(archive.read("word/document.xml"))
        numbering = ET.fromstring(archive.read("word/numbering.xml"))
    except ET.ParseError:
        return
    word_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    used = {
        element.attrib.get(qn(word_ns, "val"))
        for element in document.iter(qn(word_ns, "numId"))
        if element.attrib.get(qn(word_ns, "val"))
    }
    defined = {
        element.attrib.get(qn(word_ns, "numId"))
        for element in numbering.iter(qn(word_ns, "num"))
    }
    for num_id in sorted(used - defined, key=_relationship_sort_key):
        report.errors.append(f"Missing numbering definition for numId {num_id}")


def _validate_equation_reference_residue(xml_text: str, report: ValidationReport) -> None:
    unresolved = set(re.findall(r"\[(eq:[A-Za-z0-9_.:-]+)\]", xml_text))
    unresolved.update(re.findall(r"\\eqref\{(eq:[A-Za-z0-9_.:-]+)\}", xml_text))
    for label in sorted(unresolved):
        report.errors.append(f"Unresolved equation reference found: [{label}]")


def _validate_author_year_citation_links(archive: zipfile.ZipFile, names: set[str], report: ValidationReport) -> None:
    if "word/document.xml" not in names:
        return
    try:
        document = ET.fromstring(archive.read("word/document.xml"))
    except ET.ParseError:
        return
    paragraphs = document.findall(f".//{qn(W_NS, 'body')}/{qn(W_NS, 'p')}")
    reference_paragraphs: set[int] = set()
    label_to_anchor: dict[str, str] = {}
    previous_authors = ""
    for paragraph in paragraphs:
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
        bookmark = paragraph.find(qn(W_NS, "bookmarkStart"))
        anchor = bookmark.attrib.get(qn(W_NS, "name")) if bookmark is not None else ""
        for label in _citation_labels(authors, match.group(0)):
            label_to_anchor.setdefault(label, anchor)
        reference_paragraphs.add(id(paragraph))
    if not label_to_anchor:
        return
    labels = sorted(label_to_anchor, key=len, reverse=True)
    for paragraph in paragraphs:
        if id(paragraph) in reference_paragraphs:
            continue
        text = _paragraph_text(paragraph)
        if not text:
            continue
        linked = {
            label
            for hyperlink in paragraph.findall(f".//{qn(W_NS, 'hyperlink')}")
            for label in labels
            if label in _paragraph_text(hyperlink)
            and hyperlink.attrib.get(qn(W_NS, "anchor")) == label_to_anchor[label]
        }
        for label in labels:
            if label in text and label not in linked:
                report.errors.append(f"Citation is not linked to bibliography: {label}")


def _paragraph_text(paragraph: ET.Element) -> str:
    return "".join(text.text or "" for text in paragraph.findall(f".//{qn(W_NS, 't')}"))


def _paragraph_style(paragraph: ET.Element) -> str | None:
    style = paragraph.find(f"{qn(W_NS, 'pPr')}/{qn(W_NS, 'pStyle')}")
    return style.attrib.get(qn(W_NS, "val")) if style is not None else None


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


def _validate_mc_ignorable_prefixes(archive: zipfile.ZipFile, names: set[str], report: ValidationReport) -> None:
    for name in names:
        if not name.endswith(".xml"):
            continue
        data = archive.read(name).decode("utf-8", errors="replace")
        root_start = _root_start(data)
        if "Ignorable=" not in root_start:
            continue
        declarations = set(__import__("re").findall(r"\sxmlns:([A-Za-z0-9_]+)=", root_start))
        values = __import__("re").findall(r"\s[A-Za-z0-9_]+:Ignorable=['\"]([^'\"]*)['\"]", root_start)
        for value in values:
            for prefix in value.split():
                if prefix not in declarations:
                    report.errors.append(f"Undeclared mc:Ignorable prefix {prefix} in {name}")


def _root_start(data: str) -> str:
    if data.startswith("<?xml"):
        end = data.find("?>")
        if end >= 0:
            data = data[end + 2 :]
    data = data.lstrip()
    end = data.find(">")
    return data[: end + 1] if end >= 0 else data
