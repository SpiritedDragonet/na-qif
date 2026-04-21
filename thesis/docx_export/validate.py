from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

from .openxml import M_NS, PKG_REL_NS, R_NS, qn


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

            media_parts = [name for name in names if name.startswith("word/media/")]
            for media in media_parts:
                if Path(media).suffix.lower() in UNSUPPORTED_MEDIA:
                    report.errors.append(f"Unsupported media in DOCX: {media}")

            _validate_media_relationships(archive, names, report)
            _validate_used_relationships(archive, names, report)
            _validate_numbering(archive, names, report)
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
