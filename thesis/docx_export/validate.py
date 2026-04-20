from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

from .openxml import M_NS, PKG_REL_NS, qn


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

