from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
W14_NS = "http://schemas.microsoft.com/office/word/2010/wordml"
W15_NS = "http://schemas.microsoft.com/office/word/2012/wordml"
W16SE_NS = "http://schemas.microsoft.com/office/word/2015/wordml/symex"
WP14_NS = "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

ET.register_namespace("w", W_NS)
ET.register_namespace("r", R_NS)
ET.register_namespace("m", M_NS)
ET.register_namespace("mc", MC_NS)
ET.register_namespace("w14", W14_NS)
ET.register_namespace("w15", W15_NS)
ET.register_namespace("w16se", W16SE_NS)
ET.register_namespace("wp14", WP14_NS)
ET.register_namespace("wp", WP_NS)
ET.register_namespace("a", A_NS)
ET.register_namespace("pic", PIC_NS)
ET.register_namespace("", PKG_REL_NS)


def qn(namespace: str, tag: str) -> str:
    return f"{{{namespace}}}{tag}"


@dataclass
class DocxPackage:
    parts: dict[str, bytes]

    @classmethod
    def read(cls, path: str | Path) -> "DocxPackage":
        with zipfile.ZipFile(path) as archive:
            return cls({name: archive.read(name) for name in archive.namelist()})

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(self.parts):
                archive.writestr(name, self.parts[name])

    def xml_part(self, name: str) -> ET.Element:
        return ET.fromstring(self.parts[name])

    def set_xml_part(self, name: str, element: ET.Element) -> None:
        xml = ET.tostring(element, encoding="utf-8", xml_declaration=True)
        self.parts[name] = repair_mc_ignorable(xml)

    def next_relationship_id(self, rels_xml: bytes) -> str:
        try:
            root = ET.fromstring(rels_xml)
            ids = [
                int(match.group(1))
                for relationship in root
                if (match := re.fullmatch(r"rId(\d+)", relationship.attrib.get("Id", "")))
            ]
        except ET.ParseError:
            ids = [int(match.group(1)) for match in re.finditer(rb'Id="rId(\d+)"', rels_xml)]
        return f"rId{max(ids, default=0) + 1}"


def empty_relationships() -> ET.Element:
    return ET.Element(qn(PKG_REL_NS, "Relationships"))


def relationship_elements(root: ET.Element) -> list[ET.Element]:
    return list(root.findall(qn(PKG_REL_NS, "Relationship")))


def repair_mc_ignorable(xml: bytes) -> bytes:
    text = xml.decode("utf-8")
    root_start_begin = 0
    if text.startswith("<?xml"):
        declaration_end = text.find("?>")
        if declaration_end >= 0:
            root_start_begin = declaration_end + 2
            while root_start_begin < len(text) and text[root_start_begin].isspace():
                root_start_begin += 1
    root_start_end = text.find(">", root_start_begin)
    if root_start_end < 0 or "Ignorable=" not in text[:root_start_end]:
        return xml
    prefix_text = text[:root_start_begin]
    root_start = text[root_start_begin : root_start_end + 1]
    declarations = set(re.findall(r"\sxmlns:([A-Za-z0-9_]+)=", root_start))

    def replace(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        quote = match.group("quote")
        values = match.group("value").split()
        kept = [value for value in values if value in declarations]
        if not kept:
            return ""
        return f" {prefix}:Ignorable={quote}{' '.join(kept)}{quote}"

    repaired_root = re.sub(
        r"\s(?P<prefix>[A-Za-z0-9_]+):Ignorable=(?P<quote>['\"])(?P<value>[^'\"]*)(?P=quote)",
        replace,
        root_start,
    )
    return (prefix_text + repaired_root + text[root_start_end + 1 :]).encode("utf-8")
