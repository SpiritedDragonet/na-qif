from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

ET.register_namespace("w", W_NS)
ET.register_namespace("r", R_NS)
ET.register_namespace("m", M_NS)
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
        self.parts[name] = ET.tostring(element, encoding="utf-8", xml_declaration=True)

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

