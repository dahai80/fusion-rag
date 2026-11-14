# Office format parsers: PPTX + XLSX (server-side, python-pptx + openpyxl).

from __future__ import annotations

from pathlib import Path

from fusion_rag.engine.document import EXTENSION_MAP, DocumentParser, DocumentType


def _make_xlsx(path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["Region", "Amount"])
    ws.append(["North", 120])
    ws.append(["South", None])
    ws2 = wb.create_sheet("Empty")
    ws2["A1"] = None
    wb.save(path)


def _make_pptx(path: Path) -> None:
    from pptx import Presentation

    prs = Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[1])
    s1.shapes.title.text = "Quarterly Review"
    s1.placeholders[1].text = "Revenue up 20%"
    s1.notes_slide.notes_text_frame.text = "mention the roadmap"
    prs.slides.add_slide(prs.slide_layouts[6])  # blank slide, no text
    prs.save(path)


class TestXlsxParser:
    async def test_parse_xlsx(self, tmp_path):
        f = tmp_path / "book.xlsx"
        _make_xlsx(f)
        result = await DocumentParser().parse(f)
        assert result.error == ""
        assert result.doc_type == DocumentType.XLSX
        assert "## Sheet: Sales" in result.content
        assert "Region | Amount" in result.content
        assert "North | 120" in result.content
        # empty sheet produces no section
        assert "Empty" not in result.content
        assert result.chars == len(result.content)

    async def test_xlsx_ext_map(self):
        assert EXTENSION_MAP[".xlsx"] == DocumentType.XLSX


class TestPptxParser:
    async def test_parse_pptx(self, tmp_path):
        f = tmp_path / "deck.pptx"
        _make_pptx(f)
        result = await DocumentParser().parse(f)
        assert result.error == ""
        assert result.doc_type == DocumentType.PPTX
        assert "## Slide 1" in result.content
        assert "Quarterly Review" in result.content
        assert "Revenue up 20%" in result.content
        # speaker notes captured
        assert "[notes] mention the roadmap" in result.content
        # text-only slide 2 is skipped, so no "## Slide 2" section
        assert "## Slide 2" not in result.content

    async def test_pptx_ext_map(self):
        assert EXTENSION_MAP[".pptx"] == DocumentType.PPTX
