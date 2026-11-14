"""Document parser — extract text from multiple file formats.

Supported formats: PDF, DOCX, EPUB, MD, TXT, HTML, and code files.
All processing is local — except scanned PDFs, which are handed to the
fusion-drawing-parser OCR service when FUSION_RAG_OCR_URL is set (issue #84).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    MARKDOWN = "markdown"
    TXT = "txt"
    HTML = "html"
    EPUB = "epub"
    CODE_PYTHON = "code_python"
    CODE_SWIFT = "code_swift"
    CODE_CPP = "code_cpp"
    CODE_JS = "code_js"
    CODE_SHELL = "code_shell"
    CODE_OTHER = "code_other"
    UNKNOWN = "unknown"


# Map file extensions to document types
EXTENSION_MAP: dict[str, DocumentType] = {
    ".pdf": DocumentType.PDF,
    ".docx": DocumentType.DOCX,
    ".pptx": DocumentType.PPTX,
    ".xlsx": DocumentType.XLSX,
    ".md": DocumentType.MARKDOWN,
    ".markdown": DocumentType.MARKDOWN,
    ".txt": DocumentType.TXT,
    ".html": DocumentType.HTML,
    ".htm": DocumentType.HTML,
    ".epub": DocumentType.EPUB,
    ".py": DocumentType.CODE_PYTHON,
    ".pyw": DocumentType.CODE_PYTHON,
    ".swift": DocumentType.CODE_SWIFT,
    ".c": DocumentType.CODE_CPP,
    ".cpp": DocumentType.CODE_CPP,
    ".h": DocumentType.CODE_CPP,
    ".hpp": DocumentType.CODE_CPP,
    ".js": DocumentType.CODE_JS,
    ".ts": DocumentType.CODE_JS,
    ".jsx": DocumentType.CODE_JS,
    ".tsx": DocumentType.CODE_JS,
    ".sh": DocumentType.CODE_SHELL,
    ".bash": DocumentType.CODE_SHELL,
    ".zsh": DocumentType.CODE_SHELL,
    ".rs": DocumentType.CODE_OTHER,
    ".go": DocumentType.CODE_OTHER,
    ".java": DocumentType.CODE_OTHER,
    ".rb": DocumentType.CODE_OTHER,
    ".php": DocumentType.CODE_OTHER,
    ".yaml": DocumentType.CODE_OTHER,
    ".yml": DocumentType.CODE_OTHER,
    ".json": DocumentType.CODE_OTHER,
    ".toml": DocumentType.CODE_OTHER,
    ".cfg": DocumentType.CODE_OTHER,
    ".ini": DocumentType.CODE_OTHER,
}

# D5: single source of truth for content-type string → DocumentType. The inline
# ingest path (routes_docs) previously held a second, divergent map — a new
# type added only to EXTENSION_MAP was invisible there. Co-locate both maps on
# document.py so one edit updates every entry path.
CONTENT_TYPE_MAP: dict[str, DocumentType] = {
    "markdown": DocumentType.MARKDOWN,
    "html": DocumentType.HTML,
    "csv": DocumentType.TXT,
    "text": DocumentType.TXT,
    "epub": DocumentType.EPUB,
    "pptx": DocumentType.PPTX,
    "xlsx": DocumentType.XLSX,
}


@dataclass
class ParseResult:
    """Result of parsing a single document."""

    file_path: str
    file_name: str
    doc_type: DocumentType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: int = 0
    chars: int = 0
    error: str = ""
    # Issue #77: (start, end, page_no) char-offset spans mapping content
    # positions back to PDF pages — lets the chunker stamp per-chunk page
    # metadata for evidence anchoring. Empty for non-PDF sources.
    page_spans: list[tuple[int, int, int]] = field(default_factory=list)
    # Issue #84/#85: OCR text_chunks from fusion-drawing-parser for scanned
    # PDFs. Each dict: {chunk_id, text, page, bbox, slice_id, ocr_confidence,
    # low_confidence, kind}. Non-empty only for scanned PDFs with an OCR
    # service configured; these bypass the chunker (already slice-sized).
    ocr_chunks: list[dict] = field(default_factory=list)


class DocumentParser:
    """Parse documents of various formats into plain text."""

    @staticmethod
    def detect_type(file_path: str | Path) -> DocumentType:
        ext = Path(file_path).suffix.lower()
        return EXTENSION_MAP.get(ext, DocumentType.UNKNOWN)

    @staticmethod
    def is_code_file(doc_type: DocumentType) -> bool:
        return doc_type.value.startswith("code_")

    async def parse(self, file_path: str | Path, pdf_backend: str = "") -> ParseResult:
        """Parse a single file and return its text content.

        Issue #131 (R9): `pdf_backend` selects the PDF extraction backend for
        this call — "docling" (layout-aware) or "legacy" (PyMuPDF). Empty
        defers to RuntimeConfig.pdf_backend (FUSION_RAG_PDF_BACKEND). Any
        Docling failure falls back to legacy (logged), never fails ingest.
        """
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return ParseResult(
                file_path=str(path),
                file_name=path.name,
                doc_type=DocumentType.UNKNOWN,
                content="",
                error=f"File not found: {path}",
            )

        doc_type = self.detect_type(path)
        file_name = path.name
        result = ParseResult(
            file_path=str(path),
            file_name=file_name,
            doc_type=doc_type,
            content="",
            metadata={"size": path.stat().st_size},
        )

        try:
            if doc_type == DocumentType.PDF:
                content, pages, page_spans, backend_used = await self._parse_pdf_with_backend(path, pdf_backend)
                result.metadata["pdf_backend"] = backend_used
                result.content = content
                result.pages = pages
                result.page_spans = page_spans
                # Issue #84: pages exist but zero extractable text = scanned PDF.
                # Hand to the OCR service if configured; fail visibly otherwise
                # (previously scanned pages were silently skipped and the doc
                # ingested as empty content).
                if pages > 0 and not content.strip():
                    result.ocr_chunks = await self._ocr_scanned_pdf(path, result)
            elif doc_type == DocumentType.DOCX:
                result.content = self._parse_docx(path)
            elif doc_type == DocumentType.PPTX:
                result.content = self._parse_pptx(path)
            elif doc_type == DocumentType.XLSX:
                result.content = self._parse_xlsx(path)
            elif doc_type == DocumentType.MARKDOWN or doc_type == DocumentType.TXT:
                result.content = path.read_text(encoding="utf-8", errors="replace")
            elif doc_type == DocumentType.HTML:
                result.content = self._parse_html(path)
            elif doc_type == DocumentType.EPUB:
                result.content = self._parse_epub(path)
            elif doc_type.value.startswith("code_"):
                result.content = path.read_text(encoding="utf-8", errors="replace")
                result.metadata["language"] = doc_type.value.replace("code_", "")
            else:
                # Fallback: try to read as text
                result.content = path.read_text(encoding="utf-8", errors="replace")
                result.doc_type = DocumentType.TXT

            result.chars = len(result.content)
            result.metadata["chars"] = result.chars

        except Exception as e:
            result.error = f"Parse error: {e}"

        return result

    async def _parse_pdf_with_backend(
        self, path: Path, pdf_backend: str = ""
    ) -> tuple[str, int, list[tuple[int, int, int]], str]:
        """Issue #131 (R9): route PDF parsing through the selected backend.

        Backend resolution: explicit per-call value ("docling"/"legacy") wins,
        else RuntimeConfig.pdf_backend (FUSION_RAG_PDF_BACKEND, default
        "legacy"). A "docling" attempt that raises for ANY reason (missing
        dependency, model download failure, conversion error) falls back to
        the legacy PyMuPDF path with a loud warning — ingest degrades, the
        caller sees pdf_backend="legacy" in the doc metadata.
        """
        backend = (pdf_backend or "").strip().lower()
        if not backend:
            from .runtime_config import get_runtime_config

            backend = get_runtime_config().pdf_backend
        if backend == "docling":
            try:
                from .docling_parser import DoclingParser

                content, pages, spans = await DoclingParser().parse_pdf(path)
                return content, pages, spans, "docling"
            except Exception as e:
                logger.warning(
                    "docling PDF backend failed for %s (%s: %s) — falling back to legacy PyMuPDF",
                    path.name,
                    type(e).__name__,
                    e,
                )
        elif backend not in ("legacy", ""):
            logger.warning("unknown pdf_backend %r — using legacy PyMuPDF", backend)
        content, pages, spans = self._parse_pdf(path)
        return content, pages, spans, "legacy"

    def _parse_pdf(self, path: Path) -> tuple[str, int, list[tuple[int, int, int]]]:
        """Extract text from PDF using PyMuPDF, tracking per-page char spans."""
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        pages = len(doc)
        content = ""
        spans = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if not text:
                continue
            if content:
                content += "\n\n"
            start = len(content)
            content += text
            spans.append((start, len(content), i + 1))
        doc.close()
        logger.info("parsed PDF %s: %d pages, %d non-empty spans", path.name, pages, len(spans))
        return content, pages, spans

    def _parse_docx(self, path: Path) -> str:
        """Extract text from DOCX."""
        from docx import Document

        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    def _parse_pptx(self, path: Path) -> str:
        """Extract text from PPTX: one '## Slide N' section per slide.

        Text frames in reading order + speaker notes; empty slides skipped
        so a deck of images-only slides does not produce empty content."""
        from pptx import Presentation

        prs = Presentation(str(path))
        parts: list[str] = []
        for idx, slide in enumerate(prs.slides, start=1):
            texts: list[str] = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        line = "".join(run.text for run in para.runs).strip()
                        if line:
                            texts.append(line)
                # Grouped shapes recurse one level (common in real decks).
                if getattr(shape, "shape_type", None) is not None and hasattr(shape, "shapes"):
                    for sub in shape.shapes:  # type: ignore[attr-defined]
                        if sub.has_text_frame:
                            line = sub.text_frame.text.strip()
                            if line:
                                texts.append(line)
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    texts.append(f"[notes] {notes}")
            if texts:
                parts.append(f"## Slide {idx}\n" + "\n".join(texts))
        return "\n\n".join(parts)

    def _parse_xlsx(self, path: Path) -> str:
        """Extract text from XLSX: one '## Sheet: name' section per sheet.

        Rows render as ' | '-joined cells — line-oriented text that survives
        the semantic chunker and keeps cell context together. Empty trailing
        cells trimmed; formula cells contribute their cached value."""
        from openpyxl import load_workbook

        wb = load_workbook(str(path), read_only=True, data_only=True)
        parts: list[str] = []
        for ws in wb.worksheets:
            lines: list[str] = []
            for row in ws.iter_rows(values_only=True):
                cells = ["" if v is None else str(v).strip() for v in row]
                while cells and cells[-1] == "":
                    cells.pop()
                if cells:
                    lines.append(" | ".join(cells))
            if lines:
                parts.append(f"## Sheet: {ws.title}\n" + "\n".join(lines))
        wb.close()
        return "\n\n".join(parts)

    def _parse_html(self, path: Path) -> str:
        """Extract text from HTML."""
        from markdownify import markdownify as md

        html = path.read_text(encoding="utf-8", errors="replace")
        return md(html, heading_style="ATX", strip=["script", "style"])

    def _parse_epub(self, path: Path) -> str:
        """Extract text from EPUB: each XHTML chapter becomes a '# <name>' headed section."""
        import ebooklib
        from ebooklib import epub
        from markdownify import markdownify as md

        book = epub.read_epub(str(path), options={"ignore_ncx": True})
        parts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            html = item.get_content().decode("utf-8", errors="replace")
            # ebooklib hands back the raw XHTML incl. the <?xml ...?> declaration,
            # which markdownify renders as stray "xml version=..." text.
            html = re.sub(r"<\?xml[^>]*\?>", "", html)
            text = md(html, heading_style="ATX", strip=["script", "style"]).strip()
            if text:
                parts.append(f"# {item.get_name()}\n\n{text}")
        logger.info("parsed EPUB %s: %d chapters", path.name, len(parts))
        return "\n\n".join(parts)

    async def _ocr_scanned_pdf(self, path: Path, result: ParseResult) -> list[dict]:
        """Issue #84: OCR a scanned PDF via fusion-drawing-parser (FUSION_RAG_OCR_URL).

        Returns schema-v1 text_chunks; fails visibly (result.error, no chunks)
        when no OCR service is configured or the task fails."""
        from fusion_rag.engine.ocr_hook import OcrHook, OcrUnavailable, ocr_service_configured

        if not ocr_service_configured():
            msg = (
                "Scanned PDF (no extractable text) and no OCR service configured; "
                "set FUSION_RAG_OCR_URL to enable OCR ingestion"
            )
            logger.warning("scanned PDF rejected, no OCR service: %s", path.name)
            result.error = msg
            return []
        try:
            chunks = await OcrHook().parse_file(str(path))
        except OcrUnavailable as e:
            logger.error("OCR failed for %s: %s", path.name, e)
            result.error = f"OCR error: {e}"
            return []
        if not chunks:
            result.error = f"OCR returned no text chunks for {path.name}"
            return []
        result.pages = result.pages or 0
        return chunks

    async def parse_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
        max_files: int = 1000,
        pdf_backend: str = "",
    ) -> list[ParseResult]:
        """Parse all supported files in a directory."""
        path = Path(dir_path).expanduser().resolve()
        if not path.is_dir():
            return []

        results = []
        pattern = "**/*" if recursive else "*"
        files_scanned = 0

        for f in sorted(path.glob(pattern)):
            if not f.is_file():
                continue
            if self.detect_type(f) == DocumentType.UNKNOWN:
                continue
            if files_scanned >= max_files:
                break
            result = await self.parse(f, pdf_backend=pdf_backend)
            results.append(result)
            files_scanned += 1

        return results
