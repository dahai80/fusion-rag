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

    async def parse(self, file_path: str | Path) -> ParseResult:
        """Parse a single file and return its text content."""
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
                content, pages, page_spans = self._parse_pdf(path)
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
        self, dir_path: str | Path, recursive: bool = True, max_files: int = 1000
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
            result = await self.parse(f)
            results.append(result)
            files_scanned += 1

        return results
