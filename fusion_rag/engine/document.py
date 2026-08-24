"""Document parser — extract text from multiple file formats.

Supported formats: PDF, DOCX, MD, TXT, HTML, and code files.
All processing is local — no network calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    TXT = "txt"
    HTML = "html"
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
                content, pages = self._parse_pdf(path)
                result.content = content
                result.pages = pages
            elif doc_type == DocumentType.DOCX:
                result.content = self._parse_docx(path)
            elif doc_type == DocumentType.MARKDOWN or doc_type == DocumentType.TXT:
                result.content = path.read_text(encoding="utf-8", errors="replace")
            elif doc_type == DocumentType.HTML:
                result.content = self._parse_html(path)
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

    def _parse_pdf(self, path: Path) -> tuple[str, int]:
        """Extract text from PDF using PyMuPDF."""
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        pages = len(doc)
        texts = []
        for page in doc:
            texts.append(page.get_text())
        doc.close()
        return "\n\n".join(texts), pages

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
