"""Document chunker — split parsed text into chunks for embedding.

Supports semantic splitting (by paragraph/section), fixed-size splitting,
and code-specific splitting (by function/class).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .document import DocumentParser, ParseResult

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk of text from a document."""

    text: str
    index: int
    doc_path: str = ""
    doc_name: str = ""
    doc_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens: int = 0
    # Issue #77: 1-based source page for evidence anchoring (PDF page spans).
    # 0 = no page info (non-PDF source or untracked strategy).
    page: int = 0


class Chunker:
    """Split document text into chunks for embedding."""

    def __init__(self, strategy: str = "semantic", chunk_size: int = 512, chunk_overlap: int = 64):
        # F10: overlap >= chunk_size makes the fixed-stride step <= 0, so the
        # while loop never advances → infinite loop / memory blowup. Reject early.
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})")
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def chunk(self, result: ParseResult) -> list[Chunk]:
        """Split a parsed document into chunks."""
        if not result.content.strip():
            return []

        if result.error:
            return []

        if self.strategy == "ast" or self._should_use_ast(result):
            return self._chunk_ast(result)
        elif self.strategy == "code" or DocumentParser.is_code_file(result.doc_type):
            return self._chunk_code(result)
        elif self.strategy == "semantic":
            return self._chunk_semantic(result)
        else:
            return self._chunk_fixed(result)

    def _should_use_ast(self, result: ParseResult) -> bool:
        # D2: compare to the enum's .value (lowercase "code_python"), not the
        # enum NAME "CODE_PYTHON". result.doc_type.value is lowercase; the prior
        # uppercase compare was always False, so Python AST auto-detection was
        # dead code (a .py file only got the AST path via explicit strategy="ast").
        dtype = result.doc_type.value if hasattr(result.doc_type, "value") else str(result.doc_type)
        return dtype == "code_python" and self.strategy != "code"

    def _page_at(self, result: ParseResult, offset: int) -> int:
        # Issue #77: map a content char offset back to its source page via the
        # parser's page_spans. Spans are ordered; -1 (unknown offset) or a
        # miss (gap between spans) means no page info.
        if offset < 0 or not result.page_spans:
            return 0
        for start, end, page_no in result.page_spans:
            if start <= offset < end:
                return page_no
        return 0

    def _chunk_semantic(self, result: ParseResult) -> list[Chunk]:
        """Split by semantic boundaries (paragraphs, sections)."""
        text = result.content
        # Split by markdown headings or double newlines
        sections = re.split(r"(?=\n#|\n##|\n###|\n####|\n#####|\n######|^#)", text, flags=re.MULTILINE)
        chunks = []
        current = []
        # Issue #77: parallel offset list — re.split parts concatenate back to
        # the original text exactly, so a running cursor gives each section's
        # char offset for page mapping.
        current_offsets = []
        current_len = 0
        pos = 0

        for raw in sections:
            section = raw.strip()
            section_offset = pos + (len(raw) - len(raw.lstrip()))
            pos += len(raw)
            if not section:
                continue
            sec_len = len(section)
            if current_len + sec_len > self.chunk_size and current:
                chunks.append(self._make_chunk("\n\n".join(current), result, len(chunks), offset=current_offsets[0]))
                # Keep overlap
                overlap = []
                overlap_offsets = []
                overlap_len = 0
                for c, co in zip(reversed(current), reversed(current_offsets)):
                    if overlap_len + len(c) > self.chunk_overlap:
                        break
                    overlap.insert(0, c)
                    overlap_offsets.insert(0, co)
                    overlap_len += len(c)
                current = overlap
                current_offsets = overlap_offsets
                current_len = overlap_len
            current.append(section)
            current_offsets.append(section_offset)
            current_len += sec_len

        if current:
            chunks.append(self._make_chunk("\n\n".join(current), result, len(chunks), offset=current_offsets[0]))

        return chunks if chunks else [self._make_chunk(text, result, 0)]

    def _chunk_fixed(self, result: ParseResult) -> list[Chunk]:
        """Split by fixed character count with overlap."""
        text = result.content
        chunks = []
        start = 0
        # F10: stride must stay positive or the loop never advances → infinite loop.
        step = max(self.chunk_size - self.chunk_overlap, 1)
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(self._make_chunk(chunk_text, result, len(chunks), offset=start))
            start += step
            if start >= len(text):
                break
        return chunks if chunks else [self._make_chunk(text, result, 0)]

    def _chunk_ast(self, result: ParseResult) -> list[Chunk]:
        """AST-based chunking for Python files — uses built-in ast module."""
        try:
            from .ast_chunker import ASTChunker

            ast_chunker = ASTChunker()
            ast_chunks = ast_chunker.chunk(result.content, result.file_path)
            chunks = []
            for ac in ast_chunks:
                chunks.append(
                    Chunk(
                        text=ac.text.strip(),
                        index=len(chunks),
                        doc_path=result.file_path,
                        doc_name=result.file_name,
                        doc_type=result.doc_type.value,
                        metadata={
                            **result.metadata,
                            "symbol_name": ac.symbol_name,
                            "symbol_type": ac.symbol_type,
                            "line_start": ac.line_start,
                            "line_end": ac.line_end,
                            "decorators": ac.decorators,
                        },
                        tokens=len(ac.text) // 4,
                    )
                )
            if not chunks:
                return self._chunk_code(result)
            return chunks
        except Exception as e:
            logger.warning("AST chunking failed, falling back to code chunking: %s", e)
            return self._chunk_code(result)

    def _chunk_code(self, result: ParseResult) -> list[Chunk]:
        """Split code files by function/class boundaries."""
        text = result.content
        # Try to split by function/class definitions
        patterns = [
            r"(?=^def\s+\w+\s*\()",  # Python
            r"(?=^class\s+\w+)",  # Python/Java/C++
            r"(?=^func\s+\w+\s*\()",  # Go/Swift
            r"(?=^public\s+(static\s+)?\w+\s+\w+\s*\()",  # Java/C#
            r"(?=^function\s+\w+\s*\()",  # JS/TS
            r"(?=^\w+\s*:\s*function\s*\()",  # JS object method
        ]

        for pattern in patterns:
            parts = re.split(pattern, text, flags=re.MULTILINE)
            if len(parts) > 1:
                chunks = []
                for part in parts:
                    part = part.strip()
                    if part:
                        if len(part) > self.chunk_size:
                            # Sub-chunk large functions
                            sub_chunks = self._chunk_fixed(
                                ParseResult(
                                    file_path=result.file_path,
                                    file_name=result.file_name,
                                    doc_type=result.doc_type,
                                    content=part,
                                )
                            )
                            chunks.extend(sub_chunks)
                        else:
                            chunks.append(self._make_chunk(part, result, len(chunks)))
                return chunks if chunks else [self._make_chunk(text, result, 0)]

        # Fallback: fixed-size chunking
        return self._chunk_fixed(result)

    def _make_chunk(self, text: str, result: ParseResult, index: int, offset: int = -1) -> Chunk:
        """Create a Chunk from text and metadata."""
        return Chunk(
            text=text.strip(),
            index=index,
            doc_path=result.file_path,
            doc_name=result.file_name,
            doc_type=result.doc_type.value,
            metadata=result.metadata,
            tokens=len(text) // 4,  # Rough token estimate
            page=self._page_at(result, offset),
        )
