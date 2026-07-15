"""Document chunker — split parsed text into chunks for embedding.

Supports semantic splitting (by paragraph/section), fixed-size splitting,
and code-specific splitting (by function/class).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .document import DocumentType, ParseResult


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


class Chunker:
    """Split document text into chunks for embedding."""

    def __init__(self, strategy: str = "semantic", chunk_size: int = 512,
                 chunk_overlap: int = 64):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def chunk(self, result: ParseResult) -> list[Chunk]:
        """Split a parsed document into chunks."""
        if not result.content.strip():
            return []

        if result.error:
            return []

        if self.strategy == "code" or DocumentParser.is_code_file(result.doc_type):
            return self._chunk_code(result)
        elif self.strategy == "semantic":
            return self._chunk_semantic(result)
        else:
            return self._chunk_fixed(result)

    def _chunk_semantic(self, result: ParseResult) -> list[Chunk]:
        """Split by semantic boundaries (paragraphs, sections)."""
        text = result.content
        # Split by markdown headings or double newlines
        sections = re.split(r"(?=\n#|\n##|\n###|\n####|\n#####|\n######|^#)", text, flags=re.MULTILINE)
        chunks = []
        current = []
        current_len = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue
            sec_len = len(section)
            if current_len + sec_len > self.chunk_size and current:
                chunks.append(self._make_chunk("\n\n".join(current), result, len(chunks)))
                # Keep overlap
                overlap = []
                overlap_len = 0
                for c in reversed(current):
                    if overlap_len + len(c) > self.chunk_overlap:
                        break
                    overlap.insert(0, c)
                    overlap_len += len(c)
                current = overlap
                current_len = overlap_len
            current.append(section)
            current_len += sec_len

        if current:
            chunks.append(self._make_chunk("\n\n".join(current), result, len(chunks)))

        return chunks if chunks else [self._make_chunk(text, result, 0)]

    def _chunk_fixed(self, result: ParseResult) -> list[Chunk]:
        """Split by fixed character count with overlap."""
        text = result.content
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(self._make_chunk(chunk_text, result, len(chunks)))
            start += self.chunk_size - self.chunk_overlap
            if start >= len(text):
                break
        return chunks if chunks else [self._make_chunk(text, result, 0)]

    def _chunk_code(self, result: ParseResult) -> list[Chunk]:
        """Split code files by function/class boundaries."""
        text = result.content
        # Try to split by function/class definitions
        patterns = [
            r"(?=^def\s+\w+\s*\()",      # Python
            r"(?=^class\s+\w+)",           # Python/Java/C++
            r"(?=^func\s+\w+\s*\()",       # Go/Swift
            r"(?=^public\s+(static\s+)?\w+\s+\w+\s*\()",  # Java/C#
            r"(?=^function\s+\w+\s*\()",   # JS/TS
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
                            sub_chunks = self._chunk_fixed(ParseResult(
                                file_path=result.file_path, file_name=result.file_name,
                                doc_type=result.doc_type, content=part,
                            ))
                            chunks.extend(sub_chunks)
                        else:
                            chunks.append(self._make_chunk(part, result, len(chunks)))
                return chunks if chunks else [self._make_chunk(text, result, 0)]

        # Fallback: fixed-size chunking
        return self._chunk_fixed(result)

    def _make_chunk(self, text: str, result: ParseResult, index: int) -> Chunk:
        """Create a Chunk from text and metadata."""
        return Chunk(
            text=text.strip(),
            index=index,
            doc_path=result.file_path,
            doc_name=result.file_name,
            doc_type=result.doc_type.value,
            metadata=result.metadata,
            tokens=len(text) // 4,  # Rough token estimate
        )


from .document import DocumentParser