"""Document preprocessor — cleaning, normalization, and deduplication."""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """Preprocesses documents before chunking: clean, normalize, deduplicate."""

    @staticmethod
    def clean(text: str) -> str:
        """Clean text by removing excessive whitespace, null chars, and control characters."""
        text = text.replace("\x00", "")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        text = re.sub(r" {3,}", " ", text)
        text = re.sub(r"\t+", " ", text)
        return text.strip()

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text: lowercase, unicode NFKC, strip."""
        import unicodedata
        text = unicodedata.normalize("NFKC", text)
        text = text.strip()
        return text

    @staticmethod
    def deduplicate(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate or near-duplicate chunks."""
        seen = set()
        unique = []
        for chunk in chunks:
            text = chunk.get("text", "")
            sig = text[:100]  # Simple signature
            if sig not in seen:
                seen.add(sig)
                unique.append(chunk)
        return unique

    @staticmethod
    def strip_boilerplate(text: str) -> str:
        """Remove common boilerplate (headers, footers, nav)."""
        lines = text.split("\n")
        filtered = []
        boilerplate_patterns = [
            r"^navigation$", r"^footer$", r"^header$",
            r"^cookie", r"^terms of service", r"^privacy policy",
            r"^all rights reserved", r"^copyright",
        ]
        for line in lines:
            stripped = line.strip().lower()
            if any(re.match(p, stripped) for p in boilerplate_patterns):
                continue
            if len(stripped) < 3:
                continue
            filtered.append(line)
        return "\n".join(filtered)


class RecursiveChunker:
    """Recursively splits text by multiple separators, like LangChain."""

    SEPARATORS = None

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if self.SEPARATORS is None:
            self.SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str) -> list[str]:
        """Recursively split text into chunks."""
        return self._split(text, self.SEPARATORS)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split using available separators."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separator = separators[0]
        if not separator:
            return self._split_by_chars(text)

        splits = text.split(separator) if separator else list(text)
        if len(splits) == 1:
            return self._split(text, separators[1:]) if len(separators) > 1 else self._split_by_chars(text)

        chunks = []
        current = []
        current_len = 0
        for split in splits:
            split_len = len(split)
            if current_len + split_len > self.chunk_size and current:
                chunks.append(separator.join(current))
                overlap = self._get_overlap(current, separator)
                current = overlap
                current_len = sum(len(s) for s in overlap)
            current.append(split)
            current_len += split_len

        if current:
            chunks.append(separator.join(current))
        return chunks

    def _split_by_chars(self, text: str) -> list[str]:
        """Fallback: split by character count."""
        overlap = min(self.chunk_overlap, self.chunk_size - 1)
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if start >= end:
                break
            chunks.append(text[start:end])
            next_start = end - overlap if end < len(text) else len(text)
            if next_start <= start:
                next_start = start + 1
            start = next_start
        return chunks

    @staticmethod
    def _get_overlap(current: list[str], separator: str) -> list[str]:
        """Get overlap from the end of current."""
        overlap_text = separator.join(current)
        overlap_len = len(overlap_text)
        # Keep last ~quarter as overlap
        keep_len = overlap_len // 4
        result = []
        char_count = 0
        for item in reversed(current):
            if char_count + len(item) > keep_len:
                break
            result.insert(0, item)
            char_count += len(item)
        return result if result else current[-1:]
