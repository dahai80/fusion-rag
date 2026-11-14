"""Issue #131 (R9): Docling-based layout-aware PDF ingestion.

Docling (IBM/LF AI) understands PDF layout — reading order, table structure,
sections — where plain PyMuPDF text extraction scrambles column-major tables
and interleaved layouts. This module is the optional "docling" PDF backend:

- Tables export as markdown pipe tables (TableFormer ACCURATE), keeping row
  alignment that per-cell text extraction loses.
- Page anchoring is EXACT here: the DoclingDocument is exported per page
  (export_to_markdown(page_no=...)) and the per-page fragments join into the
  same content/spans shape the legacy _parse_pdf produces, so the chunker's
  page mapping (#77) keeps working unchanged.

Failure contract: any Docling error (import, model download, conversion)
falls back to the legacy PyMuPDF path — ingestion degrades, never fails
(the caller logs the fallback). The backend is opt-in via RuntimeConfig
(FUSION_RAG_PDF_BACKEND) or per-KB config (pdf_backend).
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_CONVERT_LOCK = threading.Lock()


class DoclingParser:
    """Layout-aware PDF parser backed by docling. Heavy deps are lazy."""

    _converter = None

    @classmethod
    def _get_converter(cls):
        if cls._converter is not None:
            return cls._converter
        with _CONVERT_LOCK:
            if cls._converter is not None:
                return cls._converter
            from docling.datamodel.accelerator_options import AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            opts = PdfPipelineOptions()
            # Text PDFs only: scanned PDFs keep the #84 OCR-service path
            # (fusion-drawing-parser), which carries confidence metadata the
            # evidence contract (#85) surfaces. Docling OCR here would produce
            # unattributed text.
            opts.do_ocr = False
            opts.do_table_structure = True
            # Apple Silicon: TableFormer hits "Cannot convert a MPS Tensor to
            # float64" under the auto-selected MPS device — pin the docling
            # models to CPU (conversion is seconds per PDF either way).
            opts.accelerator_options = AcceleratorOptions(num_threads=4, device="cpu")
            cls._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )
            logger.info("DoclingParser: converter initialized (table_structure=accurate, ocr=off)")
        return cls._converter

    async def parse_pdf(self, path: Path) -> tuple[str, int, list[tuple[int, int, int]]]:
        """Convert a PDF to markdown, returning (content, pages, page_spans).

        Runs the (blocking, torch-backed) conversion in a worker thread; the
        event loop stays responsive during multi-second conversions. Per-page
        export gives exact (start, end, page_no) spans for #77 page anchoring.
        """
        return await asyncio.to_thread(self._parse_pdf_sync, Path(path))

    def _parse_pdf_sync(self, path: Path) -> tuple[str, int, list[tuple[int, int, int]]]:
        from docling_core.types.doc.base import ImageRefMode

        converter = self._get_converter()
        with _CONVERT_LOCK:
            result = converter.convert(str(path))
        document = result.document
        pages = len(document.pages)
        content = ""
        spans: list[tuple[int, int, int]] = []
        for page_no in sorted(document.pages.keys()):
            # Per-page export: exact page anchoring + reading order preserved.
            page_text = document.export_to_markdown(
                page_no=page_no,
                image_mode=ImageRefMode.PLACEHOLDER,
            ).strip()
            # Drop the picture placeholder comments — they are chunker noise.
            page_text = re.sub(r"<!--\s*image\s*-->\n?", "", page_text).strip()
            if not page_text:
                continue
            if content:
                content += "\n\n"
            start = len(content)
            content += page_text
            spans.append((start, len(content), page_no))
        logger.info(
            "docling parsed PDF %s: %d pages, %d non-empty spans, %d chars",
            path.name,
            pages,
            len(spans),
            len(content),
        )
        return content, pages, spans
