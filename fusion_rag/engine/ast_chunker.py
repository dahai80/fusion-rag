import ast
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ASTChunk:
    text: str
    symbol_name: str
    symbol_type: str
    line_start: int
    line_end: int
    decorators: list[str] = field(default_factory=list)
    docstring: str = ""


class ASTChunker:
    def chunk(self, source_code: str, file_path: str = "") -> list[ASTChunk]:
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            # L18: AST parse failure previously returned the WHOLE file as one
            # chunk — a large broken file exceeded embed context and got
            # silently truncated. Fall back to RecursiveChunker so the text is
            # split into embed-sized pieces, then map each piece back to line
            # ranges for ASTChunk.
            logger.warning(
                "Syntax error in %s at line %s: %s, falling back to recursive chunker",
                file_path,
                e.lineno,
                e.msg,
            )
            return self._recursive_fallback(source_code, file_path)

        source_lines = source_code.splitlines()
        chunks: list[ASTChunk] = []

        import_nodes: list[ast.Import | ast.ImportFrom] = []
        constant_ranges: list[tuple[int, int]] = []

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_nodes.append(node)
                continue

            if import_nodes:
                chunks.append(self._build_imports_chunk(import_nodes, source_lines, source_code))
                import_nodes = []

            if isinstance(node, ast.ClassDef):
                chunks.append(self._extract_class_chunk(node, source_lines, source_code))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(self._extract_function_chunk(node, source_lines, source_code))
            else:
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                if start is not None and end is not None:
                    constant_ranges.append((start, end))
                else:
                    logger.debug("Skipping node without line info: %s", type(node).__name__)

        if import_nodes:
            chunks.append(self._build_imports_chunk(import_nodes, source_lines, source_code))

        if constant_ranges:
            chunks.append(self._build_constants_chunk(constant_ranges, source_lines, source_code))

        chunks.sort(key=lambda c: c.line_start)
        return chunks

    def _recursive_fallback(self, source_code: str, file_path: str) -> list[ASTChunk]:
        # L18: AST parse failed — split with RecursiveChunker instead of
        # returning the whole file as one chunk. Map each piece to a line
        # range by accumulating character offsets.
        from .preprocessor import RecursiveChunker

        if not source_code or not source_code.strip():
            return []
        pieces = RecursiveChunker(chunk_size=512, chunk_overlap=64).chunk(source_code)
        if not pieces:
            return []
        line_starts = [1]
        for ch in source_code:
            if ch == "\n":
                line_starts.append(line_starts[-1] + 1)
        # char offset → line number
        line_offsets = {}
        offset = 0
        for idx, line in enumerate(source_code.splitlines(keepends=True), start=1):
            line_offsets[offset] = idx
            offset += len(line)
        line_offsets[offset] = len(line_starts)

        def _line_at(char_pos: int) -> int:
            keys = [k for k in line_offsets if k <= char_pos]
            return line_offsets[max(keys)] if keys else 1

        chunks = []
        search_from = 0
        for i, piece in enumerate(pieces):
            pos = source_code.find(piece, search_from)
            if pos == -1:
                pos = search_from
            ls = _line_at(pos)
            le = _line_at(pos + len(piece) - 1) if piece else ls
            chunks.append(
                ASTChunk(
                    text=piece,
                    symbol_name=f"fallback_{i}",
                    symbol_type="constants",
                    line_start=ls,
                    line_end=max(le, ls),
                )
            )
            search_from = pos + len(piece)
        logger.info("AST fallback produced %d recursive chunks for %s", len(chunks), file_path)
        return chunks

    def _extract_class_chunk(
        self,
        node: ast.ClassDef,
        source_lines: list[str],
        source_code: str,
    ) -> ASTChunk:
        line_start = node.lineno
        if node.decorator_list:
            line_start = node.decorator_list[0].lineno
        line_end = node.end_lineno or node.lineno
        text = self._slice_lines(source_lines, line_start, line_end)
        decorators = self._get_decorators(node)
        docstring = self._get_docstring(node)
        return ASTChunk(
            text=text,
            symbol_name=node.name,
            symbol_type="class",
            line_start=line_start,
            line_end=line_end,
            decorators=decorators,
            docstring=docstring,
        )

    def _extract_function_chunk(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_lines: list[str],
        source_code: str,
    ) -> ASTChunk:
        line_start = node.lineno
        if node.decorator_list:
            line_start = node.decorator_list[0].lineno
        line_end = node.end_lineno or node.lineno
        text = self._slice_lines(source_lines, line_start, line_end)
        decorators = self._get_decorators(node)
        docstring = self._get_docstring(node)
        symbol_type = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
        return ASTChunk(
            text=text,
            symbol_name=node.name,
            symbol_type=symbol_type,
            line_start=line_start,
            line_end=line_end,
            decorators=decorators,
            docstring=docstring,
        )

    def _get_decorators(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        names: list[str] = []
        for dec in node.decorator_list:
            names.append(self._decorator_name(dec))
        return names

    def _decorator_name(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._decorator_name(node.value)
            return f"{base}.{node.attr}"
        if isinstance(node, ast.Call):
            return self._decorator_name(node.func)
        if isinstance(node, ast.Subscript):
            return self._decorator_name(node.value)
        return ast.dump(node)

    def _get_docstring(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        ds = ast.get_docstring(node, clean=True)
        return ds if ds is not None else ""

    def _build_imports_chunk(
        self,
        nodes: list[ast.Import | ast.ImportFrom],
        source_lines: list[str],
        source_code: str,
    ) -> ASTChunk:
        line_start = nodes[0].lineno
        line_end = nodes[-1].end_lineno or nodes[-1].lineno
        text = self._slice_lines(source_lines, line_start, line_end)
        return ASTChunk(
            text=text,
            symbol_name="imports",
            symbol_type="imports",
            line_start=line_start,
            line_end=line_end,
        )

    def _build_constants_chunk(
        self,
        ranges: list[tuple[int, int]],
        source_lines: list[str],
        source_code: str,
    ) -> ASTChunk:
        line_start = ranges[0][0]
        line_end = ranges[-1][1]
        text = self._slice_lines(source_lines, line_start, line_end)
        return ASTChunk(
            text=text,
            symbol_name="module_level",
            symbol_type="constants",
            line_start=line_start,
            line_end=line_end,
        )

    @staticmethod
    def _slice_lines(source_lines: list[str], line_start: int, line_end: int) -> str:
        return "\n".join(source_lines[line_start - 1 : line_end])
