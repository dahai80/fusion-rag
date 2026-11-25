"""Issue #161: structured XLSX query (text-to-SQL over sheet schema) tests.

Ingest indexes each sheet as a real SQLite table in the per-KB sheets.db; the
query path is LLM-planner (mocked here) -> deterministic SQL validation ->
read-only execution with an enforced row cap. Markdown chunks stay as the
fallback tier — both paths are asserted."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app
from fusion_rag.embed.client import EmbeddingClient
from fusion_rag.engine.llm_errors import LLMUnavailable
from fusion_rag.engine.xlsx_query import validate_sql


def _make_xlsx(path, rows_a, rows_b=None):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "销售"
    ws.append(["产品", "区域", "销售额"])
    for r in rows_a:
        ws.append(r)
    if rows_b is not None:
        ws2 = wb.create_sheet("成本")
        ws2.append(["产品", "成本"])
        for r in rows_b:
            ws2.append(r)
    wb.save(path)


ROWS_A = [
    ["键盘", "华东", 1200],
    ["键盘", "华东", 800],
    ["鼠标", "华南", 500],
]
ROWS_B = [["键盘", 900], ["鼠标", 200]]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSION_RAG_DATA_DIR", str(tmp_path))
    with patch.object(
        EmbeddingClient,
        "embed_batch",
        new_callable=AsyncMock,
        side_effect=lambda texts: [[0.01] * 1024 for _ in texts],
    ), patch.object(
        EmbeddingClient, "embed", new_callable=AsyncMock, return_value=[0.01] * 1024
    ), TestClient(create_app()) as tc:
        yield tc


@pytest.fixture
def kb_with_xlsx(client, tmp_path):
    xlsx = tmp_path / "q3-sales.xlsx"
    _make_xlsx(xlsx, ROWS_A, ROWS_B)
    kb_id = client.post("/kb/bases", json={"name": "xlsx-kb"}).json()["id"]
    resp = client.post(f"/kb/bases/{kb_id}/documents", json={"file_path": str(xlsx)})
    assert resp.status_code == 200, resp.text
    doc_id = resp.json()["doc_id"]
    return kb_id, doc_id


class TestSchemaExtraction:
    def test_ingest_indexes_sheets(self, client, kb_with_xlsx):
        # Acceptance: schema (column names/types/ranges) extracted at ingest.
        kb_id, doc_id = kb_with_xlsx
        resp = client.get(f"/kb/bases/{kb_id}/xlsx/sheets")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["count"] == 2
        sales = next(s for s in body["sheets"] if s["sheet_name"] == "销售")
        assert sales["doc_id"] == doc_id
        assert sales["row_count"] == 3
        names = [c["name"] for c in sales["columns"]]
        headers = [c["header"] for c in sales["columns"]]
        assert names == ["_c1", "_c2", "_c3"]
        assert headers == ["产品", "区域", "销售额"]
        dtypes = [c["dtype"] for c in sales["columns"]]
        assert dtypes == ["TEXT", "TEXT", "INTEGER"]

    def test_sheets_404_without_xlsx(self, client):
        kb_id = client.post("/kb/bases", json={"name": "empty-kb"}).json()["id"]
        assert client.get(f"/kb/bases/{kb_id}/xlsx/sheets").status_code == 404

    def test_markdown_chunks_still_ingested(self, client, kb_with_xlsx):
        # Fallback tier intact: the XLSX also produced searchable markdown chunks.
        kb_id, _doc_id = kb_with_xlsx
        docs = client.get(f"/kb/bases/{kb_id}/documents").json()
        assert isinstance(docs, list) and len(docs) == 1


class TestSqlGuards:
    def test_rejects_non_select(self):
        for bad in ["DROP TABLE t_x", "INSERT INTO t_x VALUES (1)", ""]:
            with pytest.raises(ValueError):
                validate_sql(bad)

    def test_rejects_multiple_statements(self):
        with pytest.raises(ValueError):
            validate_sql("SELECT 1; DROP TABLE t_x")

    def test_rejects_forbidden_keyword(self):
        with pytest.raises(ValueError, match="PRAGMA"):
            validate_sql("SELECT * FROM t ORDER BY 1 -- pragma note")
        with pytest.raises(ValueError):
            validate_sql("ATTACH DATABASE 'x' AS y")

    def test_strips_markdown_fence(self):
        sql = validate_sql("```sql\nSELECT 1\n```")
        assert sql == "SELECT 1"

    def test_allows_select_and_cte(self):
        assert validate_sql("SELECT SUM(_c3) FROM t_x") == "SELECT SUM(_c3) FROM t_x"
        assert validate_sql("WITH a AS (SELECT 1 AS x) SELECT x FROM a")


class TestQueryPath:
    @staticmethod
    def _table_for(ddl, sheet):
        import re as _re

        m = _re.search(
            r'-- sheet "' + sheet + r'"[^;]*?CREATE TABLE "(t_[0-9a-z_]+)"', ddl, _re.DOTALL
        )
        return m.group(1)

    def test_aggregation_computed_by_sql_with_citations(self, client, kb_with_xlsx):
        # Acceptance: aggregation answered from computed rows + citations,
        # not LLM arithmetic over chunks.
        kb_id, doc_id = kb_with_xlsx

        def plan(ddl, question):
            # Pull the real table name out of the DDL the route built.

            tname = TestQueryPath._table_for(ddl, "销售")
            return f"SELECT _c1, SUM(_c3) AS total FROM {tname} WHERE _c2 = '华东'"

        with patch(
            "fusion_rag.engine.xlsx_query.XlsxQueryPlanner.generate_sql",
            new_callable=AsyncMock,
            side_effect=plan,
        ), patch(
            "fusion_rag.engine.xlsx_query.XlsxQueryPlanner.compose_answer",
            new_callable=AsyncMock,
            return_value="华东 keyboard sales total 2000, from sheet 销售.",
        ):
            resp = client.post(
                f"/kb/bases/{kb_id}/xlsx/query",
                json={"question": "Q3 华东区键盘销售额合计", "answer": True},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["columns"] == ["_c1", "total"]
        assert body["rows"] == [["键盘", 2000]]
        assert body["truncated"] is False
        assert body["citations"], body
        assert body["citations"][0]["doc_id"] == doc_id
        assert body["citations"][0]["sheet_name"] == "销售"
        assert body["answer"] == "华东 keyboard sales total 2000, from sheet 销售."

    def test_limit_enforced_when_planner_omits_it(self, client, kb_with_xlsx):
        kb_id, _doc_id = kb_with_xlsx

        def plan(ddl, question):
            tname = TestQueryPath._table_for(ddl, "销售")
            return f"SELECT _c1 FROM {tname}"

        with patch(
            "fusion_rag.engine.xlsx_query.XlsxQueryPlanner.generate_sql",
            new_callable=AsyncMock,
            side_effect=plan,
        ):
            resp = client.post(
                f"/kb/bases/{kb_id}/xlsx/query",
                json={"question": "all products", "max_rows": 2},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["row_count"] == 2
        assert body["truncated"] is True
        assert "LIMIT 3" in body["sql"]

    def test_bad_sql_generation_400(self, client, kb_with_xlsx):
        kb_id, _doc_id = kb_with_xlsx
        with patch(
            "fusion_rag.engine.xlsx_query.XlsxQueryPlanner.generate_sql",
            new_callable=AsyncMock,
            return_value="DELETE FROM nothing",
        ):
            resp = client.post(f"/kb/bases/{kb_id}/xlsx/query", json={"question": "boom"})
        assert resp.status_code == 400, resp.text
        assert "rejected" in resp.json()["detail"]

    def test_execution_error_400(self, client, kb_with_xlsx):
        kb_id, _doc_id = kb_with_xlsx
        with patch(
            "fusion_rag.engine.xlsx_query.XlsxQueryPlanner.generate_sql",
            new_callable=AsyncMock,
            return_value="SELECT _c99 FROM t_does_not_exist",
        ):
            resp = client.post(f"/kb/bases/{kb_id}/xlsx/query", json={"question": "boom"})
        assert resp.status_code == 400, resp.text

    def test_llm_unavailable_502(self, client, kb_with_xlsx):
        kb_id, _doc_id = kb_with_xlsx
        with patch(
            "fusion_rag.engine.xlsx_query.XlsxQueryPlanner.generate_sql",
            new_callable=AsyncMock,
            side_effect=LLMUnavailable("mlx down"),
        ):
            resp = client.post(f"/kb/bases/{kb_id}/xlsx/query", json={"question": "q"})
        assert resp.status_code == 502, resp.text

    def test_missing_question_400(self, client, kb_with_xlsx):
        kb_id, _doc_id = kb_with_xlsx
        assert client.post(f"/kb/bases/{kb_id}/xlsx/query", json={}).status_code == 400

    def test_no_sheets_404(self, client):
        kb_id = client.post("/kb/bases", json={"name": "kb2"}).json()["id"]
        resp = client.post(f"/kb/bases/{kb_id}/xlsx/query", json={"question": "q"})
        assert resp.status_code == 404

    def test_answer_failure_degrades_not_500(self, client, kb_with_xlsx):
        kb_id, _doc_id = kb_with_xlsx

        def plan(ddl, question):
            tname = TestQueryPath._table_for(ddl, "销售")
            return f"SELECT COUNT(*) FROM {tname}"

        with patch(
            "fusion_rag.engine.xlsx_query.XlsxQueryPlanner.generate_sql",
            new_callable=AsyncMock,
            side_effect=plan,
        ), patch(
            "fusion_rag.engine.xlsx_query.XlsxQueryPlanner.compose_answer",
            new_callable=AsyncMock,
            side_effect=LLMUnavailable("down"),
        ):
            resp = client.post(
                f"/kb/bases/{kb_id}/xlsx/query", json={"question": "count", "answer": True}
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["rows"] == [[3]]
        assert body["answer"] is None


class TestLifecycle:
    def test_delete_doc_drops_sheet_tables(self, client, kb_with_xlsx):
        kb_id, doc_id = kb_with_xlsx
        assert client.delete(f"/kb/bases/{kb_id}/documents/{doc_id}").status_code == 200
        assert client.get(f"/kb/bases/{kb_id}/xlsx/sheets").status_code == 404
