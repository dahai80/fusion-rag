"""Issue #156: standalone attachment file service tests.

Covers: upload (multipart + raw body), list, byte-range segment walk with
the server-side max_bytes clamp, text/binary round-trips (incl. multibyte
char split across segment boundaries), delete + 404s, and the optional
promote-to-KB ingest. The file service is KB-independent — the round trip
never touches a knowledge base.
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app
from fusion_rag.embed.client import EmbeddingClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Isolated data dir -> file blobs + files.db land under tmp_path.
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


class TestFileService:
    def test_roundtrip_upload_segment_delete_no_kb(self, client):
        # Acceptance: upload -> segment-walk -> delete with NO knowledge base.
        text = "hello fusion file service " * 40
        resp = client.post("/api/files?filename=note.txt", content=text.encode("utf-8"))
        assert resp.status_code == 200, resp.text
        meta = resp.json()
        assert meta["filename"] == "note.txt"
        assert meta["size"] == len(text.encode("utf-8"))
        assert meta["media_kind"] == "text"
        assert len(meta["checksum"]) == 64

        listed = client.get("/api/files").json()
        assert listed["count"] == 1
        assert listed["files"][0]["file_id"] == meta["file_id"]

        # Walk segments with a tiny cap and reassemble byte-exactly.
        pieces = []
        offset = 0
        while True:
            seg = client.get(
                f"/api/files/{meta['file_id']}/segments",
                params={"offset": offset, "max_bytes": 64},
            )
            assert seg.status_code == 200, seg.text
            body = seg.json()
            assert body["total"] == meta["size"]
            assert body["media_kind"] == "text"
            pieces.append(body["content"])
            assert body["next_offset"] > offset or body["eof"]
            offset = body["next_offset"]
            if body["eof"]:
                break
        assert "".join(pieces).encode("utf-8") == text.encode("utf-8")

        assert client.delete(f"/api/files/{meta['file_id']}").json()["deleted"] is True
        assert client.get("/api/files").json()["count"] == 0
        assert client.delete(f"/api/files/{meta['file_id']}").status_code == 404

    def test_multipart_upload(self, client):
        resp = client.post(
            "/api/files",
            files={"file": ("report.txt", b"multipart body content", "text/plain")},
        )
        assert resp.status_code == 200, resp.text
        meta = resp.json()
        assert meta["filename"] == "report.txt"
        assert meta["media_kind"] == "text"
        seg = client.get(f"/api/files/{meta['file_id']}/segments").json()
        assert seg["content"] == "multipart body content"
        assert seg["eof"] is True

    def test_binary_roundtrip_base64(self, client):
        # Acceptance: binary content round-trips correctly (base64 segments).
        blob = bytes(range(256)) * 3
        resp = client.post("/api/files?filename=blob.bin", content=blob)
        meta = resp.json()
        assert meta["media_kind"] == "binary"

        pieces = []
        offset = 0
        while True:
            seg = client.get(
                f"/api/files/{meta['file_id']}/segments",
                params={"offset": offset, "max_bytes": 100},
            ).json()
            pieces.append(base64.b64decode(seg["content"]))
            offset = seg["next_offset"]
            if seg["eof"]:
                break
        assert b"".join(pieces) == blob

    def test_max_bytes_clamped_not_error(self, client):
        # Acceptance: oversize max_bytes CLAMPS server-side, never errors.
        resp = client.post("/api/files?filename=big.txt", content=b"x" * 1000)
        file_id = resp.json()["file_id"]
        seg = client.get(
            f"/api/files/{file_id}/segments", params={"offset": 0, "max_bytes": 10_000_000}
        )
        assert seg.status_code == 200, seg.text
        assert seg.json()["next_offset"] <= 1000
        # Zero/negative also clamp to the cap instead of misbehaving.
        seg = client.get(
            f"/api/files/{file_id}/segments", params={"offset": 0, "max_bytes": 0}
        )
        assert seg.status_code == 200
        client.delete(f"/api/files/{file_id}")

    def test_multibyte_char_split_across_segments(self, client):
        # A segment boundary inside a multibyte char must not corrupt it —
        # the partial bytes are re-served by the next segment.
        text = "a" * 63 + "融合平台" + "b" * 50
        raw = text.encode("utf-8")
        resp = client.post("/api/files?filename=cjk.txt", content=raw)
        file_id = resp.json()["file_id"]
        pieces = []
        offset = 0
        while True:
            seg = client.get(
                f"/api/files/{file_id}/segments",
                params={"offset": offset, "max_bytes": 64},
            ).json()
            pieces.append(seg["content"].encode("utf-8"))
            assert seg["next_offset"] <= len(raw)
            offset = seg["next_offset"]
            if seg["eof"]:
                break
        assert b"".join(pieces) == raw

    def test_offset_past_eof(self, client):
        resp = client.post("/api/files?filename=small.txt", content=b"12345")
        file_id = resp.json()["file_id"]
        seg = client.get(
            f"/api/files/{file_id}/segments", params={"offset": 999}
        ).json()
        assert seg["content"] == ""
        assert seg["eof"] is True
        assert seg["next_offset"] == 999
        client.delete(f"/api/files/{file_id}")

    def test_unknown_file_404(self, client):
        assert client.get(f"/api/files/{'0' * 32}/segments").status_code == 404
        assert client.delete(f"/api/files/{'0' * 32}").status_code == 404
        # Malformed ids never reach the blob path.
        assert client.get("/api/files/../../etc/passwd/segments").status_code in (404, 400)
        assert client.delete("/api/files/not-a-hex-id").status_code == 404

    def test_empty_upload_rejected(self, client):
        resp = client.post("/api/files?filename=empty.txt", content=b"")
        assert resp.status_code == 400

    def test_ingest_promotes_into_kb(self, client):
        # Optional promote: file -> existing KB pipeline (parse -> chunk ->
        # index); embed calls are mocked, the rest is real.
        content = "fusion file promotion test document with enough text to chunk properly " * 3
        file_id = client.post("/api/files?filename=promo.txt", content=content.encode("utf-8")).json()[
            "file_id"
        ]
        kb_id = client.post("/kb/bases", json={"name": "file-promo-kb"}).json()["id"]
        resp = client.post(f"/api/files/{file_id}/ingest", json={"kb_id": kb_id, "contextualize": False})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["file_id"] == file_id
        assert body["filename"] == "promo.txt"
        assert body["chunks"] > 0
        # The promoted doc is retrievable through the normal KB surface.
        docs = client.get(f"/kb/bases/{kb_id}/documents").json()
        assert isinstance(docs, list) and len(docs) == 1
        # Ingesting a missing file into a KB 404s.
        assert (
            client.post(f"/api/files/{'0' * 32}/ingest", json={"kb_id": kb_id}).status_code
            == 404
        )
        # Missing kb_id 400s.
        assert client.post(f"/api/files/{file_id}/ingest", json={}).status_code == 400

    def test_subpath_mount_serves_files(self, client):
        # Console subpath parity: /shell/fusion-rag/api/files must answer.
        resp = client.post(
            "/shell/fusion-rag/api/files?filename=sub.txt", content=b"subpath"
        )
        assert resp.status_code == 200, resp.text
        file_id = resp.json()["file_id"]
        assert client.get("/shell/fusion-rag/api/files").json()["count"] == 1
        client.delete(f"/shell/fusion-rag/api/files/{file_id}")
        assert client.get("/shell/fusion-rag/api/files").json()["count"] == 0
