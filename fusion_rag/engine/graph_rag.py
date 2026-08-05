"""Lightweight GraphRAG — entity extraction and relationship-aware retrieval.

callers: routes.py ask endpoint, KnowledgeBase advanced search
API: GraphRAG.extract_entities(text), .build_graph(chunks), .search(query, graph)
schema: entities table (id, name, type), relations table (source, target, label)
user instruction: "按照你的方案和计划落地所有phase阶段的需求"
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

ENTITY_PROMPT = (
    "Extract entities and their relationships from the following text. "
    "Output ONLY a JSON object with two keys:\n"
    '- "entities": array of {{"name": str, "type": str}} where type is one of: '
    "PERSON, ORG, LOCATION, CONCEPT, EVENT, PRODUCT\n"
    '- "relations": array of {{"source": str, "target": str, "label": str}}\n\n'
    "Text: {text}\n\nJSON:"
)


class GraphRAG:
    """Lightweight graph-based RAG with entity extraction via LLM."""

    def __init__(self, db_path: str = "", mlx_url: str = "http://127.0.0.1:11432/v1", model: str = "qwen3.5-9b"):
        if not db_path:
            db_path = str(Path.home() / ".fusion-rag" / "graph.db")
        self.db_path = db_path
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'CONCEPT',
                chunk_id TEXT NOT NULL DEFAULT '',
                kb_id TEXT NOT NULL DEFAULT '',
                UNIQUE(name, type, chunk_id)
            );
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_kb ON entities(kb_id);
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                label TEXT NOT NULL DEFAULT 'related_to',
                kb_id TEXT NOT NULL DEFAULT '',
                UNIQUE(source, target, label, kb_id)
            );
            CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source);
            CREATE INDEX IF NOT EXISTS idx_relations_kb ON relations(kb_id);
        """)
        conn.commit()
        conn.close()

    async def extract_entities(self, text: str) -> dict[str, Any]:
        """Extract entities and relations from text via LLM."""
        if not text.strip():
            return {"entities": [], "relations": []}
        prompt = ENTITY_PROMPT.format(text=text[:3000])
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.mlx_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 500,
                        "temperature": 0.0,
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                return self._parse_extraction(content)
        except Exception as e:
            logger.warning("Entity extraction failed: %s", e)
            return {"entities": [], "relations": []}

    async def build_graph(self, chunks: list[dict], kb_id: str = "") -> dict[str, int]:
        """Extract entities from chunks and store in graph DB."""
        total_entities = 0
        total_relations = 0
        conn = self._get_conn()
        try:
            for chunk in chunks:
                text = chunk.get("text", "")
                chunk_id = chunk.get("id", "")
                result = await self.extract_entities(text)
                for ent in result.get("entities", []):
                    try:
                        conn.execute(
                            "INSERT OR IGNORE INTO entities (name, type, chunk_id, kb_id) VALUES (?, ?, ?, ?)",
                            (ent.get("name", ""), ent.get("type", "CONCEPT"), chunk_id, kb_id),
                        )
                        total_entities += 1
                    except Exception as e:
                        logger.warning("Failed to insert entity: %s", e)
                for rel in result.get("relations", []):
                    try:
                        conn.execute(
                            "INSERT OR IGNORE INTO relations (source, target, label, kb_id) VALUES (?, ?, ?, ?)",
                            (rel.get("source", ""), rel.get("target", ""), rel.get("label", "related_to"), kb_id),
                        )
                        total_relations += 1
                    except Exception as e:
                        logger.warning("Failed to insert relation: %s", e)
            conn.commit()
        finally:
            conn.close()
        logger.info(
            "GraphRAG: built graph with %d entities, %d relations for kb=%s", total_entities, total_relations, kb_id
        )
        return {"entities": total_entities, "relations": total_relations}

    def search(self, query: str, kb_id: str = "", max_hops: int = 2) -> list[dict[str, Any]]:
        """Find entities matching query and expand via graph relations."""
        conn = self._get_conn()
        try:
            # Find matching entities
            if kb_id:
                rows = conn.execute(
                    "SELECT name, type, chunk_id FROM entities WHERE name LIKE ? AND kb_id = ?",
                    (f"%{query}%", kb_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT name, type, chunk_id FROM entities WHERE name LIKE ?",
                    (f"%{query}%",),
                ).fetchall()

            if not rows:
                return []

            # Expand via relations
            entity_names = [r["name"] for r in rows]
            chunk_ids = {r["chunk_id"] for r in rows if r["chunk_id"]}
            visited = set(entity_names)
            current = entity_names

            for _ in range(max_hops):
                next_entities = []
                for name in current:
                    if kb_id:
                        rels = conn.execute(
                            "SELECT source, target FROM relations WHERE (source = ? OR target = ?) AND kb_id = ?",
                            (name, name, kb_id),
                        ).fetchall()
                    else:
                        rels = conn.execute(
                            "SELECT source, target FROM relations WHERE source = ? OR target = ?",
                            (name, name),
                        ).fetchall()
                    for rel in rels:
                        for field in ("source", "target"):
                            neighbor = rel[field]
                            if neighbor not in visited:
                                visited.add(neighbor)
                                next_entities.append(neighbor)
                if not next_entities:
                    break

                # Find chunk_ids for new entities
                for name in next_entities:
                    if kb_id:
                        e_rows = conn.execute(
                            "SELECT chunk_id FROM entities WHERE name = ? AND kb_id = ?",
                            (name, kb_id),
                        ).fetchall()
                    else:
                        e_rows = conn.execute(
                            "SELECT chunk_id FROM entities WHERE name = ?",
                            (name,),
                        ).fetchall()
                    for e in e_rows:
                        if e["chunk_id"]:
                            chunk_ids.add(e["chunk_id"])
                current = next_entities

            return [{"chunk_id": cid, "entity_count": len(visited)} for cid in chunk_ids]
        finally:
            conn.close()

    def get_entity_count(self, kb_id: str = "") -> int:
        conn = self._get_conn()
        try:
            if kb_id:
                row = conn.execute("SELECT COUNT(*) as cnt FROM entities WHERE kb_id = ?", (kb_id,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) as cnt FROM entities").fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    def get_relation_count(self, kb_id: str = "") -> int:
        conn = self._get_conn()
        try:
            if kb_id:
                row = conn.execute("SELECT COUNT(*) as cnt FROM relations WHERE kb_id = ?", (kb_id,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) as cnt FROM relations").fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    @staticmethod
    def _parse_extraction(content: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"entities": [], "relations": []}
