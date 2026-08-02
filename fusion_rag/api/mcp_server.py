"""MCP Server — Model Context Protocol server for Fusion-RAG.

callers: External MCP clients (Claude Desktop, Cursor, etc.)
API: MCP tool definitions for kb_list, kb_search, kb_ask, kb_create
schema: JSON-RPC 2.0 protocol, tools following MCP specification
user instruction: "按照你的方案和计划落地所有phase阶段的需求"
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["mcp"])

MCP_TOOLS = [
    {
        "name": "kb_list",
        "description": "List all knowledge bases",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "kb_create",
        "description": "Create a new knowledge base",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Knowledge base name"},
                "description": {"type": "string", "description": "Optional description"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "kb_search",
        "description": "Search for relevant chunks in a knowledge base",
        "inputSchema": {
            "type": "object",
            "properties": {
                "kb_id": {"type": "string", "description": "Knowledge base ID"},
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Max results (default 5)"},
            },
            "required": ["kb_id", "query"],
        },
    },
    {
        "name": "kb_ask",
        "description": "Ask a question and get an AI-generated answer with sources",
        "inputSchema": {
            "type": "object",
            "properties": {
                "kb_id": {"type": "string", "description": "Knowledge base ID"},
                "question": {"type": "string", "description": "Question to ask"},
                "top_k": {"type": "integer", "description": "Max context chunks (default 5)"},
            },
            "required": ["kb_id", "question"],
        },
    },
    {
        "name": "kb_upload",
        "description": "Upload a document to a knowledge base",
        "inputSchema": {
            "type": "object",
            "properties": {
                "kb_id": {"type": "string", "description": "Knowledge base ID"},
                "file_path": {"type": "string", "description": "Path to document file"},
            },
            "required": ["kb_id", "file_path"],
        },
    },
    {
        "name": "kb_status",
        "description": "Get knowledge base statistics",
        "inputSchema": {
            "type": "object",
            "properties": {
                "kb_id": {"type": "string", "description": "Knowledge base ID"},
            },
            "required": ["kb_id"],
        },
    },
]


@router.post("")
async def mcp_handler(request: Request) -> JSONResponse:
    """Handle MCP JSON-RPC requests."""
    body = await request.json()
    method = body.get("method", "")
    request_id = body.get("id")
    params = body.get("params", {})

    if method == "initialize":
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "fusion-rag", "version": "0.3.0"},
                },
            }
        )

    if method == "tools/list":
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": MCP_TOOLS},
            }
        )

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        result = await _dispatch_tool(tool_name, arguments)
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]},
            }
        )

    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    )


async def _dispatch_tool(name: str, args: dict) -> Any:
    """Dispatch MCP tool calls to internal API."""
    from ..store.vector_store import VectorStore
    from .routes import _get_base, _get_embed_client, _get_kb_manager

    try:
        if name == "kb_list":
            return _get_kb_manager().list()

        if name == "kb_create":
            kb = _get_kb_manager().create(
                name=args["name"],
                description=args.get("description", ""),
            )
            return {"id": kb.id, "name": kb.config.name, "status": "created"}

        if name == "kb_search":
            kb_id = args["kb_id"]
            kb = _get_base(kb_id)
            embed = _get_embed_client()
            vec_store = VectorStore(kb.vector_path)
            query_vector = await embed.embed(args["query"])
            if not query_vector or all(v == 0.0 for v in query_vector):
                return {"error": "Embedding failed"}
            top_k = args.get("top_k", kb.config.max_results)
            results = vec_store.search(query_vector, top_k=top_k)
            return results

        if name == "kb_ask":
            from .routes import _generate_answer

            kb_id = args["kb_id"]
            kb = _get_base(kb_id)
            embed = _get_embed_client()
            vec_store = VectorStore(kb.vector_path)
            question = args["question"]
            query_vector = await embed.embed(question)
            if not query_vector or all(v == 0.0 for v in query_vector):
                return {"error": "Embedding failed"}
            top_k = args.get("top_k", kb.config.max_results)
            chunks = vec_store.search(query_vector, top_k=top_k)
            if not chunks:
                return {"answer": "No relevant documents found.", "sources": []}
            context = "\n\n".join(f"[{c['doc_name']}] {c['text'][:2000]}" for c in chunks)
            return await _generate_answer(question, context, chunks)

        if name == "kb_upload":
            from .routes import upload_document

            return await upload_document(args["kb_id"], args)

        if name == "kb_status":
            kb = _get_base(args["kb_id"])
            return kb.to_dict()

        return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        logger.error("MCP tool '%s' failed: %s", name, e)
        return {"error": str(e)}
