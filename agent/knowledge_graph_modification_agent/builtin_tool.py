from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

from database.knowledge_repository import get_knowledge_repository


def _read_document(arguments: dict[str, Any]) -> dict[str, Any]:
    path = Path(arguments["path"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".py", ".yaml", ".yml"}:
        text = path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".json":
        text = json.dumps(
            json.loads(path.read_text(encoding="utf-8")),
            ensure_ascii=False,
            indent=2,
        )
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", errors="replace") as file:
            text = "\n".join(", ".join(row) for row in csv.reader(file))
    elif suffix == ".pdf":
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    elif suffix == ".docx":
        from docx import Document

        text = "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    else:
        raise ValueError(f"Unsupported document type: {suffix}")

    limit = max(1000, min(int(arguments.get("max_chars", 50000)), 200000))
    return {
        "path": str(path),
        "name": path.name,
        "content": text[:limit],
        "truncated": len(text) > limit,
    }


BUILTIN_TOOLS: dict[str, dict[str, Any]] = {
    "read_document": {
        "config": {
            "name": "read_document",
            "description": "Read a local document for Knowledge graph analysis.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 1000},
                },
                "required": ["path"],
            },
        },
        "run": _read_document,
    },
    "inspect_knowledge": {
        "config": {
            "name": "inspect_knowledge",
            "description": "Return a compact view of the external Knowledge graph.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200}
                },
            },
        },
        "run": lambda args: get_knowledge_repository().summary(args.get("limit", 50)),
    },
    "node_create": {
        "config": {
            "name": "node_create",
            "description": "Create or update a KnowledgeDocument, KnowledgeChunk, or KnowledgeEntity.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "kind": {
                        "enum": [
                            "KnowledgeDocument",
                            "KnowledgeChunk",
                            "KnowledgeEntity",
                        ]
                    },
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "properties": {"type": "object"},
                    "parent_id": {"type": ["string", "null"]},
                },
                "required": ["id", "kind", "name"],
            },
        },
        "run": lambda args: get_knowledge_repository().create_node(
            node_id=args["id"],
            kind=args["kind"],
            name=args["name"],
            description=args.get("description", ""),
            properties=args.get("properties", {}),
            parent_id=args.get("parent_id"),
        ),
    },
    "relationship_create": {
        "config": {
            "name": "relationship_create",
            "description": "Create an allowed relationship inside the Knowledge domain.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "type": {"type": "string"},
                    "properties": {"type": "object"},
                },
                "required": ["source_id", "target_id", "type"],
            },
        },
        "run": lambda args: get_knowledge_repository().create_relationship(
            args["source_id"],
            args["target_id"],
            args["type"],
            args.get("properties", {}),
        ),
    },
    "node_delete": {
        "config": {
            "name": "node_delete",
            "description": "Delete a node connected to Knowledge. The root is protected.",
            "inputSchema": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
        "run": lambda args: get_knowledge_repository().delete_node(args["id"]),
    },
    "relationship_delete": {
        "config": {
            "name": "relationship_delete",
            "description": "Delete an allowed relationship inside the Knowledge domain.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "type": {"type": "string"},
                },
                "required": ["source_id", "target_id", "type"],
            },
        },
        "run": lambda args: get_knowledge_repository().delete_relationship(
            args["source_id"],
            args["target_id"],
            args["type"],
        ),
    },
    "validate_knowledge": {
        "config": {
            "name": "validate_knowledge",
            "description": "Validate that Knowledge-domain nodes remain connected to the root.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        "run": lambda args: get_knowledge_repository().validate(),
    },
}


def get_tool_spec() -> list[dict[str, Any]]:
    return [tool["config"] for tool in BUILTIN_TOOLS.values()]


def execute(name: str, arguments: dict[str, Any]) -> Any:
    tool = BUILTIN_TOOLS.get(name)
    if not tool:
        raise KeyError(f"Unknown built-in tool: {name}")
    run_function: Callable = tool["run"]
    return run_function(arguments)
