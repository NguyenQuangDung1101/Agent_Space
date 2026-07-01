from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

from database.knowledge_repository import get_knowledge_repository
from share.schemas import ModificationPlan


MUTATING_TOOLS = {
    "apply_modification_plan",
    "node_create",
    "relationship_create",
    "node_delete",
    "relationship_delete",
}


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
            text = "\n".join(
                ", ".join(row) for row in csv.reader(file)
            )
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


def _save_plan(arguments: dict[str, Any]) -> dict[str, Any]:
    return ModificationPlan.model_validate(arguments["plan"]).model_dump(
        mode="json"
    )


def _apply_plan(arguments: dict[str, Any]) -> dict[str, Any]:
    plan = ModificationPlan.model_validate(arguments["plan"])
    return get_knowledge_repository().apply_plan(plan).model_dump(mode="json")


def _node_create(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_knowledge_repository().create_node(
        node_id=arguments["id"],
        kind=arguments["kind"],
        name=arguments["name"],
        description=arguments.get("description", ""),
        properties=arguments.get("properties", {}),
        parent_id=arguments.get("parent_id"),
    )


def _relationship_create(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_knowledge_repository().create_relationship(
        arguments["source_id"],
        arguments["target_id"],
        arguments["type"],
        arguments.get("properties", {}),
    )


def _node_delete(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_knowledge_repository().delete_node(arguments["id"])


def _relationship_delete(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_knowledge_repository().delete_relationship(
        arguments["source_id"],
        arguments["target_id"],
        arguments["type"],
    )


BUILTIN_TOOLS: dict[str, dict[str, Any]] = {
    "read_document": {
        "config": {
            "name": "read_document",
            "description": "Read a local document attachment for knowledge analysis.",
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
    "save_modification_plan": {
        "config": {
            "name": "save_modification_plan",
            "description": "Validate and save an ImportPlan or DeletePlan before review.",
            "inputSchema": {
                "type": "object",
                "properties": {"plan": {"type": "object"}},
                "required": ["plan"],
            },
        },
        "run": _save_plan,
    },
    "apply_modification_plan": {
        "config": {
            "name": "apply_modification_plan",
            "description": "Apply an approved modification plan and validate the result.",
            "inputSchema": {
                "type": "object",
                "properties": {"plan": {"type": "object"}},
                "required": ["plan"],
            },
        },
        "run": _apply_plan,
    },
    "node_create": {
        "config": {
            "name": "node_create",
            "description": "Create a protected KnowledgeDocument, KnowledgeChunk, or KnowledgeEntity.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"enum": ["KnowledgeDocument", "KnowledgeChunk", "KnowledgeEntity"]},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "properties": {"type": "object"},
                    "parent_id": {"type": ["string", "null"]},
                },
                "required": ["id", "kind", "name"],
            },
        },
        "run": _node_create,
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
        "run": _relationship_create,
    },
    "node_delete": {
        "config": {
            "name": "node_delete",
            "description": "Delete a node connected to Knowledge. The Knowledge root is protected.",
            "inputSchema": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
        "run": _node_delete,
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
        "run": _relationship_delete,
    },
    "validate_knowledge": {
        "config": {
            "name": "validate_knowledge",
            "description": "Validate that all Knowledge-domain nodes remain connected to the protected root.",
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
