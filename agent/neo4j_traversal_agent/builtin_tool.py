from __future__ import annotations

from typing import Any, Callable

from database.knowledge_repository import get_knowledge_repository


BUILTIN_TOOLS: dict[str, dict[str, Any]] = {
    "get_anchor_node": {
        "config": {
            "name": "get_anchor_node",
            "description": (
                "Find the most relevant Knowledge-domain nodes with hybrid "
                "lexical and Ollama embedding similarity."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
            },
        },
        "run": lambda args: get_knowledge_repository().get_anchor_node(
            args["query"], args.get("top_k", 5)
        ),
    },
    "k_hop_expansion": {
        "config": {
            "name": "k_hop_expansion",
            "description": "Expand from Knowledge-domain node IDs by a bounded hop count.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "node_ids": {"type": "array", "items": {"type": "string"}},
                    "hops": {"type": "integer", "minimum": 1, "maximum": 5},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["node_ids"],
            },
        },
        "run": lambda args: get_knowledge_repository().k_hop_expansion(
            args["node_ids"], args.get("hops", 1), args.get("top_k", 30)
        ),
    },
    "path_finding": {
        "config": {
            "name": "path_finding",
            "description": "Find short undirected paths between two Knowledge-domain nodes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                    "max_hops": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["source_id", "target_id"],
            },
        },
        "run": lambda args: get_knowledge_repository().path_finding(
            args["source_id"],
            args["target_id"],
            args.get("top_k", 3),
            args.get("max_hops", 8),
        ),
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
