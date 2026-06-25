from typing import Any, Callable


BUILTIN_TOOLS: dict[str, dict[str, Any]] = {}


def get_tool_spec() -> list[dict[str, Any]]:
    return [
        tool["config"]
        for tool in BUILTIN_TOOLS.values()
    ]


def has_tool(name: str) -> bool:
    return name in BUILTIN_TOOLS


def execute(
    name: str,
    arguments: dict[str, Any],
) -> Any:
    if name not in BUILTIN_TOOLS:
        raise KeyError(
            f"Unknown built-in tool: {name}"
        )

    run_function: Callable = BUILTIN_TOOLS[name]["run"]
    return run_function(arguments)