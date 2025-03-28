import re
from typing import Any, Callable, Dict, List

from agents import RunContextWrapper
from pydantic import BaseModel
from deepsearch_agents import conf
from deepsearch_agents.context import TaskContext
from deepsearch_agents.log import logger


tool_instructions: Dict[str, Callable[[TaskContext | None], str]] = {}


def get_tool_instructions(ctx: TaskContext, tool_names: List[str]) -> str:
    """
    Get the instructions for the tools.
    """
    instructions = []
    for tool_name in tool_names:
        if tool_name in tool_instructions:
            instructions.append(
                f"""<action-{tool_name}>
    {tool_instructions[tool_name](ctx)}
</action-{tool_name}>"""
            )
    return "\n\n".join(instructions)


def track_usage(
    func: Callable[[RunContextWrapper[TaskContext], Any], Any],
) -> Callable[[RunContextWrapper[TaskContext], Any], Any]:
    def wrapper(ctx: RunContextWrapper[TaskContext], *args: Any, **kwargs: Any) -> Any:
        result = func(ctx, *args, **kwargs)
        ctx.usage.add(result.usage)
        return result

    return wrapper


def log_action(
    ctx: RunContextWrapper[TaskContext],
    action: str,
    think: str,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Log the action and its arguments.
    """
    tolog = {
        "query": ctx.context.current_task().query,
        "action": action,
        "think": think,
    }
    for k, v in kwargs.items():
        if isinstance(v, list):
            tolog[k] = [str(item) for item in v]
        else:
            tolog[k] = str(v)

    logger.info(
        f"Task: {ctx.context.current_task_id()}, Turn: {ctx.context.current_task().turn}"
    )

    logger.info(tolog)


def remove_markdown_link(content: str) -> str:
    return re.sub(
        r"\[([^\]]*)\]\([^\)]+\)",
        lambda m: m.group(1) if m.group(1) else "",
        content,
    )
