from typing import Any, Callable, Dict, List

from agents import RunContextWrapper
from pydantic import BaseModel
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
        "task": ctx.context.current_task_id(),
        "action": action,
        "think": think,
    }
    for k, v in kwargs.items():
        if isinstance(v, list):
            tolog[k] = [str(item) for item in v]
        else:
            tolog[k] = str(v)
    logger.info(tolog)
