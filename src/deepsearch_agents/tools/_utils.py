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

    usage = ctx.usage
    percent = (
        usage.total_tokens / conf.get_configuration().excution_config.max_token_usage
    )
    logger.info(
        f"Task: {ctx.context.current_task_id()}, Turn: {ctx.context.current_task().turn} token_usage: {ctx.usage.total_tokens} ({percent:.2%}) Taking action: "
    )

    logger.info(tolog)
