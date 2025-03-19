from typing import Callable, Dict, List
from agents import RunContextWrapper
from deepsearch_agents.context import TaskContext


tool_instructions: Dict[str, Callable[[TaskContext], str]] = {}


def get_tool_instructions(ctx: TaskContext, tool_names: List[str]) -> str:
    """
    Get the instructions for the tools.
    """
    instructions = []
    for tool_name in tool_names:
        if tool_name in tool_instructions:
            instructions.append(
                f""" <action-{tool_name}>
                                {tool_instructions[tool_name](ctx)}"""
            )
    return "\n".join(instructions)
