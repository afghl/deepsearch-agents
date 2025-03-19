import contextvars
from dataclasses import dataclass
import time
from typing import Dict, List

from agents import Agent, RunContextWrapper, Runner

from deepsearch_agents.agent import answer, reflect, search, visit
from deepsearch_agents.context import Task, TaskContext


@dataclass
class Manager(Agent[TaskContext]):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.instructions = _build_instructions_and_tools
        self.start_datatime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.executors = []
        self.tools = [
            search,
            reflect,
            visit,
            answer,
        ]

    def tool_names(self) -> List[str]:
        return [tool.name for tool in self.tools]

    def _rebuild_tools(self, ctx: RunContextWrapper[TaskContext]) -> None:
        """
        Rebuild the tools with the current context.
        """
        # last_used = ctx.context.last_used_tool
        last_used = "answer"
        if last_used is not None:
            # remove last used tool from the list
            self.tools = [tool for tool in self.tools if tool.name != last_used]
            print(
                f"Rebuild tools: {self.tool_names()}, remove last used tool: {last_used}"
            )


def _build_instructions_and_tools(
    ctx: RunContextWrapper[TaskContext], agent: Agent[TaskContext]
) -> str:
    agent._rebuild_tools(ctx)
    return f"""
Current Date: {agent.start_datatime}

You are an advanced AI research agent from DeepSearch AI. You are specialized in multistep reasoning. 
Using your best knowledge, conversation with the user and lessons learned, answer the user question with absolute certainty.
"""
