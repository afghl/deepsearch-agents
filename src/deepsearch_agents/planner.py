import contextvars
from dataclasses import dataclass
import time
from typing import Callable, Dict, List

from agents import Agent, AgentHooks, RunContextWrapper, RunHooks, Runner

from deepsearch_agents.agent import Executor, answer, reflect, search, visit
from deepsearch_agents.context import Task, TaskContext


@dataclass
class Planner(Agent[TaskContext]):
    def __init__(
        self,
        name: str,
        *,
        task_generator_tool_name: str = "task_generator",
        on_new_task_generated: Callable | None = None,
        hooks: RunHooks[TaskContext] = None,
    ):
        super().__init__(name=name)
        self.instructions = _build_instructions_and_tools
        self.start_datatime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.task_generator_tool_name = task_generator_tool_name
        self.on_new_task_generated = on_new_task_generated
        self.executors: List[Executor] = []
        self.tools = [
            search,
            reflect,
            visit,
            answer,
        ]
        self.all_tools = self.tools
        self.hooks = hooks or PlannerHooks()

    def tool_names(self) -> List[str]:
        return [tool.name for tool in self.tools]

    def _rebuild_tools(
        self, ctx: RunContextWrapper[TaskContext], last_used: str
    ) -> None:
        """
        Rebuild the tools with the current context.
        """
        # last_used = ctx.context.last_used_tool
        if last_used is not None:
            self.tools = [tool for tool in self.all_tools if tool.name != last_used]
            print(
                f"Rebuild tools: {self.tool_names()}, remove last used tool: {last_used}"
            )

    def _build_new_tasks(self, result: str) -> List[Task]:
        """
        Build new tasks from the result.
        """
        print(f"Build new tasks from result: {result}")
        return []


# TODO: seperate planner hooks and agent hooks
class PlannerHooks(AgentHooks[TaskContext]):
    async def on_tool_end(self, context, agent, tool, result):
        agent._rebuild_tools(context, tool.name)
        if tool.name == agent.task_generator_tool_name and result:
            for t in agent._build_new_tasks(result):
                agent.on_new_task_generated(context, t)


def _build_instructions_and_tools(
    ctx: RunContextWrapper[TaskContext], agent: Agent[TaskContext]
) -> str:
    return f"""
Current Date: {agent.start_datatime}

You are an advanced AI research agent from DeepSearch AI. You are specialized in multistep reasoning. 
Using your best knowledge, conversation with the user and lessons learned, answer the user question with absolute certainty.
"""
