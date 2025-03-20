import asyncio
import contextvars
from dataclasses import dataclass, field
import json
import time
from typing import Any, Callable, Dict, List, Optional

from agents import (
    Agent,
    AgentHooks,
    OpenAIProvider,
    RunConfig,
    RunContextWrapper,
    RunHooks,
    Runner,
)

from deepsearch_agents._utils import Scope
from deepsearch_agents.conf import MAX_TASK_DEPTH, OPENAI_API_KEY, OPENAI_BASE_URL
from deepsearch_agents.context import Task, TaskContext
from deepsearch_agents.tools import answer, search, visit, reflect
from deepsearch_agents.tools._utils import get_tool_instructions


@dataclass
class Planner(Agent[TaskContext]):
    def __init__(
        self,
        name: str = "DeepSearch Agent",
    ):
        super().__init__(name=name)
        self.instructions = _build_instructions_and_tools
        self.start_datatime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.tools = [
            # search,
            reflect,
            # visit,
            answer,
        ]
        self.all_tools = self.tools
        self.hooks = PlannerHooks()

    def _rebuild_tools(
        self, ctx: RunContextWrapper[TaskContext], last_used: Optional[str] = None
    ) -> None:
        """
        Rebuild the tools with the current context.
        """
        # last_used = ctx.context.last_used_tool

        forbid = []
        if ctx.context.current_task().level >= MAX_TASK_DEPTH:
            forbid.append(self.task_generator_tool_name)
        if last_used and last_used not in forbid:
            forbid.append(last_used)
        new_tools = [tool for tool in self.all_tools if tool.name not in forbid]
        # print(
        #     f"rebuild tools for task: {ctx.context.current_task().id}, level is {ctx.context.current_task().level}, last_used: {last_used}, new_tools: {[t.name for t in new_tools]}"
        # )
        self.tools = new_tools

    def _build_new_tasks(
        self, ctx: RunContextWrapper[TaskContext], result: str
    ) -> List[Task]:
        """
        Build new tasks from the result.
        """
        print(f"Build new tasks from result: {result}")
        # TODO:
        # question_list = json.loads(result)
        # if not isinstance(question_list, list):
        #     raise ValueError(f"Invalid question list: {question_list}")
        tasks = []
        # print(f"Build new tasks from result: {result}, type: {type(result)}")
        question_list = result.split("|")
        curr = ctx.context.current_task()
        if isinstance(question_list, str):
            question_list = json.loads(question_list)

        cnt = 0
        for q in question_list:
            task = Task(
                id=f"{curr.id}_{cnt+1}",
                origin_query=ctx.context.origin_query,
                query=q,
                level=curr.level + 1,
                parent=curr,
            )
            cnt += 1
            print(
                f"curr: {curr.id}Create new task: {task.id}, ctx: {Scope.get_current_task_id()}"
            )
            curr.sub_tasks[task.id] = task
            ctx.context.tasks[task.id] = task
            tasks.append(task)
        return tasks

    @property
    def tool_names(self) -> List[str]:
        return [tool.name for tool in self.tools]


# TODO: seperate planner hooks and agent hooks
class PlannerHooks(AgentHooks[TaskContext]):
    # async def on_tool_start(self, context, agent, tool):
    async def on_start(
        self, context: RunContextWrapper[TaskContext], agent: Agent[TaskContext]
    ) -> None:
        agent._rebuild_tools(context)
        print(
            f"Planner on_start: {context.context.current_task().id}, tools: {agent.tool_names}"
        )

    async def on_tool_end(self, context, a, tool, result):
        agent: Planner[TaskContext] = a
        agent._rebuild_tools(context, tool.name)
        if tool.name != agent.task_generator_tool_name or not result:
            return
        to_wait = []
        new_tasks = agent._build_new_tasks(context, result)
        for t in new_tasks:
            if hasattr(agent, "on_new_task_generated"):
                to_wait.append(
                    asyncio.create_task(agent.on_new_task_generated(context, t))
                )
        await asyncio.gather(*to_wait)


def _build_instructions_and_tools(
    ctx: RunContextWrapper[TaskContext], agent: Agent[TaskContext]
) -> str:
    # TODO: set start_datetime in context
    return f"""
Current Date: {agent.start_datatime}

You are an advanced AI research agent from DeepSearch AI. You are specialized in multistep reasoning. 
Using your best knowledge, conversation with the user and lessons learned, answer the user question with absolute certainty.

Here's the actions provided, read the docs blow. Choose one of the following actions:
<actions>
{get_tool_instructions(ctx.context, agent.tool_names)}
</actions>

Think step by step, choose the action carefully.
"""
