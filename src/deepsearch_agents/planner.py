import asyncio
import contextvars
from dataclasses import dataclass, field
import json
import time
from typing import Callable, Dict, List

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
from deepsearch_agents.conf import OPENAI_API_KEY, OPENAI_BASE_URL
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
        # self.executors: List[Executor] = []
        self.tools = [
            # search,
            reflect,
            # visit,
            answer,
        ]
        self.all_tools = self.tools
        self.hooks = PlannerHooks()

    def _rebuild_tools(
        self, ctx: RunContextWrapper[TaskContext], last_used: str
    ) -> None:
        """
        Rebuild the tools with the current context.
        """
        # last_used = ctx.context.last_used_tool
        self.tools = [tool for tool in self.all_tools if tool.name != last_used]

    def _build_new_tasks(
        self, ctx: RunContextWrapper[TaskContext], result: list[str]
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
        question_list = result.split("seq|seq")
        curr = ctx.context.current_task()
        print(
            f"curr: {curr.id}, question_list: {question_list}, type of question_list: {type(question_list)}, len: {len(question_list)}"
        )
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
    async def on_tool_end(self, context, agent, tool, result):
        agent._rebuild_tools(context, tool.name)
        if tool.name == agent.task_generator_tool_name and result:
            new_tasks = agent._build_new_tasks(context, result)
            for t in new_tasks:
                if hasattr(agent, "on_new_task_generated"):
                    asyncio.create_task(agent.on_new_task_generated(context, t))


def _build_instructions_and_tools(
    ctx: RunContextWrapper[TaskContext], agent: Agent[TaskContext]
) -> str:
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
