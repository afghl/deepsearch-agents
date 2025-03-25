from abc import abstractmethod
import asyncio
import contextvars
from dataclasses import dataclass, field
import json
import time
from typing import Any, Callable, Dict, List, Optional, cast

from agents import (
    Agent,
    AgentHooks,
    ModelSettings,
    RunContextWrapper,
    Tool,
)

from deepsearch_agents import conf
from deepsearch_agents.log import logger
from deepsearch_agents.context import TaskContext, Task
from deepsearch_agents.tools import get_tool_instructions


def _build_instructions_and_tools(
    ctx: RunContextWrapper[TaskContext], agent: Agent[TaskContext]
) -> str:
    tool_names = "\n".join([f"{i}. {tool.name}" for i, tool in enumerate(agent.tools)])
    curr = ctx.context.current_task()
    if curr.query == curr.origin_query:
        question = curr.query
    else:
        question = f"The Original Question is: {curr.origin_query}\n And you are currently focusing on this aspect of it. \n You are trying to answer this question: {curr.query}"

    return f"""
Current Date: {ctx.context.start_date_time}

You are an advanced AI research agent from Deepsearch AI. You are specialized in multistep reasoning. 
Using your best knowledge, conversation with the user and lessons learned, answer the user question with absolute certainty.

-Goal-
Given a question in any domain, do research to find the answer. Provide a detailed, comprehensive and factually accurate answer.

-Rules-
1. Think step by step, choose the action carefully.
2. ALWAYS show your thinking process before taking any action. Explain why you think it helps towards the end goal.
3. You job is to provide the best answer, So the conversation does not end until the user is satisfied with the answer.

-Question-
{question}

Using your best knowledge, conversation with the user and lessons learned, answer the user question with absolute certainty.

Here's the actions provided. YOU CAN ONLY chose one of these actions. DON'T use any other actions not listed here.

-Available actions-
{tool_names}

Below are the details documentation of each action.

-Action details-
{get_tool_instructions(ctx.context, agent.tool_names)}

"""


class PlannerHooks(AgentHooks[TaskContext]):

    @abstractmethod
    async def on_new_task_generated(
        self, context: RunContextWrapper[TaskContext], agent: "Planner", task: Task
    ) -> None:
        pass


def _hooks(planner_hooks: PlannerHooks | None) -> AgentHooks[TaskContext] | None:
    """
    This function creates a new hooks instance that extends the original hooks with the planner hooks.
    """
    if not planner_hooks:
        return None

    async def on_tool_end(
        context: RunContextWrapper[TaskContext],
        agent: Agent[TaskContext],
        tool: Tool,
        result: str,
    ) -> None:
        # First call the original hooks' on_tool_end
        await planner_hooks.on_tool_end(context, agent, tool, result)

        planner = cast(Planner, agent)
        if tool.name == planner.task_generator and result:
            new_tasks = planner._build_new_tasks(context, result)
            await asyncio.gather(
                *[
                    planner_hooks.on_new_task_generated(context, planner, task)
                    for task in new_tasks
                ]
            )

    hooks_instance = AgentHooks[TaskContext]()
    hooks_instance.on_start = planner_hooks.on_start
    hooks_instance.on_end = planner_hooks.on_end
    hooks_instance.on_handoff = planner_hooks.on_handoff
    hooks_instance.on_tool_start = planner_hooks.on_tool_start
    hooks_instance.on_tool_end = on_tool_end
    return hooks_instance


@dataclass
class Planner(Agent[TaskContext]):
    """
    A Planner agent that manages task planning and execution .
    The Planner is responsible for breaking down complex tasks into subtasks,
    managing the execution flow, and coordinating tool usage.

    The Planner uses a task generation tool to create subtasks from a given task.
    """

    task_generator: str | None = None
    """
    Optional string identifier for the task generation tool.
        
    When specified, the Planner can use this tool to generate subtasks from a given task. If None, task decomposition will not be available.
    """

    def __init__(
        self,
        name: str,
        tools: List[Tool],
        task_generator: str | None = None,
        planner_hooks: PlannerHooks | None = None,
        hooks: AgentHooks[TaskContext] | None = None,
        model: str | None = None,
        model_settings: ModelSettings | None = None,
    ):
        super().__init__(
            name=name,
            instructions=_build_instructions_and_tools,
            tools=tools,
            hooks=_hooks(planner_hooks) if planner_hooks else hooks,
            model=model,
            model_settings=model_settings,
        )
        self.task_generator = task_generator
        self.all_tools = tools

    def rebuild_tools(
        self, ctx: RunContextWrapper[TaskContext], last_used: str | None = None
    ) -> None:
        """
        This method updates the available tools for the Planner based on the current context and the last used tool.
        The method filters out tools that should not be available for the current task. Specifically, it excludes:
        - The last used tool to avoid immediate repetition.
        - The task generator tool if the current task depth exceeds the maximum allowed depth.

        The filtered list of tools is then assigned to the Planner's tools attribute.
        """

        def forbid(name: str) -> bool:
            return name == last_used or (
                ctx.context.current_task().level
                >= conf.get_configuration().excution_config.max_task_depth
                and name == self.task_generator
            )

        available = [tool for tool in self.all_tools if not forbid(tool.name)]
        self.tools = available

    def _build_new_tasks(
        self, ctx: RunContextWrapper[TaskContext], result: str
    ) -> List[Task]:
        """
        Build new tasks from the result.
        """

        logger.info(f"Build new tasks from result: {result}")
        tasks = []
        question_list = result.split("|")
        curr = ctx.context.current_task()
        if isinstance(question_list, str):
            question_list = json.loads(question_list)

        cnt = 0
        for q in question_list:
            task = Task(
                id=f"{curr.id}_{cnt+1}",
                origin_query=curr.origin_query,
                query=q,
                level=curr.level + 1,
                parent=curr,
            )
            cnt += 1
            logger.info(
                f"curr: {curr.id} Create new task: {task.id}, ctx: {ctx.context.current_task_id()}"
            )
            curr.sub_tasks[task.id] = task
            ctx.context.tasks[task.id] = task
            tasks.append(task)
        return tasks

    @property
    def tool_names(self) -> List[str]:
        return [tool.name for tool in self.tools]
