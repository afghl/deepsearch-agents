from abc import abstractmethod
import asyncio
import contextvars
from dataclasses import dataclass, field
import json
from typing import List, cast

from agents import (
    Agent,
    AgentHooks,
    FunctionTool,
    ModelSettings,
    RunContextWrapper,
    Runner,
    Tool,
)

from deepsearch_agents import conf
from deepsearch_agents.log import logger
from deepsearch_agents.context import TaskContext, Task
from deepsearch_agents.tools import get_tool_instructions, sep


def _build_instructions_and_tools(
    ctx: RunContextWrapper[TaskContext], agent: Agent[TaskContext]
) -> str:
    tool_names = "\n".join([f"{i}. {tool.name}" for i, tool in enumerate(agent.tools)])
    curr = ctx.context.current_task()
    if curr.query == curr.origin_query:
        question = f"The Question you are trying to answer is: {curr.query}"
    else:
        question = f"The Original Question is: {curr.origin_query}\n And you are currently focusing on this aspect of it. \n You are trying to answer this question: {curr.query}"

    if not agent._running_out_of_token(ctx):
        return f"""
Current Date: {ctx.context.start_date_time}

You are an advanced AI research agent from Deepsearch AI. You are specialized in multistep reasoning. 
Using your best knowledge, conversation with the user and lessons learned, answer the user question with absolute certainty.

-Goal-

Given a question in any domain, do research to find the answer. Provide a detailed, comprehensive and factually accurate answer.

-Rules-

1. Think step by step, choose the action carefully.
2. ALWAYS show your thinking process before taking any action. Reflect on what you have already known first, and then explain the reason on your next move.
3. No rush to answer the question, Examine the question and the evidence carefully before answering.
4. You job is to provide the best answer, So the conversation does not end until the user is satisfied with the answer.

-Question-
{question}

-Available actions-

Here's the actions provided. YOU CAN ONLY chose one of these actions. DON'T use any other actions not listed here:

{tool_names}

-Action details-

Below are the details documentation of each action.

{get_tool_instructions(ctx.context, agent.tool_names)}

Think step by step, choose the action carefully.
"""
    else:
        logger.info(
            f"We are running out of token, take a best try to answer the question."
        )
        agent.model_settings.tool_choice = "auto"
        return f"""Current Date: {ctx.context.start_date_time}

You are an advanced AI research agent from Deepsearch AI. You are specialized in multistep reasoning. 
Using your best knowledge, conversation with the user and lessons learned, answer the user question with absolute certainty.

-Question-
{curr.origin_query}

-Goals-
- Don't hesitate, just respond!
- Partial responses are fine, but make sure they're well-informed.
- Feel free to refer to our previous conversations if it helps.
- When unsure, base your response on what we know so far.

Let's keep things smooth and on track.

Base on the background information, take a best try, answer the question. 
"""


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
        hooks: AgentHooks[TaskContext] | None = None,
        model: str | None = None,
        model_settings: ModelSettings | None = None,
    ):
        super().__init__(
            name=name,
            instructions=_build_instructions_and_tools,
            tools=tools,
            hooks=hooks,
            model=model,
            model_settings=model_settings,
        )
        self.task_generator = task_generator
        if task_generator:
            self._build_task_generate_tool()
        self.all_tools = self.tools  # type: ignore

    def rebuild_tools(
        self, ctx: RunContextWrapper[TaskContext], last_used: str | None = None
    ) -> None:
        """
        This method updates the available tools for the Planner based on the current context and the last used tool.
        The method filters out tools that should not be available for the current task. Specifically, it excludes:
        - The last used tool to avoid immediate repetition.
        - The task generator tool if the current task depth exceeds the maximum allowed depth.
        - If the agent is running out of tokens, it returns the answer action exclusively.

        The filtered list of tools is then assigned to the Planner's tools attribute.
        """
        if self._running_out_of_token(ctx):
            self.tools = [tool for tool in self.all_tools if tool.name == "answer"]
            return
        config = conf.get_configuration().execution_config
        available = []
        for tool in self.all_tools:
            # Skip if this is the last used tool
            if tool.name == last_used:
                continue
            # Check task generator conditions
            if tool.name == self.task_generator:
                if ctx.context.current_task().level >= config.max_task_depth:
                    continue
                if len(ctx.context.current_task().sub_tasks) >= config.max_tasks_count:
                    continue

            available.append(tool)
        self.tools = available

    def _build_new_tasks(
        self, ctx: RunContextWrapper[TaskContext], result: str
    ) -> List[Task]:
        """
        Build new tasks from the result.
        """

        logger.info(f"Building new tasks from result: {result}")
        tasks = []
        question_list = result.split(sep)
        curr = ctx.context.current_task()
        if isinstance(question_list, str):
            question_list = json.loads(question_list)

        cnt = len(curr.sub_tasks)
        for q in question_list:
            sub_task = Task(
                id=f"{curr.id}_{cnt+1}",
                origin_query=curr.origin_query,
                query=q,
                level=curr.level + 1,
                parent=curr,
            )
            cnt += 1
            logger.info(
                f"curr: {curr.id} Create new task: {sub_task.id}, ctx: {ctx.context.current_task_id()}"
            )
            curr.sub_tasks[sub_task.id] = sub_task
            ctx.context.tasks[sub_task.id] = sub_task
            tasks.append(sub_task)
        return tasks

    @property
    def tool_names(self) -> List[str]:
        return [tool.name for tool in self.tools]

    def _running_out_of_token(self, ctx: RunContextWrapper[TaskContext]) -> bool:
        return (
            ctx.usage.total_tokens
            > conf.get_configuration().execution_config.max_token_usage * 0.85
        )

    def _build_task_generate_tool(self) -> None:
        tool = next(tool for tool in self.tools if tool.name == self.task_generator)
        if not tool:
            return
        assert isinstance(
            tool, FunctionTool
        ), f"Task generator tool {self.task_generator} must be a FunctionTool"

        async def execute_task(ctx: RunContextWrapper[TaskContext], input: str) -> str:
            ret = await tool.on_invoke_tool(ctx, input)
            if not ret:
                return "No new tasks generated."
            tasks = self._build_new_tasks(ctx, ret)
            await asyncio.gather(*[self._execute_sub_task(ctx, task) for task in tasks])
            return "\n".join(
                [
                    (
                        f"For Question: {task.query}\nYou did some research. Here is the answer: {task.answer.answer}"  # type: ignore
                        if task.solved()
                        else f"For Question: {task.query}\n Cannot find information for it"
                    )
                    for task in tasks
                ]
            )

        # remove the original tool
        self.tools = [tool for tool in self.tools if tool.name != self.task_generator]
        self.tools.append(
            FunctionTool(
                name=tool.name,
                description=tool.description,
                params_json_schema=tool.params_json_schema,
                on_invoke_tool=execute_task,
                strict_json_schema=tool.strict_json_schema,
            )
        )

    async def _execute_sub_task(
        self, context: RunContextWrapper[TaskContext], new_task: Task
    ) -> None:
        """
        Execute a sub task.
        """

        async def run():
            # Set the current task id in the context
            new_task.set_as_current()
            p = Planner(
                name=f"DeepSearch Agent-{new_task.id}",
                tools=self.tools,
                task_generator=self.task_generator,
                hooks=self.hooks,
                model=self.model,
                model_settings=self.model_settings,
            )
            try:
                await Runner.run(
                    starting_agent=p,
                    input=new_task.query,
                    context=context.context,
                )
            except Exception as e:
                print(f"Error running sub task: {e}")
            print(f"task is finish run: {new_task.id}")

        ctx = contextvars.copy_context()
        await ctx.run(run)
