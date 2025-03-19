import asyncio
import copy
from typing import Any, Callable, cast

from agents import (
    Agent,
    RunConfig,
    RunContextWrapper,
    RunHooks,
    RunResult,
    Runner,
    TContext,
    TResponseInputItem,
    Tool,
)

from deepsearch_agents._utils import Scope
from deepsearch_agents.context import Task, TaskContext


class DeepsearchRunner(Runner):
    pass


class TaskSolver:

    @classmethod
    async def solve(
        cls,
        starting_agent: Agent[TaskContext],
        query: str,
        *,
        task_generator_name: str = None,
        on_new_task: Callable[[TaskContext, Task], None] | None = None,
        max_turns: int = 10,
        run_config: RunConfig | None = None,
    ) -> RunResult:
        task = Task(question=query, origin_query=query)
        print(f"task.id: {task.id}")
        Scope.set_current_task_id(task.id)
        context = TaskContext(task=task)

        result = await Runner.run(
            starting_agent=starting_agent,
            input=query,
            context=context,
            max_turns=max_turns,
            hooks=cls._get_hooks(on_new_task),
            run_config=run_config,
        )
        return result

    @classmethod
    def _get_hooks(
        cls,
        task_generator_name: str | None = None,
        on_new_task: Callable[[TaskContext], None] | None = None,
    ) -> RunHooks[TaskContext]:
        if task_generator_name is None:
            return None

        class CustomRunHooks(RunHooks[TaskContext]):
            async def on_tool_end(
                self,
                context: RunContextWrapper[TaskContext],
                agent: Agent[TaskContext],
                tool: Tool,
                result: str,
            ):
                print(
                    f"Tool {tool.name} finished with result: {result}, context: {context}, agent: {agent}"
                )

        return CustomRunHooks()
