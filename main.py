import argparse
import asyncio
import contextvars
from agents import OpenAIProvider, RunConfig, RunContextWrapper, Runner

import os

from deepsearch_agents._utils import Scope
from deepsearch_agents.conf import OPENAI_API_KEY, OPENAI_BASE_URL
from deepsearch_agents.context import Task, TaskContext, build_task_context
from deepsearch_agents.planner import Planner


async def main():
    parser = argparse.ArgumentParser(description="DeepSearch Agents CLI")
    parser.add_argument("query", type=str, help="query string to search")

    # args = parser.parse_args()
    # q = args.query.strip()
    q = "why is spx down 10% in last month?"

    conf = RunConfig(
        model_provider=OpenAIProvider(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            use_responses=False,
        ),
        tracing_disabled=True,
    )
    context = build_task_context(q)

    async def run_sub_task(
        self, context: RunContextWrapper[TaskContext], new_task: Task
    ) -> None:
        """
        Callback for when a new task is generated.
        """
        ctx = contextvars.copy_context()
        Scope.set_current_task_id(new_task.id)
        await ctx.run(
            lambda: Runner.run(
                starting_agent=self,
                input=new_task.query,
                context=context.context,
                conf=conf,
            )
        )

    planner = Planner(
        name="DeepSearch Manger",
        task_generator_tool_name="reflect",
        on_new_task_generated=run_sub_task,
    )
    result = await Runner.run(
        starting_agent=planner,
        input=q,
        context=context,
        max_turns=10,
        run_config=conf,
    )

    answer = context.final_answer
    print("final output----------\n")
    print(result.final_output)
    print("final answer----------\n")
    answer


if __name__ == "__main__":
    asyncio.run(main())
