import argparse
import asyncio
import contextvars
import os
from dataclasses import dataclass
import json
from typing import Any
import uuid
from agents import (
    Agent,
    OpenAIProvider,
    RunConfig,
    RunContextWrapper,
    Runner,
    Tool,
    trace,
)
from agents.tracing.processors import TracingExporter

from deepsearch_agents._utils import Scope
from deepsearch_agents.conf import MY_OPENAI_API_KEY, OPENAI_BASE_URL
from deepsearch_agents.context import Task, TaskContext, build_task_context

from deepsearch_agents.planner import Planner, PlannerHooks
from deepsearch_agents.tools import answer, reflect, search, visit


class Hooks(PlannerHooks):
    def __init__(self, conf: RunConfig):
        super().__init__()
        self.conf = conf

    async def on_tool_end(
        self,
        ctx: RunContextWrapper[TaskContext],
        agent: Agent[TaskContext],
        tool: Tool,
        result: str,
    ) -> None:
        agent.rebuild_tools(ctx)

    async def on_new_task_generated(
        self, context: RunContextWrapper[TaskContext], agent: Planner, new_task: Task
    ) -> None:
        """
        Callback for when a new task is generated.
        """

        async def run():
            # Set the current task id in the context
            context.context.set_as_current(new_task)
            p = Planner(
                name=f"DeepSearch Agent-{new_task.id}",
                tools=agent.tools,
                task_generator=agent.task_generator,
                hooks=agent.hooks,
            )
            p.task_generator_tool_name = "reflect"
            try:
                await Runner.run(
                    starting_agent=p,
                    input=new_task.query,
                    context=context.context,
                    run_config=self.conf,
                )
            except Exception as e:
                print(f"Error running sub task: {e}")
            print(f"task is finish run: {new_task.id}")

        ctx = contextvars.copy_context()
        await ctx.run(run)
        return


async def main():
    parser = argparse.ArgumentParser(description="DeepSearch Agents CLI")
    parser.add_argument("query", type=str, help="query string to search")
    # set_trace_processors([BatchTraceProcessor(ConsoleSpanExporter())])
    # args = parser.parse_args()
    # q = args.query.strip()
    # q = "why is spx down 10% in last month?"
    q = "What are the recent policies released by Trump, and how will these policies affect the United States and other countries?"

    trace_id = f"trace_{uuid.uuid4().hex}"
    new_trace = trace(
        workflow_name="deepsearch",
        trace_id=trace_id,
    )
    new_trace.start(mark_as_current=True)
    conf = RunConfig(
        model_provider=OpenAIProvider(
            base_url=OPENAI_BASE_URL,
            api_key=MY_OPENAI_API_KEY,
            use_responses=False,
        ),
    )
    context = build_task_context(q)

    planner = Planner(
        name="DeepSearch Agent",
        tools=[search, visit, answer, reflect],
        task_generator="reflect",
        planner_hooks=Hooks(conf),
    )

    result = await Runner.run(
        starting_agent=planner,
        input=q,
        context=context,
        max_turns=15,
        run_config=conf,
    )
    new_trace.finish()
    print("final output----------\n")
    print(result.final_output)
    print("final answer----------\n")
    context.final_answer

    print("logs----------\n")


if __name__ == "__main__":
    asyncio.run(main())
