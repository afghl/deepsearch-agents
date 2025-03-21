import argparse
import asyncio
import contextvars
from dataclasses import dataclass
import json
from typing import Any
import uuid
from agents import (
    OpenAIProvider,
    RunConfig,
    RunContextWrapper,
    Runner,
    Span,
    Trace,
    set_trace_processors,
    trace,
)
from agents.tracing.processors import (
    BatchTraceProcessor,
    TracingExporter,
)

import os

from deepsearch_agents._utils import Scope
from deepsearch_agents.conf import MY_OPENAI_API_KEY, OPENAI_BASE_URL
from deepsearch_agents.context import Task, TaskContext, build_task_context

from deepsearch_agents.planner import Planner
from deepsearch_agents.tools import answer, search, visit

consolg_logs: list[str] = []


@dataclass
class ConsoleSpanExporter(TracingExporter):
    def export(self, items: list[Trace | Span[Any]]) -> None:
        for item in items:
            if isinstance(item, Trace):
                # print(f"[Exporter] Export trace_id={item.trace_id}, name={item.name}, ")
                log = f"[Exporter] Export trace_id={item.trace_id}, name={item.name}, "
            else:
                # print(f"[Exporter] Export span: {item.export()}")
                log = f"[Exporter] Export span: {item.export()}"
            consolg_logs.append(log)


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
        tracing_disabled=True,
        trace_id=trace_id,
        model="DeepSeek-R1",
    )
    context = build_task_context(q)

    planner = Planner()
    planner.name = "DeepSearch Agent"
    planner.task_generator_tool_name = "reflect"

    async def run_sub_task(
        context: RunContextWrapper[TaskContext], new_task: Task
    ) -> None:
        """
        Callback for when a new task is generated.
        """

        async def set_task_id_and_run():
            # Set the current task id in the context
            Scope.set_current_task_id(new_task.id)
            p = Planner(name=f"DeepSearch Agent-{new_task.id}")
            p.task_generator_tool_name = "reflect"
            try:
                await Runner.run(
                    starting_agent=p,
                    input=new_task.query,
                    context=context.context,
                    run_config=conf,
                )
            except Exception as e:
                print(f"Error running sub task: {e}")
            print(f"task is finish run: {new_task.id}")

        ctx = contextvars.copy_context()
        await ctx.run(set_task_id_and_run)

    planner.on_new_task_generated = run_sub_task
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

    for log in consolg_logs:
        print(log)


if __name__ == "__main__":
    asyncio.run(main())
