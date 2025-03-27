import argparse
import asyncio
from contextlib import asynccontextmanager
import contextvars
import uuid
from agents import (
    Agent,
    AgentHooks,
    RunContextWrapper,
    Runner,
    Tool,
    gen_trace_id,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_export_api_key,
    trace,
)
from openai import AsyncOpenAI

from deepsearch_agents.log import logger
from deepsearch_agents.conf import get_configuration
from deepsearch_agents.context import Task, TaskContext, build_task_context

from deepsearch_agents.planner import Planner, PlannerHooks
from deepsearch_agents.tools import answer, reflect, search, visit


class Hooks(AgentHooks[TaskContext]):

    async def on_tool_start(
        self,
        ctx: RunContextWrapper[TaskContext],
        _: Agent[TaskContext],
        tool: Tool,
    ) -> None:
        ctx.context.current_task().turn += 1

    async def on_tool_end(
        self,
        ctx: RunContextWrapper[TaskContext],
        agent: Agent[TaskContext],
        tool: Tool,
        result: str,
    ) -> None:
        logger.info(f"finish action {tool.name} result: {result}")
        agent.rebuild_tools(ctx)


async def main():
    config = get_configuration()
    client = AsyncOpenAI(
        base_url=config.openai_base_url if config else None,
        api_key=config.openai_api_key,
    )
    set_default_openai_client(client)
    set_tracing_export_api_key(config.tracing_openai_api_key)
    set_default_openai_api("chat_completions")
    parser = argparse.ArgumentParser(description="DeepSearch Agents CLI")
    parser.add_argument("query", type=str, help="query string to search")
    # args = parser.parse_args()
    # q = args.query.strip()
    q = "How has the SPX performed in the last 30 days? What specific reasons have driven the market recently?"
    logger.info(f"query: {q}")
    trace_id = gen_trace_id()
    with trace(workflow_name="deepsearch", trace_id=trace_id):
        context = build_task_context(q)
        planner_conf = config.get_model_config("planner")
        planner = Planner(
            name="DeepSearch Agent",
            tools=[search, visit, answer, reflect],
            task_generator="reflect",
            hooks=Hooks(),
            model=planner_conf.model_name,
            model_settings=planner_conf.as_model_settings(),
        )

        ret = await Runner.run(
            starting_agent=planner,
            input=q,
            context=context,
            max_turns=config.execution_config.max_turns,
        )
    logger.info(ret.final_output)
    logger.info("final answer----------\n")
    logger.info(context.final_answer())


if __name__ == "__main__":
    asyncio.run(main())
