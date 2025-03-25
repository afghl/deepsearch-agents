import argparse
import asyncio
from contextlib import asynccontextmanager
import contextvars
import uuid
from agents import (
    Agent,
    OpenAIProvider,
    RunConfig,
    RunContextWrapper,
    Runner,
    Tool,
    trace,
    ModelSettings,
)

from deepsearch_agents.log import logger
from deepsearch_agents.conf import get_configuration
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
        logger.info(
            f"tool: {tool.name} end current task: {ctx.context.current_task_id()}"
        )
        agent.rebuild_tools(ctx)

    async def on_new_task_generated(
        self, context: RunContextWrapper[TaskContext], agent: Planner, new_task: Task
    ) -> None:
        """
        Callback for when a new task is generated.
        """

        async def run():
            # Set the current task id in the context
            new_task.set_as_current()
            p = Planner(
                name=f"DeepSearch Agent-{new_task.id}",
                tools=agent.tools,
                task_generator=agent.task_generator,
                hooks=agent.hooks,
                model=agent.model,
                model_settings=agent.model_settings,
            )
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
    config = get_configuration()
    parser = argparse.ArgumentParser(description="DeepSearch Agents CLI")
    parser.add_argument("query", type=str, help="query string to search")
    # args = parser.parse_args()
    # q = args.query.strip()
    q = "why is spx down 10% in last month?"
    logger.info(f"query: {q}")
    trace_id = f"trace_{uuid.uuid4().hex}"
    new_trace = trace(
        workflow_name="deepsearch",
        trace_id=trace_id,
    )
    new_trace.start(mark_as_current=True)
    conf = RunConfig(
        model_provider=OpenAIProvider(
            base_url=config.openai_base_url,
            api_key=config.get_openai_api_key(),
            use_responses=False,
        ),
    )
    context = build_task_context(q)
    planner_conf = config.get_model_config("planner")
    planner = Planner(
        name="DeepSearch Agent",
        tools=[search, visit, answer, reflect],
        task_generator="reflect",
        planner_hooks=Hooks(conf),
        model=planner_conf.model_name,
        model_settings=planner_conf.as_model_settings(),
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
    print(context.final_answer())

    print("logs----------\n")


if __name__ == "__main__":
    asyncio.run(main())
