import argparse
import asyncio
from agents import OpenAIProvider, RunConfig

import os

from deepsearch_agents.conf import OPENAI_API_KEY, OPENAI_BASE_URL
from deepsearch_agents.context import TaskContext
from deepsearch_agents.run import DeepsearchRunner, TaskSolver
from deepsearch_agents.manager import Manager


async def main():
    parser = argparse.ArgumentParser(description="DeepSearch Agents CLI")
    parser.add_argument("query", type=str, help="query string to search")

    # args = parser.parse_args()
    # q = args.query.strip()
    q = "why is spx down 10% in last month?"
    manager = Manager(name="DeepSearch Manger")
    conf = RunConfig(
        model_provider=OpenAIProvider(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            use_responses=False,
        ),
        tracing_disabled=True,
    )
    result = await TaskSolver.solve(
        manager, q, task_generator_name="reflect", run_config=conf
    )
    print("final output----------\n")
    print(result.final_output)
    print("raw responses----------\n")
    for raw_response in result.raw_responses:
        print(raw_response)


if __name__ == "__main__":
    asyncio.run(main())
