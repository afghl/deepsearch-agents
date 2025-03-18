import argparse
import asyncio
from agents import OpenAIProvider, RunConfig
import dotenv

import os

from deepsearch_agents.context import TaskContext
from deepsearch_agents.run import DeepsearchRunner
from deepsearch_agents.manager import Manager

dotenv.load_dotenv()


async def main():
    parser = argparse.ArgumentParser(description="DeepSearch Agents CLI")
    parser.add_argument("query", type=str, help="query string to search")

    args = parser.parse_args()
    q = args.query.strip()
    manager = Manager(name="DeepSearch Manger")
    ctx = TaskContext(original_query=q)
    conf = RunConfig(
        model_provider=OpenAIProvider(
            base_url=os.getenv("OPENAI_BASE_URL", None),
            api_key=os.getenv("OPENAI_API_KEY", None),
            use_responses=False,
        )
    )
    result = await DeepsearchRunner.run(manager, q, context=ctx, run_config=conf)
    print("----------\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
