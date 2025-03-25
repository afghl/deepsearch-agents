import asyncio
from typing import List, Literal, Optional
from pydantic import BaseModel
import requests
from typing_extensions import TypedDict

from agents import (
    Agent,
    FunctionTool,
    ModelSettings,
    ModelTracing,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    function_tool,
)

from deepsearch_agents.conf import get_configuration
from deepsearch_agents.log import logger
from deepsearch_agents.context import Knowledge, Task, TaskContext
from deepsearch_agents.llm import get_response
from deepsearch_agents.tools import summarize
from deepsearch_agents.tools._utils import tool_instructions


class SummarizeResult(BaseModel):
    reason: str
    summarize: str
    evaluate: Literal["useful", "not_related", "unavailable"]


def visit_description(ctx: Optional[TaskContext] = None) -> str:
    return f"""
- Access and retrieve comprehensive content from web URLs, including full text, publication metadata, and last updated timestamps.
    """


tool_instructions["visit"] = visit_description


@function_tool()
async def visit(
    ctx: RunContextWrapper[TaskContext],
    urls: List[str],
) -> List[SummarizeResult]:
    """
    - Access and retrieve comprehensive content from web URLs, including full text, publication metadata, and last updated timestamps.

    Args:
        urls: Must be an array of URLs, choose up the most relevant 5 URLs to visit
    """
    contents = []
    print(
        f"Perform Visit. URLs: {len(urls)}, current_task_id: {ctx.context.current_task_id()}"
    )  # For debugging purposes, remove in production
    urls_to_process = urls[:5]

    # 创建所有URL的任务列表
    tasks = []

    for url in urls_to_process:
        tasks.append(visit_url_and_summarize(ctx.context, url))

    # 并发执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # 处理结果
    for url, result in zip(urls_to_process, results):
        if isinstance(result, BaseException):
            logger.error(f"Error processing URL {url}: {result}")
        elif result:
            contents.append(result)
            curr = ctx.context.current_task()
            curr.knowledges.append(Knowledge(reference=url, answer=result))  # type: ignore

    print(
        f"Finished visiting {len(urls_to_process)} URLs. Got {len(contents)} valid contents. Current_task_id: {ctx.context.current_task_id()}"
    )

    return contents


async def visit_url_and_summarize(ctx: TaskContext, url: str) -> SummarizeResult:
    """
    - Crawl and read full content from URLs
    """
    config = get_configuration()
    url = f"https://r.jina.ai/{url}"
    headers = {"Authorization": f"Bearer {config.jina_api_key}"}
    response = requests.get(url, headers=headers)

    origin_query = ctx.origin_query
    query = ctx.current_task().query
    summarize_result = await summarize(ctx, query, origin_query, response.text)
    return summarize_result
