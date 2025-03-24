from abc import ABC, abstractmethod
import asyncio
from typing import Any, Callable, Generic, List, Literal, Optional, cast
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
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

from deepsearch_agents._utils import Scope
from deepsearch_agents.conf import (
    JINA_API_KEY,
    MY_OPENAI_API_KEY,
    OPENAI_BASE_URL,
)
from deepsearch_agents.context import Knowledge, Task, TaskContext
from deepsearch_agents.llm import get_response
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


@function_tool(description_override=visit_description())
async def visit(
    ctx: RunContextWrapper[TaskContext],
    urls: List[str],
) -> List[SummarizeResult]:
    """
    _

    Args:
        urls: Must be an array of URLs, choose up the most relevant 5 URLs to visit
    """
    contents = []
    print(
        f"Perform Visit. URLs: {len(urls)}, current_task_id: {Scope.get_current_task_id()}"
    )  # For debugging purposes, remove in production
    urls_to_process = urls[:5]

    # 创建所有URL的任务列表
    tasks = []
    for url in urls_to_process:
        tasks.append(get_content(ctx, url))

    # 并发执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # 处理结果
    for url, result in zip(urls_to_process, results):
        if isinstance(result, BaseException):
            print(f"Error processing URL {url}: {result}")
        elif result:
            contents.append(result)
            curr = ctx.context.current_task()
            curr.knowledges.append(Knowledge(reference=url, answer=result))  # type: ignore

    print(
        f"Finished visiting {len(urls_to_process)} URLs. Got {len(contents)} valid contents. Current_task_id: {Scope.get_current_task_id()}"
    )

    return contents


def main():
    print(visit.description)
    print(visit)


if __name__ == "__main__":
    main()


def summarize_description() -> str:
    return """
You are an advanced AI analysis system specialized in extracting and synthesizing relevant information from web content.
        
Your task is to analyze web page content to gather information that addresses the following inquiry: 
{origin_query}
More specifically: {query}

Guidelines for Analysis:
- Conduct a thorough evaluation of the content's relevance and utility for addressing the inquiry. Document your analytical reasoning in the <reason> field.
- Prioritize direct quotations from the original content when appropriate, while employing thoughtful paraphrasing to enhance clarity when necessary.
- If the web resource is inaccessible (e.g., 404 error, access restrictions, rate limitations), classify as evaluate="unavailable".
- If the content lacks relevance to the inquiry after careful analysis, classify as evaluate="not_related".
"""


async def get_content(ctx: RunContextWrapper[TaskContext], url: str) -> SummarizeResult:
    """
    - Crawl and read full content from URLs
    """
    url = f"https://r.jina.ai/{url}"
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
    response = requests.get(url, headers=headers)

    origin_query = ctx.context.origin_query
    query = ctx.context.current_task().query
    ret = await get_response(
        model="gpt-4o",
        input=response.text,
        output_type=SummarizeResult,
        system_instructions=summarize_description().format(
            origin_query=origin_query,
            query=query,
        ),
    )
    print(f"Summarize result: {ret}")
    return ret
