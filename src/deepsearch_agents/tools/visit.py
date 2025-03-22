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


def visit_instuctions(ctx: TaskContext | None = None) -> str:
    # TODO: fix duplicate code
    return f"""
- Crawl and read full content from URLs, you can get the fulltext, last updated datetime etc of any URL.
    """


tool_instructions["visit"] = visit_instuctions


@function_tool(description_override=visit_instuctions())
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


def sumarize_instuctions() -> str:
    return """
You are an advanced AI agent for summarizing and extracting useful information from the web page content.
        
You are working on a task to gather information to answer the following question: 
You are given a web page content, your task is to extract the most useful information from the content and summarize it.

-Question-
{origin_query}
more specifically, {query}

-Rules-
- Think carefully and deeply about the content, and evaluate why is this information useful for answering the question. Give your reasoning in the <reason> field
- Try to use the original content of the content in your summary, but you can also paraphrase if needed.
- Sometimes, the web page could be unavailable (404 page, forbid crawling, rate limit etc.). Report as evaluate="unavailable".
- If the content is not related to the question, report as evaluate="not_related".
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
        system_instructions=sumarize_instuctions().format(
            origin_query=origin_query,
            query=query,
        ),
    )
    print(f"Summarize result: {ret}")
    return ret
