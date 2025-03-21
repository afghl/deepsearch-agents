from abc import ABC, abstractmethod
import asyncio
from typing import Any, Callable, Generic, List, Optional
from openai import AsyncOpenAI
from pydantic import BaseModel
import requests
import serpapi
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
from deepsearch_agents.tools._utils import tool_instructions


def visit_instuctions(ctx: TaskContext) -> str:
    # TODO: fix duplicate code
    return f"""
    - Crawl and read full content from URLs, you can get the fulltext, last updated datetime etc of any URL.
    """


tool_instructions["visit"] = visit_instuctions


@function_tool
async def visit(ctx: RunContextWrapper[TaskContext], urls: List[str]) -> str:
    """
    - Crawl and read full content from URLs, you can get the fulltext, last updated datetime etc of any URL.

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
        if isinstance(result, Exception):
            print(f"Error processing URL {url}: {result}")
        elif result:
            contents.append(result)
            curr = ctx.context.current_task()
            curr.knowledges.append(Knowledge(reference=url, answer=result))

    print(
        f"Finished visiting {len(urls_to_process)} URLs. Got {len(contents)} valid contents. Current_task_id: {Scope.get_current_task_id()}"
    )

    return "\n".join(contents)


async def get_content(ctx: RunContextWrapper[TaskContext], url: str) -> str:
    """
    - Crawl and read full content from URLs
    """
    url = f"https://r.jina.ai/{url}"
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
    response = requests.get(url, headers=headers)
    model = OpenAIChatCompletionsModel(
        "gpt-4o-mini",
        openai_client=AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=MY_OPENAI_API_KEY),
    )
    q = ctx.context.origin_query
    summarize_res = await model.get_response(
        system_instructions=f"""You are an advanced AI agent for summarizing and extracting useful information from the web page content.
        
        You are working on a task to gather information to answer the following question: {q}
        You are given a web page content, your task is to extract the most useful information from the content and summarize it.
        
        - Think carefully and deeply about the content, why is this information useful for answering the question.
        - Try to use the original wording of the content, but you can also paraphrase if needed.
        """,
        input=response.text,
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
    )
    return summarize_res.output[0].content[0].text
