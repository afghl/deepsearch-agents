import asyncio
import datetime
import re
from typing import List, Literal, Optional
from pydantic import BaseModel
import requests

from agents import RunContextWrapper, function_tool

from deepsearch_agents.conf import get_configuration
from deepsearch_agents.log import logger
from deepsearch_agents.context import (
    Knowledge,
    Reference,
    TaskContext,
    build_task_context,
)
from deepsearch_agents.tools._utils import (
    log_action,
    remove_markdown_link,
    tool_instructions,
)


class PageContent(BaseModel):
    title: str
    description: str
    content: str
    warning: str | None = None


def visit_description(ctx: Optional[TaskContext] = None) -> str:
    return f"""
    - Retrieve and analyze content from web URLs to gather relevant information for your query
    - Use this tool to access full webpage content when search results are promising
    - Ideal for extracting detailed information that search snippets don't provide
    """


tool_instructions["visit"] = visit_description


@function_tool()
async def visit(
    ctx: RunContextWrapper[TaskContext],
    think: str,
    urls: List[str],
) -> str:
    """
    - Visit the URLs and extract useful content for the query

    Args:
        think: A very concise explain of why choose to visit these URLs.
        urls: Must be an array of URLs, choose up to 5 URLs to visit
    """
    knowledges: List[Knowledge] = []
    log_action(ctx, "visit", think, urls=urls)  # type: ignore
    urls_to_process = urls[:5]

    # 创建所有URL的任务列表
    tasks = []

    for url in urls_to_process:
        tasks.append(fetch_url(url))

    # 并发执行所有任务
    results: List[PageContent | BaseException] = await asyncio.gather(
        *tasks, return_exceptions=True
    )
    # 处理结果
    for url, result in zip(urls_to_process, results):
        if isinstance(result, BaseException):
            logger.error(f"Error processing URL {url}: {result}")
        elif result.warning:
            logger.warning(
                f"URL {url}, something wrong with the content, {result.warning}"
            )
        else:
            knowledges.append(
                Knowledge(
                    reference=Reference(url=url, title=result.title),
                    summary=f"Title: {result.title}\nDescription: {result.description}.",
                    quotes=[result.content],
                )
            )

    ctx.context.current_task().knowledges.extend(knowledges)
    content_str = "\n".join(
        [
            f"URL: {k.reference.url}\nPublication Date: {k.reference.datetime}\nSummary: {k.summary}"
            for k in knowledges
        ]
    )
    if len(knowledges) > 0:
        return f"""
        Successfully visited {len(urls_to_process)} URLs, {len(knowledges)} of them contain clues to answer the question.
        Here are the details:
        {content_str}
        """
    else:
        return f"""
        Try to visit {len(urls_to_process)} URLs, but found nothing useful. Maybe try another set of URLs.
        """


async def fetch_url(url: str) -> PageContent:
    """
    - Crawl and read full content from URLs
    """
    config = get_configuration()
    url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": f"Bearer {config.jina_api_key}",
        "Accept": "application/json",
    }
    response = requests.get(url, headers=headers).json()

    if response.get("code") != 200:
        message = response.get("message", "Unknown error occurred while fetching URL")
        raise ValueError(f"API request failed: {message}")
    data = response.get("data")
    if not data:
        raise ValueError("No data returned from API")
    try:
        return PageContent(
            title=data["title"],
            description=data["description"],
            content=remove_markdown_link(data["content"]),
            warning=data.get("warning"),
        )
    except Exception as e:
        raise ValueError(f"Error parsing data: {e}") from e
