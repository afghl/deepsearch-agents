import asyncio
from typing import List, Optional
import requests

from agents import RunContextWrapper, function_tool

from deepsearch_agents.conf import get_configuration
from deepsearch_agents.log import logger
from deepsearch_agents.context import Knowledge, TaskContext, build_task_context
from deepsearch_agents.tools.summarize import SummarizeResult, summarize
from deepsearch_agents.tools._utils import tool_instructions


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
) -> List[SummarizeResult]:
    """
    - Visit the URLs and extract useful content for the query

    Args:
        think: A very concise explain of why choose to visit these URLs.
        urls: Must be an array of URLs, choose up to 5 URLs to visit
    """
    contents: List[SummarizeResult] = []
    logger.info(
        f"Perform Visit. URLs: {len(urls)}, current_task_id: {ctx.context.current_task_id()}"
    )  # For debugging purposes, remove in production
    urls_to_process = urls[:5]

    # 创建所有URL的任务列表
    tasks = []

    for url in urls_to_process:
        tasks.append(visit_url_and_summarize(ctx.context, url))

    # 并发执行所有任务
    results: List[SummarizeResult | BaseException] = await asyncio.gather(
        *tasks, return_exceptions=True
    )
    # 处理结果
    for url, result in zip(urls_to_process, results):
        if isinstance(result, BaseException):
            logger.error(f"Error processing URL {url}: {result}")
        elif result and result.evaluate == "useful":
            contents.append(result)
            curr = ctx.context.current_task()
            curr.knowledges.append(
                Knowledge(reference=url, answer=result.summarize, quotes=result.quotes)
            )
        else:
            logger.warning(
                f"URL {url}, something wrong with the content, {result.reason}"
            )

    content_str = "\n".join(
        [f"URL: {c.reference}\nSummary: {c.summarize}\n" for c in contents]
    )
    if len(contents) > 0:
        ret = f"""
        Successfully visited {len(urls_to_process)} URLs, {len(contents)} of them contain clues to answer the question.
        Here are the details:
        {content_str}
        """
    else:
        ret = f"""
        Try to visit {len(urls_to_process)} URLs, but found nothing useful. Maybe try another set of URLs.
        """

    return ret


async def visit_url_and_summarize(ctx: TaskContext, url: str) -> SummarizeResult:
    """
    - Crawl and read full content from URLs
    """
    config = get_configuration()
    url = f"https://r.jina.ai/{url}"
    headers = {"Authorization": f"Bearer {config.jina_api_key}"}
    response = requests.get(url, headers=headers)

    task = ctx.current_task()
    origin_query = task.origin_query
    query = task.query
    summarize_result = await summarize(query, origin_query, response.text)
    return summarize_result


if __name__ == "__main__":
    url = "https://www.cnbc.com/2025/03/14/us-stock-market-loses-5-trillion-in-value-in-three-weeks.html"
    asyncio.run(
        visit_url_and_summarize(
            build_task_context("why is spx down 10% in last month?"), url
        )
    )
