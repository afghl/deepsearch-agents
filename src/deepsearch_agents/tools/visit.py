"""Visit tool for fetching and processing web page content."""

import asyncio
from typing import List, Optional
from pydantic import BaseModel
import requests

from agents import RunContextWrapper, function_tool

from deepsearch_agents.conf import get_configuration
from deepsearch_agents.log import logger
from deepsearch_agents.context import (
    Knowledge,
    Reference,
    TaskContext,
)
from deepsearch_agents.tools._utils import (
    log_action,
    remove_markdown_link,
    tool_instructions,
)
from deepsearch_agents.tools.summarize import summarize


class PageContent(BaseModel):
    """Content fetched from a web page."""
    title: str
    description: str
    content: str
    url: str  # Original URL
    warning: str | None = None


def visit_description(ctx: Optional[TaskContext] = None) -> str:
    return """
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
    log_action(ctx, "visit", think, urls=urls)  # type: ignore
    urls_to_process = urls[:5]

    curr_task = ctx.context.current_task()
    origin_query = curr_task.origin_query
    query = curr_task.query

    # Fetch all URLs in parallel
    fetch_tasks = [fetch_url(url) for url in urls_to_process]
    fetch_results: List[PageContent | BaseException] = await asyncio.gather(
        *fetch_tasks, return_exceptions=True
    )

    # Filter successful fetches
    successful_fetches: List[PageContent] = []
    for url, result in zip(urls_to_process, fetch_results):
        if isinstance(result, BaseException):
            logger.error(f"Error fetching URL {url}: {result}")
        elif result.warning:
            logger.warning(f"URL {url} has warning: {result.warning}")
        else:
            successful_fetches.append(result)

    if not successful_fetches:
        return f"""
Tried to visit {len(urls_to_process)} URLs, but all failed to fetch.
Please try different URLs or use search to find alternative sources.
"""

    # Summarize all fetched content in parallel
    summarize_tasks = [
        summarize(ctx, query, origin_query, page.content)
        for page in successful_fetches
    ]
    summarize_results = await asyncio.gather(*summarize_tasks, return_exceptions=True)

    # Process summarization results and create Knowledge objects
    knowledges: List[Knowledge] = []
    useful_count = 0
    
    for page, summary_result in zip(successful_fetches, summarize_results):
        if isinstance(summary_result, BaseException):
            logger.error(f"Error summarizing {page.url}: {summary_result}")
            continue
        
        if summary_result.evaluate == "useful":
            useful_count += 1
            knowledge = Knowledge(
                reference=Reference(
                    url=page.url,
                    title=page.title,
                    datetime=summary_result.datetime,
                ),
                quotes=summary_result.quotes,
                summary=summary_result.summarize,
            )
            knowledges.append(knowledge)
            logger.info(f"Extracted useful knowledge from {page.url}")
        else:
            logger.info(f"URL {page.url} marked as {summary_result.evaluate}: {summary_result.reason}")

    # Store knowledges in the current task
    curr_task.knowledges.extend(knowledges)

    # Build response
    if knowledges:
        content_lines = []
        for k in knowledges:
            content_lines.append(f"**{k.reference.title}** ({k.reference.url})")
            if k.reference.datetime:
                content_lines.append(f"Published: {k.reference.datetime}")
            content_lines.append(f"Summary: {k.summary}")
            if k.quotes:
                content_lines.append("Key quotes:")
                for q in k.quotes[:2]:  # Limit quotes shown
                    content_lines.append(f"  - \"{q[:150]}{'...' if len(q) > 150 else ''}\"")
            content_lines.append("")
        
        return f"""
Successfully visited {len(urls_to_process)} URLs. {useful_count} contained useful information.

{chr(10).join(content_lines)}

Total knowledge items collected for this task: {len(curr_task.knowledges)}
        """
    else:
        return f"""
Visited {len(urls_to_process)} URLs, but none contained directly useful information for the query.
Reasons: content was either not related or unavailable.
Consider trying different URLs or refining your search.
        """


async def fetch_url(url: str) -> PageContent:
    """
    Fetch content from a URL using Jina Reader API.
    
    Args:
        url: The URL to fetch.
        
    Returns:
        PageContent with the fetched data.
        
    Raises:
        ValueError: If the fetch fails.
    """
    config = get_configuration()
    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": f"Bearer {config.jina_api_key}",
        "Accept": "application/json",
    }
    
    # Use asyncio to run blocking request in thread pool
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, 
        lambda: requests.get(jina_url, headers=headers, timeout=30)
    )
    
    response_data = response.json()

    if response_data.get("code") != 200:
        message = response_data.get("message", "Unknown error occurred while fetching URL")
        raise ValueError(f"API request failed: {message}")
    
    data = response_data.get("data")
    if not data:
        raise ValueError("No data returned from API")
    
    try:
        return PageContent(
            title=data.get("title", "Untitled"),
            description=data.get("description", ""),
            content=remove_markdown_link(data.get("content", "")),
            url=url,  # Store original URL
            warning=data.get("warning"),
        )
    except Exception as e:
        raise ValueError(f"Error parsing data: {e}") from e
