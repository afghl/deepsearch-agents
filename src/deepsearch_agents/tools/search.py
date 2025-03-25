from abc import ABC, abstractmethod
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

from deepsearch_agents.log import logger
from deepsearch_agents.conf import get_configuration
from deepsearch_agents.context import TaskContext
from deepsearch_agents.llm.llm import get_response
from deepsearch_agents.tools._utils import tool_instructions
from deepsearch_agents.tools.rewrite import rewrite_search_query


def search_description(ctx: Optional[TaskContext] = None) -> str:
    return f"""
    - Useful To get external information to answer the question.
    - Use web search to find relevant information.
    - Build a search request based on the deep intention behind the original question and the expected answer format
    - Always prefer a single search request, only add another request if the original question covers multiple aspects or elements and one query is not enough, each request focus on one specific aspect of the original question.
    """


tool_instructions["search"] = search_description


MAX_SEARCH_RESULTS = 10
TOTAL_SEARCH_RESULTS = 10


class SearchResult(BaseModel):
    """
    Search result from SERPAPI
    """

    title: str
    link: str
    snippet: str
    date: str | None = None
    source: str | None = None


@function_tool()
async def search(
    ctx: RunContextWrapper[TaskContext], think: str, search_queries: List[str]
) -> list[SearchResult]:
    """
    - Perform a search
    - Search query should be search engine-friendly, concise, using relevant keywords, avoiding unnecessary stop words.

    Args:
        think: A very concise explain of why choose to search these queries.
        search_queries: Always prefer a single request, only add another request if the original question covers multiple aspects or elements and one search request is definitely not enough, each request focus on one specific aspect of the original question. Minimize mutual information between each request. Maximum 3 search requests.
    """

    print(f"Perform Search: {search_queries}")
    if search_queries is None or len(search_queries) == 0:
        return []
    queries = await rewrite_search_query(ctx.context, search_queries)
    logger.info(f"Rewrite original query: {search_queries} -> {queries}")

    res: list[SearchResult] = []
    for query in queries.queries:
        r = _search(query)
        res.extend(r)
    reranked_ret = _rerank(res)

    print(
        f"Perform Search. queries: {search_queries}, search results: {len(res)}, current_task_id: {ctx.context.current_task_id()}"
    )
    return reranked_ret[:TOTAL_SEARCH_RESULTS]


def _rerank(results: List[SearchResult]) -> List[SearchResult]:
    return results


def _search(query: str) -> List[SearchResult]:
    config = get_configuration()
    r = serpapi.search(
        q=query, engine="google", hl="en", gl="us", api_key=config.serpapi_api_key
    )
    return [SearchResult(**r) for r in r["organic_results"][:MAX_SEARCH_RESULTS]]
