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

from deepsearch_agents._utils import Scope
from deepsearch_agents.conf import (
    JINA_API_KEY,
    MAX_SEARCH_RESULTS,
    OPENAI_BASE_URL,
    SERPAPI_KEY,
)
from deepsearch_agents.context import TaskContext
from deepsearch_agents.tools._utils import tool_instructions


def search_description(ctx: Optional[TaskContext] = None) -> str:
    return f"""
    - Useful To get external information to answer the question.
    - Use web search to find relevant information.
    - Build a search request based on the deep intention behind the original question and the expected answer format
    - Always prefer a single search request, only add another request if the original question covers multiple aspects or elements and one query is not enough, each request focus on one specific aspect of the original question.
    """


tool_instructions["search"] = search_description


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
def search(
    ctx: RunContextWrapper[TaskContext], search_queries: List[str]
) -> list[SearchResult]:
    """
    - Perform a search
    - Search query should be search engine-friendly, concise, using relevant keywords, avoiding unnecessary stop words.

    Args:
        search_queries: Always prefer a single request, only add another request if the original question covers multiple aspects or elements and one search request is definitely not enough, each request focus on one specific aspect of the original question. Minimize mutual information between each request. Maximum 3 search requests.
    """

    print(f"Perform Search: {search_queries}")
    if search_queries is None or len(search_queries) == 0:
        return []
    quereis = rewrite(search_queries)
    # rewrite the query
    res = serpapi.search(
        q=search_queries[0], engine="google", hl="en", gl="us", api_key=SERPAPI_KEY
    )
    res = [
        SearchResult(
            title=r["title"],
            link=r["link"],
            snippet=r["snippet"],
            date=r.get("date"),
            source=r.get("source"),
        )
        for r in res["organic_results"][:MAX_SEARCH_RESULTS]
    ]
    reranked_ret = rerank(res)

    print(
        f"Perform Search. queries: {search_queries}, search results: {len(res)}, current_task_id: {ctx.context.current_task_id()}"
    )
    return reranked_ret


def rewrite(search_queries: List[str]) -> List[str]:
    return search_queries


def rerank(results: List[SearchResult]) -> List[SearchResult]:
    return results
