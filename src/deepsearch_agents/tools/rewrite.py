from agents import RunContextWrapper
from pydantic import BaseModel
from deepsearch_agents.context import TaskContext
from deepsearch_agents.llm.llm import get_response


REWRITE_PROMPT = """
Rewrite the search query to be more specific and relevant to the question.
"""


SEARCH_REWRITE_INSTRUCTIONS = """
You are an expert search query expander. You optimize user queries by extensively analyzing potential user intents and generating comprehensive query variations.

You will be given a user query and a list of queries that are related to the user query.

Your task is to rewrite the user query to be more specific and targeted.

-Rules-
1. Make the query more specific and targeted
2. Use natural, search-friendly language
3. Include relevant keywords that search engines love
4. Keep it concise but informative
"""

SEARCH_REWRITE_PROMPT = """
My original search query is: "${origin_query}"

Now I focus on this specific aspect of the original query: "${query}"

So I am going to search these queries:
{to_rewrite}

Given those info, now please generate the best effective queries. No more than {max_queries} queries.
"""


class SearchQueries(BaseModel):
    """Represents an optimized search query"""

    explanation: str
    """Brief explanation of why this query is better"""
    queries: list[str]
    """The optimized search queries"""


async def rewrite_search_query(
    ctx: RunContextWrapper[TaskContext], to_rewrite: list[str]
) -> SearchQueries:
    """
    Rewrites the search query to be more search-engine friendly
    """
    task = ctx.context.current_task()
    origin_query = task.origin_query
    query = task.query

    llm_response = await get_response(
        model="rewrite",
        input=SEARCH_REWRITE_PROMPT.format(
            origin_query=origin_query,
            query=query,
            to_rewrite=to_rewrite,
            max_queries=len(to_rewrite),
        ),
        output_type=SearchQueries,
        system_instructions=SEARCH_REWRITE_INSTRUCTIONS,
    )
    ctx.usage.add(llm_response.usage)
    return llm_response.response
