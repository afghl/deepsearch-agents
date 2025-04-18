import asyncio
from typing import List, Literal
from agents import RunContextWrapper
from pydantic import BaseModel

from deepsearch_agents.context import TaskContext, build_task_context
from deepsearch_agents.llm.llm import get_response


SUMMARIZE_PROMPT = """
You are an advanced AI analysis system specialized in extracting and synthesizing relevant information from web content.
        
Your task is to analyze web page content to gather information to answer the following question:
{query_content}

-Guidelines for Analysis-

- Conduct a thorough evaluation of the content's relevance and utility for addressing the inquiry. Document your analytical reasoning in the <reason> field. Just give a short reason.
- For the <summarize> field, You summarize the information in the content to this specific question: {query}. Summary should be concise and to the point. No more than {max_summary_length} sentences.
- For the <quotes> field, Prioritize direct quotations from the original content when appropriate, while employing thoughtful paraphrasing to enhance clarity when necessary.
- If the web resource is inaccessible (e.g., 404 error, access restrictions, rate limitations), classify as evaluate="unavailable".
- If the content lacks relevance to the inquiry after careful analysis, classify as evaluate="not_related".
- See if you can infer the publication date of this content based on the original text, and if possible, return it in <datetime>. format like "2024-01-01".
"""


class SummarizeResult(BaseModel):
    reason: str
    summarize: str
    quotes: List[str]
    datetime: str | None
    evaluate: Literal["useful", "not_related", "unavailable"]


MAX_SUMMARY_LENGTH = 5


async def summarize(
    ctx: RunContextWrapper[TaskContext], query: str, origin_query: str, content: str
) -> SummarizeResult:
    """
    - Summarize the content of the web page
    """
    ret = await get_response(
        model="summarize",
        input=f"the web page content is: \n{content}",
        output_type=SummarizeResult,
        system_instructions=SUMMARIZE_PROMPT.format(
            query_content=query_content(origin_query, query),
            query=query,
            max_summary_length=MAX_SUMMARY_LENGTH,
        ),
    )
    ctx.usage.add(ret.usage)
    return ret.response


def query_content(origin_query: str, query: str) -> str:
    if origin_query == query:
        return query
    else:
        return f"Original query: {origin_query}\nMore specifically: {query}"
