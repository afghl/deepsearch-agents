from deepsearch_agents.log import logger
from typing import Any, List, Tuple
from agents import RunContextWrapper, Usage
import numpy as np
from pydantic import BaseModel
import asyncio

from deepsearch_agents.context import Task, TaskContext, build_task_context
from deepsearch_agents.llm.emb import get_embedding


async def pick_content(
    ctx: RunContextWrapper[TaskContext],
    content: str,
    max_length: int,
    window_step: int,
) -> str:
    """
    Pick the most relevant content, according to the question embeddings.
    TODO: This is a naive implementation, we should use a more sophisticated method.
    Args:
        ctx: Run context wrapper containing task context
        content: The content to pick from
        max_length: Maximum length of the picked content
        window_step: Step size for sliding window
    Returns:
        The picked content
    """
    if len(content) < max_length:
        return content
    curr_task = ctx.context.current_task()
    question_embeddings = curr_task.question_embeddings
    window_size = max_length // window_step

    # Get question embeddings if not cached
    if question_embeddings is None:
        question_embeddings = await _get_embeddings(ctx, "embedding", curr_task.query)
        curr_task.question_embeddings = question_embeddings

    async def process_window(start_idx: int) -> float:
        """Process a single window and compute its similarity score."""
        window_content = content[start_idx : start_idx + window_size]
        content_emb = await _get_embeddings(ctx, "embedding", window_content)
        similarity = _cosine_similarity(question_embeddings, content_emb)
        return similarity

    # Run all tasks in parallel and collect results
    res = await asyncio.gather(
        *[
            process_window(i)
            for i in range(0, len(content) - window_size * window_step, window_size)
        ]
    )
    window_similarity_scores = np.array(res)
    assert len(window_similarity_scores) > 0, "No window similarity scores"
    # Calculate average scores for each large window
    avg_scores = [
        np.mean(window_similarity_scores[i : i + window_step])
        for i in range(0, len(window_similarity_scores) - window_step + 1)
    ]
    # Find the best window with highest average score
    start = np.array(avg_scores).argmax() * window_size  # type: ignore
    return content[start : start + max_length]


async def _get_embeddings(
    ctx: RunContextWrapper[TaskContext], model: str, text: str
) -> List[float]:
    response = await get_embedding(model=model, text=text)
    ctx.usage.add(response.usage)
    return response.embedding


def _cosine_similarity(
    question_embeddings: List[float], content_embeddings: List[float]
) -> float:
    return np.dot(question_embeddings, content_embeddings) / (
        np.linalg.norm(question_embeddings) * np.linalg.norm(content_embeddings)
    )
