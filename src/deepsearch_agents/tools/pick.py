from typing import List
from agents import RunContextWrapper, Usage
import numpy as np
from pydantic import BaseModel
import asyncio

from deepsearch_agents.context import Task, TaskContext, build_task_context
from deepsearch_agents.llm.emb import get_embedding


class PickResult(BaseModel):
    start: int
    score: float


async def pick_content(
    ctx: RunContextWrapper[TaskContext],
    content: str,
    max_length: int,
    window_step: int,
) -> PickResult:
    """
    Find the most relevant section of text by comparing embeddings similarity between the task query
    and content windows. Returns the start index and score of the best matching window.
    """

    curr_task = ctx.context.current_task()
    question_embeddings = curr_task.question_embeddings
    window_size = max_length // window_step
    if question_embeddings is None:
        question_embeddings = await _get_embeddings(ctx, "embedding", curr_task.query)
        curr_task.question_embeddings = question_embeddings

    # Create tasks for parallel embedding computation
    tasks = []
    for i in range(0, len(content) - window_size * window_step, window_size):
        window_content = content[i : i + window_size]
        tasks.append(_get_embeddings(ctx, "embedding", window_content))

    # Run all embedding tasks in parallel
    content_embeddings_list = await asyncio.gather(*tasks)

    similarity_scores = [
        _cosine_similarity(question_embeddings, content_emb)
        for content_emb in content_embeddings_list
    ]
    window_similarity_scores: List[float] = []
    for i in range(len(similarity_scores)):
        window_score = np.mean(similarity_scores[i : i + window_size - 1])
        window_similarity_scores.append(window_score)  # type: ignore

    # Find the best snippet with the highest score
    max_score = max(window_similarity_scores)
    best_start_index = window_similarity_scores.index(max_score)

    return PickResult(start=best_start_index * window_size, score=max_score)


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
