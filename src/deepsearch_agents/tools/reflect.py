from typing import List, Optional
from agents import RunContextWrapper, function_tool

from deepsearch_agents.context import TaskContext
from deepsearch_agents.log import logger
from deepsearch_agents.tools._utils import tool_instructions


def reflect_description(ctx: Optional[TaskContext] = None) -> str:
    return f"""
    - ONLY reflect AFTER you have a fundamental understanding of the question. because you will have a better idea of the specific knowledge gap.
    - Think slowly and planning lookahead. Examine <question>, <context>, previous conversation with users to identify knowledge gaps.
    - Reflect the gaps and plan a list key clarifying questions that deeply related to the original question and lead to the answer.
    - The questions should be specific and clear, and should not be too broad or vague. Each question should be independent and understandable without context.
    - Phrase each question carefully so they don't rely on each other and can be answered independently.
    - Maximun 3 questions.
    """


tool_instructions["reflect"] = reflect_description


@function_tool()
def reflect(
    ctx: RunContextWrapper[TaskContext],
    think: str,
    origin_question: str,
    questions_to_answer: list[str],
) -> str:
    """
    - Reflect the gaps and plan a list key clarifying questions that deeply related to the original question and lead to the answer.
    - New questions will be delegated to the new agent and be answered.

    Args:
        think: Based on the original question and the context, think about the knowledge gaps and what you need to know to answer the question. Explain the reason why you need to ask these questions, why would it help towards the end goal.
        origin_question: The original question you are trying to answer.
        questions_to_answer: Reflection and planing, generate a list of most important questions to fill the knowledge gaps to original question.
    """
    logger.info(
        f"Perform Reflect. Questions to answer: {len(questions_to_answer)}, origin question: {origin_question}, think: {think}, current_task_id: {ctx.context.current_task_id()}"
    )  # For debugging purposes, remove in production
    return "|".join(questions_to_answer)
