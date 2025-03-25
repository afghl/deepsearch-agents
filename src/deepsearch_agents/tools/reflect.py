from typing import List, Optional
from agents import RunContextWrapper, function_tool

from deepsearch_agents._utils import Scope
from deepsearch_agents.context import TaskContext
from deepsearch_agents.tools._utils import tool_instructions


def reflect_description(ctx: Optional[TaskContext] = None) -> str:
    return f"""
    - Engage in reflection only after establishing a comprehensive understanding of the inquiry, as this allows for more accurate identification of knowledge gaps.
    - Employ methodical analysis and forward planning when examining the question, context, and previous user interactions to identify areas requiring additional information.
    - Formulate a concise list of essential clarifying questions that are directly relevant to the original inquiry and will facilitate a comprehensive response.
    - Ensure each question is specific, precise, and avoids ambiguity or excessive breadth. Each inquiry should be independently comprehensible without requiring additional context.
    - Construct each question with careful consideration to ensure independence from other inquiries, allowing for autonomous responses to each question.
    - Limit the reflection to a maximum of three focused questions.
    """


tool_instructions["reflect"] = reflect_description


@function_tool()
def reflect(
    ctx: RunContextWrapper[TaskContext],
    origin_question: str,
    reason: str,
    questions_to_answer: list[str],
) -> str:
    """
    - Reflect the gaps and plan a list key clarifying questions that deeply related to the original question and lead to the answer.
    - New questions will be delegated to the new agent and be answered.

    Args:
        origin_question: The original question you are trying to answer.
        reason: Based on the original question and the context, think about the knowledge gaps and what you need to know to answer the question. Explain the reason why you need to ask these questions.
        questions_to_answer: Reflection and planing, generate a list of most important questions to fill the knowledge gaps to original question.
    """
    print(
        f"Perform Reflect. Questions to answer: {len(questions_to_answer)}, origin question: {origin_question}, reason: {reason}, current_task_id: {ctx.context.current_task_id()}"
    )  # For debugging purposes, remove in production
    return "|".join(questions_to_answer)
