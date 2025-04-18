from typing import List, Optional
from agents import RunContextWrapper, function_tool

from deepsearch_agents.context import TaskContext
from deepsearch_agents.log import logger
from ._utils import log_action, tool_instructions

sep = "<|sub_task_separator|>"


def reflect_description(ctx: Optional[TaskContext] = None) -> str:
    return f"""
    - ONLY reflect AFTER you have a fundamental understanding of the question. because you will have a better idea of the specific knowledge gap.
    - Think slowly and planning lookahead. Examine <question>, <context>, previous conversation with users to identify knowledge gaps.
    - Reflect the gaps and plan a list key clarifying questions that deeply related to the original question and lead to the answer.
    - The questions should be specific and clear, and should not be too broad or vague. Each question should be independent and understandable without context.
    - Phrase each question carefully so they don't rely on each other and can be answered independently.
    - Maximun 2 questions.
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
        questions_to_answer: Reflection and planing, generate a list of most important questions to fill the knowledge gaps to original question. Maximun 2 questions.
    """
    log_action(ctx, "reflect", think, origin_question=origin_question, questions_to_answer=questions_to_answer)  # type: ignore
    return sep.join(questions_to_answer[:3])
