import asyncio
from typing import List

from agents import RunContextWrapper
from deepsearch_agents import conf
from deepsearch_agents.context import (
    Evaluation,
    Knowledge,
    Reference,
    TaskContext,
    Task,
    build_task_context,
)
from deepsearch_agents.llm.llm import get_response
from deepsearch_agents.log import logger


EVALUATION_PROMPT = """
You are a ruthless answer evaluator trained to REJECT answers. 
Given a question-answer pair, your job is to find ANY weakness in the presented answer. 

-Rules- 
1. Extremely strict standards of evidence apply. 
2. Identity EVERY missing detail. 
3. First, argue AGAINST the answer with the strongest possible case. 
4. Then, argue FOR the answer. 
5. Only after considering both perspectives, synthesize a final improvement plan starts with "For the best answer, you must...".

The user will also provide the knowledge items he used to answer the question. Note that some of them may not be directly related to the question/answer user provided. 


"""

USER_PROMPT = """
the question is: {question}.

Here are some information I found to answer the question:
{knowledge}

Here is the answer I provided:
{answer}

Here are the references I used:
{references}
"""


async def evaluate_answer(
    ctx: RunContextWrapper[TaskContext],
    answer: str,
    references: List[Reference],
) -> Evaluation:
    curr = ctx.context.current_task()
    config = conf.get_configuration()
    curr.attempt += 1
    if (
        not curr.is_origin_query()
        or curr.attempt > config.execution_config.max_critical_attempts
    ):
        # only evaluate the answer of the origin query
        return Evaluation(
            is_pass=True,
            critic="",
            improvement="",
            reason="",
        )

    question = curr.query
    knowledge = knowledge_list(curr)
    ret = await get_response(
        model="evaluate",
        input=USER_PROMPT.format(
            question=question, answer=answer, knowledge=knowledge, references=references
        ),
        output_type=Evaluation,
        system_instructions=EVALUATION_PROMPT,
    )
    ctx.usage.add(ret.usage)
    return ret.response


def knowledge_list(task: Task) -> str:
    if not task.knowledges:
        return ""

    ret = task.list_out_knowledge()
    if any(sub_task.answer for sub_task in task.sub_tasks.values()):
        ret += "\n\nIn order to dig deeper, And provide a more comprehensive answer, I did some research on the following aspects: \n"
        for sub_task in task.sub_tasks.values():
            if sub_task.answer:
                ret += f"{sub_task.query}\n"
                ret += f"After some research, I concluded on this answer: \n{sub_task.answer.answer}\n"

    return ret
