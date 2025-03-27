import contextvars
from dataclasses import dataclass, field
import time
from typing import Dict, List, Literal
import uuid

from pydantic import BaseModel


_current_task_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_task_id", default=None
)


class Evaluation(BaseModel):
    reason: str
    is_pass: bool
    critic: str
    improvement: str


class Reference(BaseModel):
    url: str
    datetime: str | None

    def __str__(self) -> str:
        return self.model_dump_json()


class Answer(BaseModel):
    "The final answer to the question."

    evaluation: Evaluation | None = None
    answer: str | None = None
    references: List[Reference] = field(default_factory=list)


class Knowledge(BaseModel):
    "A piece of knowledge is an answer found by an agent to a certain aspect of a question, from outside sources."

    reference: Reference
    "The source of the knowledge."
    summary: str
    "How the reference answers the question."
    quotes: List[str]
    "Quotes from the reference that support the answer."

    def __str__(self) -> str:
        return self.model_dump_json()


_list_out_knowledge_template = """
{i}. I visited this website: {knowledge.reference}. \nHere is the answer I got: \n{knowledge.summary}, \nAnd some quotes are: \n{knowledge.quotes}
"""


@dataclass
class Task:
    origin_query: str
    query: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:6])
    level: int = 1
    turn: int = 0
    parent: "Task | None" = None
    sub_tasks: Dict[str, "Task"] = field(default_factory=dict)
    progress: List[str] = field(default_factory=list)
    knowledges: List[Knowledge] = field(default_factory=list)
    answer: Answer | None = None
    attempt: int = 0

    def is_origin_query(self) -> bool:
        return self.origin_query == self.query

    def set_as_current(self) -> contextvars.Token:
        """Set the current task as the current task"""
        return _current_task_id.set(self.id)

    def list_out_knowledge(self, template: str = _list_out_knowledge_template) -> str:
        return "\n\n".join(
            [
                template.format(
                    knowledge=knowledge,
                    i=i + 1,
                )
                for i, knowledge in enumerate(self.knowledges)
            ]
        )

    def solved(self) -> bool:
        return bool(
            self.answer
            and self.answer.answer
            and self.answer.evaluation
            and self.answer.evaluation.is_pass
        )


@dataclass
class TaskContext:
    start_date_time: str
    tasks: Dict[str, Task] = field(default_factory=dict)

    def __init__(self, task: Task):
        self.tasks = {}
        self.tasks[task.id] = task
        self.start_date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def current_task_id(self) -> str:
        """Get the current active task ID"""
        return _current_task_id.get()  # type: ignore

    def current_task(self) -> Task:
        """Get the current active task"""
        return self.tasks[self.current_task_id()]

    def reset_current_task(self, token: contextvars.Token) -> None:
        """Restore the previous task context"""
        _current_task_id.reset(token)

    def final_answer(self) -> Answer | None:
        root_task = next((task for task in self.tasks.values() if task.parent is None))
        return root_task.answer


def build_task_context(query: str) -> TaskContext:
    """Create a new task context and set it in the current coroutine context"""
    task = Task(origin_query=query, query=query)
    task_context = TaskContext(task)
    task.set_as_current()
    return task_context
