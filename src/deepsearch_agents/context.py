import contextvars
from dataclasses import dataclass, field
import time
from typing import Dict, List, Literal
import uuid

from pydantic import BaseModel


_current_task_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_task_id", default=None
)


@dataclass
class Evaluation:
    pass


@dataclass
class Answer:
    evaluation: Evaluation | None = None
    answer: str | None = None
    # TODO: need this ?
    references: List[str] = field(default_factory=list)


class Knowledge(BaseModel):
    reference: str
    answer: str
    quotes: List[str]


@dataclass
class Task:
    origin_query: str
    query: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: Literal["unresolved", "solved", "failed"] = "unresolved"
    level: int = 1
    turn: int = 1
    parent: "Task | None" = None
    sub_tasks: Dict[str, "Task"] = field(default_factory=dict)
    progress: List[str] = field(default_factory=list)
    knowledges: List[Knowledge] = field(default_factory=list)
    answer: Answer | None = None

    def set_as_current(self) -> contextvars.Token:
        """Set the current task as the current task"""
        return _current_task_id.set(self.id)


@dataclass
class TaskContext:
    start_date_time: str
    tasks: Dict[str, Task] = field(default_factory=dict)

    def __init__(self, task):
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
