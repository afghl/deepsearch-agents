import contextvars
from dataclasses import dataclass, field
import time
from typing import Dict, Any, List, Literal, Optional
import uuid

from pydantic import BaseModel

from deepsearch_agents._utils import Scope


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
    references: List[str] = field(default_factory=list)


class Knowledge(BaseModel):
    reference: str
    answer: str


@dataclass
class Task:
    origin_query: str
    query: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # TODO: 这里需要这个字段么？ 还是说一个bool值 is_solved就可以？
    # 这个字段可能会让模型来判断，比如是good solve 还是bad solve
    status: Literal["unresolved", "solved", "failed"] = "unresolved"
    level: int = 1
    turn: int = 1
    parent: "Task | None" = None
    sub_tasks: Dict[str, "Task"] = field(default_factory=dict)
    progress: List[str] = field(default_factory=list)
    knowledges: List[Knowledge] = field(default_factory=list)
    answer: Answer | None = None


@dataclass
class TaskContext:
    start_date_time: str
    origin_query: str
    final_answer: Answer | None = None
    tasks: Dict[str, Task] = field(default_factory=dict)

    def __init__(self, task):
        self.origin_query = task.origin_query
        self.tasks = {}
        self.tasks[task.id] = task
        self.start_date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def current_task_id(self) -> str:
        """获取当前活动任务ID"""
        return _current_task_id.get()  # type: ignore

    def current_task(self) -> Task:
        """获取当前活动任务"""
        return self.tasks[self.current_task_id()]

    def set_as_current(self, task: Task) -> contextvars.Token:
        """将指定任务设置为当前任务"""
        if task.id not in self.tasks:
            raise ValueError(f"Task ID {task.id} does not exist in this context")
        return _current_task_id.set(task.id)

    def reset_current_task(self, token: contextvars.Token) -> None:
        """恢复之前的任务上下文"""
        _current_task_id.reset(token)


def build_task_context(query: str) -> TaskContext:
    """创建一个新的任务上下文并设置为当前协程的上下文"""
    task = Task(origin_query=query, query=query)
    task_context = TaskContext(task)
    # 为了兼容现有的_utils.py中的Scope类，也设置全局的task_id
    task_context.set_as_current(task)
    return task_context
