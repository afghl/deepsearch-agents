from dataclasses import dataclass, field
from typing import Dict, Any, List, Literal, Optional
import uuid

from deepsearch_agents._utils import Scope


@dataclass
class Evaluation:
    pass


@dataclass
class Answer:
    evaluation: Evaluation = None
    answer: str | None = None
    references: List[str] = field(default_factory=list)


@dataclass
class Task:
    id: Optional[str] = field(default_factory=lambda: uuid.uuid4().hex)
    origin_query: str = None
    query: str = None
    # TODO: 这里需要这个字段么？ 还是说一个bool值 is_solved就可以？
    # 这个字段可能会让模型来判断，比如是good solve 还是bad solve
    status: Literal["unresolved", "solved", "failed"] = "unresolved"
    level: int = 0
    parent: "Task | None" = None
    sub_tasks: Dict[str, "Task"] = field(default_factory=dict)
    progress: List[str] = field(default_factory=list)
    answer: Answer | None = None


@dataclass
class TaskContext:
    origin_query: str
    final_answer: Answer | None = None
    tasks: Dict[str, Task] = field(default_factory=dict)

    def __init__(self, task):
        self.origin_query = task.origin_query
        self.tasks = {}
        self.tasks[task.id] = task

    def current_task(self) -> Task:
        current_task_id = Scope.get_current_task_id()
        if not current_task_id:
            print(f"No current task id found in context")
        return self.tasks[current_task_id]


def build_task_context(query: str) -> TaskContext:
    task = Task(origin_query=query)
    task_context = TaskContext(task)
    # Set the current task id in the context
    Scope.set_current_task_id(task.id)
    return task_context
