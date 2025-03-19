from dataclasses import dataclass, field
from typing import Dict, Any, List, Literal, Optional
import uuid


@dataclass
class Task:
    id: Optional[str] = field(default_factory=lambda: uuid.uuid4().hex)
    origin_query: str = None
    question: str = None
    # TODO: 这里需要这个字段么？ 还是说一个bool值 is_solved就可以？
    # 这个字段可能会让模型来判断，比如是good solve 还是bad solve
    status: Literal["unresolved", "solved", "failed"] = "unresolved"
    level: int = 0
    parent: "Task | None" = None
    sub_tasks: Dict[str, "Task"] = field(default_factory=dict)
    progress: List[str] = field(default_factory=list)

    # def __post_init__(self):
    #     if self.id is None:
    #         self.id = uuid.uuid4().hex

    @property
    def key(self) -> str:
        return self.question


@dataclass
class TaskContext:
    origin_query: str
    tasks: Dict[str, Task] = field(default_factory=dict)
    last_used_tool: str = None

    def __init__(self, task):
        self.origin_query = task.origin_query
        self.tasks = {}
        self.tasks[task.id] = task

    def current_task(self) -> Task:
        return self.tasks[self.origin_query]
