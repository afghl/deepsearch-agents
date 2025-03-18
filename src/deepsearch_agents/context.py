from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Task:
    task_id: str
    # TODO: 这里需要这个字段么？ 还是说一个bool值 is_solved就可以？
    # 这个字段可能会让模型来判断，比如是good solve 还是bad solve
    status: str


@dataclass
class TaskContext:
    original_query: str
    tasks: Dict[str, Task] = field(default_factory=dict)
