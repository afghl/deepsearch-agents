import contextvars
from typing import Callable, Dict, List


_current_task_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_task_id", default=None
)


class Scope:
    @classmethod
    def get_current_task_id(cls) -> str | None:
        return _current_task_id.get()

    @classmethod
    def set_current_task_id(cls, task: str) -> contextvars.Token[str | None]:
        return _current_task_id.set(task)

    @classmethod
    def reset_current_task(cls, token: contextvars.Token[str | None]) -> None:
        _current_task_id.reset(token)
