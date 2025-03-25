import functools
from typing import (
    Any,
    Callable,
    Concatenate,
    List,
    Mapping,
    ParamSpec,
    Sequence,
    TypeVar,
    overload,
)
from deepsearch_agents.log import logger
from agents import RunContextWrapper

from deepsearch_agents.context import TaskContext

P = ParamSpec("P")
R = TypeVar("R")

ToolFunction = Callable[Concatenate[RunContextWrapper[TaskContext], P], R]


def log_start(
    ctx: RunContextWrapper[TaskContext],
    *args: Sequence[Any],
    **kwargs: Mapping[str, Any],
):
    logger.info(f"Start: {args} {kwargs}")


def log_end(
    ctx: RunContextWrapper[TaskContext],
    *args: Sequence[Any],
    **kwargs: Mapping[str, Any],
):
    logger.info(f"End: {args} {kwargs}")


@overload
def trace(
    fn: ToolFunction,
    *,
    on_starts: List[Callable[..., None]] = [log_start],
    on_ends: List[Callable[..., None]] = [log_end],
) -> ToolFunction:
    """Overload for usage as @trace(fn)."""
    ...


@overload
def trace(
    *,
    on_starts: List[Callable[..., None]] = [log_start],
    on_ends: List[Callable[..., None]] = [log_end],
) -> Callable[[ToolFunction], ToolFunction]:
    """Overload for usage as @trace(...)."""
    ...


def trace(
    fn: ToolFunction | None = None,
    *,
    on_starts: List[Callable[..., None]] = [log_start],
    on_ends: List[Callable[..., None]] = [log_end],
) -> ToolFunction | Callable[[ToolFunction], ToolFunction]:
    """
    Decorator to trace function execution. It can be used in two ways:
    1. As @trace - directly decorate a function
    2. As @trace(...) - decorate a function with custom on_starts and on_ends handlers

    Args:
        fn: The function to trace
        on_starts: List of functions to call before executing the traced function
        on_ends: List of functions to call after executing the traced function

    Returns:
        The decorated function that preserves the original signature
    """

    if callable(fn):
        return _create_traced_fn(fn, on_starts, on_ends)
    else:
        return _create_trace_decorator(on_starts, on_ends)


def _create_traced_fn(
    fn: ToolFunction,
    on_starts: List[Callable[..., None]],
    on_ends: List[Callable[..., None]],
) -> ToolFunction:
    def traced_fn(ctx: RunContextWrapper[TaskContext], *args, **kwargs):
        for on_start_fn in on_starts:
            on_start_fn(ctx, *args, **kwargs)
        result = fn(ctx, *args, **kwargs)
        for on_end_fn in on_ends:
            on_end_fn(ctx, *args, **kwargs)
        return result

    return traced_fn


def _create_trace_decorator(
    on_starts: List[Callable[..., None]], on_ends: List[Callable[..., None]]
) -> Callable[[ToolFunction], ToolFunction]:
    def decorator(fn: ToolFunction) -> ToolFunction:
        return _create_traced_fn(fn, on_starts, on_ends)

    return decorator


if __name__ == "__main__":

    @trace
    def test(
        ctx: RunContextWrapper[TaskContext], think: str, search_queries: List[str]
    ) -> int:
        logger.info(f"test: {think}")
        return 1

    def my_start(ctx: RunContextWrapper[TaskContext], value: int) -> None:
        logger.info(f"my_start: {value}")

    # Example with custom parameters
    @trace(on_starts=[log_start, my_start])
    def test2(ctx: RunContextWrapper[TaskContext], value: int) -> str:
        return f"Value: {value}"

    logger.info(test(ctx=1, think="test", search_queries=["test"]))
    logger.info(test2(ctx=1, value=42))
