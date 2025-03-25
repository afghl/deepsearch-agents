from abc import ABC, abstractmethod

from agents import RunContextWrapper, Tool

from deepsearch_agents.context import TaskContext


class RecordableTool(Tool):

    @abstractmethod
    def execute(
        self, ctx: RunContextWrapper[TaskContext], think: str, *args, **kwargs
    ) -> str:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class MyTool(RecordableTool):
    def execute(
        self,
        ctx: RunContextWrapper[TaskContext],
        think: str,
        additional_param: str,
        *args,
        **kwargs,
    ) -> str:
        # 实现具体逻辑
        pass

    def get_name(self) -> str:
        return "MyTool"
