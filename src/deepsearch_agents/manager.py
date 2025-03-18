from dataclasses import dataclass
from typing import Dict

from agents import Agent, RunContextWrapper, Runner

from deepsearch_agents.agent import answer, reflect, search, visit
from deepsearch_agents.context import TaskContext


@dataclass
class Manager(Agent[TaskContext]):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.instructions = self._get_instructions
        self.tools = [
            search,
            reflect,
            visit,
            answer,
        ]

    def _get_instructions(
        self, ctx: RunContextWrapper[TaskContext], agent: Agent[TaskContext]
    ) -> str:

        return """
    You are an advanced AI research agent from Jina AI. You are specialized in multistep reasoning. 
    Using your best knowledge, conversation with the user and lessons learned, answer the user question with absolute certainty.
    """
