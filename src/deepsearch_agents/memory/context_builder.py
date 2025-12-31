"""Context builder for dynamic LLM context construction.

This module provides a unified way to build optimized context
for LLM prompts, managing token budgets across different sections.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Callable

from deepsearch_agents.log import logger
from .knowledge_store import KnowledgeStore, ManagedKnowledge
from .scratchpad import Scratchpad


@dataclass
class ContextBudget:
    """Token budget allocation for different context sections.
    
    The total budget is divided among sections, with conversation_buffer
    automatically calculated as the remaining space.
    """
    
    total: int = 100000
    """Total token budget for the entire context."""
    
    system_prompt: int = 2000
    """Budget for system prompt and instructions."""
    
    task_definition: int = 500
    """Budget for current task/question definition."""
    
    knowledge: int = 30000
    """Budget for knowledge store content."""
    
    scratchpad: int = 2000
    """Budget for action history."""
    
    sub_task_results: int = 5000
    """Budget for sub-task results summary."""
    
    @property
    def conversation_buffer(self) -> int:
        """Remaining space for conversation history."""
        used = (
            self.system_prompt 
            + self.task_definition 
            + self.knowledge 
            + self.scratchpad
            + self.sub_task_results
        )
        return max(0, self.total - used)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "total": self.total,
            "system_prompt": self.system_prompt,
            "task_definition": self.task_definition,
            "knowledge": self.knowledge,
            "scratchpad": self.scratchpad,
            "sub_task_results": self.sub_task_results,
            "conversation_buffer": self.conversation_buffer,
        }


@dataclass
class TaskState:
    """Current state of the task for context building."""
    
    origin_query: str
    """The original user query."""
    
    current_query: str
    """The current query (may be a sub-question)."""
    
    current_turn: int = 0
    """Current turn number."""
    
    task_level: int = 1
    """Depth level of the current task."""
    
    attempt_count: int = 0
    """Number of answer attempts made."""
    
    sub_task_summaries: List[str] = field(default_factory=list)
    """Summaries from completed sub-tasks."""
    
    def is_sub_task(self) -> bool:
        """Check if this is a sub-task (not the root query)."""
        return self.origin_query != self.current_query


@dataclass
class ContextBuilder:
    """Builds optimized context for LLM prompts.
    
    Combines knowledge, action history, and task state into
    a coherent context string, respecting token budgets.
    """
    
    knowledge_store: KnowledgeStore
    """Knowledge storage component."""
    
    scratchpad: Scratchpad
    """Action history tracking component."""
    
    budget: ContextBudget = field(default_factory=ContextBudget)
    """Token budget allocation."""
    
    def build_knowledge_section(
        self, 
        task_state: Optional[TaskState] = None,
    ) -> str:
        """Build the knowledge section of the context.
        
        Args:
            task_state: Optional task state for relevance calculation.
            
        Returns:
            Formatted knowledge context string.
        """
        if not self.knowledge_store.items:
            return ""
        
        current_turn = task_state.current_turn if task_state else 0
        
        content = self.knowledge_store.get_context_string(
            max_tokens=self.budget.knowledge,
            current_turn=current_turn,
        )
        
        if content and content != "(No knowledge collected yet)":
            return f"""## Collected Knowledge

The following knowledge has been gathered from research:

{content}"""
        
        return ""
    
    def build_scratchpad_section(self) -> str:
        """Build the action history section.
        
        Returns:
            Formatted scratchpad context string.
        """
        if not self.scratchpad.entries:
            return ""
        
        return self.scratchpad.to_context_string(include_stats=True)
    
    def build_status_section(self, task_state: TaskState) -> str:
        """Build the current status section.
        
        Args:
            task_state: Current task state.
            
        Returns:
            Formatted status string.
        """
        lines = ["## Current Status"]
        
        lines.append(f"- Turn: {task_state.current_turn}")
        lines.append(f"- Knowledge items: {len(self.knowledge_store.items)}")
        lines.append(f"- Actions taken: {len(self.scratchpad.entries)}")
        
        if task_state.attempt_count > 0:
            lines.append(f"- Answer attempts: {task_state.attempt_count}")
        
        if task_state.is_sub_task():
            lines.append(f"- Task level: {task_state.task_level} (sub-task)")
        
        # Add knowledge store stats
        store_stats = self.knowledge_store.stats()
        usage_pct = float(store_stats["usage_ratio"].rstrip("%"))
        if usage_pct > 50:
            lines.append(f"- Knowledge budget usage: {store_stats['usage_ratio']}")
        
        return "\n".join(lines)
    
    def build_sub_task_results_section(self, task_state: TaskState) -> str:
        """Build the sub-task results section.
        
        Args:
            task_state: Task state containing sub-task summaries.
            
        Returns:
            Formatted sub-task results string.
        """
        if not task_state.sub_task_summaries:
            return ""
        
        lines = ["## Research from Sub-questions"]
        lines.append("")
        lines.append("The following research was done on related sub-questions:")
        lines.append("")
        
        for i, summary in enumerate(task_state.sub_task_summaries, 1):
            # Truncate if too long
            if len(summary) > 500:
                summary = summary[:500] + "..."
            lines.append(f"{i}. {summary}")
            lines.append("")
        
        return "\n".join(lines)
    
    def build_full_context(
        self,
        task_state: TaskState,
        base_instructions: str,
    ) -> str:
        """Build the complete enhanced context.
        
        Combines all sections into a coherent context string
        for LLM prompt injection.
        
        Args:
            task_state: Current task state.
            base_instructions: Base system instructions to enhance.
            
        Returns:
            Complete context string.
        """
        sections = [base_instructions]
        
        # Add knowledge section
        knowledge_section = self.build_knowledge_section(task_state)
        if knowledge_section:
            sections.append("")
            sections.append(knowledge_section)
        
        # Add sub-task results
        sub_task_section = self.build_sub_task_results_section(task_state)
        if sub_task_section:
            sections.append("")
            sections.append(sub_task_section)
        
        # Add scratchpad
        scratchpad_section = self.build_scratchpad_section()
        if scratchpad_section:
            sections.append("")
            sections.append(scratchpad_section)
        
        # Add status
        sections.append("")
        sections.append(self.build_status_section(task_state))
        
        return "\n".join(sections)
    
    async def prepare_context(
        self,
        task_state: TaskState,
        base_instructions: str,
    ) -> str:
        """Prepare context with automatic compression if needed.
        
        This method checks if compression is needed and performs it
        before building the context.
        
        Args:
            task_state: Current task state.
            base_instructions: Base system instructions.
            
        Returns:
            Complete context string after any necessary compression.
        """
        # Check and perform compression if needed
        compressed_count = await self.knowledge_store.compress_if_needed(
            current_turn=task_state.current_turn
        )
        
        if compressed_count > 0:
            logger.info(f"Compressed {compressed_count} knowledge items before context build")
        
        return self.build_full_context(task_state, base_instructions)
    
    def estimate_context_tokens(self, task_state: TaskState) -> dict:
        """Estimate token usage for each section.
        
        Args:
            task_state: Current task state.
            
        Returns:
            Dictionary with estimated tokens per section.
        """
        knowledge_section = self.build_knowledge_section(task_state)
        scratchpad_section = self.build_scratchpad_section()
        status_section = self.build_status_section(task_state)
        sub_task_section = self.build_sub_task_results_section(task_state)
        
        # Rough estimate: 4 chars per token
        return {
            "knowledge": len(knowledge_section) // 4,
            "scratchpad": len(scratchpad_section) // 4,
            "status": len(status_section) // 4,
            "sub_tasks": len(sub_task_section) // 4,
            "total_estimated": (
                len(knowledge_section) 
                + len(scratchpad_section) 
                + len(status_section)
                + len(sub_task_section)
            ) // 4,
        }
    
    def clear(self) -> None:
        """Clear all stored context data."""
        self.knowledge_store.clear()
        self.scratchpad.clear()
    
    def stats(self) -> dict:
        """Get combined statistics.
        
        Returns:
            Dictionary with stats from all components.
        """
        return {
            "knowledge": self.knowledge_store.stats(),
            "scratchpad": self.scratchpad.stats(),
            "budget": self.budget.to_dict(),
        }


