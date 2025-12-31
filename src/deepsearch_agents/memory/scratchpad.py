"""Scratchpad for tracking agent action history.

This module provides a structured way to record and summarize
the agent's decision history, enabling better context awareness
for subsequent decisions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import time


class ActionType(str, Enum):
    """Types of actions the agent can take."""

    SEARCH = "search"
    VISIT = "visit"
    REFLECT = "reflect"
    ANSWER = "answer"
    OTHER = "other"


@dataclass
class ActionEntry:
    """A single action record in the scratchpad."""

    action_type: ActionType
    """Type of action performed."""

    summary: str
    """Brief description of the action (e.g., 'searched for SPX performance')."""

    turn: int
    """Turn number when this action was taken."""

    outcome: str = ""
    """Brief outcome description (e.g., 'found 25 results')."""

    is_success: bool = True
    """Whether the action was successful."""

    details: Optional[str] = None
    """Optional additional details (not shown in compact view)."""

    timestamp: float = field(default_factory=time.time)
    """When this action was recorded."""

    def to_compact_string(self) -> str:
        """Generate a compact single-line representation.

        Returns:
            Compact string like "[search] ✓ searched SPX → 25 results"
        """
        status = "✓" if self.is_success else "✗"
        result = f"[{self.action_type.value}] {status} {self.summary}"
        if self.outcome:
            result += f" → {self.outcome}"
        return result

    def to_detailed_string(self) -> str:
        """Generate a detailed multi-line representation.

        Returns:
            Detailed string with all information.
        """
        lines = [self.to_compact_string()]
        if self.details:
            lines.append(f"   Details: {self.details}")
        return "\n".join(lines)


@dataclass
class ScratchpadConfig:
    """Configuration for Scratchpad."""

    max_entries: int = 15
    """Maximum number of entries to keep."""

    max_tokens: int = 2000
    """Maximum tokens for context output."""

    compress_after: int = 10
    """Start compressing entries after this count."""

    show_details_for_recent: int = 3
    """Show details for the N most recent entries."""


@dataclass
class Scratchpad:
    """Agent action history tracker.

    Records actions taken by the agent and provides formatted
    summaries for context injection, helping the agent understand
    its own decision history.
    """

    entries: List[ActionEntry] = field(default_factory=list)
    config: ScratchpadConfig = field(default_factory=ScratchpadConfig)

    # Summary of compressed (oldest) entries
    _compressed_summary: str = ""
    _compressed_count: int = 0

    def record(
        self,
        action_type: ActionType,
        summary: str,
        turn: int,
        outcome: str = "",
        is_success: bool = True,
        details: Optional[str] = None,
    ) -> ActionEntry:
        """Record a new action.

        Args:
            action_type: Type of action performed.
            summary: Brief description of the action.
            turn: Current turn number.
            outcome: Brief outcome description.
            is_success: Whether the action succeeded.
            details: Optional additional details.

        Returns:
            The created ActionEntry.
        """
        entry = ActionEntry(
            action_type=action_type,
            summary=summary,
            turn=turn,
            outcome=outcome,
            is_success=is_success,
            details=details,
        )
        self.entries.append(entry)
        self._trim_if_needed()
        return entry

    def record_search(
        self,
        queries: List[str],
        turn: int,
        result_count: int = 0,
    ) -> ActionEntry:
        """Convenience method to record a search action.

        Args:
            queries: Search queries used.
            turn: Current turn number.
            result_count: Number of results found.

        Returns:
            The created ActionEntry.
        """
        summary = f"searched: {', '.join(queries[:2])}"
        if len(queries) > 2:
            summary += f" (+{len(queries)-2} more)"

        outcome = f"{result_count} results" if result_count > 0 else "no results"

        return self.record(
            action_type=ActionType.SEARCH,
            summary=summary,
            turn=turn,
            outcome=outcome,
            is_success=result_count > 0,
        )

    def record_visit(
        self,
        urls: List[str],
        turn: int,
        useful_count: int = 0,
    ) -> ActionEntry:
        """Convenience method to record a visit action.

        Args:
            urls: URLs visited.
            turn: Current turn number.
            useful_count: Number of URLs that had useful content.

        Returns:
            The created ActionEntry.
        """
        summary = f"visited {len(urls)} URL(s)"
        outcome = f"{useful_count} useful" if useful_count > 0 else "nothing useful"

        return self.record(
            action_type=ActionType.VISIT,
            summary=summary,
            turn=turn,
            outcome=outcome,
            is_success=useful_count > 0,
        )

    def record_reflect(
        self,
        questions: List[str],
        turn: int,
    ) -> ActionEntry:
        """Convenience method to record a reflect action.

        Args:
            questions: Sub-questions generated.
            turn: Current turn number.

        Returns:
            The created ActionEntry.
        """
        summary = f"identified {len(questions)} sub-question(s)"
        outcome = "; ".join(q[:50] for q in questions[:2])

        return self.record(
            action_type=ActionType.REFLECT,
            summary=summary,
            turn=turn,
            outcome=outcome,
            is_success=True,
        )

    def record_answer(
        self,
        turn: int,
        is_pass: bool,
        feedback: str = "",
    ) -> ActionEntry:
        """Convenience method to record an answer action.

        Args:
            turn: Current turn number.
            is_pass: Whether the answer passed evaluation.
            feedback: Evaluation feedback if failed.

        Returns:
            The created ActionEntry.
        """
        summary = "provided answer"
        outcome = "accepted" if is_pass else f"rejected: {feedback[:50]}"

        return self.record(
            action_type=ActionType.ANSWER,
            summary=summary,
            turn=turn,
            outcome=outcome,
            is_success=is_pass,
        )

    def _trim_if_needed(self) -> None:
        """Trim entries if exceeding max count, compressing old ones."""
        if len(self.entries) <= self.config.max_entries:
            return

        # Compress oldest entries into summary
        excess = len(self.entries) - self.config.max_entries
        to_compress = self.entries[:excess]

        # Build compressed summary
        action_counts: dict[ActionType, int] = {}
        for entry in to_compress:
            action_counts[entry.action_type] = (
                action_counts.get(entry.action_type, 0) + 1
            )

        summary_parts = [
            f"{count}x {action.value}" for action, count in action_counts.items()
        ]
        new_summary = f"Earlier: {', '.join(summary_parts)}"

        if self._compressed_summary:
            self._compressed_summary = f"{self._compressed_summary}; {new_summary}"
        else:
            self._compressed_summary = new_summary

        self._compressed_count += excess
        self.entries = self.entries[excess:]

    def get_action_counts(self) -> dict[ActionType, int]:
        """Get count of each action type.

        Returns:
            Dictionary mapping action types to counts.
        """
        counts: dict[ActionType, int] = {}
        for entry in self.entries:
            counts[entry.action_type] = counts.get(entry.action_type, 0) + 1
        return counts

    def get_last_action(self) -> Optional[ActionEntry]:
        """Get the most recent action entry.

        Returns:
            Most recent ActionEntry or None if empty.
        """
        return self.entries[-1] if self.entries else None

    def get_actions_by_type(self, action_type: ActionType) -> List[ActionEntry]:
        """Get all entries of a specific action type.

        Args:
            action_type: The type to filter by.

        Returns:
            List of matching entries.
        """
        return [e for e in self.entries if e.action_type == action_type]

    def to_context_string(self, include_stats: bool = True) -> str:
        """Generate a formatted string for LLM context injection.

        Args:
            include_stats: Whether to include summary statistics.

        Returns:
            Formatted action history string.
        """
        if not self.entries and not self._compressed_summary:
            return "## Action History\n(No actions taken yet)"

        lines = ["## Action History"]

        # Add compressed summary if exists
        if self._compressed_summary:
            lines.append(
                f"[{self._compressed_count} earlier actions: {self._compressed_summary}]"
            )
            lines.append("")

        # Add recent entries
        recent_count = self.config.show_details_for_recent
        for i, entry in enumerate(self.entries):
            is_recent = i >= len(self.entries) - recent_count
            if is_recent and entry.details:
                lines.append(
                    f"{i + self._compressed_count + 1}. {entry.to_detailed_string()}"
                )
            else:
                lines.append(
                    f"{i + self._compressed_count + 1}. {entry.to_compact_string()}"
                )

        # Add stats if requested
        if include_stats:
            lines.append("")
            counts = self.get_action_counts()
            stats_str = ", ".join(f"{v} {k.value}(s)" for k, v in counts.items())
            total = len(self.entries) + self._compressed_count
            lines.append(f"_Total: {total} actions ({stats_str})_")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all entries and compressed history."""
        self.entries.clear()
        self._compressed_summary = ""
        self._compressed_count = 0

    def stats(self) -> dict:
        """Get statistics about the scratchpad.

        Returns:
            Dictionary with scratchpad statistics.
        """
        return {
            "total_entries": len(self.entries) + self._compressed_count,
            "active_entries": len(self.entries),
            "compressed_count": self._compressed_count,
            "action_counts": {k.value: v for k, v in self.get_action_counts().items()},
            "has_failures": any(not e.is_success for e in self.entries),
        }
