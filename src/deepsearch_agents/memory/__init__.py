"""Memory management module for context engineering.

This module provides:
- KnowledgeStore: Hierarchical knowledge storage with compression levels
- Scratchpad: Action history tracking for agent decision transparency
- ContextBuilder: Dynamic context construction with budget management
- KnowledgeCompressor: LLM-based intelligent compression
"""

from .knowledge_store import (
    CompressionLevel,
    ManagedKnowledge,
    KnowledgeStore,
    KnowledgeStoreConfig,
)
from .scratchpad import (
    ActionType,
    ActionEntry,
    Scratchpad,
    ScratchpadConfig,
)
from .compression import (
    KnowledgeCompressor,
    CompressedContent,
)
from .context_builder import (
    ContextBudget,
    ContextBuilder,
    TaskState,
)

__all__ = [
    # Knowledge Store
    "CompressionLevel",
    "ManagedKnowledge",
    "KnowledgeStore",
    "KnowledgeStoreConfig",
    # Scratchpad
    "ActionType",
    "ActionEntry",
    "Scratchpad",
    "ScratchpadConfig",
    # Compression
    "KnowledgeCompressor",
    "CompressedContent",
    # Context Builder
    "ContextBudget",
    "ContextBuilder",
    "TaskState",
]
