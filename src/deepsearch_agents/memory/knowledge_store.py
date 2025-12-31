"""Knowledge storage with hierarchical compression levels.

This module implements a knowledge store that supports progressive summarization
through multiple compression levels, enabling efficient context window management.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Callable, Awaitable
import time


class CompressionLevel(IntEnum):
    """Compression levels for knowledge items.
    
    Higher levels indicate more compression (fewer tokens).
    """
    RAW = 0        # Original full content with all quotes and details
    CONDENSED = 1  # Compressed summary with key quotes only
    ESSENCE = 2    # Single sentence core insight


@dataclass
class ManagedKnowledge:
    """A knowledge item with compression metadata.
    
    Represents a piece of information gathered from external sources,
    with support for progressive compression to manage context window budget.
    """
    
    # Reference information
    reference_url: str
    reference_title: str
    reference_datetime: Optional[str] = None
    
    # Content (changes based on compression level)
    summary: str = ""
    quotes: List[str] = field(default_factory=list)
    
    # Compression metadata
    level: CompressionLevel = CompressionLevel.RAW
    token_count: int = 0
    original_token_count: int = 0  # Token count before any compression
    
    # Relevance and timing
    relevance_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    turn_created: int = 0
    
    # Unique identifier
    id: str = field(default_factory=lambda: f"k_{int(time.time() * 1000) % 100000}")
    
    def __post_init__(self) -> None:
        """Initialize token count if not set."""
        if self.token_count == 0:
            self.token_count = self.estimate_tokens()
        if self.original_token_count == 0:
            self.original_token_count = self.token_count
    
    def estimate_tokens(self) -> int:
        """Estimate token count (rough: ~4 chars per token)."""
        text = self.summary + " ".join(self.quotes)
        text += self.reference_url + self.reference_title
        return max(1, len(text) // 4)
    
    def to_context_string(self, include_quotes: bool = True) -> str:
        """Convert to a string suitable for LLM context injection.
        
        Args:
            include_quotes: Whether to include quotes in output.
            
        Returns:
            Formatted string representation of this knowledge.
        """
        level_marker = ["📄", "📝", "💡"][self.level]
        lines = [
            f"{level_marker} [{self.reference_title}]({self.reference_url})",
        ]
        
        if self.reference_datetime:
            lines[0] += f" ({self.reference_datetime})"
        
        if self.summary:
            lines.append(f"   {self.summary}")
        
        if include_quotes and self.quotes and self.level == CompressionLevel.RAW:
            lines.append("   Quotes:")
            for q in self.quotes[:3]:  # Limit quotes shown
                lines.append(f"   - \"{q[:200]}{'...' if len(q) > 200 else ''}\"")
        
        return "\n".join(lines)
    
    def compression_priority(self, current_turn: int) -> float:
        """Calculate priority for compression (higher = compress first).
        
        Factors:
        - Age (older = higher priority)
        - Relevance (lower relevance = higher priority)
        - Current compression level (lower level = higher priority)
        
        Args:
            current_turn: Current agent turn number.
            
        Returns:
            Priority score (higher means should compress sooner).
        """
        age_factor = (current_turn - self.turn_created) / max(current_turn, 1)
        relevance_factor = 1.0 - self.relevance_score
        level_factor = 1.0 - (self.level / 2.0)  # RAW=1.0, CONDENSED=0.5, ESSENCE=0
        
        return (age_factor * 0.4) + (relevance_factor * 0.3) + (level_factor * 0.3)


@dataclass
class KnowledgeStoreConfig:
    """Configuration for KnowledgeStore."""
    
    budget_tokens: int = 30000
    """Total token budget for knowledge storage."""
    
    max_level0: int = 3
    """Maximum number of RAW (uncompressed) items to keep."""
    
    max_level1: int = 10
    """Maximum number of CONDENSED items to keep."""
    
    compress_threshold: float = 0.8
    """Trigger compression when usage exceeds this fraction of budget."""


# Type alias for async compression function
CompressorFunc = Callable[
    ["ManagedKnowledge", CompressionLevel], 
    Awaitable["ManagedKnowledge"]
]


@dataclass
class KnowledgeStore:
    """Hierarchical knowledge storage with automatic compression.
    
    Manages a collection of knowledge items across multiple compression levels,
    automatically compressing older/less relevant items when the token budget
    is exceeded.
    """
    
    items: List[ManagedKnowledge] = field(default_factory=list)
    config: KnowledgeStoreConfig = field(default_factory=KnowledgeStoreConfig)
    
    # Optional async compressor function (to be set externally)
    _compressor: Optional[CompressorFunc] = field(default=None, repr=False)
    
    def set_compressor(self, compressor: CompressorFunc) -> None:
        """Set the compression function.
        
        Args:
            compressor: Async function that compresses a knowledge item.
        """
        self._compressor = compressor
    
    @property
    def total_tokens(self) -> int:
        """Total token count across all items."""
        return sum(k.token_count for k in self.items)
    
    @property
    def usage_ratio(self) -> float:
        """Current usage as a fraction of budget."""
        return self.total_tokens / self.config.budget_tokens
    
    def add(self, knowledge: ManagedKnowledge) -> None:
        """Add a knowledge item to the store.
        
        Args:
            knowledge: The knowledge item to add.
        """
        # Check for duplicates by URL
        for existing in self.items:
            if existing.reference_url == knowledge.reference_url:
                # Update existing if new one has more content
                if knowledge.token_count > existing.token_count:
                    self.items.remove(existing)
                    self.items.append(knowledge)
                return
        
        self.items.append(knowledge)
    
    def get_items_by_level(self, level: CompressionLevel) -> List[ManagedKnowledge]:
        """Get all items at a specific compression level.
        
        Args:
            level: The compression level to filter by.
            
        Returns:
            List of knowledge items at that level.
        """
        return [k for k in self.items if k.level == level]
    
    def needs_compression(self) -> bool:
        """Check if compression is needed based on current usage."""
        # Check token budget
        if self.usage_ratio > self.config.compress_threshold:
            return True
        
        # Check level counts
        if len(self.get_items_by_level(CompressionLevel.RAW)) > self.config.max_level0:
            return True
        if len(self.get_items_by_level(CompressionLevel.CONDENSED)) > self.config.max_level1:
            return True
        
        return False
    
    def select_for_compression(self, current_turn: int = 0) -> List[ManagedKnowledge]:
        """Select items that should be compressed.
        
        Args:
            current_turn: Current agent turn for age calculation.
            
        Returns:
            List of items to compress, sorted by priority.
        """
        candidates = [k for k in self.items if k.level < CompressionLevel.ESSENCE]
        
        # Sort by compression priority (highest first)
        candidates.sort(key=lambda k: k.compression_priority(current_turn), reverse=True)
        
        # Determine how many to compress
        to_compress = []
        
        # First, handle level count violations
        level0_items = self.get_items_by_level(CompressionLevel.RAW)
        if len(level0_items) > self.config.max_level0:
            excess = len(level0_items) - self.config.max_level0
            level0_sorted = sorted(
                level0_items, 
                key=lambda k: k.compression_priority(current_turn), 
                reverse=True
            )
            to_compress.extend(level0_sorted[:excess])
        
        level1_items = self.get_items_by_level(CompressionLevel.CONDENSED)
        if len(level1_items) > self.config.max_level1:
            excess = len(level1_items) - self.config.max_level1
            level1_sorted = sorted(
                level1_items, 
                key=lambda k: k.compression_priority(current_turn), 
                reverse=True
            )
            to_compress.extend(level1_sorted[:excess])
        
        # Then, handle token budget if still over
        if self.usage_ratio > self.config.compress_threshold:
            for candidate in candidates:
                if candidate not in to_compress:
                    to_compress.append(candidate)
                    # Estimate new usage after compression
                    # (rough estimate: compression reduces tokens by ~70%)
                    estimated_savings = candidate.token_count * 0.7
                    current_tokens = self.total_tokens - estimated_savings
                    if current_tokens / self.config.budget_tokens < self.config.compress_threshold:
                        break
        
        return to_compress
    
    async def compress_if_needed(self, current_turn: int = 0) -> int:
        """Check and perform compression if needed.
        
        Args:
            current_turn: Current agent turn for priority calculation.
            
        Returns:
            Number of items compressed.
        """
        if not self.needs_compression():
            return 0
        
        if self._compressor is None:
            # No compressor set, use simple fallback
            return self._simple_compress(current_turn)
        
        to_compress = self.select_for_compression(current_turn)
        compressed_count = 0
        
        for item in to_compress:
            target_level = CompressionLevel(min(item.level + 1, CompressionLevel.ESSENCE))
            try:
                compressed = await self._compressor(item, target_level)
                # Update the item in place
                idx = self.items.index(item)
                self.items[idx] = compressed
                compressed_count += 1
            except Exception:
                # If compression fails, skip this item
                continue
        
        return compressed_count
    
    def _simple_compress(self, current_turn: int) -> int:
        """Simple compression without LLM (truncation-based).
        
        Used as fallback when no compressor is set.
        
        Args:
            current_turn: Current agent turn.
            
        Returns:
            Number of items compressed.
        """
        to_compress = self.select_for_compression(current_turn)
        compressed_count = 0
        
        for item in to_compress:
            if item.level == CompressionLevel.RAW:
                # RAW -> CONDENSED: Keep first quote, truncate summary
                item.quotes = item.quotes[:1] if item.quotes else []
                if len(item.summary) > 200:
                    item.summary = item.summary[:200] + "..."
                item.level = CompressionLevel.CONDENSED
            elif item.level == CompressionLevel.CONDENSED:
                # CONDENSED -> ESSENCE: Remove quotes, keep short summary
                item.quotes = []
                if len(item.summary) > 100:
                    item.summary = item.summary[:100] + "..."
                item.level = CompressionLevel.ESSENCE
            
            item.token_count = item.estimate_tokens()
            compressed_count += 1
        
        return compressed_count
    
    def get_context_string(
        self, 
        max_tokens: Optional[int] = None,
        current_turn: int = 0,
    ) -> str:
        """Generate a formatted string for LLM context injection.
        
        Args:
            max_tokens: Maximum tokens to include (uses budget if None).
            current_turn: Current turn for relevance calculation.
            
        Returns:
            Formatted knowledge context string.
        """
        if not self.items:
            return "(No knowledge collected yet)"
        
        max_tokens = max_tokens or self.config.budget_tokens
        
        # Sort items: prioritize high relevance and recent items
        sorted_items = sorted(
            self.items,
            key=lambda k: (
                -k.relevance_score,  # Higher relevance first
                -k.created_at,       # More recent first
                k.level,             # Lower compression level first
            )
        )
        
        lines = []
        current_tokens = 0
        
        for item in sorted_items:
            item_str = item.to_context_string()
            item_tokens = len(item_str) // 4  # Rough estimate
            
            if current_tokens + item_tokens > max_tokens:
                break
            
            lines.append(item_str)
            current_tokens += item_tokens
        
        if not lines:
            return "(Knowledge items too large for budget)"
        
        return "\n\n".join(lines)
    
    def clear(self) -> None:
        """Clear all knowledge items."""
        self.items.clear()
    
    def stats(self) -> dict:
        """Get statistics about the knowledge store.
        
        Returns:
            Dictionary with store statistics.
        """
        return {
            "total_items": len(self.items),
            "total_tokens": self.total_tokens,
            "usage_ratio": f"{self.usage_ratio:.1%}",
            "by_level": {
                "raw": len(self.get_items_by_level(CompressionLevel.RAW)),
                "condensed": len(self.get_items_by_level(CompressionLevel.CONDENSED)),
                "essence": len(self.get_items_by_level(CompressionLevel.ESSENCE)),
            },
            "budget": self.config.budget_tokens,
        }


