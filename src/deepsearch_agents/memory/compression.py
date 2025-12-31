"""Knowledge compression using LLM.

This module provides intelligent compression of knowledge items
using language models to preserve the most important information
while reducing token count.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from deepsearch_agents.llm.llm import get_response
from deepsearch_agents.log import logger
from .knowledge_store import ManagedKnowledge, CompressionLevel


class CompressedContent(BaseModel):
    """Result of knowledge compression."""
    
    summary: str = Field(
        description="Compressed summary (1-2 sentences for CONDENSED, single sentence for ESSENCE)"
    )
    key_quote: Optional[str] = Field(
        default=None,
        description="The most important quote to preserve (only for CONDENSED level)"
    )
    core_insight: str = Field(
        description="Single sentence capturing the absolute core insight"
    )


COMPRESSION_SYSTEM_PROMPT = """You are a knowledge compressor specialized in preserving critical information while minimizing token usage.

Your task is to compress research knowledge while retaining:
1. The core factual claims
2. Key evidence or data points  
3. Source attribution

Compression levels:
- CONDENSED: Keep 1-2 most important sentences and one key quote
- ESSENCE: Reduce to a single sentence capturing the core insight

Be ruthless about removing:
- Redundant information
- Background context
- Hedging language
- Examples (unless crucial)
"""

COMPRESSION_USER_PROMPT = """Compress this knowledge to level: {target_level}

Source: [{title}]({url})
{datetime_line}

Current Summary:
{summary}

Current Quotes:
{quotes}

Compress this to the target level while preserving the most critical information."""


class KnowledgeCompressor:
    """Compresses knowledge items using LLM.
    
    Provides intelligent compression that preserves key information
    while reducing token count for context window management.
    """
    
    def __init__(self, model: str = "summarize"):
        """Initialize the compressor.
        
        Args:
            model: Model configuration name to use for compression.
        """
        self.model = model
    
    async def compress(
        self,
        knowledge: ManagedKnowledge,
        target_level: CompressionLevel,
    ) -> ManagedKnowledge:
        """Compress a knowledge item to the target level.
        
        Args:
            knowledge: The knowledge item to compress.
            target_level: The target compression level.
            
        Returns:
            A new ManagedKnowledge with compressed content.
        """
        if knowledge.level >= target_level:
            logger.warning(
                f"Knowledge already at level {knowledge.level}, "
                f"cannot compress to {target_level}"
            )
            return knowledge
        
        # Format quotes for prompt
        quotes_text = "\n".join(f"- \"{q}\"" for q in knowledge.quotes) if knowledge.quotes else "(no quotes)"
        
        # Format datetime line
        datetime_line = f"Published: {knowledge.reference_datetime}" if knowledge.reference_datetime else ""
        
        try:
            response = await get_response(
                model=self.model,
                input=COMPRESSION_USER_PROMPT.format(
                    target_level=target_level.name,
                    title=knowledge.reference_title,
                    url=knowledge.reference_url,
                    datetime_line=datetime_line,
                    summary=knowledge.summary,
                    quotes=quotes_text,
                ),
                output_type=CompressedContent,
                system_instructions=COMPRESSION_SYSTEM_PROMPT,
            )
            
            compressed = response.response
            
            # Create new knowledge with compressed content
            new_knowledge = ManagedKnowledge(
                reference_url=knowledge.reference_url,
                reference_title=knowledge.reference_title,
                reference_datetime=knowledge.reference_datetime,
                level=target_level,
                relevance_score=knowledge.relevance_score,
                created_at=knowledge.created_at,
                turn_created=knowledge.turn_created,
                original_token_count=knowledge.original_token_count or knowledge.token_count,
                id=knowledge.id,
            )
            
            # Set content based on target level
            if target_level == CompressionLevel.CONDENSED:
                new_knowledge.summary = compressed.summary
                new_knowledge.quotes = [compressed.key_quote] if compressed.key_quote else []
            elif target_level == CompressionLevel.ESSENCE:
                new_knowledge.summary = compressed.core_insight
                new_knowledge.quotes = []
            
            # Recalculate token count
            new_knowledge.token_count = new_knowledge.estimate_tokens()
            
            logger.info(
                f"Compressed knowledge '{knowledge.reference_title}' "
                f"from {knowledge.level.name} to {target_level.name}: "
                f"{knowledge.token_count} -> {new_knowledge.token_count} tokens"
            )
            
            return new_knowledge
            
        except Exception as e:
            logger.error(f"Failed to compress knowledge: {e}")
            # Return original on failure
            return knowledge
    
    async def batch_compress(
        self,
        knowledges: List[ManagedKnowledge],
        target_level: CompressionLevel,
    ) -> List[ManagedKnowledge]:
        """Compress multiple knowledge items in parallel.
        
        Args:
            knowledges: List of knowledge items to compress.
            target_level: Target compression level for all items.
            
        Returns:
            List of compressed knowledge items.
        """
        import asyncio
        
        tasks = [self.compress(k, target_level) for k in knowledges]
        return await asyncio.gather(*tasks)
    
    def create_compressor_func(self):
        """Create a compressor function for KnowledgeStore.
        
        Returns:
            Async function compatible with KnowledgeStore.set_compressor().
        """
        async def compressor(
            knowledge: ManagedKnowledge, 
            target_level: CompressionLevel
        ) -> ManagedKnowledge:
            return await self.compress(knowledge, target_level)
        
        return compressor


