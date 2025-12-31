"""Tests for memory module."""

import asyncio
from deepsearch_agents.memory import (
    CompressionLevel,
    ManagedKnowledge,
    KnowledgeStore,
    KnowledgeStoreConfig,
    ActionType,
    Scratchpad,
    ScratchpadConfig,
    ContextBuilder,
    ContextBudget,
    TaskState,
)


def test_managed_knowledge_basic():
    """Test ManagedKnowledge creation and methods."""
    knowledge = ManagedKnowledge(
        reference_url="https://example.com/article",
        reference_title="Test Article",
        summary="This is a test summary about the market.",
        quotes=["Quote 1", "Quote 2"],
        turn_created=1,
    )
    
    assert knowledge.level == CompressionLevel.RAW
    assert knowledge.token_count > 0
    assert "Test Article" in knowledge.to_context_string()
    assert "Quote 1" in knowledge.to_context_string()
    
    print("✓ ManagedKnowledge basic test passed")


def test_knowledge_store_add_and_dedup():
    """Test KnowledgeStore add and deduplication."""
    store = KnowledgeStore()
    
    # Add first knowledge
    k1 = ManagedKnowledge(
        reference_url="https://example.com/1",
        reference_title="Article 1",
        summary="Summary 1",
    )
    store.add(k1)
    assert len(store.items) == 1
    
    # Add second knowledge
    k2 = ManagedKnowledge(
        reference_url="https://example.com/2",
        reference_title="Article 2",
        summary="Summary 2",
    )
    store.add(k2)
    assert len(store.items) == 2
    
    # Add duplicate URL - should not increase count
    k3 = ManagedKnowledge(
        reference_url="https://example.com/1",
        reference_title="Article 1 Updated",
        summary="Summary 1",
    )
    store.add(k3)
    assert len(store.items) == 2
    
    print("✓ KnowledgeStore add and dedup test passed")


def test_knowledge_store_compression_trigger():
    """Test compression triggering logic."""
    config = KnowledgeStoreConfig(
        budget_tokens=1000,
        max_level0=2,
        compress_threshold=0.5,
    )
    store = KnowledgeStore(config=config)
    
    # Add items that don't trigger compression
    for i in range(2):
        store.add(ManagedKnowledge(
            reference_url=f"https://example.com/{i}",
            reference_title=f"Article {i}",
            summary="Short summary",
        ))
    
    assert not store.needs_compression()
    
    # Add one more to exceed max_level0
    store.add(ManagedKnowledge(
        reference_url="https://example.com/3",
        reference_title="Article 3",
        summary="Short summary",
    ))
    
    assert store.needs_compression()
    
    print("✓ KnowledgeStore compression trigger test passed")


def test_knowledge_store_simple_compression():
    """Test simple compression without LLM."""
    config = KnowledgeStoreConfig(
        budget_tokens=1000,
        max_level0=1,
    )
    store = KnowledgeStore(config=config)
    
    # Add two RAW items (exceeds max_level0=1)
    store.add(ManagedKnowledge(
        reference_url="https://example.com/1",
        reference_title="Article 1",
        summary="This is a longer summary that should be compressed.",
        quotes=["Quote 1", "Quote 2", "Quote 3"],
        turn_created=1,
    ))
    store.add(ManagedKnowledge(
        reference_url="https://example.com/2",
        reference_title="Article 2",
        summary="Another summary",
        quotes=["Another quote"],
        turn_created=2,
    ))
    
    # Run simple compression
    compressed_count = store._simple_compress(current_turn=3)
    assert compressed_count > 0
    
    # Check that one item was compressed
    level1_items = store.get_items_by_level(CompressionLevel.CONDENSED)
    assert len(level1_items) >= 1
    
    print("✓ KnowledgeStore simple compression test passed")


def test_scratchpad_basic():
    """Test Scratchpad basic operations."""
    scratchpad = Scratchpad()
    
    # Record some actions
    scratchpad.record(
        action_type=ActionType.SEARCH,
        summary="searched for SPX performance",
        turn=1,
        outcome="25 results",
        is_success=True,
    )
    
    scratchpad.record(
        action_type=ActionType.VISIT,
        summary="visited bloomberg.com",
        turn=2,
        outcome="found useful data",
        is_success=True,
    )
    
    assert len(scratchpad.entries) == 2
    assert scratchpad.get_last_action().action_type == ActionType.VISIT
    
    counts = scratchpad.get_action_counts()
    assert counts[ActionType.SEARCH] == 1
    assert counts[ActionType.VISIT] == 1
    
    print("✓ Scratchpad basic test passed")


def test_scratchpad_convenience_methods():
    """Test Scratchpad convenience recording methods."""
    scratchpad = Scratchpad()
    
    # Test record_search
    entry = scratchpad.record_search(
        queries=["SPX", "market performance"],
        turn=1,
        result_count=25,
    )
    assert entry.action_type == ActionType.SEARCH
    assert "SPX" in entry.summary
    
    # Test record_visit
    entry = scratchpad.record_visit(
        urls=["https://a.com", "https://b.com"],
        turn=2,
        useful_count=1,
    )
    assert entry.action_type == ActionType.VISIT
    assert "2 URL" in entry.summary
    
    # Test record_reflect
    entry = scratchpad.record_reflect(
        questions=["What drives SPX?", "Fed impact?"],
        turn=3,
    )
    assert entry.action_type == ActionType.REFLECT
    
    # Test record_answer
    entry = scratchpad.record_answer(
        turn=4,
        is_pass=False,
        feedback="Need more evidence",
    )
    assert entry.action_type == ActionType.ANSWER
    assert not entry.is_success
    
    print("✓ Scratchpad convenience methods test passed")


def test_scratchpad_trimming():
    """Test Scratchpad auto-trimming."""
    config = ScratchpadConfig(max_entries=3)
    scratchpad = Scratchpad(config=config)
    
    # Add 5 entries (exceeds max of 3)
    for i in range(5):
        scratchpad.record(
            action_type=ActionType.SEARCH,
            summary=f"search {i}",
            turn=i + 1,
        )
    
    # Should have trimmed to max_entries
    assert len(scratchpad.entries) == 3
    assert scratchpad._compressed_count == 2
    assert scratchpad._compressed_summary != ""
    
    # Context string should mention compressed entries
    context = scratchpad.to_context_string()
    assert "earlier" in context.lower()
    
    print("✓ Scratchpad trimming test passed")


def test_context_generation():
    """Test context string generation for both modules."""
    # Setup knowledge store
    store = KnowledgeStore()
    store.add(ManagedKnowledge(
        reference_url="https://bloomberg.com/spx",
        reference_title="SPX Analysis",
        summary="SPX gained 2.3% this month.",
        quotes=["The rally was driven by tech sector"],
        relevance_score=0.9,
    ))
    
    knowledge_context = store.get_context_string()
    assert "SPX Analysis" in knowledge_context
    assert "bloomberg.com" in knowledge_context
    
    # Setup scratchpad
    scratchpad = Scratchpad()
    scratchpad.record_search(["SPX performance"], turn=1, result_count=25)
    scratchpad.record_visit(["https://bloomberg.com"], turn=2, useful_count=1)
    
    scratchpad_context = scratchpad.to_context_string()
    assert "Action History" in scratchpad_context
    assert "search" in scratchpad_context.lower()
    
    print("✓ Context generation test passed")


def test_context_builder():
    """Test ContextBuilder integration."""
    # Setup components
    store = KnowledgeStore()
    store.add(ManagedKnowledge(
        reference_url="https://example.com/1",
        reference_title="Article 1",
        summary="Important finding about topic X.",
        quotes=["Quote from article"],
        relevance_score=0.9,
        turn_created=1,
    ))
    store.add(ManagedKnowledge(
        reference_url="https://example.com/2",
        reference_title="Article 2",
        summary="Another finding about topic Y.",
        relevance_score=0.7,
        turn_created=2,
    ))
    
    scratchpad = Scratchpad()
    scratchpad.record_search(["topic X", "topic Y"], turn=1, result_count=25)
    scratchpad.record_visit(["https://example.com/1"], turn=2, useful_count=1)
    
    # Create context builder
    builder = ContextBuilder(
        knowledge_store=store,
        scratchpad=scratchpad,
        budget=ContextBudget(total=50000, knowledge=10000),
    )
    
    # Create task state
    task_state = TaskState(
        origin_query="What is topic X?",
        current_query="What is topic X?",
        current_turn=3,
        task_level=1,
        attempt_count=0,
    )
    
    # Build full context
    base_instructions = "You are a research agent."
    full_context = builder.build_full_context(task_state, base_instructions)
    
    assert "research agent" in full_context
    assert "Article 1" in full_context
    assert "Action History" in full_context
    assert "Knowledge items: 2" in full_context
    
    # Test stats
    stats = builder.stats()
    assert stats["knowledge"]["total_items"] == 2
    assert stats["scratchpad"]["total_entries"] == 2
    
    print("✓ ContextBuilder test passed")


def test_task_state():
    """Test TaskState creation and methods."""
    # Root task
    root_state = TaskState(
        origin_query="Main question",
        current_query="Main question",
        current_turn=5,
    )
    assert not root_state.is_sub_task()
    
    # Sub task
    sub_state = TaskState(
        origin_query="Main question",
        current_query="Sub question",
        current_turn=3,
        task_level=2,
    )
    assert sub_state.is_sub_task()
    
    print("✓ TaskState test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("Running Memory Module Tests")
    print("=" * 50 + "\n")
    
    test_managed_knowledge_basic()
    test_knowledge_store_add_and_dedup()
    test_knowledge_store_compression_trigger()
    test_knowledge_store_simple_compression()
    test_scratchpad_basic()
    test_scratchpad_convenience_methods()
    test_scratchpad_trimming()
    test_context_generation()
    test_context_builder()
    test_task_state()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_all_tests()

