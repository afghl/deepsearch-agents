"""Planner agent for task planning and execution with memory management."""

import asyncio
import contextvars
from dataclasses import dataclass, field
import json
from typing import List, Optional, Coroutine, Any, TypeVar

T = TypeVar("T")


def _run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously.

    Handles the case where we're already inside an event loop
    (which is the case when called from Agent's instructions).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(coro)
    else:
        # Already in a running loop, use nest_asyncio or create new thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()


from agents import (
    Agent,
    AgentHooks,
    FunctionTool,
    ModelSettings,
    RunContextWrapper,
    Runner,
    Tool,
)

from deepsearch_agents import conf
from deepsearch_agents.log import logger
from deepsearch_agents.context import TaskContext, Task
from deepsearch_agents.tools import get_tool_instructions, sep
from deepsearch_agents.memory import (
    KnowledgeStore,
    KnowledgeStoreConfig,
    Scratchpad,
    ScratchpadConfig,
    ContextBuilder,
    ContextBudget,
    TaskState,
    KnowledgeCompressor,
    ManagedKnowledge,
    CompressionLevel,
)


def _build_base_instructions(
    ctx: RunContextWrapper[TaskContext],
    agent: Agent[TaskContext],
) -> str:
    """Build the base system instructions without memory context."""
    tool_names = "\n".join(
        [f"{i+1}. {tool.name}" for i, tool in enumerate(agent.tools)]
    )
    curr = ctx.context.current_task()

    if curr.query == curr.origin_query:
        question = f"The Question you are trying to answer is: {curr.query}"
    else:
        question = (
            f"The Original Question is: {curr.origin_query}\n"
            f"And you are currently focusing on this aspect of it.\n"
            f"You are trying to answer this question: {curr.query}"
        )

    return f"""Current Date: {ctx.context.start_date_time}

You are an advanced AI research agent from Deepsearch AI. You are specialized in multistep reasoning. 
Using your best knowledge, conversation with the user and lessons learned, answer the user question with absolute certainty.

-Goal-

Given a question in any domain, do research to find the answer. Provide a detailed, comprehensive and factually accurate answer.

-Rules-

1. Think step by step, choose the action carefully.
2. ALWAYS show your thinking process before taking any action. Reflect on what you have already known first, and then explain the reason on your next move.
3. No rush to answer the question, Examine the question and the evidence carefully before answering.
4. Your job is to provide the best answer. The conversation does not end until the answer is verified.
5. Use the collected knowledge effectively - don't repeat searches for information you already have.

-Question-
{question}

-Available actions-

Here are the actions provided. YOU CAN ONLY choose one of these actions:

{tool_names}

-Action details-

{get_tool_instructions(ctx.context, agent.tool_names)}

Think step by step, choose the action carefully.
"""


def _build_low_token_instructions(ctx: RunContextWrapper[TaskContext]) -> str:
    """Build simplified instructions when running low on tokens."""
    curr = ctx.context.current_task()
    return f"""Current Date: {ctx.context.start_date_time}

You are an advanced AI research agent from Deepsearch AI.

-Question-
{curr.origin_query}

-Urgent Instructions-
- You are running low on token budget. Provide your best answer NOW.
- Use all the knowledge collected so far.
- Partial but well-informed responses are acceptable.
- Base your response on what we know so far.

Answer the question with the information available.
"""


def _build_instructions_and_tools(
    ctx: RunContextWrapper[TaskContext], agent: Agent[TaskContext]
) -> str:
    """Build instructions with integrated memory context.

    This function is called before each LLM call to construct the system prompt.
    It performs:
    1. Sync knowledge from task to knowledge store
    2. Compress knowledge if needed (runs async code synchronously)
    3. Build the full context with all memory components
    """

    # Check if running out of tokens
    if agent._running_out_of_token(ctx):
        logger.info("Running out of tokens, switching to simplified mode")
        agent.model_settings.tool_choice = "auto"
        return _build_low_token_instructions(ctx)

    # Build base instructions
    base_instructions = _build_base_instructions(ctx, agent)

    # If planner has context builder, use it to enhance instructions
    if hasattr(agent, "context_builder") and agent.context_builder is not None:
        curr = ctx.context.current_task()

        # Sync knowledge from task to knowledge store
        if hasattr(agent, "sync_knowledge_from_task"):
            agent.sync_knowledge_from_task(curr)

        # Compress knowledge if needed (run async code synchronously)
        if hasattr(agent, "knowledge_store") and agent.knowledge_store is not None:
            try:
                compressed = _run_sync(
                    agent.knowledge_store.compress_if_needed(curr.turn)
                )
                if compressed > 0:
                    logger.info(
                        f"Compressed {compressed} knowledge items before LLM call"
                    )
            except Exception as e:
                logger.warning(f"Failed to compress knowledge: {e}")

        # Build task state
        task_state = TaskState(
            origin_query=curr.origin_query,
            current_query=curr.query,
            current_turn=curr.turn,
            task_level=curr.level,
            attempt_count=curr.attempt,
            sub_task_summaries=[
                f"Q: {st.query} -> A: {st.answer.answer[:200] if st.answer and st.answer.answer else 'No answer'}"
                for st in curr.sub_tasks.values()
                if st.answer and st.answer.answer
            ],
        )

        return agent.context_builder.build_full_context(task_state, base_instructions)

    return base_instructions


@dataclass
class Planner(Agent[TaskContext]):
    """
    A Planner agent that manages task planning and execution with memory.

    The Planner is responsible for:
    - Breaking down complex tasks into subtasks
    - Managing the execution flow and coordinating tool usage
    - Tracking collected knowledge and action history
    - Building optimized context for LLM decisions
    """

    task_generator: str | None = None
    """Optional string identifier for the task generation tool."""

    # Memory components
    knowledge_store: Optional[KnowledgeStore] = None
    """Hierarchical knowledge storage with compression."""

    scratchpad: Optional[Scratchpad] = None
    """Action history tracking."""

    context_builder: Optional[ContextBuilder] = None
    """Context construction with budget management."""

    _compressor: Optional[KnowledgeCompressor] = None
    """LLM-based knowledge compressor."""

    def __init__(
        self,
        name: str,
        tools: List[Tool],
        task_generator: str | None = None,
        hooks: AgentHooks[TaskContext] | None = None,
        model: str | None = None,
        model_settings: ModelSettings | None = None,
        # Memory configuration
        enable_memory: bool = True,
        knowledge_config: Optional[KnowledgeStoreConfig] = None,
        scratchpad_config: Optional[ScratchpadConfig] = None,
        context_budget: Optional[ContextBudget] = None,
    ):
        super().__init__(
            name=name,
            instructions=_build_instructions_and_tools,
            tools=tools,
            hooks=hooks,
            model=model,
            model_settings=model_settings,
        )
        self.task_generator = task_generator
        if task_generator:
            self._build_task_generate_tool()
        self.all_tools = self.tools  # type: ignore

        # Initialize memory components if enabled
        if enable_memory:
            self._init_memory(knowledge_config, scratchpad_config, context_budget)

    def _init_memory(
        self,
        knowledge_config: Optional[KnowledgeStoreConfig] = None,
        scratchpad_config: Optional[ScratchpadConfig] = None,
        context_budget: Optional[ContextBudget] = None,
    ) -> None:
        """Initialize memory management components."""
        self.knowledge_store = KnowledgeStore(
            config=knowledge_config or KnowledgeStoreConfig()
        )
        self.scratchpad = Scratchpad(config=scratchpad_config or ScratchpadConfig())
        self.context_builder = ContextBuilder(
            knowledge_store=self.knowledge_store,
            scratchpad=self.scratchpad,
            budget=context_budget or ContextBudget(),
        )

        # Initialize compressor and wire it to knowledge store
        self._compressor = KnowledgeCompressor(model="summarize")
        self.knowledge_store.set_compressor(self._compressor.create_compressor_func())

        logger.info(f"Memory components initialized for {self.name}")

    def sync_knowledge_from_task(self, task: Task) -> int:
        """Sync knowledge from task context to knowledge store.

        This converts the legacy Knowledge format to ManagedKnowledge.

        Args:
            task: The task to sync knowledge from.

        Returns:
            Number of knowledge items synced.
        """
        if not self.knowledge_store:
            return 0

        synced = 0
        for knowledge in task.knowledges:
            managed = ManagedKnowledge(
                reference_url=knowledge.reference.url,
                reference_title=knowledge.reference.title,
                reference_datetime=knowledge.reference.datetime,
                summary=knowledge.summary or "",
                quotes=knowledge.quotes,
                turn_created=task.turn,
                level=CompressionLevel.RAW,
            )
            self.knowledge_store.add(managed)
            synced += 1

        return synced

    async def maybe_compress_knowledge(self, current_turn: int = 0) -> int:
        """Trigger knowledge compression if needed.

        Should be called from Hooks.on_tool_end after each tool execution.

        Args:
            current_turn: Current turn number for priority calculation.

        Returns:
            Number of items compressed.
        """
        if not self.knowledge_store:
            return 0

        compressed = await self.knowledge_store.compress_if_needed(current_turn)
        if compressed > 0:
            logger.info(f"Compressed {compressed} knowledge items")
        return compressed

    def rebuild_tools(
        self, ctx: RunContextWrapper[TaskContext], last_used: str | None = None
    ) -> None:
        """
        Update available tools based on current context.

        Excludes:
        - The last used tool (to avoid immediate repetition)
        - Task generator if depth limit reached
        - All tools except 'answer' if running out of tokens
        """
        # Sync knowledge from task before rebuilding
        if self.knowledge_store:
            curr = ctx.context.current_task()
            self.sync_knowledge_from_task(curr)

        if self._running_out_of_token(ctx):
            self.tools = [tool for tool in self.all_tools if tool.name == "answer"]
            return

        config = conf.get_configuration().execution_config
        available = []

        for tool in self.all_tools:
            # Skip last used tool
            if tool.name == last_used:
                continue

            # Check task generator conditions
            if tool.name == self.task_generator:
                curr = ctx.context.current_task()
                if curr.level >= config.max_task_depth:
                    continue
                if len(curr.sub_tasks) >= config.max_tasks_count:
                    continue

            available.append(tool)

        self.tools = available

    def _build_new_tasks(
        self, ctx: RunContextWrapper[TaskContext], result: str
    ) -> List[Task]:
        """Build new sub-tasks from the task generator result."""
        logger.info(f"Building new tasks from result: {result}")
        tasks = []
        question_list = result.split(sep)
        curr = ctx.context.current_task()

        if isinstance(question_list, str):
            question_list = json.loads(question_list)

        cnt = len(curr.sub_tasks)
        for q in question_list:
            sub_task = Task(
                id=f"{curr.id}_{cnt+1}",
                origin_query=curr.origin_query,
                query=q,
                level=curr.level + 1,
                parent=curr,
            )
            cnt += 1
            logger.info(f"Created sub-task: {sub_task.id} for query: {q[:50]}...")
            curr.sub_tasks[sub_task.id] = sub_task
            ctx.context.tasks[sub_task.id] = sub_task
            tasks.append(sub_task)

        # Record in scratchpad
        if self.scratchpad:
            self.scratchpad.record_reflect(
                questions=[t.query for t in tasks],
                turn=curr.turn,
            )

        return tasks

    @property
    def tool_names(self) -> List[str]:
        return [tool.name for tool in self.tools]

    def _running_out_of_token(self, ctx: RunContextWrapper[TaskContext]) -> bool:
        threshold = conf.get_configuration().execution_config.max_token_usage * 0.85
        return ctx.usage.total_tokens > threshold

    def _build_task_generate_tool(self) -> None:
        """Wrap the task generator tool to execute sub-tasks."""
        tool = next((t for t in self.tools if t.name == self.task_generator), None)
        if not tool:
            return

        assert isinstance(
            tool, FunctionTool
        ), f"Task generator tool {self.task_generator} must be a FunctionTool"

        async def execute_task(ctx: RunContextWrapper[TaskContext], input: str) -> str:
            ret = await tool.on_invoke_tool(ctx, input)
            if not ret:
                return "No new tasks generated."

            tasks = self._build_new_tasks(ctx, ret)
            await asyncio.gather(*[self._execute_sub_task(ctx, task) for task in tasks])

            results = []
            for task in tasks:
                if task.solved():
                    results.append(
                        f"For Question: {task.query}\n"
                        f"Research completed. Answer: {task.answer.answer}"  # type: ignore
                    )
                else:
                    results.append(
                        f"For Question: {task.query}\n"
                        f"Could not find sufficient information."
                    )

            return "\n\n".join(results)

        # Replace original tool with wrapped version
        self.tools = [t for t in self.tools if t.name != self.task_generator]
        self.tools.append(
            FunctionTool(
                name=tool.name,
                description=tool.description,
                params_json_schema=tool.params_json_schema,
                on_invoke_tool=execute_task,
                strict_json_schema=tool.strict_json_schema,
            )
        )

    async def _execute_sub_task(
        self, context: RunContextWrapper[TaskContext], new_task: Task
    ) -> None:
        """Execute a sub-task with its own Planner instance."""

        async def run():
            new_task.set_as_current()

            # Create sub-planner (without memory to avoid duplication)
            sub_planner = Planner(
                name=f"DeepSearch Agent-{new_task.id}",
                tools=self.tools,
                task_generator=self.task_generator,
                hooks=self.hooks,
                model=self.model,
                model_settings=self.model_settings,
                enable_memory=False,  # Sub-tasks don't need separate memory
            )

            try:
                await Runner.run(
                    starting_agent=sub_planner,
                    input=new_task.query,
                    context=context.context,
                )
            except Exception as e:
                logger.error(f"Error running sub-task {new_task.id}: {e}")

            logger.info(f"Sub-task completed: {new_task.id}")

        # Run in copied context to isolate contextvars
        ctx = contextvars.copy_context()
        await ctx.run(run)

    def get_memory_stats(self) -> dict:
        """Get statistics about memory usage.

        Returns:
            Dictionary with memory component statistics.
        """
        if not self.context_builder:
            return {"memory_enabled": False}

        return {
            "memory_enabled": True,
            **self.context_builder.stats(),
        }
