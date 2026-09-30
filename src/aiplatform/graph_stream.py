"""Run a LangGraph graph and stream the events its nodes emit, one at a time.

Nodes call ``await emitter()(event)``. Each emit returns only after the consumer has
taken the event, like ``yield`` in a plain async generator. LangGraph's own custom
stream doesn't wait, so a node would keep running (and storing a reply) after the
client disconnected. Here, when the consumer stops, the graph is cancelled where it
is and nothing after that point runs.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Any

from langgraph.config import get_config

from aiplatform.tracing import Tracing

_DONE = object()

Emit = Callable[[Any], Awaitable[None]]


def emitter() -> Emit:
    """The current run's emit function (call it inside a graph node)."""
    return get_config()["configurable"]["emit"]


async def stream_graph(graph, state: Any, config: dict,
                       tracing: Tracing | None = None) -> AsyncIterator[Any]:
    queue: asyncio.Queue = asyncio.Queue()
    tracing = tracing or Tracing()

    async def emit(event: Any) -> None:
        await queue.put(event)
        await queue.join()  # wait until the consumer has handled it

    async def run() -> None:
        try:
            # Entered inside the task, so the tracing context never leaks to the caller.
            with tracing.run(config) as run_config:
                await graph.ainvoke(state, {**run_config, "configurable": {"emit": emit}})
        finally:
            queue.put_nowait(_DONE)

    task = asyncio.create_task(run())
    try:
        while (event := await queue.get()) is not _DONE:
            yield event
            queue.task_done()
        await task  # re-raise the graph's error, if any
    finally:
        if not task.done():  # the consumer stopped early
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
