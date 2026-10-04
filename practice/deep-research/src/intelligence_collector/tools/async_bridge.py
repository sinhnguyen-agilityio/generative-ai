"""Run async MCP clients from synchronous callers while preserving trace context."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context


def run_async(function, *args):
    async def invoke():
        return await function(*args)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # asyncio checks the actual loop, rather than inherited async-library context.
        return asyncio.run(invoke())
    context = copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(context.run, lambda: asyncio.run(invoke())).result()
