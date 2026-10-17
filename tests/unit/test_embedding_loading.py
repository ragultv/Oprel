"""Loading an embedding model must not freeze the server, and concurrent callers wait for the same load."""

import asyncio


def test_embedding_load_does_not_block_event_loop_and_is_serialized(monkeypatch):
    import threading
    import time

    from oprel.server.services import generation

    state = {"running": 0, "max": 0, "calls": 0}
    guard = threading.Lock()

    def slow_load(model_id):
        with guard:
            state["running"] += 1
            state["max"] = max(state["max"], state["running"])
            state["calls"] += 1
        time.sleep(0.3)
        with guard:
            state["running"] -= 1

    monkeypatch.setattr(generation, "load_model", slow_load)

    async def scenario():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        t = asyncio.create_task(ticker())
        params = generation.EmbeddingParams(input="hi", model="nomic-embed-text")
        results = await asyncio.gather(
            generation.get_embeddings(params), generation.get_embeddings(params), return_exceptions=True
        )
        t.cancel()
        return ticks, results

    ticks, results = asyncio.run(scenario())
    assert ticks >= 10  # the loop kept running while models "loaded" (it was blocked before)
    assert state["calls"] == 2 and state["max"] == 1  # callers wait for each other
    assert all(isinstance(r, Exception) for r in results)  # no real model exists in this test
