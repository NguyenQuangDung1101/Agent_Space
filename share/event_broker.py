from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import AsyncIterator, Optional

from share.schemas import ExecutionEvent


TERMINAL_EVENTS = {
    "session_completed",
    "session_failed",
}


class EventBroker:
    def __init__(self) -> None:
        self._history: dict[str, list[ExecutionEvent]] = defaultdict(list)
        self._subscribers: dict[
            str,
            set[asyncio.Queue[ExecutionEvent]],
        ] = defaultdict(set)

    def seed(
        self,
        session_id: str,
        events: list[ExecutionEvent],
    ) -> None:
        if session_id not in self._history:
            self._history[session_id] = list(events)

    def publish(
        self,
        session_id: str,
        event: ExecutionEvent,
    ) -> None:
        self._history[session_id].append(event)

        for queue in list(self._subscribers.get(session_id, set())):
            queue.put_nowait(event)

    async def stream(
        self,
        session_id: str,
        initial_events: Optional[list[ExecutionEvent]] = None,
    ) -> AsyncIterator[Optional[ExecutionEvent]]:
        self.seed(session_id, initial_events or [])

        queue: asyncio.Queue[ExecutionEvent] = asyncio.Queue()
        self._subscribers[session_id].add(queue)
        history = list(self._history[session_id])

        try:
            for event in history:
                yield event
                if event.event_type in TERMINAL_EVENTS:
                    return

            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=15,
                    )
                except asyncio.TimeoutError:
                    yield None
                    continue

                yield event

                if event.event_type in TERMINAL_EVENTS:
                    return
        finally:
            self._subscribers[session_id].discard(queue)
