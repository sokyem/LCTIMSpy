"""In-process event bus feeding the single multiplexed SSE stream.

Job threads publish from worker threads; SSE handlers consume on the asyncio
loop. Every event gets a monotonically increasing id and is kept in a ring
buffer, so a browser that reconnects with ``Last-Event-ID`` (EventSource does
this natively) or ``?since=`` replays what it missed instead of losing progress
mid-extraction. If the gap is older than the buffer, a ``resync`` event tells
the client to refetch state.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field

BUFFER_SIZE = 2000


@dataclass(frozen=True)
class Event:
    id: int
    type: str
    data: dict
    time: float = field(default_factory=time.time)


class EventBus:
    def __init__(self, maxlen: int = BUFFER_SIZE):
        self._events: deque[Event] = deque(maxlen=maxlen)
        self._next_id = 1
        self._lock = threading.Lock()
        self._subscribers: set[Subscription] = set()

    @property
    def subscriber_count(self) -> int:
        """Connected SSE clients; the browser fallback shuts down when this stays at 0."""
        with self._lock:
            return len(self._subscribers)

    @property
    def last_id(self) -> int:
        with self._lock:
            return self._next_id - 1

    def publish(self, type: str, data: dict) -> Event:
        with self._lock:
            event = Event(self._next_id, type, data)
            self._next_id += 1
            self._events.append(event)
            subscribers = list(self._subscribers)
        for sub in subscribers:
            sub._deliver(event)
        return event

    def since(self, last_id: int) -> tuple[list[Event], bool]:
        """Events after ``last_id``, and whether some were already dropped."""
        with self._lock:
            events = [e for e in self._events if e.id > last_id]
            first = self._events[0].id if self._events else self._next_id
        return events, last_id < first - 1

    def subscribe(self, last_id: int = 0) -> Subscription:
        """Replay after ``last_id``, then follow live events. Call from the event loop."""
        sub = Subscription(self, asyncio.get_running_loop(), last_id)
        with self._lock:
            self._subscribers.add(sub)        # register before replay: no gap
        backlog, gap = self.since(last_id)
        if gap:
            sub._backlog.append(Event(0, "resync", {"reason": "events were dropped; refetch state"}))
        sub._backlog.extend(backlog)
        return sub

    def _unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            self._subscribers.discard(sub)


class Subscription:
    """One client's view of the stream. ``next()`` never misses or repeats an event.

    A plain object rather than an async generator: timing out a wait on a
    generator would cancel and finalize it, ending the stream.
    """

    def __init__(self, bus: EventBus, loop: asyncio.AbstractEventLoop, last_id: int):
        self._bus = bus
        self._loop = loop
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._backlog: deque[Event] = deque()
        self._seen = last_id

    def _deliver(self, event: Event) -> None:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)
        except RuntimeError:          # loop closed; the client is gone
            pass

    async def next(self, timeout: float | None = None) -> Event | None:
        """The next event, or None if ``timeout`` elapses first."""
        while self._backlog:
            event = self._backlog.popleft()
            if event.id == 0 or event.id > self._seen:
                self._seen = max(self._seen, event.id)
                return event
        while True:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout)
            except asyncio.TimeoutError:
                return None
            if event.id > self._seen:
                self._seen = event.id
                return event

    def close(self) -> None:
        self._bus._unsubscribe(self)
