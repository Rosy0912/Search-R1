"""A single FIFO consumer with a fixed collection window and bounded batches."""
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
import threading
import time
import uuid


@dataclass
class PendingRequest:
    queries: list
    k: int
    return_scores: bool
    compare_original: bool = False
    enqueued: float = field(default_factory=time.monotonic)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    future: Future = field(default_factory=Future)


class RequestQueue:
    def __init__(self, collection_seconds=0.025, max_pending=128):
        if not 0 <= collection_seconds <= 1:
            raise ValueError("Collection window must be between 0 and 1 seconds")
        self.collection_seconds = collection_seconds
        self.max_pending = max_pending
        self.pending = deque()
        self.condition = threading.Condition()
        self.closed = False

    def submit(self, request):
        if not 0 < len(request.queries) <= 2560:
            raise ValueError("Queue accepts 1..2560 queries per request")
        with self.condition:
            if self.closed:
                raise RuntimeError("Retriever queue is closed")
            if len(self.pending) >= self.max_pending:
                raise RuntimeError("Retriever queue is full")
            self.pending.append(request)
            self.condition.notify_all()
        return request.future

    def next_batch(self):
        with self.condition:
            while not self.pending and not self.closed:
                self.condition.wait()
            if not self.pending:
                return []
            # A fixed deadline is not extended by newly arriving requests.
            deadline = time.monotonic() + self.collection_seconds
            while not self.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=remaining)
            batch, total = [], 0
            while self.pending and len(batch) < 8:
                request = self.pending[0]
                if total + len(request.queries) > 20480:
                    break
                batch.append(self.pending.popleft())
                total += len(request.queries)
            return batch

    def close(self, reason):
        with self.condition:
            self.closed = True
            while self.pending:
                request = self.pending.popleft()
                request.future.set_exception(RuntimeError(reason))
            self.condition.notify_all()
