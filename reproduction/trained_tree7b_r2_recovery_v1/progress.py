"""Bound transient progress observations without reading or changing predictions."""
import json
from pathlib import Path
import time


class ProgressObserver:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.last = {}
        self.errors_since = {}

    def read(self, output):
        now = self.clock()
        for rank in range(8):
            path = Path(output) / ('rank-%d.progress.json' % rank)
            try:
                value = json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                # No progress file during initial worker construction is normal.
                # After a prior observation, absence cannot be hidden indefinitely.
                if rank not in self.last and isinstance(exc, FileNotFoundError):
                    continue
                start = self.errors_since.setdefault(rank, now)
                if now - start >= 60:
                    raise TimeoutError('Progress observation unavailable for 60 seconds; inspect without retry') from exc
                continue
            assert value['rank'] == rank and type(value['completed']) is int
            assert type(value['total']) is int and 0 <= value['completed'] <= value['total']
            assert value['total'] == len(range(rank, 51713, 8))
            if rank in self.last:
                assert value['completed'] >= self.last[rank]['completed']
            self.last[rank] = value
            self.errors_since.pop(rank, None)
        return list(self.last.values())
