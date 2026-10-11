"""CPU lifecycle proofs only; this does not execute CUDA copies or measure speed."""
import unittest
from .pinned_overlap import DoubleBufferTiles, Tile, memory_plan, tile_plan


class FakeBackend:
    def __init__(self):
        self.events = []
        self.dma_pending = [False, False]
        self.current = [None, None]

    def sync_ready(self, slot):
        self.events.append(('sync_ready', slot))
        self.dma_pending[slot] = False

    def wait_used(self, slot):
        self.events.append(('wait_used', slot))

    def stage(self, tile):
        if self.dma_pending[tile.slot]:
            raise AssertionError('Host overwritten before prior DMA completes')
        self.events.append(('stage', tile.number))
        self.current[tile.slot] = tile
        self.dma_pending[tile.slot] = True

    def acquire(self, tile):
        self.events.append(('compute_wait_ready', tile.number))
        assert self.current[tile.slot] == tile
        return tile

    def consumed(self, tile):
        self.events.append(('record_consumed', tile.number))

    def close(self):
        self.events.append(('close_both_streams',))
        self.dma_pending = [False, False]


class Checks(unittest.TestCase):
    def test_exact_original_tile_boundaries_and_tail(self):
        plan = tile_plan(21015324)
        self.assertEqual(len(plan), 642)
        self.assertEqual(plan[0], Tile(0, 0, 32768))
        self.assertEqual(plan[-1].stop, 21015324)
        self.assertEqual(plan[-1].stop-plan[-1].start, 11036)
        self.assertEqual(sum(x.stop-x.start for x in plan), 21015324)

    def test_original_scan_bytes_not_reduced_and_buffers_bounded(self):
        plan = memory_plan(21015324, 768)
        self.assertEqual(plan['index_scan_bytes'], 64559075328)
        self.assertEqual(plan['pinned_cpu_buffers_bytes'], 201326592)
        self.assertEqual(plan['device_buffers_bytes'], 201326592)
        self.assertEqual(plan['max_one_query_micro_distances_bytes'], 8388608)

    def test_next_tile_staged_before_current_compute_but_same_slot_protected(self):
        backend = FakeBackend(); buffer = DoubleBufferTiles(backend)
        a,b,c = tile_plan(32768*2+7)
        buffer.stage(a); buffer.stage(b)
        self.assertEqual(buffer.acquire(a), a); buffer.consumed(a)
        buffer.stage(c)
        self.assertEqual(backend.events[:2], [('stage',0), ('stage',1)])
        self.assertLess(backend.events.index(('sync_ready',0)), backend.events.index(('stage',2)))
        self.assertLess(backend.events.index(('wait_used',0)), backend.events.index(('stage',2)))
        self.assertEqual(buffer.acquire(b), b); buffer.consumed(b)
        self.assertEqual(buffer.acquire(c), c); buffer.consumed(c)
        buffer.close()
        self.assertEqual(backend.events[-1], ('close_both_streams',))

    def test_pending_device_consumer_cannot_be_overwritten(self):
        backend = FakeBackend(); buffer = DoubleBufferTiles(backend)
        a,b,c = tile_plan(32768*3)
        buffer.stage(a); buffer.stage(b); buffer.acquire(a)
        with self.assertRaisesRegex(ValueError, 'consumer'):
            buffer.stage(c)
        self.assertNotIn(('stage',2), backend.events)

    def test_wrong_tile_duplicate_completion_and_exception_cleanup(self):
        backend = FakeBackend(); buffer = DoubleBufferTiles(backend)
        a,b,c = tile_plan(32768*3)
        buffer.stage(a)
        with self.assertRaises(ValueError): buffer.acquire(c)
        buffer.acquire(a); buffer.consumed(a)
        with self.assertRaises(ValueError): buffer.consumed(a)
        try:
            buffer.stage(b)
            raise RuntimeError('simulated caller compute failure')
        except RuntimeError:
            buffer.close()
        self.assertFalse(any(backend.dma_pending))


if __name__ == '__main__': unittest.main()
