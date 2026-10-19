"""Real CPU interprocess sockets/lock checks, no model or GPU allocation."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import socket
import tempfile
import time
import unittest
from unittest.mock import patch

from . import port_recovery as module


def probe(root, rank, queue):
    os.environ['VLLM_PORT'] = str(20000 + rank * 2048)
    def constructor(*args, **kwargs):
        active = Path(root) / 'active'
        with active.open('x'):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                sock.listen()
                time.sleep(.02)
                result = (args, kwargs)
        active.unlink()
        return result
    result = module.construct_serialized(constructor, root, rank, ('fixture',), {'seed': 0})
    queue.put((rank, result))


class PortRecoveryTests(unittest.TestCase):
    def test_eight_process_constructor_only_lock(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'output').mkdir()
            ctx = mp.get_context('spawn'); queue = ctx.Queue()
            processes = [ctx.Process(target=probe, args=(root, rank, queue)) for rank in range(8)]
            for process in processes: process.start()
            result = [queue.get(timeout=60) for _ in processes]
            for process in processes:
                process.join(60); self.assertEqual(process.exitcode, 0)
            self.assertEqual({x[0] for x in result}, set(range(8)))
            self.assertTrue(all(x[1] == (('fixture',), {'seed': 0}) for x in result))
            starts = [json.loads(p.read_text()) for p in (Path(root) / 'output').glob('startup-rank-*.json') if '.complete.' not in p.name]
            self.assertEqual(len(starts), 8)
            self.assertEqual(len({x['vllm_port'] for x in starts}), 8)

    def test_arguments_rng_and_exact_once(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, VLLM_PORT='22048'):
            (Path(root) / 'output').mkdir(); before = random.getstate(); calls = []
            def constructor(*args, **kwargs):
                calls.append((args, kwargs)); return object()
            module.construct_serialized(constructor, root, 1, ('model',), dict(seed=0, temperature=0))
            self.assertEqual(random.getstate(), before)
            self.assertEqual(calls, [(('model',), dict(seed=0, temperature=0))])
            with self.assertRaises(FileExistsError):
                module.construct_serialized(constructor, root, 1, (), {})
            self.assertEqual(len(calls), 1)

    def test_constructor_failure_preserved_not_retried(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, VLLM_PORT='20000'):
            (Path(root) / 'output').mkdir(); calls = []
            def constructor():
                calls.append(1); raise RuntimeError('fixture')
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                module.construct_serialized(constructor, root, 0, (), {})
            self.assertEqual(calls, [1])
            self.assertFalse((Path(root) / 'output/startup-rank-0.complete.json').exists())

    def test_wrong_scientific_repeat_refused(self):
        with self.assertRaises(AssertionError):
            module.verify_recovery(dict(kind='searchr1', repeat=2, job=module.NEW_JOB, startup_recovery={}))


if __name__ == '__main__': unittest.main()
