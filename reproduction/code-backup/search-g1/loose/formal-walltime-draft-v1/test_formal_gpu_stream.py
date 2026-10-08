"""UNEXECUTED offline test text. Requires new exact-byte static PASS first.

No transport, scheduler, /proc, training module or GPU call is used. The sealed
v8 stdlib core is loaded for pure validator parity after external review only.
"""
import copy
import hashlib
import os
import tempfile
import time
import types
import unittest
from pathlib import Path

import formal_gpu_stream as stream

CORE_PATH = Path('/Users/shaoshitong/Documents/Search-G1/preparation/formal-progress-draft-v8/formal_progress_core.py')
CORE_SHA = '45c0a4660df4e4bd00d32ab20bcc5b124bc4154237bacea952d824524d948b47'
raw = CORE_PATH.read_bytes()
if hashlib.sha256(raw).hexdigest() != CORE_SHA:
    raise ValueError('Sealed parity core differs')
core = types.ModuleType('pure_v8_core')
exec(compile(raw, str(CORE_PATH), 'exec'), core.__dict__)


def inventory():
    return {index: dict(uuid='GPU-%08x-0000-0000-0000-000000000000' % index, total_mib=81559)
            for index in range(8)}


def sampler():
    return dict(pid=50, start_ticks=70, sid=50, uid=os.getuid())


def sample(start=101, end=102, memory=200):
    return dict(schema_version=1, role='actual_common8_gpu_occupancy', run_id=core.RUN,
        slurm_job_id=core.JOB, source_map_sha256=core.MAP_SHA, sampler_identity=sampler(),
        sampling_interval_seconds=2, query_started_unix=start, unix=end,
        gpus=[dict(index=index, uuid=inventory()[index]['uuid'], used_mib=memory, total_mib=81559,
                   util_percent=25) for index in range(8)])


def policy(duration=172800, line=4096):
    count = (duration + 21600) // 2 + 1
    return dict(total_walltime_seconds=duration, capture_allowance_seconds=21600, max_rows=count,
        max_bytes=count * (line + 1) + line, max_line_bytes=line, max_tail_bytes=line,
        chunk_bytes=37, read_seconds=120, summary_bytes=2097152)


def scan(path, *, now=150, duration=172800, previous=None, selected_policy=None):
    return stream.scan_gpu(path, core, inventory(), selected_policy or policy(duration), duration=duration,
        now=now, run_started_unix=100, training_claim_unix=103, expected_sampler=sampler(),
        deadline=time.monotonic() + 120, previous=previous)


class GPUStream(unittest.TestCase):
    def test_parity_split_chunks_gap_straddlers_and_tail(self):
        rows = [sample(101, 102, 200), sample(102, 104, 999), sample(106, 107, 300), sample(115, 116, 350)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'samples.jsonl'
            path.write_bytes(b''.join(core.canonical(row) + b'\n' for row in rows) + b'{"partial":')
            result, provenance, recheck = scan(path)
            expected = core.validate_gpu_samples(rows, inventory(), now=150, run_started_unix=100, training_claim_unix=103)
            expected['observation_incomplete'] = True
            self.assertEqual(result, expected)
            self.assertEqual(provenance['complete_rows'], 4)
            self.assertEqual(provenance['partial_tail_bytes'], 11)
            self.assertTrue(provenance['verified_second_pass'])
            self.assertIsNone(recheck)
            self.assertEqual(result['windows']['trainer_window']['sampled_peak_mib']['0'], 350)
            self.assertEqual(result['cumulative_sampled_peak_mib']['0'], 999)

    def test_append_completion_of_old_partial_tail_and_exact_prefix_recheck(self):
        rows = [sample(), sample(106, 107)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'samples.jsonl'
            line = core.canonical(rows[1]) + b'\n'
            path.write_bytes(core.canonical(rows[0]) + b'\n' + line[:10])
            _, old, _ = scan(path)
            with path.open('ab') as file:
                file.write(line[10:])
            result, new, recheck = scan(path, previous=old)
            self.assertEqual(result['sample_count'], 2)
            self.assertGreater(new['complete_prefix_bytes'], old['complete_prefix_bytes'])
            self.assertEqual(recheck, dict(prefix_bytes=old['complete_prefix_bytes'], sha256=old['complete_prefix_sha256']))
            # Equal-size prefix content corruption still shares inode, but fails.
            data = bytearray(path.read_bytes()); data[20] ^= 1; path.write_bytes(data)
            with self.assertRaises((ValueError, core.ObservationError)):
                scan(path, previous=old)

    def test_line_row_byte_deadline_and_identity_caps_are_not_zero_peaks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'samples.jsonl'
            path.write_bytes(core.canonical(sample()) + b'\n')
            for key, value in (('slurm_job_id', '3310'), ('sampling_interval_seconds', True),
                               ('source_map_sha256', 'a' * 64), ('unix', float('inf'))):
                row = sample(); row[key] = value
                accumulator = stream.GPUAccumulator(core, inventory(), now=150, run_started_unix=100,
                    training_claim_unix=103, expected_sampler=sampler())
                with self.assertRaises((ValueError, core.ObservationError)):
                    accumulator.consume(row)
            with self.assertRaises(ValueError):
                stream.scan_prefix(path, owner_uid=os.getuid(), max_bytes=1, max_rows=1, max_line_bytes=4096,
                    max_tail_bytes=4096, chunk_bytes=10, deadline=time.monotonic() + 10, consume=lambda row: None,
                    parse=core.strict_json)
            with self.assertRaises(ValueError):
                stream.scan_prefix(path, owner_uid=os.getuid(), max_bytes=10000, max_rows=1, max_line_bytes=100,
                    max_tail_bytes=100, chunk_bytes=10, deadline=time.monotonic() - 1, consume=lambda row: None,
                    parse=core.strict_json)
            with self.assertRaises(ValueError):
                stream.validate_policy(dict(policy(), max_rows=99999999), 172800)

    def test_constant_state_and_bounded_twenty_gap_examples(self):
        accumulator = stream.GPUAccumulator(core, inventory(), now=10000, run_started_unix=100,
            training_claim_unix=103, expected_sampler=sampler())
        for index in range(101):
            accumulator.consume(sample(101 + index * 10, 102 + index * 10))
        result = accumulator.summary()
        self.assertEqual(result['observed_gap_count'], 100)
        self.assertEqual(len(result['gap_examples']), 20)
        self.assertTrue(result['gap_examples_truncated'])
        self.assertFalse(any(isinstance(value, list) and len(value) > 20 for value in vars(accumulator).values()))

    def test_more_than_100000_rows_and_128mib_without_rows_materialization(self):
        # Explicit finite longrun capacity, independently synthetic. This is no
        # selected live job extension. Fixture writer itself streams rows.
        duration = 300000; count = 110001; selected = policy(duration)
        selected['chunk_bytes'] = 65536
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'samples.jsonl'
            with path.open('wb') as file:
                for index in range(count):
                    file.write(core.canonical(sample(101 + index * 2, 102 + index * 2, 200 + index % 17)) + b'\n')
            self.assertGreater(path.stat().st_size, 128 * 1024 * 1024)
            result, provenance, _ = scan(path, now=102 + count * 2, duration=duration, selected_policy=selected)
            self.assertEqual(result['sample_count'], count)
            self.assertEqual(provenance['complete_rows'], count)
            self.assertEqual(result['cumulative_sampled_peak_mib']['0'], 216)

    def test_replacement_truncation_symlink_empty_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'samples.jsonl'
            path.write_bytes(core.canonical(sample()) + b'\n')
            _, old, _ = scan(path)
            path.unlink(); path.write_bytes(core.canonical(sample()) + b'\n')
            with self.assertRaises(ValueError):
                scan(path, previous=old)
            path.write_bytes(b'')
            with self.assertRaises(ValueError):
                scan(path, previous=old)
            result, _, _ = scan(path)
            self.assertIsNone(result['cumulative_sampled_peak_mib'])
            path.write_bytes(b'{"run_id":"x","run_id":"x"}\n')
            with self.assertRaises((ValueError, core.ObservationError)):
                scan(path)
            target = path.with_name('target'); path.rename(target); path.symlink_to(target)
            with self.assertRaises(ValueError):
                scan(path)


if __name__ == '__main__':
    unittest.main()
