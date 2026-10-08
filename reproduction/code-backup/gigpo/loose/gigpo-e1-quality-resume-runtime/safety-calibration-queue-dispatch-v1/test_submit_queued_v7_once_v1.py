"""CPU-only protocol tests using synthetic files/commands, never Slurm or GPU."""
import concurrent.futures
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import submit_queued_v7_once_v1 as q


BASE = Path(__file__).resolve().parents[1] / 'safety-calibration-deployment-v7'
sys.path.insert(0, str(BASE))
import deployment_contract_v7 as frozen


class Fixture:
    def __init__(self, directory):
        self.root = Path(directory)
        self.logs = self.root / 'logs'
        self.logs.mkdir()
        self.helpers = self.root / 'tools/helpers'
        self.helpers.mkdir(parents=True)
        self.launcher = self.helpers / 'calibration_v1.sbatch'
        self.launcher.write_bytes((BASE.parent / 'safety-calibration-v7/calibration_v1.sbatch').read_bytes())
        self.intent = self.logs / 'policy-safety-calibration-52befe9-submit-intent-v7.json'
        self.receipt = self.logs / 'policy-safety-calibration-52befe9-submit-receipt-v7.json'
        self.binding_path = self.logs / 'binding.json'
        self.review_path = self.logs / 'review.json'
        self.binding = dict(repo='/immutable/source52', helpers=str(self.helpers), files=[dict(
            path=str(self.launcher), bytes=self.launcher.stat().st_size, sha256=q.digest(self.launcher))],
            evidence=[], source_native_acceptance={'job_id': '3267'})
        self.review = dict(status='PASS', scope=frozen.SCOPE, binding_sha256=q.EXPECTED_BINDING_SHA,
            submit_helper_sha256=q.digest(q.__file__), submit_helper_path=str(q.TARGET_HELPER),
            submit_mode=q.SUBMIT_MODE, shared_intent_path=str(self.intent), shared_receipt_path=str(self.receipt),
            frozen_contract_sha256=q.CONTRACT_SHA, frozen_original_once_sha256=q.OLD_ONCE_SHA)
        self.ready_calls = 0
        self.claim_barrier = None
        self.calls = []
        self.lock = threading.Lock()
        self.queue = '3278|search-g1|RUNNING\n'
        self.gpu = '999,/own/other/slurm/job,1024 MiB\n'
        self.sbatch_result = subprocess.CompletedProcess([], 0, '99999\n', '')
        self.command_identity = str(self.helpers / 'calibration_v1.sbatch')
        self.fail_command = None
        self.contract = types.SimpleNamespace(SCOPE=frozen.SCOPE, read_bound=self.read_bound,
            validate_ready=self.validate_ready, write_once=self.write_once)

    def read_bound(self, item):
        if item['path'] == str(self.binding_path):
            if item['sha256'] != q.EXPECTED_BINDING_SHA:
                raise ValueError('Synthetic wrong binding identity')
            return self.binding
        if item['path'] == str(self.review_path) and item['sha256'] == 'reviewsha':
            return self.review
        raise ValueError('Unknown synthetic review input')

    def validate_ready(self, binding):
        self.ready_calls += 1
        if binding.get('not_ready'):
            raise ValueError('Synthetic frozen READY rejection')

    def write_once(self, path, value):
        if Path(path) == self.intent and self.claim_barrier:
            self.claim_barrier.wait(timeout=5)
        frozen.write_once(path, value)

    def run(self, command, **kwargs):
        with self.lock:
            self.calls.append(command)
        if self.fail_command == command[0]:
            return subprocess.CompletedProcess(command, 1, '', 'read failed')
        if command[0] == 'squeue':
            return subprocess.CompletedProcess(command, 0, self.queue, '')
        if command[0] == 'nvidia-smi':
            return subprocess.CompletedProcess(command, 0, self.gpu, '')
        if command[0] == 'scontrol':
            return subprocess.CompletedProcess(command, 0, 'JobId=1 Command=' + self.command_identity + '\n', '')
        if command[0] == 'sbatch':
            if isinstance(self.sbatch_result, BaseException):
                raise self.sbatch_result
            return self.sbatch_result
        raise AssertionError('Unexpected command')

    def patches(self):
        return patch.multiple(q, ROOT=self.root, HELPERS=self.helpers, INTENT=self.intent,
                              RECEIPT=self.receipt, EXPECTED_BINDING=self.binding_path)

    def dispatch(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return q.dispatch(str(self.binding_path), q.EXPECTED_BINDING_SHA,
                              str(self.review_path), 'reviewsha', self.contract, self.run)

    def sbatch_count(self):
        return sum(c[0] == 'sbatch' for c in self.calls)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = Fixture(self.tmp.name)
        self.patch = self.f.patches()
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def refuse_no_claim(self):
        with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
            self.f.dispatch()
        self.assertFalse(self.f.intent.exists())
        self.assertFalse(self.f.receipt.exists())
        self.assertEqual(self.f.sbatch_count(), 0)

    def test_foreign_allocation_and_gpu_allow_queue_once(self):
        result = self.f.dispatch()
        self.assertEqual(result['status'], 'SUBMITTED_EXACTLY_ONCE')
        self.assertEqual(result['job_id'], '99999')
        self.assertIsNone(result['scheduler_state_observed'])
        self.assertFalse(result['runtime_resource_accepted'])
        self.assertFalse(result['probability_accepted'])
        self.assertEqual(self.f.sbatch_count(), 1)
        command = next(c for c in self.f.calls if c[0] == 'sbatch')
        self.assertEqual(command, ['sbatch', '--parsable', str(self.f.launcher),
                                  str(self.f.binding_path), q.EXPECTED_BINDING_SHA,
                                  self.f.binding['repo'], self.f.binding['helpers']])
        self.assertFalse(any(c[0] in ('scancel', 'srun', 'salloc') for c in self.f.calls))
        with self.assertRaises(ValueError):
            self.f.dispatch()
        self.assertEqual(self.f.sbatch_count(), 1)

    def test_empty_node_allows_once(self):
        self.f.queue = self.f.gpu = ''
        self.assertEqual(self.f.dispatch()['status'], 'SUBMITTED_EXACTLY_ONCE')

    def test_pending_other_job_without_compute_allows_queue(self):
        self.f.queue = '3278|search-g1|PENDING\n'
        self.f.gpu = ''
        self.assertEqual(self.f.dispatch()['status'], 'SUBMITTED_EXACTLY_ONCE')

    def test_empty_queue_nonempty_compute_refused_and_preserved(self):
        self.f.queue = ''
        self.refuse_no_claim()
        logs = list(self.f.logs.glob('policy-safety-queued-occupancy-*.json'))
        self.assertEqual(len(logs), 1)
        self.assertEqual(json.loads(logs[0].read_text())['occupancy']['gpu_compute']['stdout'], self.f.gpu)

    def test_pending_only_nonempty_compute_not_allocation(self):
        self.f.queue = '3278|search-g1|PENDING\n'
        self.refuse_no_claim()

    def test_existing_intent_blocks(self):
        self.f.intent.write_text('{}')
        with self.assertRaises(ValueError):
            self.f.dispatch()
        self.assertEqual(self.f.calls, [])

    def test_existing_receipt_blocks(self):
        self.f.receipt.write_text('{}')
        with self.assertRaises(ValueError):
            self.f.dispatch()
        self.assertEqual(self.f.calls, [])

    def test_dangling_intent_symlink_blocks(self):
        self.f.intent.symlink_to(self.f.root / 'absent')
        with self.assertRaises(ValueError):
            self.f.dispatch()
        self.assertEqual(self.f.calls, [])

    def test_wrong_binding_path(self):
        with self.assertRaises(ValueError):
            q.dispatch('/wrong', q.EXPECTED_BINDING_SHA, str(self.f.review_path), 'reviewsha', self.f.contract, self.f.run)
        self.assertFalse(self.f.intent.exists())

    def test_wrong_binding_sha(self):
        with self.assertRaises(ValueError):
            q.dispatch(str(self.f.binding_path), 'bad', str(self.f.review_path), 'reviewsha', self.f.contract, self.f.run)
        self.assertFalse(self.f.intent.exists())

    def test_native_ready_refusal(self):
        self.f.binding['not_ready'] = True
        self.refuse_no_claim()

    def test_review_status(self):
        self.f.review['status'] = 'FAILED'
        self.refuse_no_claim()

    def test_review_helper_sha(self):
        self.f.review['submit_helper_sha256'] = q.OLD_ONCE_SHA
        self.refuse_no_claim()

    def test_review_target_path(self):
        self.f.review['submit_helper_path'] = '/wrong/helper'
        self.refuse_no_claim()

    def test_review_mode(self):
        self.f.review['submit_mode'] = 'oldonce'
        self.refuse_no_claim()

    def test_review_shared_intent(self):
        self.f.review['shared_intent_path'] = '/different/claim'
        self.refuse_no_claim()

    def test_review_shared_receipt(self):
        self.f.review['shared_receipt_path'] = '/different/receipt'
        self.refuse_no_claim()

    def test_review_contract_sha(self):
        self.f.review['frozen_contract_sha256'] = 'changed'
        self.refuse_no_claim()

    def test_review_legacy_sha(self):
        self.f.review['frozen_original_once_sha256'] = 'changed'
        self.refuse_no_claim()

    def test_script_sha_changes_refused(self):
        self.f.launcher.write_text(self.f.launcher.read_text() + '#changed\n')
        self.refuse_no_claim()

    def test_same_script_active_duplicate_refused(self):
        self.f.queue = '3278|' + q.JOB_NAME + '|PENDING\n'
        self.refuse_no_claim()

    def test_same_name_different_absolute_script_is_only_observed(self):
        self.f.queue = '3278|' + q.JOB_NAME + '|RUNNING\n'
        self.f.command_identity = '/another/thread/own-calibration.sbatch'
        self.assertEqual(self.f.dispatch()['status'], 'SUBMITTED_EXACTLY_ONCE')
        self.assertFalse(any(c[0] == 'scancel' for c in self.f.calls))

    def test_same_name_relative_command_is_ambiguous(self):
        self.f.queue = '3278|' + q.JOB_NAME + '|RUNNING\n'
        self.f.command_identity = 'calibration_v1.sbatch'
        self.refuse_no_claim()

    def test_failed_queue_read_no_claim(self):
        self.f.fail_command = 'squeue'
        self.refuse_no_claim()

    def test_failed_compute_read_no_claim(self):
        self.f.fail_command = 'nvidia-smi'
        self.refuse_no_claim()

    def test_malformed_queue_no_claim(self):
        self.f.queue = 'malformed'
        self.refuse_no_claim()

    def test_timeout_preserves_claim_receipt_no_retry(self):
        self.f.sbatch_result = subprocess.TimeoutExpired('sbatch', 60)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.f.dispatch()
        self.assertTrue(self.f.intent.exists())
        self.assertEqual(json.loads(self.f.receipt.read_text())['status'], 'UNCERTAIN_OR_FAILED_NO_RETRY')
        with self.assertRaises(ValueError):
            self.f.dispatch()
        self.assertEqual(self.f.sbatch_count(), 1)

    def test_sbatch_failure_preserves_claim(self):
        self.f.sbatch_result = subprocess.CompletedProcess([], 1, '', 'rejected')
        with self.assertRaises(ValueError):
            self.f.dispatch()
        self.assertTrue(self.f.intent.exists())
        self.assertEqual(json.loads(self.f.receipt.read_text())['status'], 'UNCERTAIN_OR_FAILED_NO_RETRY')

    def test_nonjob_stdout_preserves_claim(self):
        self.f.sbatch_result = subprocess.CompletedProcess([], 0, '99999\nwarning', '')
        with self.assertRaises(ValueError):
            self.f.dispatch()
        self.assertEqual(json.loads(self.f.receipt.read_text())['status'], 'UNCERTAIN_OR_FAILED_NO_RETRY')

    def test_cluster_job_response_accepted_not_running(self):
        self.f.sbatch_result = subprocess.CompletedProcess([], 0, '99999;cluster\n', '')
        self.assertIsNone(self.f.dispatch()['scheduler_state_observed'])

    def test_two_queued_dispatchers_share_atomic_claim(self):
        self.f.claim_barrier = threading.Barrier(2)
        def attempt():
            try:
                return self.f.dispatch()['status']
            except FileExistsError:
                return 'CLAIM_ALREADY_EXISTS'
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            rows = list(pool.map(lambda _: attempt(), range(2)))
        self.assertEqual(sorted(rows), ['CLAIM_ALREADY_EXISTS', 'SUBMITTED_EXACTLY_ONCE'])
        self.assertEqual(self.f.sbatch_count(), 1)

    def test_legacy_once_race_uses_same_atomic_claim(self):
        spec = importlib.util.spec_from_file_location('frozen_legacy_once_for_test', BASE / 'submit_once_v7.py')
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        legacy_review = dict(status='PASS', scope=frozen.SCOPE, binding_sha256=q.EXPECTED_BINDING_SHA,
                             submit_helper_sha256=q.digest(BASE / 'submit_once_v7.py'))
        def legacy_read(item):
            return self.f.binding if item['path'] == str(self.f.binding_path) else legacy_review
        self.f.claim_barrier = threading.Barrier(2)
        def old_attempt():
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    return legacy.main(str(self.f.binding_path), q.EXPECTED_BINDING_SHA, '/legacy/review', 'sha')['status']
            except FileExistsError:
                return 'CLAIM_ALREADY_EXISTS'
        def queued_attempt():
            try:
                return self.f.dispatch()['status']
            except FileExistsError:
                return 'CLAIM_ALREADY_EXISTS'
        with patch.multiple(legacy, INTENT=self.f.intent, RECEIPT=self.f.receipt, HELPERS=self.f.helpers,
                read_bound=legacy_read, validate_ready=self.f.validate_ready, write_once=self.f.write_once), \
             patch.object(legacy.subprocess, 'check_output', return_value=''), \
             patch.object(legacy.subprocess, 'run', side_effect=self.f.run):
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                rows = [pool.submit(old_attempt), pool.submit(queued_attempt)]
                rows = [task.result() for task in rows]
        self.assertEqual(sorted(rows), ['CLAIM_ALREADY_EXISTS', 'SUBMITTED_EXACTLY_ONCE'])
        self.assertEqual(self.f.sbatch_count(), 1)

    def test_wrong_management_host_before_contract_read(self):
        with patch.object(q.socket, 'gethostname', return_value='other-host'), \
             patch.object(q, 'load_frozen_contract') as loader:
            with self.assertRaises(ValueError):
                q.main('a', 'b', 'c', 'd')
            loader.assert_not_called()

    def test_wrong_helper_installation_before_contract_read(self):
        with patch.object(q.socket, 'gethostname', return_value='h100-8'), \
             patch.object(q, 'load_frozen_contract') as loader:
            with self.assertRaises(ValueError):
                q.main('a', 'b', 'c', 'd')
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
