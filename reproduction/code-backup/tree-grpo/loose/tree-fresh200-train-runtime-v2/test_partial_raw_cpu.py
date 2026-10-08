"""CPU-only fixtures: plain dictionaries and tiny non-PT bytes, no Torch import."""
import copy
import hashlib
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import partial_raw_evidence as raw
from check_partial_raw_cpu import held_state, verify


def digest(value):
    result = hashlib.sha256()
    def visit(value):
        if isinstance(value, dict):
            for key in sorted(value, key=repr):
                visit(key)
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            result.update(type(value).__name__.encode())
            for entry in value:
                visit(entry)
        else:
            result.update(repr((type(value).__name__, value)).encode())
    visit(value)
    return result.hexdigest()


def finite(value):
    if isinstance(value, dict):
        for entry in value.values():
            finite(entry)
    elif isinstance(value, (tuple, list)):
        for entry in value:
            finite(entry)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite fixture')


def ledger():
    return [dict(index=index, rows=16, local_row_indices=list(range((index-1)*16, index*16)),
                 status='SKIPPED_EMPTY', actual_writes=0) for index in range(1, 21)]


class PartialRawTests(unittest.TestCase):
    def setUp(self):
        # An explicit workspace directory avoids /var -> /private/var on Mac.
        self.temporary = tempfile.TemporaryDirectory(prefix='.partial-raw-fixture-', dir=Path(__file__).absolute().parent)
        self.root = Path(self.temporary.name)
        self.run = self.root / 'run-123'
        self.identity = dict(job_id='123', run_id='run-123', code_commit='a'*40, config_sha256='b'*64,
            data_identity='c'*64, model_identity='d'*64, training_protocol=raw.PROFILE,
            attempted_outer=1, completed_outer=0, attempt_id=1)
        self.directory = raw.partial_directory(self.run, self.identity)
        self.directory.mkdir(parents=True)
        workers = []
        for rank in range(8):
            workers.append(dict(self.file_record(f'worker-{rank:02d}.pt'), rank=rank, world_size=8,
                identity=dict(self.identity, rank=rank, world_size=8), status='PARTIAL_WORKER_STATE_SAVED',
                committed_optimizer_calls=0, scheduler_step_before_actual=0, scheduler_step_after_actual=0,
                optimizer_opportunity_ledger=ledger(), model_adam_finite=True, gradients_finite=False,
                state_finite=False, resume_permitted=False, automatic_rollback=False, outer_completed=False,
                total_committed_optimizer_calls=0))
        self.partial = dict(protocol=raw.PROTOCOL, identity=self.identity,
            worker_outcome=dict(protocol=raw.PROTOCOL, identity=self.identity, status='PARTIAL_WORKER_STATE_SAVED', worker_states=workers),
            worker_states_verified=True, verification_error=None, outer_completed=False, automatic_rollback=False,
            automatic_resume_supported=False, attempted_outer=1, completed_outer_updates=0,
            driver=dict(self.file_record('driver.pt', include_bytes=False), completed_outer_updates=0,
                        consumed_cursor=512, consumed_epoch=0),
            actor_inputs=dict(self.file_record('actor-inputs.pt'), rows=2560,
                              fields=['responses', 'attention_mask', 'loss_mask', 'old_log_probs', 'sampled_old_log_probs']))

    def tearDown(self):
        self.temporary.cleanup()

    def file_record(self, name, include_bytes=True):
        path = self.directory / name
        data = ('tiny non-PT fixture: ' + name).encode()
        path.write_bytes(data)
        record = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())
        if include_bytes:
            record['bytes'] = len(data)
        return record

    def worker(self):
        return dict(version='tree_partial_worker_v1', identity=dict(self.identity, rank=0, world_size=8),
            parameters={'p': [1.0]}, local_shards={'p': [1.0]}, buffers={}, gradients={'p': [float('nan')]},
            private_gradient_tensors={'p': {}}, optimizer={'state': {}}, scheduler={'last_epoch': 0},
            parameter_layout={'p': {'shape': [1]}}, rng={'python': (1,), 'numpy': (1,), 'torch': b'rng'},
            module_training_flags=[{'name': '', 'training': True}], resume_permitted=False,
            automatic_rollback=False, outer_completed=False, state_finite=False,
            model_adam_finite=True, gradients_finite=False, committed_optimizer_calls=0,
            scheduler_step_before_actual=0, scheduler_step_after_actual=0, optimizer_opportunity_ledger=ledger())

    def test_complete_ten_file_mapping_no_driver_size_invented(self):
        mapping = raw.verify_partial_raw_files(self.run, self.partial)
        self.assertEqual(len(mapping), 10)
        self.assertEqual(set(mapping), {str(path) for path, _, _ in raw.raw_file_records(self.run, self.partial)})
        self.assertNotIn('bytes', self.partial['driver'])
        self.assertNotIn('torch', sys.modules)

    def test_every_missing_worker_driver_and_actor_rejected(self):
        for path, _, _ in raw.raw_file_records(self.run, self.partial):
            with self.subTest(path=path.name):
                data = path.read_bytes()
                path.unlink()
                with self.assertRaises((ValueError, OSError)):
                    raw.verify_partial_raw_files(self.run, self.partial)
                path.write_bytes(data)
        bad = copy.deepcopy(self.partial)
        bad['worker_outcome']['worker_states'].pop()
        with self.assertRaises(ValueError):
            raw.verify_partial_raw_files(self.run, bad)

    def test_sha_and_byte_count_drift_rejected(self):
        for group in ('driver', 'actor_inputs'):
            with self.subTest(group=group):
                bad = copy.deepcopy(self.partial)
                bad[group]['sha256'] = '0'*64
                with self.assertRaises(ValueError):
                    raw.verify_partial_raw_files(self.run, bad)
        bad = copy.deepcopy(self.partial)
        bad['worker_outcome']['worker_states'][3]['sha256'] = '0'*64
        with self.assertRaises(ValueError):
            raw.verify_partial_raw_files(self.run, bad)
        for group in ('driver', 'actor_inputs'):
            bad = copy.deepcopy(self.partial)
            bad[group]['bytes'] = 1
            with self.assertRaises(ValueError):
                raw.verify_partial_raw_files(self.run, bad)
        bad = copy.deepcopy(self.partial)
        bad['worker_outcome']['worker_states'][0]['bytes'] = True
        with self.assertRaises(ValueError):
            raw.verify_partial_raw_files(self.run, bad)

    def test_file_and_directory_symlinks_rejected(self):
        path = self.directory / 'worker-03.pt'
        target = self.root / 'target.bin'
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaises((ValueError, OSError)):
            raw.verify_partial_raw_files(self.run, self.partial)
        path.unlink()
        target.rename(path)
        relocated = self.root / 'relocated'
        self.directory.rename(relocated)
        self.directory.symlink_to(relocated, target_is_directory=True)
        with self.assertRaises((ValueError, OSError)):
            raw.verify_partial_raw_files(self.run, self.partial)

    def test_wrong_attempt_path_rank_and_identity_rejected(self):
        for mutation in ('path', 'rank', 'identity'):
            bad = copy.deepcopy(self.partial)
            if mutation == 'path':
                bad['driver']['path'] = str(self.root / 'driver.pt')
            elif mutation == 'rank':
                bad['worker_outcome']['worker_states'][0]['rank'] = True
            else:
                bad['worker_outcome']['worker_states'][2]['identity']['attempt_id'] = 2
            with self.assertRaises(ValueError):
                raw.verify_partial_raw_files(self.run, bad)

    def test_actual_same_size_byte_drift_rejected(self):
        path = self.directory / 'worker-07.pt'
        original = path.read_bytes()
        path.write_bytes(b'X' + original[1:])
        with self.assertRaises(ValueError):
            raw.verify_partial_raw_files(self.run, self.partial)

    def test_same_fd_loader_tiny_non_pt_bytes_and_mutation(self):
        path = self.root / 'ordinary.bin'
        data = b'tiny non-PT fixture only'
        path.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        def loader(stream, **kwargs):
            self.assertEqual(kwargs, {'map_location': 'cpu', 'weights_only': False})
            self.assertEqual(stream.tell(), 0)
            return stream.read()
        self.assertEqual(held_state(path, expected, loader), data)
        def mutation(stream, **kwargs):
            value = stream.read()
            path.write_bytes(b'changed')
            return value
        with self.assertRaises(ValueError):
            held_state(path, expected, mutation)

    def test_gradient_nan_preserved_but_model_nan_and_logical_drift_rejected(self):
        state = self.worker()
        record = self.partial['worker_outcome']['worker_states'][0]
        record['logical_state_sha256'] = digest(state)
        result = raw.verify_worker_logical(state, self.partial, record, digest, finite, cumulative_calls=0)
        self.assertFalse(result['gradients_finite'])
        self.assertTrue(math.isnan(state['gradients']['p'][0]))
        for mutation in ('model', 'logical', 'scheduler', 'ledger', 'identity'):
            bad = copy.deepcopy(state)
            receipt = copy.deepcopy(record)
            if mutation == 'model':
                bad['parameters']['p'] = [float('nan')]
            elif mutation == 'logical':
                receipt['logical_state_sha256'] = '0'*64
            elif mutation == 'scheduler':
                bad['scheduler']['last_epoch'] = 1
            elif mutation == 'ledger':
                bad['optimizer_opportunity_ledger'][0]['rows'] = 15
            else:
                bad['identity']['attempt_id'] = 2
            if mutation != 'logical':
                receipt['logical_state_sha256'] = digest(bad)
            with self.assertRaises(ValueError):
                raw.verify_worker_logical(bad, self.partial, receipt, digest, finite, cumulative_calls=0)

    def test_actual_driver_sampler_rng_noresume_and_actor_context(self):
        driver = dict(version=3, snapshot_kind=raw.PROTOCOL, identity=self.identity,
            attempted_outer=1, completed_outer_updates=0, outer_completed=False, automatic_rollback=False,
            automatic_resume_supported=False, recovery_provenance=None,
            consumed_sampler=dict(rows=169028, batch_size=512, identity=self.identity['data_identity'], shuffle=True,
                cursor=512, epoch=0, order=list(range(169028)), generator=b'generator'),
            rng={'python': (1,), 'numpy': (1,), 'torch': b'rng'},
            algorithm=dict(method='tree_grpo', tree_rng=(1,), stage={}),
            guard_continue_accounting=dict(protocol='tree_guard_continue_fresh_rollout_v1', training_attempts=1, completed_outer_updates=0))
        self.partial['driver']['logical_state_sha256'] = digest(driver)
        self.assertEqual(raw.verify_driver_logical(driver, self.partial, digest, finite)['consumed_cursor'], 512)
        for mutate in ('cursor', 'rng', 'resume'):
            bad = copy.deepcopy(driver)
            if mutate == 'cursor':
                bad['consumed_sampler']['cursor'] = 0
            elif mutate == 'rng':
                bad['rng']['torch'] = None
            else:
                bad['automatic_resume_supported'] = True
            self.partial['driver']['logical_state_sha256'] = digest(bad)
            with self.assertRaises(ValueError):
                raw.verify_driver_logical(bad, self.partial, digest, finite)
        actor = dict(batch={key: [float('nan')]*2560 for key in self.partial['actor_inputs']['fields']},
            non_tensor_batch={'uid': list(range(2560))}, meta_info=dict(
                tree_partial_context=dict(protocol=raw.PROTOCOL, identity=self.identity, directory=str(self.directory)),
                baseline_outer_step=1, baseline_attempt_id=1, baseline_completed_outer=0))
        self.assertEqual(raw.verify_actor_inputs_logical(actor, self.run, self.partial)['actual_rows'], 2560)
        for mutation in ('context', 'rows'):
            bad = copy.deepcopy(actor)
            if mutation == 'context':
                bad['meta_info']['tree_partial_context']['identity']['attempt_id'] = 2
            else:
                bad['batch']['responses'].pop()
            with self.assertRaises(ValueError):
                raw.verify_actor_inputs_logical(bad, self.run, self.partial)

    def test_empty_partial_report_explicit_zero_files_and_deserializations(self):
        checkpoint = self.run / 'checkpoints/training_state/global_step_5/manifest.json'
        checkpoint.parent.mkdir(parents=True)
        value = {'identity': {'guard_continue_accounting': {'partial_evidence': []}}}
        observed = dict(training_attempts=5, record_sha256={}, all_skipped=[], credited=[
            {'outer': index, 'attempt': index, 'actual_writes_per_rank': {str(rank): 1 for rank in range(8)}} for index in range(1, 6)])
        package = dict(code_commit='a'*40, common_data_identity='c'*64, model_identity={'sha256': 'd'*64})
        with patch.dict(os_environ(), {'CUDA_VISIBLE_DEVICES': ''}), \
             patch('check_partial_raw_cpu.read_json_bound', return_value=(value, 'e'*64, b'{}')), \
             patch('production_contract.checkpoint_manifest', return_value=(value, 'e'*64)), \
             patch('attempt_accounting.attempt_results', return_value=observed), \
             patch('attempt_accounting.checkpoint_accounting', return_value={'partial_evidence': []}):
            report = verify(self.run, package, 'b'*64, 5, self.root / 'report.json')
        self.assertEqual(report['status'], 'FRESH200_PARTIAL_RAW_CPU_REVERIFIED')
        self.assertEqual(report['raw_files_checked'], 0)
        self.assertEqual(report['pt_deserialization_count'], 0)
        self.assertFalse(report['actual_pt_deserialization'])
        self.assertFalse(report['cuda_initialized'])
        self.assertEqual(report['logical_records'], [])


def os_environ():
    import os
    return os.environ


if __name__ == '__main__':
    unittest.main()
