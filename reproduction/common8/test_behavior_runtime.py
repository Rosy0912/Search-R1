"""CPU protocol fixtures and real eight-rank Gloo; no GPU/Adam/admission proof."""
import copy
import datetime
import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from reproduction.common8 import behavior_runtime as runtime
from reproduction.common8.behavior_trace import project_sampling_logprobs
from reproduction.common8.policy_calibration_plan import canonical_bytes, document_sha256
from reproduction.common8.test_behavior_probability import tiny_outer_fixture


class TinyBatch(SimpleNamespace):
    """Only the actual reader's batch surface; no DataProto/TensorDict dependency."""

    def __len__(self):
        return self.batch['responses'].shape[0]


def cpu_fixture(base):
    """Produce real CPU gate bytes, hashes and rows; patch only native namespace."""
    base = Path(base).resolve()
    (base / 'logs').mkdir()
    (base / 'repo').mkdir()
    (base / 'artifacts').mkdir()
    source = {}
    for name in ('test_behavior_runtime.py', 'behavior_runtime.py', 'behavior_trace.py',
                 'behavior_probability.py'):
        content = Path(__file__).with_name(name).read_bytes()
        target = base / 'repo' / name
        target.write_bytes(content)
        source[name] = hashlib.sha256(content).hexdigest()
    run = dict(profile='common8', run_id='cpu-only-runtime', completed_updates=0,
               source_sha256=source, physical_execution_authenticated=False)
    job = 'cpu-only-gloo'
    (base / 'logs/run-manifest.json').write_bytes(canonical_bytes(run))
    # Literal input to the light reader; this fixture grants no native allocation.
    (base / 'logs/native-allocation-admission.json').write_bytes(canonical_bytes(dict(
        accepted=True, run_id=run['run_id'], slurm_job_id=job,
        fixture_kind='cpu_only_protocol_record', physical_execution_authenticated=False)))
    trace, bindings = tiny_outer_fixture()
    upstream = [dict(cpu_fixture_group=index) for index in range(512)]
    parents = [str(index) + '-' + document_sha256(value) for index, value in enumerate(upstream)]
    for name, value in (('run_id', run['run_id']), ('source_sha256', source),
                        ('slurm_job_id', job), ('parent_group_ids', parents)):
        trace[name] = value
        bindings['expected_' + name] = value
    rows = trace['actor_rows']
    projection = project_sampling_logprobs(trace, actor_rows=rows, **bindings)
    trace_content = canonical_bytes(trace)
    trace_path = base / 'artifacts/cpu-only-private-trace.json'
    trace_path.write_bytes(trace_content)
    private_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    assert private_sha == projection['audit']['trace_sha256']
    context = dict(trace=trace, actor_rows=rows, audit=projection['audit'], bindings=bindings,
                   private=dict(private_trace_sha256=private_sha, serialization_transformed=False))
    identity = {name: bindings['expected_' + name] for name in
                ('run_id', 'source_sha256', 'slurm_job_id', 'logical_update',
                 'parent_group_ids', 'trajectory_ids', 'policy_snapshot_id')}
    identity['validation'] = False
    prompt_width = rows[0]['prompt_width']
    tensors = {name: torch.tensor([row[name] for row in rows], dtype=torch.long) for name in
               ('input_ids', 'attention_mask', 'position_ids', 'info_mask')}
    tensors.update(prompts=tensors['input_ids'][:, :prompt_width].clone(),
                   responses=tensors['input_ids'][:, prompt_width:].clone(),
                   loss_mask=tensors['info_mask'][:, prompt_width:].clone(),
                   sampling_log_probs=torch.tensor(projection['sampling_log_probs'], dtype=torch.float32),
                   sampling_mask=torch.tensor(projection['sampling_mask'], dtype=torch.bool))
    ids = projection['trajectory_ids']
    data = TinyBatch(batch=tensors,
                     non_tensor_batch=dict(behavior_trajectory_id=np.array(ids, dtype=object),
                                           uid=np.array([upstream[index // 5] for index in range(2560)], dtype=object)),
                     meta_info=dict(behavior_identity=identity, behavior_sampling=copy.deepcopy(trace['sampling']),
                                    temperature=1.0))
    hf = tensors['sampling_log_probs'].clone()
    hf[tensors['sampling_mask']] -= .01
    hf[~tensors['sampling_mask']] = float('nan')
    local_config = SimpleNamespace(log_prob_micro_batch_size=4, temperature=1.0,
                                    log_prob_use_dynamic_bsz=False, log_prob_max_token_len_per_gpu=4096)
    hf_output = TinyBatch(batch=dict(old_log_probs=hf), meta_info=runtime.rollout_logprob_metadata(local_config))
    # This explicitly substitutes a CPU fixture namespace, never a native proof.
    with patch.object(runtime, '_native_run', return_value=(base, run, job)):
        proof = runtime.freeze_after_probability_gate(data, context, hf_output)
    torch.save(data.batch, base / 'cpu-only-tensors.pt')
    (base / 'cpu-only-worker-metadata.json').write_bytes(canonical_bytes(dict(
        ids=ids, uid=data.non_tensor_batch['uid'].tolist(), meta_info=data.meta_info)))
    return data, context, hf_output, proof, run, job


def local_batch(base, rank):
    tensors = torch.load(Path(base) / 'cpu-only-tensors.pt', map_location='cpu', weights_only=True)
    metadata = json.loads((Path(base) / 'cpu-only-worker-metadata.json').read_bytes())
    start, stop = rank * 320, (rank + 1) * 320
    return TinyBatch(batch={name: value[start:stop].clone() for name, value in tensors.items()},
                     non_tensor_batch=dict(behavior_trajectory_id=np.array(metadata['ids'][start:stop], dtype=object),
                                           uid=np.array(metadata['uid'][start:stop], dtype=object)),
                     meta_info=copy.deepcopy(metadata['meta_info']))


def gloo_worker(rank, base, store):
    """Real collectives; a rank-local fault must stop every simulated write."""
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + store, rank=rank, world_size=8,
                            timeout=datetime.timedelta(seconds=60))
    base = Path(base).resolve()
    run = json.loads((base / 'logs/run-manifest.json').read_bytes())
    job = 'cpu-only-gloo'
    results = []
    try:
        for fault in ('none', 'missing_temperature', 'local319', 'wrong_lp', 'wrong_uid', 'duplicate_ids'):
            data = local_batch(base, rank)
            if rank == 1:
                if fault == 'missing_temperature':
                    del data.meta_info['temperature']
                elif fault == 'local319':
                    data.batch = {name: value[:-1] for name, value in data.batch.items()}
                    data.non_tensor_batch = {name: value[:-1] for name, value in data.non_tensor_batch.items()}
                elif fault == 'wrong_lp':
                    # Equality of old and sampling is insufficient: row bytes changed.
                    data.batch['old_log_probs'][0, 0] -= .001
                    data.batch['sampling_log_probs'][0, 0] -= .001
                elif fault == 'wrong_uid':
                    data.non_tensor_batch['uid'][0] = dict(cpu_fixture_group='different')
                elif fault == 'duplicate_ids':
                    # Both rows pass local byte/UID checks; only global coverage can reject.
                    for value in data.batch.values():
                        value[5] = value[0].clone()
                    for value in data.non_tensor_batch.values():
                        value[5] = copy.deepcopy(value[0])
            before = {name: value.clone() for name, value in data.batch.items()}
            simulated_writes, error = 0, None
            try:
                with patch.object(runtime, '_native_run', return_value=(base, run, job)):
                    runtime.verify_worker_input(data, logical_update=1, group=dist.group.WORLD)
                simulated_writes += 1  # Reached boundary only; no optimizer exists.
            except (RuntimeError, AssertionError, ValueError) as failure:
                error = str(failure)
            results.append(dict(rank=rank, fault=fault, error=error,
                                simulated_writes=simulated_writes, actual_optimizer_writes=0,
                                local_rows=len(data), rows_unchanged=all(torch.equal(value, data.batch[name])
                                                                       for name, value in before.items()),
                                production_admission=False, physical_execution_authenticated=False))
            dist.barrier()
        (base / ('cpu-only-rank-' + str(rank) + '.json')).write_bytes(canonical_bytes(results))
    finally:
        dist.destroy_process_group()


class BehaviorRuntimeTests(unittest.TestCase):
    def test_private_failed_validation_journal_step0_and_later_eval_are_not_outers(self):
        with tempfile.TemporaryDirectory(prefix='g1-cpu-validation-failure-') as directory:
            base = Path(directory).resolve()
            (base / 'artifacts').mkdir()
            run = dict(run_id='cpu-validation', source_sha256={'cpu-fixture': 'a' * 64})
            records = [dict(raw_generated_token_ids=[20, 21], selected_logprobs=[-.25, float('nan')])]
            for step in (0, 5):
                identity = dict(run_id=run['run_id'], source_sha256=run['source_sha256'], slurm_job_id='123',
                    logical_update=step, validation=True,
                    policy_snapshot_id=run['run_id'] + ':' + str(step) + ':validation-' + ('a' if step == 0 else 'b') * 32)
                # Only the actual native namespace/source boundary is a CPU adapter.
                with patch.object(runtime, '_native_run', return_value=(base, run, '123')):
                    receipt = runtime.failed_generation_evidence(identity, records, 'CPU fixture failure')
                    with self.assertRaises(FileExistsError):
                        runtime.failed_generation_evidence(identity, records, 'CPU fixture failure')
                path = Path(receipt['private_validation_trace_path'])
                self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                self.assertEqual(path.stat().st_uid, os.getuid())
                raw = path.read_bytes(); value = json.loads(raw)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), receipt['private_validation_trace_sha256'])
                self.assertEqual(value['actual_sampling_records'][0]['raw_generated_token_ids'], [20, 21])
                self.assertEqual(value['actual_sampling_records'][0]['selected_logprobs'][1],
                                 {'__behavior_trace_nonfinite_float__': 'nan'})
                self.assertEqual(value['policy_snapshot_id'], identity['policy_snapshot_id'])
                self.assertEqual(value['logical_update'], step)
                self.assertEqual(value['role'], 'common8_failed_validation_generation')
                self.assertEqual(value['source_snapshot_kind'], 'current_disk_bytes_not_loaded_module_proof')
                for key in ('accepted', 'context_verified', 'production_admission', 'healthy_checkpoint_accepted',
                            'physical_execution_authenticated', 'loaded_source_authenticated', 'actual_training_outer'):
                    self.assertIs(value[key], False)
                self.assertIs(receipt['serialization_transformed'], True)
                self.assertIs(receipt['raw_trace_must_not_be_published'], True)
            self.assertFalse((base / 'artifacts/behavior-traces').exists())
            self.assertTrue(torch.isnan(torch.tensor(records[0]['selected_logprobs'][1])))

    def test_private_validation_identity_snapshot_and_owned_path_reject(self):
        with tempfile.TemporaryDirectory(prefix='g1-cpu-validation-reject-') as directory:
            base = Path(directory).resolve(); (base / 'artifacts').mkdir()
            run = dict(run_id='cpu-validation', source_sha256={'cpu-fixture': 'a' * 64})
            identity = dict(run_id=run['run_id'], source_sha256=run['source_sha256'], slurm_job_id='123',
                logical_update=0, validation=True, policy_snapshot_id=run['run_id'] + ':0:validation-' + 'a' * 32)
            for change in ({'slurm_job_id': '124'}, {'source_sha256': {}}, {'run_id': 'other'},
                           {'logical_update': True}, {'policy_snapshot_id': '../outside'}, {'validation': 1}):
                with patch.object(runtime, '_native_run', return_value=(base, run, '123')):
                    with self.assertRaises(AssertionError):
                        runtime.failed_generation_evidence(dict(identity, **change), [], 'CPU fixture')
            outside = base / 'outside'; outside.mkdir(mode=0o700)
            parent = base / 'artifacts/behavior-validation-failures'; parent.symlink_to(outside)
            with patch.object(runtime, '_native_run', return_value=(base, run, '123')):
                with self.assertRaises(AssertionError):runtime.failed_generation_evidence(identity, [], 'CPU fixture')
            self.assertEqual(list(outside.iterdir()), [])
            parent.unlink(); parent.mkdir(mode=0o755)
            with patch.object(runtime, '_native_run', return_value=(base, run, '123')):
                with self.assertRaises(AssertionError):runtime.failed_generation_evidence(identity, [], 'CPU fixture')
            self.assertEqual(list(parent.iterdir()), [])

    def test_training_failure_keeps_existing_outer_writer(self):
        base = Path('/CPU-only')
        run = dict(run_id='cpu-train', source_sha256={})
        identity = dict(run_id=run['run_id'], source_sha256={}, slurm_job_id='123',
                        logical_update=1, policy_snapshot_id='cpu-policy', validation=False)
        with patch.object(runtime, '_native_run', return_value=(base, run, '123')), \
             patch.object(runtime, 'write_private_evidence') as write:
            runtime.failed_generation_evidence(identity, [{'CPU_fixture': True}], 'CPU fixture')
            write.assert_called_once()
            self.assertEqual(write.call_args.args[1]['role'], 'common8_failed_partial_generation')
            self.assertEqual(write.call_args.args[1]['logical_update'], 1)

    def test_public_receipt_bytes_and_update_record_bind_without_raw_reread(self):
        with tempfile.TemporaryDirectory(prefix='g1-cpu-receipt-') as directory:
            data, context, _, proof, run, _ = cpu_fixture(directory)
            base = Path(directory).resolve()
            record = dict(behavior_probability_gate_sha256=data.meta_info['behavior_probability_gate']['sha256'],
                          behavior_trace_sha256=context['audit']['trace_sha256'])
            # The public reader intentionally verifies receipt metadata, not raw tokens.
            (base / 'artifacts/cpu-only-private-trace.json').unlink()
            result = runtime.read_gate_receipt(base, run, 1, update_record=record)
            self.assertEqual(result, dict(proof_sha256=record['behavior_probability_gate_sha256'],
                                          private_trace_sha256=record['behavior_trace_sha256'],
                                          included_actor_token_count=5120,
                                          max_abs_discrepancy=proof['max_abs_discrepancy'],
                                          hf_recomputation_performed=True,hf_parity_accepted=True,
                                          behavior_probability_policy=None))
            self.assertIs(proof['physical_execution_authenticated'], False)
            path = Path(data.meta_info['behavior_probability_gate']['path'])
            original = path.read_bytes()
            path.write_bytes(original + b'\n')  # Same JSON values, different original bytes.
            with self.assertRaises(AssertionError):
                runtime.read_gate_receipt(base, run, 1, update_record=record)
            path.write_bytes(original)
            for name in ('behavior_probability_gate_sha256', 'behavior_trace_sha256'):
                with self.subTest(record=name), self.assertRaises(AssertionError):
                    runtime.read_gate_receipt(base, run, 1, update_record=record | {name: '0' * 64})

    def test_public_receipt_source_job_step_and_scalar_drift_rejected(self):
        with tempfile.TemporaryDirectory(prefix='g1-cpu-receipt-') as directory:
            data, _, _, proof, run, _ = cpu_fixture(directory)
            base = Path(directory).resolve()
            path = Path(data.meta_info['behavior_probability_gate']['path'])
            original = path.read_bytes()
            mutations = [
                ('accepted', False), ('run_id', 'different-cpu-run'),
                ('source_sha256', run['source_sha256'] | {'behavior_runtime.py': '0' * 64}),
                ('slurm_job_id', 'different-cpu-job'), ('logical_update', 2),
                ('trajectory_count', 2559), ('actual_frozen_old_all_finite', False),
                ('max_abs_discrepancy', .05001), ('tolerance', .051),
                ('denominator', 'HF_recomputed_old'), ('private_trace_sha256', '0' * 64),
                ('row_tensor_sha256', dict(list(proof['row_tensor_sha256'].items())[1:])),
                ('upstream_group_sha256', dict(list(proof['upstream_group_sha256'].items())[1:])),
                ('sampling', proof['sampling'] | {'temperature': .95}),
                ('sampling', proof['sampling'] | {'top_p': .95}),
            ]
            for name, value in mutations:
                bad = copy.deepcopy(proof); bad[name] = value
                raw = canonical_bytes(bad)
                path.write_bytes(raw)
                # Hash matches the real changed bytes: semantic identity must still reject.
                record = dict(behavior_probability_gate_sha256=hashlib.sha256(raw).hexdigest(),
                              behavior_trace_sha256=proof['trace_sha256'])
                with self.subTest(field=name), self.assertRaises(AssertionError):
                    runtime.read_gate_receipt(base, run, 1, update_record=record)
            path.write_bytes(original)
            allocation_path = base / 'logs/native-allocation-admission.json'
            allocation_original = allocation_path.read_bytes()
            allocation = json.loads(allocation_original)
            for name, value in (('accepted', False), ('run_id', 'different'), ('slurm_job_id', 'different')):
                allocation_path.write_bytes(canonical_bytes(allocation | {name: value}))
                with self.subTest(allocation=name), self.assertRaises(AssertionError):
                    runtime.read_gate_receipt(base, run, 1)
            allocation_path.write_bytes(allocation_original)

    def test_write_and_both_monitor_links_reject_drift_on_every_rank(self):
        with tempfile.TemporaryDirectory(prefix='g1-cpu-links-') as directory:
            _, _, _, _, run, _ = cpu_fixture(directory)
            base = Path(directory).resolve()
            gate = runtime.read_gate_receipt(base, run, 1)
            gates = {'1': gate}
            rows = [dict(kind=kind, context=dict(logical_update=1,
                        behavior_probability_gate_sha256=gate['proof_sha256']),
                         fixture_kind='cpu_only_ledger', actual_optimizer_writes=0,
                         physical_execution_authenticated=False) for kind in
                    ('POLICY_MONITOR', 'OPTIMIZER_WRITE', 'POLICY_MONITOR')]
            # Out-of-range rows and other event kinds are outside this narrow link scan.
            rows.extend([dict(kind='POLICY_MONITOR', context=dict(logical_update=2)),
                         dict(kind='CPU_FIXTURE_NOTE')])
            for rank in range(8):
                path = base / 'logs' / ('optimizer-rank-' + str(rank) + '.jsonl')
                path.write_bytes(b'\n'.join(canonical_bytes(row) for row in rows) + b'\n')
            self.assertIsNone(runtime.verify_write_receipt_links(base, gates, 1, 1))
            for rank in range(8):
                path = base / 'logs' / ('optimizer-rank-' + str(rank) + '.jsonl')
                original = path.read_bytes()
                for index in range(3):
                    bad = copy.deepcopy(rows)
                    bad[index]['context']['behavior_probability_gate_sha256'] = '0' * 64
                    path.write_bytes(b'\n'.join(canonical_bytes(row) for row in bad) + b'\n')
                    with self.subTest(rank=rank, event=index), self.assertRaises(AssertionError):
                        runtime.verify_write_receipt_links(base, gates, 1, 1)
                    path.write_bytes(original)
            with self.assertRaises(AssertionError):
                runtime.verify_write_receipt_links(base, {'1': gate | {'proof_sha256': '0' * 64}}, 1, 1)
            self.assertIs(run['physical_execution_authenticated'], False)

    def test_logprob_metadata_reads_normalized_local_config_without_mutation(self):
        values = dict(log_prob_micro_batch_size=4, temperature=1.0,
                      log_prob_use_dynamic_bsz=False, log_prob_max_token_len_per_gpu=4096)
        config = SimpleNamespace(**values)
        self.assertEqual(runtime.rollout_logprob_metadata(config), dict(
            micro_batch_size=4, temperature=1.0, use_dynamic_bsz=False, max_token_len=4096))
        self.assertEqual(vars(config), values)
        for name, value in (('log_prob_micro_batch_size', 32), ('log_prob_micro_batch_size', True),
                            ('temperature', .95), ('temperature', True),
                            ('log_prob_use_dynamic_bsz', True), ('log_prob_max_token_len_per_gpu', 4096.)):
            bad = values | {name: value}
            with self.subTest(name=name, value=value), self.assertRaises(AssertionError):
                runtime.rollout_logprob_metadata(SimpleNamespace(**bad))

    def test_cast_only_token_fields_preserves_probability_dtype_and_bytes(self):
        fields = {name: torch.tensor([[1., 0.]]) for name in runtime.TOKEN_FIELDS}
        fields.update(sampling_log_probs=torch.tensor([[-.23, 0.]], dtype=torch.float32),
                      old_log_probs=torch.tensor([[-.24, 0.]], dtype=torch.float64),
                      sampling_mask=torch.tensor([[True, False]]), numerical_metric=torch.tensor([[.17]]))
        data = TinyBatch(batch=fields)
        numerical = {name: (value, value.clone()) for name, value in fields.items()
                     if name not in runtime.TOKEN_FIELDS}
        runtime.cast_token_fields(data)
        for name in runtime.TOKEN_FIELDS:
            self.assertEqual(data.batch[name].dtype, torch.int64)
        for name, (original, values) in numerical.items():
            self.assertIs(data.batch[name], original)
            self.assertEqual(data.batch[name].dtype, values.dtype)
            self.assertTrue(torch.equal(data.batch[name], values))

    def test_real_cpu_gate_freezes_sampling_not_hf_and_binds_every_row_hash(self):
        with tempfile.TemporaryDirectory(prefix='g1-cpu-runtime-') as directory:
            data, context, hf_output, proof, run, job = cpu_fixture(directory)
            self.assertEqual(proof['trajectory_count'], 2560)
            self.assertEqual(proof['included_actor_token_count'], 5120)
            self.assertTrue(torch.equal(data.batch['old_log_probs'], data.batch['sampling_log_probs']))
            self.assertNotEqual(data.batch['old_log_probs'].data_ptr(), data.batch['sampling_log_probs'].data_ptr())
            self.assertFalse(torch.equal(data.batch['old_log_probs'], hf_output.batch['old_log_probs']))
            self.assertTrue(torch.isfinite(data.batch['old_log_probs']).all())
            for index, identity in enumerate(data.non_tensor_batch['behavior_trajectory_id']):
                self.assertEqual(proof['row_tensor_sha256'][identity], runtime.row_hash(data, index))
                self.assertEqual(proof['upstream_group_sha256'][identity],
                                 document_sha256(data.non_tensor_batch['uid'][index]))
            path = Path(data.meta_info['behavior_probability_gate']['path'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             data.meta_info['behavior_probability_gate']['sha256'])
            self.assertEqual(json.loads(path.read_bytes()), proof)
            self.assertEqual(proof['private_trace_sha256'], context['audit']['trace_sha256'])
            for name in ('production_admission', 'healthy_checkpoint_accepted', 'physical_execution_authenticated'):
                self.assertIs(proof[name], False)
            for name, expected in run['source_sha256'].items():
                self.assertEqual(hashlib.sha256((Path(directory) / 'repo' / name).read_bytes()).hexdigest(), expected)
            original = runtime.row_hash(data, 0)
            for name in ('input_ids', 'responses', 'attention_mask', 'position_ids', 'info_mask',
                         'old_log_probs', 'sampling_log_probs', 'sampling_mask'):
                value = data.batch[name]
                before = value[0, 0].clone()
                value[0, 0] = ~before if value.dtype == torch.bool else before + 1
                self.assertNotEqual(runtime.row_hash(data, 0), original, name)
                value[0, 0] = before
                self.assertEqual(runtime.row_hash(data, 0), original)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'CPU Gloo unavailable')
    def test_real_eight_rank_320_coverage_and_one_rank_faults_collectively_reject(self):
        with tempfile.TemporaryDirectory(prefix='g1-cpu-gloo-') as directory:
            cpu_fixture(directory)
            mp.spawn(gloo_worker, args=(directory, str(Path(directory) / 'gloo-store')), nprocs=8, join=True)
            results = [json.loads((Path(directory) / ('cpu-only-rank-' + str(rank) + '.json')).read_bytes())
                       for rank in range(8)]
            for rank, records in enumerate(results):
                self.assertEqual(records[0]['rank'], rank)
                self.assertIsNone(records[0]['error'])
                self.assertEqual(records[0]['local_rows'], 320)
                self.assertEqual(records[0]['simulated_writes'], 1)
                for record in records:
                    self.assertTrue(record['rows_unchanged'])
                    self.assertEqual(record['actual_optimizer_writes'], 0)
                    self.assertIs(record['physical_execution_authenticated'], False)
                for record in records[1:]:
                    self.assertIsNotNone(record['error'], (rank, record['fault']))
                    self.assertEqual(record['simulated_writes'], 0, (rank, record['fault']))
            for case in range(1, 6):
                # Matched collectives return the same rejection to every rank.
                self.assertEqual(len({records[case]['error'] for records in results}), 1)
            self.assertIn('ranks:', results[0][1]['error'])
            self.assertIn('trajectory coverage changed', results[0][-1]['error'])


if __name__ == '__main__':
    unittest.main()
