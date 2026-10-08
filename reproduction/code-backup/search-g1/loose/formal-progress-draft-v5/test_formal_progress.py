"""Offline stdlib regressions; no network, training imports, or CUDA.

These tests are intentionally unexecuted draft source until a fresh independent
agent has reviewed the exact collector, core, and test bytes.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import formal_progress_core as core
import collect_formal_progress as collector


def budget():
    return dict(target_actual_outer_updates=200, lr_warmup_steps=57,
        world_size=8, optimizer_steps_per_outer=20, learning_rate=1e-6,
        lr_warmup_steps_ratio=0.0, label='训练')


def historical_process(pid, start_ticks):
    return dict(pid=pid, start_ticks=start_ticks, sid=80, uid=os.getuid(),
                state='S', command='synthetic offline process')


def training_claim():
    return dict(run_id=core.RUN, shell_pid=80, shell_start_ticks=90,
        slurm_job_id=core.JOB, claimed_unix=103,
        semantics='exclusive launch claim, not evidence of a completed update')


def trainer_ledger():
    leader = historical_process(80, 90)
    records = {str(leader['pid']): leader}
    records.update({str(100 + rank): historical_process(100 + rank, 200 + rank)
                    for rank in range(8)})
    return dict(run_id=core.RUN, jobs=dict(trainer=dict(leader=leader, ledger=records,
        log=core.BASE + '/logs/trainer.log', state='STARTED',
        arguments=['bash', 'reproduction/common8/train.sh'])))


def admitted_births():
    return core.validate_training_binding(training_claim(), trainer_ledger(),
        run_started_unix=100, now=200)[1]


def inspect_scheduler(rows, rank, attempts, **overrides):
    parameters = dict(contract_sha256=core.scheduler_contract_sha256(budget()),
        admitted_births=admitted_births(), run_started_unix=100, training_claim_unix=103)
    parameters.update(overrides)
    return core.inspect_scheduler_rows(rows, rank, attempts, **parameters)


def check_gpu_samples(rows, inventory, *, now=130, run_started_unix=100,
                      training_claim_unix=103):
    return core.validate_gpu_samples(rows, inventory, now=now,
        run_started_unix=run_started_unix, training_claim_unix=training_claim_unix)


def attempt(ordinal=1, completed_before=0, writes_before=0, writes=1):
    complete = writes > 0
    return dict(run_id=core.RUN, slurm_job_id=core.JOB,
        status='complete' if complete else 'recoverable_guard',
        logical_update=completed_before + 1, attempt_id=('%032x' % ordinal),
        attempt_index=ordinal, optimizer_writes_before=writes_before,
        optimizer_writes_after=writes_before + writes,
        optimizer_writes_this_attempt=writes, opportunities_processed=20,
        opportunity_statuses=['written'] * writes + ['skipped_guard'] * (20 - writes),
        guard_stage=None if complete else 'all_update_opportunities_skipped',
        guard_reasons=[] if complete else ['no_successful_adam_write'],
        complete_outer=complete, retained_partial_adam_writes=not complete,
        no_automatic_rollback=True, scheduler_advances=int(complete),
        completed_outer=completed_before + int(complete),
        all_ranks_agreed=True, real_rank_count=8,
        cumulative_aborted_attempts=ordinal - completed_before - int(complete))


def update(row):
    metrics = {name: 0.12345678901234567 for name in core.REQUIRED_METRICS}
    metrics.update({'actor/optimizer_steps_this_update': row['optimizer_writes_this_attempt'],
        'actor/update_opportunities_this_attempt': 20,
        'actor/skipped_update_opportunities': 20 - row['optimizer_writes_this_attempt']})
    return dict(run_id=core.RUN, step=row['completed_outer'], metrics=metrics,
        probe_version='probe-version-1', behavior_probability_gate_sha256='a' * 64,
        behavior_trace_sha256='b' * 64, attempt_id=row['attempt_id'],
        attempt_index=row['attempt_index'],
        optimizer_writes_before=row['optimizer_writes_before'],
        optimizer_writes_after=row['optimizer_writes_after'], opportunities_processed=20,
        actual_optimizer_writes=row['optimizer_writes_this_attempt'])


def inventory():
    return {index: dict(uuid='GPU-%08x-0000-0000-0000-000000000000' % index,
                        total_mib=81559) for index in range(8)}


def gpu_sample(ended=102, started=101, memory=200):
    physical = inventory()
    return dict(schema_version=1, role='actual_common8_gpu_occupancy',
        run_id=core.RUN, slurm_job_id=core.JOB, source_map_sha256=core.MAP_SHA,
        sampler_identity=dict(pid=50, start_ticks=70, sid=50, uid=0),
        sampling_interval_seconds=2, query_started_unix=started, unix=ended,
        gpus=[dict(index=index, uuid=physical[index]['uuid'], used_mib=memory,
                   total_mib=physical[index]['total_mib'], util_percent=25)
              for index in range(8)])


def scheduler(row, rank=0, kind='COMPLETE'):
    completed_before = row['completed_outer'] - int(row['complete_outer'])
    advanced = kind == 'COMPLETE'
    end = kind != 'BEGIN'
    after = completed_before + int(advanced)
    result = dict(schema_version=1, role='fresh200_outer_scheduler',
        contract_kind='fresh200_warmup57', rank=rank, world_size=8,
        process_pid=100 + rank, process_start_ticks=200 + rank,
        base_learning_rate=1e-6, warmup_steps=57, observed_unix=150,
        run_id=core.RUN, source_commit=core.COMMIT, slurm_job_id=core.JOB,
        source_map_sha256=core.MAP_SHA, policy_safety_mode='inline_first_five',
        scheduler_contract_sha256=core.scheduler_contract_sha256(budget()), kind=kind,
        logical_update=row['logical_update'], writes_before=row['optimizer_writes_before'],
        writes_after=row['optimizer_writes_after'] if end else row['optimizer_writes_before'],
        actual_optimizer_writes=row['optimizer_writes_this_attempt'] if end else 0,
        completed_outer_before=completed_before, completed_outer_after=after,
        consumed_learning_rates=[1e-6 * min(completed_before / 57, 1)],
        next_learning_rates=[1e-6 * min(after / 57, 1)],
        scheduler_last_epoch_before=completed_before, scheduler_last_epoch_after=after,
        scheduler_step_count_before=completed_before + 1, scheduler_step_count_after=after + 1,
        batch_gate_sha256='a' * 64, scheduler_advances=int(advanced),
        attempt_id=row['attempt_id'], attempt_index=row['attempt_index'],
        opportunities_processed=20 if end else 0, optimizer_opportunities=20)
    if kind == 'ABORTED_ATTEMPT':
        result.update(guard_stage='all_update_opportunities_skipped',
            guard_reasons=['no_successful_adam_write'], recoverable_guard=True,
            retained_partial_adam_writes=True, no_automatic_rollback=True,
            scheduler_unchanged_on_rejection=True)
    return result


def scheduler_group(row, rank=0):
    kinds = ('BEGIN', 'COMMIT_PREPARED', 'COMPLETE') if row['complete_outer'] else ('BEGIN', 'ABORTED_ATTEMPT')
    return [scheduler(row, rank, kind) for kind in kinds]


def synthetic_collection(*, missing_documents=(), current_gpu_state='observed',
                         birth_state='observed', scheduler_mode='complete',
                         historical_count=9, namespace_count=1,
                         missing_document_fields=None, extra_document_fields=None,
                         expired_budget_at=None, malformed_evaluation_raw=None):
    """Exercise the actual collector control flow with mocked scalar producers.

    All paths are inside an offline temporary directory. No network, subprocess,
    /proc, runtime source package, GPU, or actual job is accessed by this fixture.
    Field omissions and elapsed clocks are offline injection parameters only.
    """
    if expired_budget_at not in (None, 'after_driver_validation', 'during_trainer_birth',
                                 'before_current_gpu', 'during_current_gpu', 'after_namespace_birth'):
        raise AssertionError('Unknown offline budget injection stage')
    saved_validate_attempts = core.validate_attempts
    saved_actual_birth = core.actual_birth
    saved_current_gpu = core.current_gpu
    clock = dict(elapsed=0)
    def expire_budget():
        clock['elapsed'] = core.REMOTE_SECONDS + 1
    def validate_driver(values):
        result = saved_validate_attempts(values)
        if expired_budget_at == 'after_driver_validation':
            expire_budget()
        return result
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        mapping = {'synthetic/file-%03d.py' % index: 'a' * 64 for index in range(372)}
        map_sha = core.sha(core.canonical(mapping))
        with mock.patch.object(core, 'BASE', str(root)), mock.patch.object(core, 'MAP_SHA', map_sha):
            row = attempt()
            run = dict(run_id=core.RUN, source_commit=core.COMMIT, source_sha256=mapping,
                policy_safety_mode='inline_first_five', contract_kind='fresh200_warmup57',
                target_updates=200, started_unix=100, budget=budget())
            physical = inventory()
            allocation = dict(run_id=core.RUN, slurm_job_id=core.JOB, node='h100-8',
                accepted=True, actual_allocated_indices=list(physical),
                physical_gpu_inventory=[[str(index), value['uuid'], str(value['total_mib'])]
                                        for index, value in physical.items()])
            before = dict(schema_version=1, role='formal_inline_development_evaluation',
                run_id=core.RUN, step=0, policy_safety_mode='inline_first_five',
                source_map_sha256=map_sha, plan_sha256='b' * 64, validation_sha256='c' * 64,
                checked_unix=150, weighted_em=0.25, weighted_f1=0.3)
            documents = {'run-manifest.json': run, 'training-launch-claim.json': training_claim(),
                'process-ledger.json': trainer_ledger(),
                'pipeline-status.json': dict(run_id=core.RUN, stage='GRPO_1_TO_200', unix=150),
                'formal-inline-development-before.json': before,
                'native-allocation-admission.json': allocation}
            for name, fields in (missing_document_fields or {}).items():
                for field in fields:
                    documents[name].pop(field)
            for name, fields in (extra_document_fields or {}).items():
                documents[name].update(copy.deepcopy(fields))
            historical = documents['process-ledger.json']['jobs']['trainer']['ledger']
            for index in range(historical_count - len(historical)):
                pid = 1000 + index
                historical[str(pid)] = historical_process(pid, 2000 + index)
            provenance = dict(captured_sha256='d' * 64, appended_after_capture=False,
                              observation_incomplete=False)
            def capture_document(path, deadline, *, maximum=core.MAX_DOCUMENT):
                if path.name in missing_documents or path.name not in documents:
                    raise FileNotFoundError(path.name)
                expected_limit = core.MAX_PROCESS_LEDGER if path.name == 'process-ledger.json' else core.MAX_DOCUMENT
                if maximum != expected_limit:
                    raise AssertionError('Producer-specific metadata byte bound differs')
                if path.name == 'formal-inline-development-before.json' and malformed_evaluation_raw is not None:
                    return core.strict_json(malformed_evaluation_raw), dict(provenance)
                if path.name == 'native-allocation-admission.json' and expired_budget_at == 'before_current_gpu':
                    expire_budget()
                return copy.deepcopy(documents[path.name]), dict(provenance)
            def capture_rows(path, maximum, max_rows, deadline):
                if path.name == 'actual-attempts.jsonl':
                    values = [row]
                elif path.name == 'actual-updates.jsonl':
                    values = [update(row)]
                elif path.name.startswith('scheduler-rank-'):
                    rank = int(path.stem.rsplit('-', 1)[1])
                    values = scheduler_group(row, rank)
                    if scheduler_mode == 'partial':
                        values = values[:2]
                    elif scheduler_mode == 'contradictory_partial':
                        values = scheduler_group(attempt(1, 1, 9), rank)[:1]
                    elif scheduler_mode in ('different_metric_gate', 'different_partial_metric_gate'):
                        for value in values:
                            value['batch_gate_sha256'] = 'f' * 64
                        if scheduler_mode == 'different_partial_metric_gate':
                            values = values[:1]
                elif path.name == 'gpu-samples-' + core.RUN + '.jsonl':
                    values = [gpu_sample(102, 101, 9000), gpu_sample(107, 106, 300)]
                else:
                    raise FileNotFoundError(path.name)
                return copy.deepcopy(values), dict(provenance)
            def current_birth(identity, deadline):
                if expired_budget_at == 'during_trainer_birth' and identity['pid'] == 80:
                    expire_budget()
                if clock['elapsed'] > deadline:
                    # The real handler must decline /proc reads after expiry.
                    return saved_actual_birth(identity, deadline)
                if birth_state != 'observed':
                    return dict(saved_identity=identity, actual_identity=None,
                        birth_matches=None, namespace_matches=None, evidence_state=birth_state)
                observed = dict(saved_identity=identity, actual_identity=dict(identity), process_state='S',
                    birth_matches=True, namespace_matches=True, evidence_state='observed')
                if expired_budget_at == 'after_namespace_birth' and identity['pid'] >= 3000:
                    expire_budget()
                return observed
            protocol = mock.Mock()
            protocol.runtime_map.return_value = mapping
            protocol.safe_regular_bytes.return_value = b'frozen-method'
            payload = dict(base=core.BASE, run_id=core.RUN, job_id=core.JOB,
                source_commit=core.COMMIT, source_map_sha256=map_sha, source_sha256=mapping,
                method_sha256=core.sha(b'frozen-method'),
                observed_namespace_births=[dict(pid=3000 + index, start_ticks=4000 + index,
                    sid=80, uid=os.getuid()) for index in range(namespace_count)])
            current = dict(evidence_state=current_gpu_state)
            def current_gpu_query(physical_inventory, deadline, capture):
                if expired_budget_at in ('before_current_gpu', 'during_current_gpu'):
                    return saved_current_gpu(physical_inventory, deadline, capture)
                return current
            def captured_gpu(command, timeout, stdout_limit, stderr_limit):
                if (timeout <= 0 or timeout > 10 or stdout_limit != 16384 or stderr_limit != 4096):
                    raise AssertionError('Current GPU bounded capture contract differs')
                expire_budget()
                stdout = '\n'.join('%d,%s,0,%d,0' % (index, item['uuid'], item['total_mib'])
                                   for index, item in physical.items()) + '\n'
                return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr='')
            if expired_budget_at == 'during_current_gpu':
                protocol.bounded_capture.side_effect = captured_gpu
            with mock.patch.object(core, 'capture_json', side_effect=capture_document), \
                 mock.patch.object(core, 'capture_jsonl', side_effect=capture_rows), \
                 mock.patch.object(core.time, 'monotonic', side_effect=lambda: clock['elapsed']), \
                 mock.patch.object(core, 'validate_attempts', side_effect=validate_driver), \
                 mock.patch.object(core, 'actual_birth', side_effect=current_birth), \
                 mock.patch.object(core, 'current_gpu', side_effect=current_gpu_query) as current_query:
                result = core.collect_remote(payload, protocol)
            return result, current_query.call_count


class PrefixTests(unittest.TestCase):
    def test_duplicate_nested_fields_and_nonfinite_exponents_are_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":{"x":1,"x":2}}',
                    b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e400}'):
            with self.subTest(raw=raw), self.assertRaises(core.ObservationError):
                core.strict_json(raw)

    def test_torn_last_row_never_promoted_even_if_valid_json(self):
        first = core.canonical(attempt()) + b'\n'
        second = core.canonical(attempt(2, 1, 1, 3))
        for tail in (second, second[:-7], b'{bad'):
            rows, provenance = core.jsonl_prefix(first + tail, max_bytes=10000, max_lines=10)
            self.assertEqual(len(rows), 1)
            self.assertTrue(provenance['observation_incomplete'])
            self.assertEqual(provenance['partial_tail_bytes'], len(tail))
            self.assertEqual(provenance['complete_prefix_sha256'], core.sha(first))
            self.assertEqual(core.validate_attempts(rows)[0]['observed_driver_completed_outer'], 1)

    def test_empty_existing_file_and_tail_only_prefix_are_distinct(self):
        empty, empty_meta = core.jsonl_prefix(b'', max_bytes=100, max_lines=2)
        torn, torn_meta = core.jsonl_prefix(b'{"run_id":', max_bytes=100, max_lines=2)
        self.assertEqual(empty, torn)
        self.assertFalse(empty_meta['observation_incomplete'])
        self.assertTrue(torn_meta['observation_incomplete'])
        self.assertEqual(empty_meta['captured_bytes'], 0)

    def test_complete_malformed_line_is_not_silently_skipped(self):
        for raw in (b'{}\n{bad}\n', b'\n', b'[]\n'):
            with self.assertRaises(core.ObservationError):
                core.jsonl_prefix(raw, max_bytes=100, max_lines=5)

    def test_byte_line_and_tail_bounds(self):
        with self.assertRaises(core.ObservationError):
            core.jsonl_prefix(b'{}\n', max_bytes=2, max_lines=1)
        with self.assertRaises(core.ObservationError):
            core.jsonl_prefix(b'{}\n{}\n', max_bytes=20, max_lines=1)
        with mock.patch.object(core, 'MAX_LINE', 2), self.assertRaises(core.ObservationError):
            core.jsonl_prefix(b'123', max_bytes=20, max_lines=1)


class AttemptTests(unittest.TestCase):
    def test_zero_skip_then_one_real_write_advances_once(self):
        rows = [attempt(1, writes=0), attempt(2, writes=1)]
        summary, complete = core.validate_attempts(rows)
        self.assertEqual(summary['observed_driver_completed_outer'], 1)
        self.assertEqual(summary['observed_scheduler_advances'], 1)
        self.assertEqual(summary['observed_fully_skipped_attempts'], 1)
        self.assertEqual(summary['agreed_writes_at_last_return_per_rank'], 1)
        self.assertEqual(summary['observed_processed_opportunities_per_rank'], 40)
        self.assertEqual(summary['observed_skipped_opportunities_per_rank'], 39)
        self.assertTrue(summary['actual_total_writes_unavailable'])
        self.assertEqual(complete, [rows[1]])

    def test_twenty_writes_are_one_outer_not_twenty(self):
        summary, _ = core.validate_attempts([attempt(writes=20)])
        self.assertEqual(summary['observed_driver_completed_outer'], 1)
        self.assertEqual(summary['agreed_writes_at_last_return_per_rank'], 20)

    def test_wrong_identity_schema_types_partial_and_rank_disagreement(self):
        cases = {
            'run_id': 'f' * 32, 'slurm_job_id': '3298', 'real_rank_count': True,
            'all_ranks_agreed': False, 'complete_outer': 1, 'scheduler_advances': True,
            'optimizer_writes_this_attempt': True, 'opportunities_processed': 19,
            'optimizer_writes_after': 2, 'guard_reasons': ['unexpected'],
            'attempt_index': 2, 'retained_partial_adam_writes': 0,
        }
        for name, value in cases.items():
            row = attempt()
            row[name] = value
            with self.subTest(name=name), self.assertRaises(core.ObservationError):
                core.validate_attempts([row])
        row = attempt()
        row['extra'] = 1
        with self.assertRaises(core.ObservationError):
            core.validate_attempts([row])

    def test_duplicate_uuid_and_gap_and_zero_completion_are_rejected(self):
        row = attempt(2, 1, 1)
        row['attempt_id'] = attempt()['attempt_id']
        with self.assertRaises(core.ObservationError):
            core.validate_attempts([attempt(), row])
        zero = attempt(writes=0)
        zero['complete_outer'] = True
        with self.assertRaises(core.ObservationError):
            core.validate_attempts([zero])


class MetricTests(unittest.TestCase):
    def test_lagging_metrics_and_precision(self):
        rows = [attempt(), attempt(2, 1, 1, 3)]
        _, complete = core.validate_attempts(rows)
        metrics = core.validate_updates([update(rows[0])], complete)
        self.assertEqual(metrics['metrics_lag_completed_outer'], 1)
        self.assertEqual(metrics['last_metrics_completed_step'], 1)
        self.assertEqual(metrics['latest_metrics']['actor/pg_loss'], 0.12345678901234567)
        absent = core.validate_updates([], complete)
        self.assertIsNone(absent['latest_metrics'])
        self.assertEqual(absent['metrics_lag_completed_outer'], 2)

    def test_metrics_uuid_counts_nonfinite_bool_and_reorder(self):
        rows = [attempt()]
        for name, value in (('attempt_id', 'e' * 32), ('step', True),
                            ('actual_optimizer_writes', 0), ('run_id', 'e' * 32)):
            changed = update(rows[0])
            changed[name] = value
            with self.subTest(name=name), self.assertRaises(core.ObservationError):
                core.validate_updates([changed], rows)
        for value in (True, float('nan'), float('inf')):
            changed = update(rows[0])
            changed['metrics']['actor/pg_loss'] = value
            with self.assertRaises(core.ObservationError):
                core.validate_updates([changed], rows)
        with self.assertRaises(core.ObservationError):
            core.validate_updates([update(rows[0])], [])

    def test_null_probe_version_is_reported_without_inventing_version(self):
        row = attempt()
        metrics = update(row)
        metrics['probe_version'] = None
        self.assertIsNone(core.validate_updates([metrics], [row])['probe_version'])


class SchedulerTests(unittest.TestCase):
    def test_unique_inflight_suffix_must_begin_at_driver_completed_and_write_counters(self):
        valid_driver = attempt(1, writes=3)
        valid_suffix = attempt(2, completed_before=1, writes_before=3)
        rows = scheduler_group(valid_driver) + [scheduler(valid_suffix, kind='BEGIN')]
        observed, _ = inspect_scheduler(rows, 0, [valid_driver])
        self.assertEqual(observed['uncommitted_suffix_rows'], 1)
        for driver, suffix in (([], attempt(1, completed_before=100, writes_before=900)),
                               ([valid_driver], attempt(2, completed_before=2, writes_before=3)),
                               ([valid_driver], attempt(2, completed_before=1, writes_before=4))):
            known = scheduler_group(valid_driver) if driver else []
            with self.subTest(driver=driver, suffix=suffix), self.assertRaises(core.ObservationError):
                inspect_scheduler(known + [scheduler(suffix, kind='BEGIN')], 0, driver)

    def test_suffix_prepared_and_complete_cannot_disagree_on_actual_writes(self):
        begin = scheduler(attempt(), kind='BEGIN')
        prepared = scheduler(attempt(writes=1), kind='COMMIT_PREPARED')
        complete = scheduler(attempt(writes=20), kind='COMPLETE')
        with self.assertRaises(core.ObservationError):
            inspect_scheduler([begin, prepared, complete], 0, [])

    def test_no_suffix_is_allowed_after_authenticated_target_200(self):
        drivers = [attempt(index, completed_before=index-1, writes_before=index-1)
                   for index in range(1, 201)]
        rows = [value for driver in drivers for value in scheduler_group(driver)]
        self.assertEqual(len(inspect_scheduler(rows, 0, drivers)[1]), 200)
        rows.append(scheduler(attempt(201, completed_before=199, writes_before=200), kind='BEGIN'))
        with self.assertRaises(core.ObservationError):
            inspect_scheduler(rows, 0, drivers)

    def test_available_metrics_gate_must_match_every_visible_scheduler_row(self):
        driver = attempt()
        for length in (1, 2, 3):
            rows = scheduler_group(driver)[:length]
            for value in rows:
                value['batch_gate_sha256'] = 'c' * 64
            with self.subTest(length=length), self.assertRaises(core.MetricGateDisagreement):
                inspect_scheduler(rows, 0, [driver], metric_gates={driver['attempt_id']: 'a'*64})
        matched, _ = inspect_scheduler(scheduler_group(driver)[:1], 0, [driver],
            metric_gates={driver['attempt_id']: 'a'*64})
        self.assertEqual(matched['visible_metric_gate_attempts_checked'], 1)

    def test_known_partial_group_checks_driver_before_unavailable_branch(self):
        driver = attempt()
        # These rows are internally consistent and reuse the real UUID/index,
        # exactly the partial-prefix bypass described in the independent review.
        for length in (1, 2):
            for changed in (attempt(1, completed_before=1, writes_before=9),
                            attempt(1, writes_before=7), attempt(1, writes=2)):
                rows = scheduler_group(changed)[:length]
                # A BEGIN alone cannot contradict an as-yet invisible final count.
                if length == 1 and changed['optimizer_writes_this_attempt'] == 2:
                    continue
                for births, claim_time in ((admitted_births(), 103), (None, None)):
                    with self.subTest(length=length, changed=changed, bound=births is not None), \
                         self.assertRaises(core.ObservationError):
                        inspect_scheduler(rows, 0, [driver], admitted_births=births,
                                          training_claim_unix=claim_time)

    def test_actual_manifest_budget_hash_uses_scheduler_producer_encoding(self):
        # Non-ASCII budget metadata distinguishes the two real producers'
        # encodings; hashing with core.canonical would give the wrong contract.
        expected = core.sha(json.dumps(budget(), sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode())
        self.assertEqual(core.scheduler_contract_sha256(budget()), expected)
        self.assertNotEqual(expected, core.sha(core.canonical(budget())))
        rows = scheduler_group(attempt())
        for row in rows:
            row['scheduler_contract_sha256'] = 'a' * 64
        with self.assertRaises(core.ObservationError):
            inspect_scheduler(rows, 0, [attempt()])

    def test_unadmitted_birth_and_eight_ranks_claiming_one_birth_are_rejected(self):
        rows = scheduler_group(attempt())
        for row in rows:
            row['process_pid'] = 999
            row['process_start_ticks'] = 999
        with self.assertRaises(core.ObservationError):
            inspect_scheduler(rows, 0, [attempt()])
        summaries, terminals = [], []
        for rank in range(8):
            rows = scheduler_group(attempt(), rank)
            for row in rows:
                row['process_pid'] = 100
                row['process_start_ticks'] = 200
            summary, matched = inspect_scheduler(rows, rank, [attempt()])
            summaries.append(summary)
            terminals.append(matched)
        with self.assertRaises(core.ObservationError):
            core.summarize_scheduler_crosscheck(summaries, terminals)

    def test_latest_suffix_includes_gate_and_contract_shared_provenance(self):
        summaries, terminals = [], []
        for rank in range(8):
            begin = scheduler(attempt(), rank, 'BEGIN')
            if rank == 7:
                begin['batch_gate_sha256'] = 'd' * 64
            summary, matched = inspect_scheduler([begin], rank, [])
            summaries.append(summary)
            terminals.append(matched)
        result = core.summarize_scheduler_crosscheck(summaries, terminals)
        self.assertEqual(result['evidence_state'], 'inflight_disagreement')
        self.assertIsNone(result['state'])
        for name in ('batch_gate_sha256', 'scheduler_contract_sha256'):
            self.assertIn(name, summaries[0]['latest_rank_receipt_state'])
            changed = copy.deepcopy(summaries)
            for row in changed:
                row['latest_rank_receipt_state']['batch_gate_sha256'] = 'c' * 64
            changed[-1]['latest_rank_receipt_state'][name] = 'e' * 64
            self.assertEqual(core.summarize_scheduler_crosscheck(changed, terminals)
                ['evidence_state'], 'inflight_disagreement')

    def test_missing_historical_binding_cannot_publish_agreed_latest_state(self):
        summaries = [inspect_scheduler([scheduler(attempt(), rank, 'BEGIN')], rank, [],
            admitted_births=None, training_claim_unix=None)[0] for rank in range(8)]
        result = core.summarize_scheduler_crosscheck(summaries, [[] for _ in range(8)])
        self.assertEqual(result['evidence_state'], 'unavailable_or_birth_unbound')
        self.assertIsNone(result['state'])

    def test_suffix_complete_never_increments_driver_and_lr_zero_is_preserved(self):
        row = attempt()
        rank_summary, matched = inspect_scheduler(scheduler_group(row), 0, [])
        self.assertEqual(matched, [])
        self.assertEqual(rank_summary['uncommitted_suffix_rows'], 3)
        self.assertEqual(rank_summary['latest_rank_receipt_state']['consumed_learning_rates'], [0.0])
        self.assertEqual(core.validate_attempts([])[0]['observed_driver_completed_outer'], 0)
        agreed, matched = inspect_scheduler(scheduler_group(row), 0, [row])
        self.assertEqual(matched[0]['observed_scheduler_last_epoch'], 1)
        self.assertEqual(agreed['last_driver_matched_state']['consumed_learning_rates'], [0.0])

    def test_fatal_record_and_rank_driver_disagreement(self):
        row = attempt()
        bad = scheduler_group(row)
        bad[-1]['writes_after'] = 2
        with self.assertRaises(core.ObservationError):
            inspect_scheduler(bad, 0, [row])
        bad = scheduler_group(row)
        bad[-1]['kind'] = 'POST_COMMIT_FAILURE'
        with self.assertRaises(core.ObservationError):
            inspect_scheduler(bad, 0, [row])
        with self.assertRaises(core.ObservationError):
            inspect_scheduler(scheduler_group(row, 1), 0, [row])

    def test_partial_rank_return_is_separate_unavailable_crosscheck(self):
        row = attempt()
        observed, matched = inspect_scheduler(scheduler_group(row)[:2], 0, [row])
        self.assertEqual(observed['unavailable_driver_attempts'], 1)
        self.assertEqual(matched, [])
        duplicate = scheduler_group(row) + [scheduler(row)]
        with self.assertRaises(core.ObservationError):
            inspect_scheduler(duplicate, 0, [row])


class GpuAndNativeTests(unittest.TestCase):
    def test_authenticated_claim_run_job_birth_and_history_are_separate_from_liveness(self):
        claim, ledger = training_claim(), trainer_ledger()
        for name, value in (('run_id', 'f' * 32), ('slurm_job_id', '3298'),
                            ('shell_pid', 81), ('shell_start_ticks', 91),
                            ('claimed_unix', 99), ('shell_pid', True)):
            changed = dict(claim, **{name: value})
            with self.subTest(name=name), self.assertRaises(core.ObservationError):
                core.validate_training_binding(changed, ledger, run_started_unix=100, now=200)
        for alteration in ('wrong_run', 'wrong_source_command', 'foreign_session', 'missing_leader'):
            changed = copy.deepcopy(ledger)
            trainer = changed['jobs']['trainer']
            if alteration == 'wrong_run':
                changed['run_id'] = 'e' * 32
            elif alteration == 'wrong_source_command':
                trainer['arguments'] = ['bash', 'other/train.sh']
            elif alteration == 'foreign_session':
                trainer['ledger']['100']['sid'] = 999
            else:
                del trainer['ledger']['80']
            with self.subTest(alteration=alteration), self.assertRaises(core.ObservationError):
                core.validate_training_binding(claim, changed, run_started_unix=100, now=200)
        ledger['jobs']['trainer']['state'] = 'STOPPED'
        for row in ledger['jobs']['trainer']['ledger'].values():
            row['state'] = 'Z'
        ledger['jobs']['trainer']['leader']['state'] = 'Z'
        summary, births = core.validate_training_binding(claim, ledger,
            run_started_unix=100, now=200)
        self.assertEqual(summary['evidence_state'], 'observed')
        self.assertEqual(summary['recorded_trainer_state'], 'STOPPED')
        self.assertEqual(len(births), 9)
        self.assertNotIn('birth_matches', summary)
        self.assertNotIn('command', summary['trainer_leader_birth'])

    def test_setup_trainer_and_boundary_straddling_peaks_do_not_mix(self):
        rows = [gpu_sample(102, 101, 9000), gpu_sample(104, 102, 9900),
                gpu_sample(107, 106, 300)]
        summary = check_gpu_samples(rows, inventory())
        self.assertEqual(summary['cumulative_sampled_peak_mib']['0'], 9900)
        windows = summary['windows']
        self.assertEqual(windows['setup']['sampled_peak_mib']['0'], 9000)
        self.assertEqual(windows['trainer_window']['sampled_peak_mib']['0'], 300)
        self.assertEqual(windows['boundary_straddling']['sampled_peak_mib']['0'], 9900)
        self.assertEqual(windows['boundary_straddling']['sample_count'], 1)
        self.assertEqual(windows['trainer_window']['query_started_unix'], 106)
        self.assertEqual(windows['trainer_window']['window_start_unix'], 103)
        self.assertIn('initialization', windows['trainer_window']['scope'])

    def test_before_run_samples_straddle_only_and_unavailable_boundary(self):
        with self.assertRaises(core.ObservationError):
            check_gpu_samples([gpu_sample(102, 99)], inventory())
        straddle = check_gpu_samples([gpu_sample(104, 102)], inventory())
        self.assertIsNone(straddle['windows']['trainer_window']['sampled_peak_mib'])
        self.assertEqual(straddle['windows']['trainer_window']['evidence_state'], 'no_complete_samples')
        self.assertTrue(straddle['observation_incomplete'])
        unknown = check_gpu_samples([gpu_sample()], inventory(), training_claim_unix=None)
        self.assertEqual(unknown['unclassified_sample_count'], 1)
        for name in ('setup', 'trainer_window', 'boundary_straddling'):
            self.assertEqual(unknown['windows'][name]['evidence_state'], 'boundary_unavailable')
            self.assertIsNone(unknown['windows'][name]['sampled_peak_mib'])
        self.assertEqual(unknown['cumulative_sampled_peak_mib']['0'], 200)
        at_boundary = check_gpu_samples([gpu_sample(104, 103)], inventory())
        self.assertEqual(at_boundary['windows']['trainer_window']['sample_count'], 1)

    def test_gpu_peaks_window_interval_gap_and_provenance(self):
        rows = [gpu_sample(), gpu_sample(107, 106, 300), gpu_sample(119, 118, 250)]
        summary = check_gpu_samples(rows, inventory(), now=130)
        self.assertEqual(summary['sample_count'], 3)
        self.assertEqual(summary['sampling_window_seconds'], 18)
        self.assertEqual(summary['age_of_last_sample_seconds'], 11)
        self.assertEqual(summary['cumulative_sampled_peak_mib']['0'], 300)
        self.assertEqual(summary['observed_interval_max_seconds'], 12)
        self.assertEqual(summary['largest_idle_gap_seconds'], 11)

    def test_gpu_source_run_job_pid_and_inventory_mismatch(self):
        for name, value in (('run_id', 'e' * 32), ('slurm_job_id', '3298'),
                            ('source_map_sha256', 'd' * 64), ('sampling_interval_seconds', True)):
            row = gpu_sample()
            row[name] = value
            with self.subTest(name=name), self.assertRaises(core.ObservationError):
                check_gpu_samples([row], inventory(), now=130)
        row = gpu_sample()
        row['gpus'][0]['uuid'] = row['gpus'][1]['uuid']
        with self.assertRaises(core.ObservationError):
            check_gpu_samples([row], inventory(), now=130)
        second = gpu_sample(107, 106)
        second['sampler_identity']['start_ticks'] += 1
        with self.assertRaises(core.ObservationError):
            check_gpu_samples([gpu_sample(), second], inventory(), now=130)

    def test_no_samples_is_not_zero_peak_or_current_measurement(self):
        empty = check_gpu_samples([], inventory(), now=130)
        self.assertEqual(empty['sample_state'], 'no_complete_samples')
        self.assertIsNone(empty['cumulative_sampled_peak_mib'])
        self.assertIsNone(empty['sampling_window_seconds'])

    def test_current_gpu_uses_bounded_capture_and_preserves_scope(self):
        physical = inventory()
        raw = '\n'.join('%d,%s,20,81559,10' % (index, physical[index]['uuid']) for index in range(8))
        capture = mock.Mock(return_value=subprocess.CompletedProcess([], 0, raw, ''))
        result = core.current_gpu(physical, time.monotonic() + 20, capture)
        self.assertEqual(result['evidence_state'], 'observed')
        self.assertEqual(capture.call_args.args[2:], (16384, 4096))
        self.assertIn('not attribution', result['scope'])

    def test_native_purge_and_finished_stage_do_not_infer_success(self):
        observed = dict(base=core.BASE, job_id=core.JOB, job_returncode=1,
                        **{'pipeline-status.json': dict(stage='FINISHED_FULL200_FINAL51713_AND_OWNED_CLEANUP')})
        summary = core.native_summary(observed)
        self.assertEqual(summary['state'], 'unknown')
        self.assertEqual(summary['terminal_success'], 'unknown')
        observed['job_returncode'] = 0
        complete = core.native_summary(observed, dict(JobId=core.JOB, JobState='COMPLETED', ExitCode='0:0'))
        self.assertEqual(complete['terminal_success'], 'not_evaluated')


class CompletenessTests(unittest.TestCase):
    def test_auxiliary_namespace_over_bound_preserves_actual_driver_prefix(self):
        result, _ = synthetic_collection(namespace_count=core.MAX_PROCESS_BIRTHS + 1)
        self.assertEqual(result['progress']['observed_driver_completed_outer'], 1)
        self.assertEqual(result['progress']['agreed_writes_at_last_return_per_rank'], 1)
        self.assertEqual(result['current_namespace_births'], [])
        self.assertEqual(result['namespace_births_provenance']['evidence_state'], 'invalid_or_unstable')
        self.assertTrue(result['observation_incomplete'])
        rows, provenance = core.namespace_birth_component([dict(pid=1, start_ticks=2, sid=1, uid=0)] * 2)
        self.assertEqual(rows, [])
        self.assertTrue(provenance['observation_incomplete'])

    def test_different_metrics_gate_on_complete_or_partial_rank_suppresses_metrics_only(self):
        for mode in ('different_metric_gate', 'different_partial_metric_gate'):
            result, _ = synthetic_collection(scheduler_mode=mode)
            with self.subTest(mode=mode):
                self.assertEqual(result['progress']['observed_driver_completed_outer'], 1)
                self.assertEqual(result['metrics']['last_metrics_completed_step'], 1)
                self.assertEqual(result['metrics']['scheduler_gate_binding_state'], 'invalid')
                self.assertEqual(result['metrics']['metrics_state'], 'invalid_scheduler_gate_binding')
                self.assertIsNone(result['metrics']['latest_metrics'])
                self.assertIsNone(result['agreed_latest_scheduler_receipt_state']['state'])
                self.assertTrue(result['observation_incomplete'])

    def test_actual_502_historical_and_470_live_namespace_are_within_shared_bound(self):
        # Sizes observed in the real producer snapshot; identities are synthetic.
        # This exercises the actual remote collector flow without /proc or RPC.
        result, _ = synthetic_collection(historical_count=502, namespace_count=470)
        self.assertEqual(result['training_boundary']['admitted_trainer_birth_count'], 502)
        self.assertEqual(len(result['current_namespace_births']), 470)
        self.assertFalse(result['observation_incomplete'])
        self.assertEqual(result['progress']['observed_driver_completed_outer'], 1)

    def test_real_owned_wrapper_log_differs_from_canonical_training_console(self):
        ledger = trainer_ledger()
        self.assertEqual(ledger['jobs']['trainer']['log'], core.BASE + '/logs/trainer.log')
        bound, _ = core.validate_training_binding(training_claim(), ledger,
            run_started_unix=100, now=200)
        self.assertEqual(bound['evidence_state'], 'observed')
        ledger['jobs']['trainer']['log'] = core.BASE + '/logs/search-g1-common8-training.log'
        with self.assertRaises(core.ObservationError):
            core.validate_training_binding(training_claim(), ledger,
                run_started_unix=100, now=200)

    def test_actual_5287_byte_hydra_command_is_historical_metadata_only(self):
        row = historical_process(100, 200)
        row['command'] = 'x' * 5287
        identity = core.historical_identity(row)
        self.assertEqual(identity, {k: row[k] for k in ('pid', 'start_ticks', 'sid', 'uid')})
        self.assertNotIn('command', identity)
        row['command'] = 'x' * (core.MAX_PROCESS_COMMAND + 1)
        with self.assertRaises(core.ObservationError):
            core.historical_identity(row)

    def test_process_caps_and_namespace_duplicate_identity_are_still_rejected(self):
        namespace = [dict(pid=3000 + index, start_ticks=4000 + index,
            sid=80, uid=os.getuid()) for index in range(core.MAX_PROCESS_BIRTHS)]
        self.assertEqual(len(core.validate_namespace_births(namespace)), core.MAX_PROCESS_BIRTHS)
        extra = dict(pid=99999, start_ticks=99999, sid=80, uid=os.getuid())
        with self.assertRaises(core.ObservationError):
            core.validate_namespace_births(namespace + [extra])
        with self.assertRaises(core.ObservationError):
            core.validate_namespace_births(namespace[:1] * 2)
        ledger = trainer_ledger()
        recorded = ledger['jobs']['trainer']['ledger']
        for index in range(core.MAX_PROCESS_BIRTHS + 1 - len(recorded)):
            pid = 1000 + index
            recorded[str(pid)] = historical_process(pid, 2000 + index)
        with self.assertRaises(core.ObservationError):
            core.validate_training_binding(training_claim(), ledger,
                run_started_unix=100, now=200)

    def test_complete_fixture_and_missing_future_evaluation_is_not_yet_applicable(self):
        result, _ = synthetic_collection()
        complete = result['capture_completeness']
        self.assertFalse(result['observation_incomplete'])
        self.assertTrue(complete['overall_capture_complete'])
        self.assertEqual(complete['components']['development_evaluation_5']['status'], 'not_yet_applicable')
        self.assertEqual(result['progress']['observed_driver_completed_outer'], 1)
        # At the gate, the same absent evidence is now required. Unknown driver
        # progress also cannot assert that the gate has not yet been reached.
        for completed in (5, None):
            changed = copy.deepcopy(result)
            changed['progress']['observed_driver_completed_outer'] = completed
            assessed = core.observation_completeness(changed)
            self.assertTrue(assessed['observation_incomplete'])
            self.assertEqual(assessed['components']['development_evaluation_5']['status'], 'incomplete')

    def test_missing_pipeline_allocation_and_before_evaluation_keep_driver_prefix(self):
        for name, component in (('pipeline-status.json', 'pipeline'),
                                ('native-allocation-admission.json', 'gpu_allocation'),
                                ('formal-inline-development-before.json', 'development_evaluation_0')):
            result, queries = synthetic_collection(missing_documents=(name,))
            with self.subTest(name=name):
                self.assertTrue(result['observation_incomplete'])
                self.assertEqual(result['capture_completeness']['components'][component]['status'], 'incomplete')
                self.assertEqual(result['progress']['observed_driver_completed_outer'], 1)
                self.assertEqual(result['progress']['agreed_writes_at_last_return_per_rank'], 1)
            if name == 'native-allocation-admission.json':
                self.assertEqual(queries, 0)
                self.assertEqual(result['gpu_samples_provenance']['evidence_state'], 'unavailable')
                self.assertIsNone(result['gpu_samples']['cumulative_sampled_peak_mib'])

    def test_failed_current_gpu_unknown_absent_births_and_partial_crosscheck_are_incomplete(self):
        cases = [({'current_gpu_state': 'unavailable'}, 'current_gpu'),
                 ({'birth_state': 'unknown'}, 'current_namespace_births'),
                 ({'birth_state': 'absent_now'}, 'gpu_sampler_current_birth'),
                 ({'scheduler_mode': 'partial'}, 'scheduler_rank_0')]
        for parameters, component in cases:
            result, _ = synthetic_collection(**parameters)
            with self.subTest(parameters=parameters):
                self.assertTrue(result['observation_incomplete'])
                self.assertEqual(result['capture_completeness']['components'][component]['status'], 'incomplete')
                self.assertEqual(result['progress']['observed_driver_completed_outer'], 1)
        partial, _ = synthetic_collection(scheduler_mode='partial')
        self.assertEqual(partial['scheduler_crosscheck'][0]['unavailable_driver_attempts'], 1)
        self.assertEqual(partial['agreed_latest_scheduler_receipt_state']['state']['kind'], 'COMMIT_PREPARED')

    def test_invalid_partial_crosscheck_and_missing_claim_do_not_erase_valid_driver(self):
        invalid, _ = synthetic_collection(scheduler_mode='contradictory_partial')
        self.assertTrue(invalid['observation_incomplete'])
        self.assertEqual(invalid['scheduler_crosscheck'][0]['evidence_state'], 'invalid_or_unstable')
        self.assertIsNone(invalid['agreed_latest_scheduler_receipt_state']['state'])
        self.assertEqual(invalid['progress']['observed_driver_completed_outer'], 1)
        unbound, _ = synthetic_collection(missing_documents=('training-launch-claim.json',))
        self.assertEqual(unbound['pipeline']['stage'], 'GRPO_1_TO_200')
        self.assertIsNone(unbound['gpu_samples']['windows']['trainer_window']['sampled_peak_mib'])
        self.assertIsNone(unbound['agreed_latest_scheduler_receipt_state']['state'])
        self.assertEqual(unbound['progress']['observed_driver_completed_outer'], 1)
        self.assertTrue(unbound['observation_incomplete'])

    def test_torn_deferred_or_missing_component_provenance_is_not_complete(self):
        result, _ = synthetic_collection()
        for component in ('attempts_provenance', 'updates_provenance', 'gpu_samples_provenance',
                          'trainer_ledger_provenance', 'training_claim_provenance'):
            changed = copy.deepcopy(result)
            changed[component]['observation_incomplete'] = True
            with self.subTest(component=component):
                self.assertTrue(core.observation_completeness(changed)['observation_incomplete'])
        changed = copy.deepcopy(result)
        del changed['gpu_samples_provenance']
        self.assertTrue(core.observation_completeness(changed)['observation_incomplete'])
        lagging = copy.deepcopy(result)
        lagging['metrics'] = core.validate_updates([], [attempt()])
        assessed = core.observation_completeness(lagging)
        self.assertEqual(assessed['components']['update_metrics']['status'], 'incomplete')
        self.assertEqual(lagging['progress']['observed_driver_completed_outer'], 1)
        self.assertIsNone(lagging['metrics']['latest_metrics'])


class FileAndReviewTests(unittest.TestCase):
    def test_actual_2187233_byte_ledger_uses_its_own_bounded_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve() / 'process-ledger.json'
            # Match the observed size; content is synthetic and contains no commands.
            prefix = b'{"padding":"'
            suffix = b'"}'
            raw = prefix + b'x' * (2187233 - len(prefix) - len(suffix)) + suffix
            target.write_bytes(raw)
            deadline = time.monotonic() + 5
            with self.assertRaises(core.ObservationError):
                core.capture_json(target, deadline)
            captured, metadata = core.optional_json(target, deadline, maximum=core.MAX_PROCESS_LEDGER)
            self.assertEqual(metadata['captured_bytes'], 2187233)
            self.assertEqual(metadata['evidence_state'], 'observed')
            self.assertEqual(len(captured['padding']), len(raw) - len(prefix) - len(suffix))
            self.assertLess(metadata['captured_bytes'], core.MAX_PROCESS_LEDGER)

    def test_missing_versus_positive_empty_jsonl_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            missing, missing_meta = core.optional_capture(root / 'missing', 100, 2, time.monotonic() + 5)
            self.assertIsNone(missing)
            self.assertEqual(missing_meta['evidence_state'], 'unavailable')
            (root / 'empty').write_bytes(b'')
            empty, empty_meta = core.optional_capture(root / 'empty', 100, 2, time.monotonic() + 5)
            self.assertEqual(empty, [])
            self.assertEqual(empty_meta['evidence_state'], 'observed')
            self.assertFalse(empty_meta['observation_incomplete'])

    def test_symlink_and_size_bound_refuse_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'data').write_bytes(b'{}\n')
            (root / 'link').symlink_to(root / 'data')
            for path, maximum in ((root / 'link', 100), (root / 'data', 2)):
                with self.assertRaises(core.ObservationError):
                    core.snapshot(path, maximum, time.monotonic() + 5)

    def test_review_gate_rejects_unapproved_or_executed_draft_before_loading(self):
        for approved, executed in ((False, False), (True, True)):
            value = dict(role='formal_progress_independent_static_review',
                approved=approved, scope='staticonly', blocking_findings=[],
                draft_code_executed=executed, external_file_sha256={})
            with mock.patch.object(collector, 'sealed', return_value=core.canonical(value)), \
                 self.assertRaises(ValueError):
                collector.review_files(Path('/review.json'), 'f' * 64)


class DeferredAppendAndGpuFailureRegressions(unittest.TestCase):
    """Focused regression source appended without executing the draft."""

    def test_json_and_jsonl_capture_mark_deferred_append_without_manual_flag(self):
        for appended in (False, True):
            for kind, raw in (('json', b'{"value":1}'),
                              ('jsonl', b'{"value":1}\n')):
                provenance = dict(captured_sha256=core.sha(raw),
                    appended_after_capture=appended, captured_bytes=len(raw))
                with self.subTest(kind=kind, appended=appended), \
                     mock.patch.object(core, 'snapshot', return_value=(raw, provenance)):
                    if kind == 'json':
                        values, metadata = core.capture_json(Path('/synthetic/document.json'), 1)
                        self.assertEqual(values, {'value': 1})
                    else:
                        values, metadata = core.capture_jsonl(Path('/synthetic/rows.jsonl'), 100, 2, 1)
                        self.assertEqual(values, [{'value': 1}])
                self.assertEqual(metadata['observation_incomplete'], appended)
                self.assertEqual(metadata['captured_sha256'], core.sha(raw))

    def test_deferred_append_provenance_is_incomplete_without_manual_flag(self):
        baseline, _ = synthetic_collection()
        components = (
            ('run_manifest', ('run_manifest_provenance',)),
            ('driver_attempts', ('attempts_provenance',)),
            ('update_metrics', ('updates_provenance',)),
            ('scheduler_rank_0', ('scheduler_crosscheck', 0, 'provenance')),
            ('training_launch_claim', ('training_claim_provenance',)),
            ('historical_trainer_ledger', ('trainer_ledger_provenance',)),
            ('pipeline', ('pipeline', 'provenance')),
            ('development_evaluation_0', ('development_evaluations', 0, 'provenance')),
            ('gpu_allocation', ('gpu_allocation_provenance',)),
            ('gpu_sample_prefix', ('gpu_samples_provenance',)))
        for component, path in components:
            changed = copy.deepcopy(baseline)
            provenance = changed
            for key in path:
                provenance = provenance[key]
            provenance.pop('observation_incomplete', None)
            provenance['appended_after_capture'] = True
            with self.subTest(component=component):
                assessed = core.observation_completeness(changed)
                self.assertEqual(assessed['components'][component]['status'], 'incomplete')
                self.assertFalse(assessed['overall_capture_complete'])
                self.assertTrue(assessed['observation_incomplete'])
                self.assertEqual(changed['progress']['observed_driver_completed_outer'], 1)
                self.assertEqual(changed['progress']['agreed_writes_at_last_return_per_rank'], 1)

    def test_current_gpu_runtime_failure_returns_only_unavailable_component(self):
        capture = mock.Mock(side_effect=RuntimeError('sealed capture diagnostic must stay private'))
        result = core.current_gpu(inventory(), time.monotonic() + 20, capture)
        self.assertEqual(result['evidence_state'], 'unavailable')
        self.assertEqual(result['reason_code'], 'RuntimeError')
        self.assertTrue(result['observation_incomplete'])
        self.assertNotIn('sealed capture diagnostic', json.dumps(result))
        self.assertNotIn('gpus', result)
        self.assertEqual(capture.call_count, 1)
        self.assertEqual(capture.call_args.args[2:], (16384, 4096))

    def test_integrated_gpu_runtime_failure_retains_independent_driver_evidence(self):
        # Restore the real current_gpu inside the existing offline fixture so
        # the sealed protocol failure crosses the actual collector call path.
        saved_current_gpu = core.current_gpu
        saved_collect_remote = core.collect_remote
        def collect_with_gpu_failure(payload, protocol):
            protocol.bounded_capture.side_effect = RuntimeError('sealed bounded capture refused')
            core.current_gpu.side_effect = saved_current_gpu
            return saved_collect_remote(payload, protocol)
        with mock.patch.object(core, 'collect_remote', side_effect=collect_with_gpu_failure):
            result, queries = synthetic_collection()
        self.assertEqual(queries, 1)
        self.assertEqual(result['current_gpu']['evidence_state'], 'unavailable')
        self.assertEqual(result['current_gpu']['reason_code'], 'RuntimeError')
        self.assertEqual(result['progress']['observed_driver_completed_outer'], 1)
        self.assertEqual(result['progress']['agreed_writes_at_last_return_per_rank'], 1)
        self.assertEqual(result['metrics']['last_metrics_completed_step'], 1)
        self.assertEqual(result['gpu_samples']['sample_state'], 'available')
        incomplete = [name for name, state in result['capture_completeness']['components'].items()
                      if not state['capture_complete']]
        self.assertEqual(incomplete, ['current_gpu'])
        self.assertTrue(result['observation_incomplete'])


class EvaluationPresenceAndDeadlineRegressions(unittest.TestCase):
    """Unexecuted offline source exercising real component handlers and joins."""

    def assert_driver_retained(self, result):
        self.assertEqual(result['progress']['observed_driver_completed_outer'], 1)
        self.assertEqual(result['progress']['agreed_writes_at_last_return_per_rank'], 1)
        self.assertTrue(result['progress']['actual_total_writes_unavailable'])

    def test_each_missing_evaluation_field_is_invalid_without_erasing_driver(self):
        required = ('schema_version', 'role', 'run_id', 'step', 'policy_safety_mode',
            'source_map_sha256', 'plan_sha256', 'validation_sha256', 'checked_unix',
            'weighted_em', 'weighted_f1')
        for name in required:
            result, _ = synthetic_collection(missing_document_fields={
                'formal-inline-development-before.json': (name,)})
            with self.subTest(field=name):
                evaluation = result['development_evaluations'][0]
                self.assertEqual(evaluation['step'], 0)
                self.assertEqual(evaluation['evidence_state'], 'invalid_or_unstable')
                self.assertEqual(evaluation['reason_code'], 'ObservationError')
                self.assertNotIn('metrics', evaluation)
                self.assertEqual(result['capture_completeness']['components'][
                    'development_evaluation_0']['status'], 'incomplete')
                self.assertTrue(result['observation_incomplete'])
                self.assert_driver_retained(result)
                self.assertEqual(result['metrics']['last_metrics_completed_step'], 1)

    def test_producer_raw_evaluation_fields_are_allowed_and_not_exported(self):
        ids = ['offline-private-question-%d' % index for index in range(512)]
        sources = ['nq'] * 256 + ['hotpotqa'] * 256
        rows = [dict(question_id=question, data_source=source, em=int(index < 128),
                     f1=1 if index < 128 else 1 / 15, actual_calls=0, valid_action=0,
                     turns=1, answer_sha256='e' * 64, answer_present=True,
                     response_token_count=1, model_token_count=1)
                for index, (question, source) in enumerate(zip(ids, sources))]
        extras = dict(question_count=512, source_counts=dict(nq=256, hotpotqa=256),
            question_ids=ids, data_sources=sources, rows=rows,
            source_correct=dict(nq=128, hotpotqa=0), source_em=dict(nq=0.5, hotpotqa=0.0),
            executed_search_question_count=0, executed_search_call_count=0,
            valid_action_count=0, action_opportunity_count=512, production_admission=False,
            valid_action_rate_definition='synthetic definition',
            executed_search_definition='synthetic definition', semantics='synthetic paired dev512')
        result, _ = synthetic_collection(extra_document_fields={
            'formal-inline-development-before.json': extras})
        evaluation = result['development_evaluations'][0]
        self.assertEqual(evaluation['evidence_state'], 'observed')
        self.assertEqual(evaluation['metrics']['weighted_em'], 0.25)
        self.assertEqual(evaluation['metrics']['weighted_f1'], 0.3)
        self.assertFalse(result['observation_incomplete'])
        for name in extras:
            self.assertNotIn(name, evaluation)
        self.assertNotIn('offline-private-question-', json.dumps(result))
        self.assert_driver_retained(result)

    def test_budget_expiring_after_driver_validation_defers_remaining_files(self):
        result, queries = synthetic_collection(expired_budget_at='after_driver_validation')
        self.assert_driver_retained(result)
        self.assertEqual(queries, 0)
        self.assertEqual(result['updates_provenance']['reason_code'], 'time_budget_exhausted')
        self.assertEqual(result['training_claim_provenance']['reason_code'], 'time_budget_exhausted')
        self.assertTrue(all(row['provenance']['reason_code'] == 'time_budget_exhausted'
                            for row in result['scheduler_crosscheck']))
        self.assertEqual(result['collection_time_budget']['reason_code'], 'time_budget_exhausted')
        self.assertEqual(result['capture_completeness']['components'][
            'collection_time_budget']['status'], 'incomplete')
        self.assertTrue(result['observation_incomplete'])

    def test_real_trainer_birth_deadline_handler_keeps_valid_driver_and_metrics(self):
        result, queries = synthetic_collection(expired_budget_at='during_trainer_birth')
        self.assert_driver_retained(result)
        self.assertEqual(result['metrics']['last_metrics_completed_step'], 1)
        self.assertEqual(queries, 0)
        birth = result['trainer_leader_current_birth']
        self.assertEqual(birth['evidence_state'], 'unknown')
        self.assertEqual(birth['reason_code'], 'time_budget_exhausted')
        self.assertIsNone(birth['actual_identity'])
        self.assertTrue(birth['observation_incomplete'])
        self.assertTrue(result['observation_incomplete'])

    def test_real_current_gpu_deadlines_keep_driver_and_mark_budget_incomplete(self):
        for stage in ('before_current_gpu', 'during_current_gpu'):
            result, queries = synthetic_collection(expired_budget_at=stage)
            with self.subTest(stage=stage):
                self.assert_driver_retained(result)
                self.assertEqual(result['metrics']['last_metrics_completed_step'], 1)
                self.assertEqual(queries, 1)
                self.assertEqual(result['current_gpu']['evidence_state'], 'unavailable')
                self.assertEqual(result['current_gpu']['reason_code'], 'time_budget_exhausted')
                self.assertEqual(result['gpu_samples_provenance']['reason_code'], 'time_budget_exhausted')
                self.assertEqual(result['capture_completeness']['components'][
                    'collection_time_budget']['status'], 'incomplete')
                self.assertTrue(result['observation_incomplete'])

    def test_finalization_after_last_birth_retains_all_previously_known_scalars(self):
        result, _ = synthetic_collection(expired_budget_at='after_namespace_birth')
        self.assert_driver_retained(result)
        self.assertEqual(result['metrics']['last_metrics_completed_step'], 1)
        self.assertEqual(result['gpu_samples']['sample_state'], 'available')
        self.assertEqual(result['current_namespace_births'][0]['evidence_state'], 'observed')
        incomplete = [name for name, state in result['capture_completeness']['components'].items()
                      if not state['capture_complete']]
        self.assertEqual(incomplete, ['collection_time_budget'])
        self.assertTrue(result['observation_incomplete'])

    def test_expired_optional_file_and_gpu_handlers_start_no_capture(self):
        with mock.patch.object(core.time, 'monotonic', return_value=61), \
             mock.patch.object(core, 'capture_json') as document_read, \
             mock.patch.object(core, 'capture_jsonl') as row_read:
            document, document_meta = core.optional_json(Path('/offline/document'), 60)
            rows, row_meta = core.optional_capture(Path('/offline/rows'), 100, 2, 60)
            capture = mock.Mock()
            current = core.current_gpu(inventory(), 60, capture)
        document_read.assert_not_called()
        row_read.assert_not_called()
        capture.assert_not_called()
        self.assertIsNone(document)
        self.assertIsNone(rows)
        for metadata in (document_meta, row_meta, current):
            self.assertEqual(metadata['evidence_state'], 'unavailable')
            self.assertEqual(metadata['reason_code'], 'time_budget_exhausted')
            self.assertTrue(metadata['observation_incomplete'])

    def test_actual_birth_expired_on_entry_does_not_touch_proc(self):
        identity = dict(pid=80, start_ticks=90, sid=80, uid=os.getuid())
        with mock.patch.object(core.time, 'monotonic', return_value=61), \
             mock.patch.object(Path, 'stat') as stat_query, \
             mock.patch.object(core.os, 'open') as opened, \
             mock.patch.object(core.os, 'read') as read:
            result = core.actual_birth(identity, 60)
        stat_query.assert_not_called()
        opened.assert_not_called()
        read.assert_not_called()
        self.assertEqual(result['evidence_state'], 'unknown')
        self.assertEqual(result['reason_code'], 'time_budget_exhausted')
        self.assertIsNone(result['actual_identity'])
        self.assertTrue(result['observation_incomplete'])

    def test_actual_birth_expiring_during_bounded_stat_read_closes_and_defers_env(self):
        identity = dict(pid=80, start_ticks=90, sid=80, uid=os.getuid())
        fields = [b'S', b'1', b'1', b'80'] + [b'0'] * 15 + [b'90']
        raw = b'80 (offline birth) ' + b' '.join(fields)
        clock = dict(elapsed=0)
        def read_stat(fd, maximum):
            self.assertEqual((fd, maximum), (70, 4097))
            clock['elapsed'] = 61
            return raw
        with mock.patch.object(core.time, 'monotonic', side_effect=lambda: clock['elapsed']), \
             mock.patch.object(Path, 'stat', return_value=mock.Mock(st_uid=os.getuid())) as stat_query, \
             mock.patch.object(core.os, 'open', return_value=70) as opened, \
             mock.patch.object(core.os, 'read', side_effect=read_stat) as read, \
             mock.patch.object(core.os, 'close') as closed:
            result = core.actual_birth(identity, 60)
        self.assertEqual(stat_query.call_count, 1)
        self.assertEqual(opened.call_count, 1)
        self.assertEqual(opened.call_args.args[0], Path('/proc/80/stat'))
        self.assertEqual(read.call_count, 1)
        closed.assert_called_once_with(70)
        self.assertEqual(result['evidence_state'], 'unknown')
        self.assertEqual(result['reason_code'], 'time_budget_exhausted')
        self.assertIsNone(result['actual_identity'])
        self.assertTrue(result['observation_incomplete'])

    def test_malformed_birth_identity_remains_strict_even_with_expired_budget(self):
        with mock.patch.object(core.time, 'monotonic', return_value=61), \
             self.assertRaises(core.ObservationError):
            core.actual_birth(dict(pid=80), 60)


class DecoderFailureIsolationTests(unittest.TestCase):
    def test_deep_auxiliary_json_is_invalid_without_erasing_driver(self):
        depth = sys.getrecursionlimit() + 100
        raw = b'{"extra":' + b'[' * depth + b'0' + b']' * depth + b'}'
        result, _ = synthetic_collection(malformed_evaluation_raw=raw)
        self.assertEqual(result['progress']['observed_driver_completed_outer'], 1)
        self.assertEqual(result['progress']['agreed_writes_at_last_return_per_rank'], 1)
        evaluation = result['development_evaluations'][0]
        self.assertEqual(evaluation['evidence_state'], 'invalid_or_unstable')
        self.assertEqual(evaluation['reason_code'], 'ObservationError')
        self.assertTrue(result['observation_incomplete'])

    def test_deep_jsonl_prefix_is_component_invalid_not_native_recursion_failure(self):
        depth = sys.getrecursionlimit() + 100
        raw = b'{"extra":' + b'[' * depth + b'0' + b']' * depth + b'}\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'rows.jsonl'
            path.write_bytes(raw)
            rows, provenance = core.optional_capture(path, 256*1024, 2, time.monotonic()+5)
        self.assertIsNone(rows)
        self.assertEqual(provenance['reason_code'], 'ObservationError')
        self.assertTrue(provenance['observation_incomplete'])

    def test_non_utf8_strings_and_overlong_integer_literals_decline_auxiliary_only(self):
        for raw in (b'{"extra":"\\ud800"}', b'{"extra":' + b'1'*129 + b'}'):
            with self.subTest(raw=raw):
                result, _ = synthetic_collection(malformed_evaluation_raw=raw)
                self.assertEqual(result['progress']['observed_driver_completed_outer'], 1)
                self.assertEqual(result['development_evaluations'][0]['evidence_state'], 'invalid_or_unstable')
                self.assertTrue(result['observation_incomplete'])
        self.assertEqual(core.strict_json(b'{"value":' + b'1'*128 + b'}')['value'], int('1'*128))
        self.assertEqual(core.strict_json(b'{"value":"\\ud83d\\ude00"}')['value'], '\U0001f600')


if __name__ == '__main__':
    unittest.main()
