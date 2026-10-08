"""Offline stdlib regressions; no network, training imports, or CUDA.

These tests are intentionally unexecuted draft source until a fresh independent
agent has reviewed the exact collector, core, and test bytes.
"""
import copy
import json
import os
import subprocess
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
        log=core.BASE + '/logs/search-g1-common8-training.log', state='STARTED',
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
        batch_gate_sha256='c' * 64, scheduler_advances=int(advanced),
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
                         birth_state='observed', scheduler_mode='complete'):
    """Exercise the actual collector control flow with mocked scalar producers.

    All paths are inside an offline temporary directory. No network, subprocess,
    /proc, runtime source package, GPU, or actual job is accessed by this fixture.
    """
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
            provenance = dict(captured_sha256='d' * 64, appended_after_capture=False,
                              observation_incomplete=False)
            def capture_document(path, deadline):
                if path.name in missing_documents or path.name not in documents:
                    raise FileNotFoundError(path.name)
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
                elif path.name == 'gpu-samples-' + core.RUN + '.jsonl':
                    values = [gpu_sample(102, 101, 9000), gpu_sample(107, 106, 300)]
                else:
                    raise FileNotFoundError(path.name)
                return copy.deepcopy(values), dict(provenance)
            def current_birth(identity, deadline):
                if birth_state != 'observed':
                    return dict(saved_identity=identity, actual_identity=None,
                        birth_matches=None, namespace_matches=None, evidence_state=birth_state)
                return dict(saved_identity=identity, actual_identity=dict(identity), process_state='S',
                    birth_matches=True, namespace_matches=True, evidence_state='observed')
            protocol = mock.Mock()
            protocol.runtime_map.return_value = mapping
            protocol.safe_regular_bytes.return_value = b'frozen-method'
            payload = dict(base=core.BASE, run_id=core.RUN, job_id=core.JOB,
                source_commit=core.COMMIT, source_map_sha256=map_sha, source_sha256=mapping,
                method_sha256=core.sha(b'frozen-method'),
                observed_namespace_births=[gpu_sample()['sampler_identity']])
            current = dict(evidence_state=current_gpu_state)
            with mock.patch.object(core, 'capture_json', side_effect=capture_document), \
                 mock.patch.object(core, 'capture_jsonl', side_effect=capture_rows), \
                 mock.patch.object(core, 'actual_birth', side_effect=current_birth), \
                 mock.patch.object(core, 'current_gpu', return_value=current) as current_query:
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


if __name__ == '__main__':
    unittest.main()
