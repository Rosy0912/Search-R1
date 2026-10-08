"""Native-free boundary fixtures; synthetic checkpoint bytes are never runtime proof."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import author200_observation as m

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
SNAP = HERE.parent / 'actual-snapshots/3261-20261005T210622178545Z/original-files'
BASE = m.loads((SNAP / 'full_state/global_step_5/manifest.json').read_bytes())
GATE = m.frozen_gate(WORK / 'opsd-e1-300')
CTX = {'gate': GATE, 'workers': [{'pid': 1533910 + r} for r in range(8)]}


def evaluation(counts):
    metrics = {'val/qa_count/all': sum(counts.values())}
    for source, n in counts.items():
        metrics['val/qa_count/' + source] = n
        metrics['val/evaluated_count/' + source] = n
        metrics['val/test_score/' + source] = .125
    for key in m.QA_KEYS:
        value = 150.25 if key == 'generated_tokens' else (.75 if key == 'search_calls' else .125)
        for source in set(counts) | {'all'}:
            metrics[f'val/{key}/{source}'] = value
    return metrics


def outer(step):
    metrics = {k: copy.deepcopy(v) for k, v in BASE['outer_metrics'].items()
               if not k.startswith(('val/', 'opsd_stage/'))}
    for rank in range(8):
        metrics['teacher/cuda_update_entries_cumulative_per_rank/rank_' + str(rank)] = step
    if step % 50 == 0:
        metrics.update(evaluation(m.DEV_COUNTS))
    if step == 5 or step % 50 == 0:
        metrics.update({'opsd_stage/' + k: v for k, v in evaluation(m.STAGE_COUNTS).items()})
    return dict(actual_outer_updates=step, global_step=step, driver_pid=m.DRIVER_PID,
                job_id=m.JOB, source_commit=m.SOURCE, config_sha256=m.CONFIG_SHA, metrics=metrics)


def records(last, final=False):
    rows = [outer(s) for s in range(1, last + 1)]
    result = dict(outers=rows, outer_positions=[s * 4 for s in range(1, last + 1)],
                  evaluations=[evaluation(m.DEV_COUNTS)], evaluation_positions=[1],
                  final=[], final_positions=[], native_console=[])
    if last >= 5:
        result['evaluations'].append(evaluation(m.STAGE_COUNTS))
        result['evaluation_positions'].append(19)
    for step in (50, 100, 150, 200):
        if last >= step:
            result['evaluations'].extend([evaluation(m.DEV_COUNTS), evaluation(m.STAGE_COUNTS)])
            result['evaluation_positions'].extend([step * 4 - 2, step * 4 - 1])
    if final:
        result['evaluations'].append(evaluation(m.FINAL_COUNTS))
        result['evaluation_positions'].append(801)
        result['final'] = [copy.deepcopy(result['evaluations'][-1])]
        result['final_positions'] = [802]
    return result


class PrefixAndMemberTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.path = self.root / 'log'
        self.raw = b'line one\nline two\n'
        self.path.write_bytes(self.raw)
        self.pin = hashlib.sha256(self.raw).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_prefix_with_preexisting_append(self):
        self.path.write_bytes(self.raw + b'not a complete later line')
        raw, rec = m.prefix_file(self.path, self.root, len(self.raw), self.pin)
        self.assertEqual(raw, self.raw)
        self.assertGreater(rec['observed_file_size_at_close'], rec['bytes'])

    def test_append_during_read_is_not_false_member_instability(self):
        old = os.read
        calls = 0
        def read(fd, count):
            nonlocal calls
            value = old(fd, count)
            calls += 1
            if calls == 1:
                with self.path.open('ab') as out:
                    out.write(b'new tail\n')
            return value
        with patch.object(m.os, 'read', read):
            raw, rec = m.prefix_file(self.path, self.root, len(self.raw), self.pin)
        self.assertEqual(raw, self.raw)
        self.assertGreater(rec['observed_file_size_at_close'], rec['observed_file_size_at_open'])

    def test_prefix_rewrite_between_reads_rejected(self):
        old = os.read
        calls = 0
        def read(fd, count):
            nonlocal calls
            value = old(fd, count)
            calls += 1
            if calls == 1:
                self.path.write_bytes(b'X' + self.raw[1:])
            return value
        with patch.object(m.os, 'read', read), self.assertRaisesRegex(ValueError, 'prefix/identity'):
            m.prefix_file(self.path, self.root, len(self.raw), self.pin)

    def test_truncated_log_rejected(self):
        self.path.write_bytes(self.raw[:-1])
        with self.assertRaises(ValueError):
            m.prefix_file(self.path, self.root, len(self.raw), self.pin)

    def test_unterminated_prefix_rejected_even_correct_sha(self):
        raw = self.raw[:-1]
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            m.prefix_file(self.path, self.root, len(raw), hashlib.sha256(raw).hexdigest())

    def test_wrong_pin_rejected(self):
        with self.assertRaises(ValueError):
            m.prefix_file(self.path, self.root, len(self.raw), '0' * 64)

    def test_symlink_rejected(self):
        other = self.root / 'link'
        other.symlink_to(self.path)
        with self.assertRaises(ValueError):
            m.stable_file(other, self.root)

    def test_inode_replacement_during_prefix_read_rejected(self):
        old = os.read
        calls = 0
        def read(fd, count):
            nonlocal calls
            value = old(fd, count)
            calls += 1
            if calls == 1:
                replacement = self.root / 'replacement'
                replacement.write_bytes(self.raw)
                replacement.replace(self.path)
            return value
        with patch.object(m.os, 'read', read), self.assertRaises(ValueError):
            m.prefix_file(self.path, self.root, len(self.raw), self.pin)

    def test_stable_member_change_rejected_at_publication(self):
        record = m.stable_file(self.path, self.root)[0]
        self.path.write_bytes(b'X' + self.raw[1:])
        with self.assertRaisesRegex(ValueError, 'after streamed'):
            m.recheck_member(record, self.root)

    def test_new_checkpoint_hardlink_alias_rejected(self):
        other = self.root / 'other'
        os.link(self.path, other)
        recs = {'one': m.stable_file(self.path)[0], 'two': m.stable_file(other)[0]}
        with self.assertRaisesRegex(ValueError, 'inode alias'):
            m.member_unique(recs)

    def test_repeated_anchor_reference_is_legitimate_not_member_alias(self):
        ref = {'physical_path': str(self.path), 'sha256': self.pin}
        refs = {'source': ref, 'accepted_source': ref}
        self.assertEqual(refs['source'], refs['accepted_source'])
        # Anti-alias applies only to current checkpoint membership, not anchor refs.
        self.assertEqual(m.stable_file(ref['physical_path'], expected=ref['sha256'])[0]['sha256'], self.pin)

    def test_legitimate_empty_source_explicit_pin_only(self):
        self.path.write_bytes(b'')
        with self.assertRaises(ValueError):
            m.stable_file(self.path)
        self.assertEqual(m.stable_file(self.path, expected=hashlib.sha256(b'').hexdigest())[0]['bytes'], 0)

    def test_output_no_clobber(self):
        out = self.root / 'result.json'
        m.exclusive_json(out, {'scope': 'synthetic CPU boundary'})
        with self.assertRaises(ValueError):
            m.exclusive_json(out, {'scope': 'overwrite'})

    def test_json_duplicate_or_nonfinite_rejected(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                m.loads(raw)


class BudgetAndLogTests(unittest.TestCase):
    def test_all200_actual_outer_contract_not201(self):
        self.assertEqual(m.check_outer_rows(records(200)['outers'], CTX), 200)
        with self.assertRaisesRegex(ValueError, 'outer clock'):
            m.check_outer_rows(records(201)['outers'], CTX)

    def test_authoritative_driver_prefix_allows_aggregate_lag(self):
        result = m.join_log_prefixes(records(27), records(5), CTX, 25)
        self.assertTrue(result['aggregate_behind_checkpoint'])
        self.assertEqual(result['aggregate_forwarding_lag'], 22)

    def test_aggregate_future_or_conflict_rejected(self):
        with self.assertRaises(ValueError):
            m.join_log_prefixes(records(25), records(26), CTX, 25)
        a = records(5)
        a['outers'][0]['metrics']['critic/rewards/mean'] = .999
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            m.join_log_prefixes(records(25), a, CTX, 25)

    def test_missing_duplicate_or_reordered_outer_rejected(self):
        for rows in (records(25)['outers'][1:], records(25)['outers'] + [outer(25)],
                     [outer(2), outer(1)]):
            with self.assertRaises(ValueError):
                m.check_outer_rows(rows, CTX)

    def test_wrong_source_config_job_driver_rejected(self):
        for key, bad in (('source_commit', '8e'), ('config_sha256', '0' * 64),
                         ('job_id', '3210'), ('driver_pid', m.DRIVER_PID + 1)):
            rows = [outer(1)]
            rows[0][key] = bad
            with self.assertRaises(ValueError):
                m.check_outer_rows(rows, CTX)

    def test_wrong16_adam_micro_missingrank_and_teacher_clock_rejected(self):
        for key, bad in (('actor/optimizer_steps/rank_7', 20),
                         ('actor/samples/rank_3', 320),
                         ('teacher/cuda_update_entries_cumulative_per_rank/rank_4', 5),
                         ('actor/old_log_probs_finite/rank_2', 0),
                         ('actor/supervised_tokens/rank_0', 0)):
            rows = [outer(1)]
            rows[0]['metrics'][key] = bad
            with self.assertRaises((ValueError, RuntimeError)):
                m.check_outer_rows(rows, CTX)

    def test_nonfinite_and_zero_lr_rejected_negative_opsd_loss_allowed(self):
        row = outer(1)
        row['metrics']['actor/opsd_loss'] = -.2
        m.check_outer_rows([row], CTX)
        for key, bad in (('actor/lr', 0), ('actor/opsd_loss', float('nan')),
                         ('actor/grad_norm_mean/rank_0', float('inf'))):
            changed = copy.deepcopy(row)
            changed['metrics'][key] = bad
            with self.assertRaises(ValueError):
                m.check_outer_rows([changed], CTX)

    def test_actual_pinned_marker_parser_owner_and_suffix_rules(self):
        row = outer(1)
        good = '(main_task pid=1533312) OPSD_E1_OUTER_JSON=' + m.canonical(row).decode() + '\n'
        self.assertEqual(m.parse_records(good.encode(), CTX)['outers'], [row])
        for raw in (good.replace('pid=1533312', 'pid=1533313'), good.rstrip() + ' arbitrary\n',
                    good.rstrip() + ' OPSD_E1_OUTER_JSON={}\n'):
            with self.assertRaises(ValueError):
                m.parse_records(raw.encode(), CTX)

    def test_rank_transport_tag_then_owned_warning_exact_case(self):
        rank = {'pid': CTX['workers'][0]['pid'], 'rank': 0, 'step': 25}
        warning_path = Path('/verified/torch/utils/checkpoint.py')
        context = dict(CTX, checkpoint_warning_path=warning_path)
        prefix = '(WorkerDict pid=' + str(rank['pid']) + ') FULL_STATE_RANK_JSON='
        warning = ("(WorkerDict pid=" + str(rank['pid']) + ") " + str(warning_path)
                   + ":1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. "
                   + "Please use `torch.amp.autocast('cpu', args...)` instead.")
        tag = '\x1b[32m [repeated 7x across cluster]\x1b[0m'
        line = prefix + m.canonical(rank).decode() + tag + warning + '\n'
        parsed = m.parse_records(line.encode(), context)
        self.assertEqual(parsed['native_console'], [rank])
        self.assertEqual(parsed['native_transport'][0]['observed_repeat_tag'], 7)
        self.assertEqual(len(parsed['native_console']), 1)  # never expand repeated7
        for bad in (line.replace('repeated 7x', 'repeated 9x'),
                    line.replace(str(warning_path), '/unverified/checkpoint.py'),
                    line.replace('Please use', 'ARBITRARY use')):
            with self.assertRaises(ValueError):
                m.parse_records(bad.encode(), context)

    def test_frozen_gate_source_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            (repo / 'search_r1/baselines').mkdir(parents=True)
            (repo / 'search_r1/baselines/checkpoint_gate.py').write_text('print("drift")')
            with self.assertRaises(ValueError):
                m.frozen_gate(repo)

    def test_latest_contract_rejects_old_run_and_foreign_pins(self):
        args = SimpleNamespace(run_root=str(m.RUN), job_id=m.JOB, source_root=str(m.REPO),
            source_commit=m.SOURCE, accepted5_manifest_sha256=m.MANIFEST5_SHA,
            accepted5_receipt_sha256=m.RECEIPT5_SHA, public_key_sha256=m.PUBLIC_SHA,
            native_import_closure_sha256=m.CLOSURE_SHA)
        m.require_contract(args)
        for key, bad in (('job_id', '3210'), ('source_commit', '8e'),
                         ('public_key_sha256', '0' * 64), ('accepted5_receipt_sha256', '0' * 64)):
            changed = copy.deepcopy(args)
            setattr(changed, key, bad)
            with self.assertRaises(ValueError):
                m.require_contract(changed)

    def test_real_rsa_public_verification_and_tamper_rejection_no_signing(self):
        snap = HERE.parent / 'actual-snapshots/3261-20261005T210622178545Z/original-files'
        pub_raw = (snap / 'bindings/public.json').read_bytes()
        receipt_raw = (snap / 'acceptance-5-receipt.json').read_bytes()
        self.assertEqual(hashlib.sha256(pub_raw).hexdigest(), m.PUBLIC_SHA)
        self.assertEqual(hashlib.sha256(receipt_raw).hexdigest(), m.RECEIPT5_SHA)
        pub, receipt = m.loads(pub_raw), m.loads(receipt_raw)
        GATE.verify_rsa_signature(receipt['payload'], receipt['signature_hex'], pub)
        changed = copy.deepcopy(receipt['payload'])
        changed['horizon'] = 300
        with self.assertRaises(ValueError):
            GATE.verify_rsa_signature(changed, receipt['signature_hex'], pub)


class EvaluationTests(unittest.TestCase):
    def test_exact_final7_and_weighted_all(self):
        self.assertEqual(m.check_evaluation(evaluation(m.FINAL_COUNTS), m.FINAL_COUNTS)['count'], 51713)

    def test_correct_total_wrong_source_rejected(self):
        counts = dict(m.FINAL_COUNTS)
        counts['unknown'] = counts.pop('bamboogle')
        with self.assertRaisesRegex(ValueError, 'source/cardinality'):
            m.check_evaluation(evaluation(counts), m.FINAL_COUNTS)

    def test_correct_total_wrong_source_distribution_rejected(self):
        counts = dict(m.FINAL_COUNTS)
        counts['nq'] -= 1
        counts['hotpotqa'] += 1
        with self.assertRaises(ValueError):
            m.check_evaluation(evaluation(counts), m.FINAL_COUNTS)

    def test_wrong_weighted_all_and_nonfinite_or_out_of_range_rejected(self):
        for key, value in (('val/em/all', .9), ('val/f1/nq', 1.1),
                           ('val/generated_tokens/all', -1), ('val/test_score/hotpotqa', float('nan'))):
            metrics = evaluation(m.FINAL_COUNTS)
            metrics[key] = value
            with self.assertRaises(ValueError):
                m.check_evaluation(metrics, m.FINAL_COUNTS)

    def test_dev_stage_final_sequence_and_separate_scope(self):
        out = m.check_evaluation_sequence(records(200, final=True), 200, True)
        self.assertTrue(out['final_marker_present'])
        self.assertFalse(out['per_question_raw_evidence_available'])
        self.assertEqual(len(out['events']), 11)

    def test_checkpoint200_before_final_completion_is_not_final_pass(self):
        out = m.check_evaluation_sequence(records(200), 200)
        self.assertFalse(out['final_marker_present'])
        with self.assertRaises(ValueError):
            m.check_evaluation_sequence(records(200), 200, True)

    def test_missing_or_duplicate_stage_console_rejected(self):
        missing = records(25)
        missing['evaluations'].pop()
        missing['evaluation_positions'].pop()
        with self.assertRaises(ValueError):
            m.check_evaluation_sequence(missing, 25)
        duplicate = records(25)
        duplicate['evaluations'].append(copy.deepcopy(duplicate['evaluations'][1]))
        duplicate['evaluation_positions'].append(99)
        with self.assertRaises(ValueError):
            m.check_evaluation_sequence(duplicate, 25)

    def test_inflight_next_eval_is_accepted_before_next_outer_record(self):
        trace = records(49)
        trace['evaluations'].append(evaluation(m.DEV_COUNTS))
        trace['evaluation_positions'].append(198)
        self.assertEqual(len(m.check_evaluation_sequence(trace, 49)['events']), 3)

    def test_console_outer_stage_conflict_and_early_final_rejected(self):
        changed = records(25)
        changed['outers'][4]['metrics']['opsd_stage/val/em/all'] = .9
        with self.assertRaises(ValueError):
            m.check_evaluation_sequence(changed, 25)
        early = records(200, True)
        early['evaluation_positions'][-1] = 798
        with self.assertRaises(ValueError):
            m.check_evaluation_sequence(early, 200, True)


class CheckpointMetadataTests(unittest.TestCase):
    """Small fake PT/shards test only observer schema; never native content acceptance."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name).resolve()
        self.step = 25
        self.actor = self.run / 'checkpoints/actor/global_step_25'
        self.native = self.actor / 'training_state'
        self.native.mkdir(parents=True)
        self.driver = self.run / 'full_state/global_step_25'
        self.driver.mkdir(parents=True)
        self.algo = self.run / 'checkpoints/algorithm/global_step_25/state.json'
        self.algo.parent.mkdir(parents=True)
        self.config = m.loads((SNAP / 'full_state/global_step_5/resolved-config.json').read_bytes())
        self.workers = []
        self.hf_roles = {role for role in BASE['files']
                         if role.startswith('actor/') and not role.startswith('actor/training_state/')}
        for role in self.hf_roles:
            path = self.actor / role.removeprefix('actor/')
            original = SNAP / 'checkpoints/actor/global_step_5' / path.name
            path.write_bytes(original.read_bytes() if path.name == 'model.safetensors.index.json'
                             else b'SYNTHETIC_OBSERVER_ONLY_' + path.name.encode())
        for rank in range(8):
            report = m.loads((SNAP / f'checkpoints/actor/global_step_5/training_state/rank_{rank}.json').read_bytes())
            self.workers.append(dict(rank=rank, pid=report['pid'], object_ids=report['object_ids']))
            pt = self.native / f'rank_{rank}.pt'
            pt.write_bytes(b'SYNTHETIC_NOT_NATIVE_PT_' + str(rank).encode())
            pin = hashlib.sha256(pt.read_bytes()).hexdigest()
            report.update(status='FULL_STATE_SERIALIZED_OK', step=25, scheduler_last_epoch=25,
                          path=str(pt), sha256=pin, state_sha256=pin)
            report['teacher']['counters']['cuda_update_entries'] = 25
            (self.native / f'rank_{rank}.json').write_bytes(m.canonical(report))
        pt = self.driver / 'driver.pt'
        pt.write_bytes(b'SYNTHETIC_NOT_NATIVE_DRIVER_PT')
        report = dict(status='DRIVER_STATE_SERIALIZED_OK', step=25, pid=m.DRIVER_PID,
                      uid=os.getuid(), job_id=m.JOB, digest='1' * 64,
                      state_sha256=hashlib.sha256(pt.read_bytes()).hexdigest())
        (self.driver / 'driver.json').write_bytes(m.canonical(report))
        algorithm = dict(version=1, method='pure_opsd', global_step=25,
                         distillation=self.config['baselines']['distillation'],
                         resume_scope='same_live_objects_same_topology', outcome_rewards_train_policy=False)
        self.algo.write_bytes(m.canonical(algorithm))
        self.context = dict(run=self.run, workers=self.workers, config=self.config)

    def tearDown(self):
        self.tmp.cleanup()

    def members(self):
        return m.checkpoint_files(self.run, self.step, self.hf_roles)

    def update_json(self, path, field, value):
        data = m.loads(path.read_bytes())
        data[field] = value
        path.write_bytes(m.canonical(data))

    def test_metadata_valid_scope_and_all_later_steps_accepted(self):
        files = self.members()
        self.assertEqual(len(files), 31)
        ranks, driver = m.checkpoint_reports(files, 25, self.context)
        self.assertEqual([r['status'] for r in ranks], ['FULL_STATE_SERIALIZED_OK'] * 8)
        self.assertEqual(driver['status'], 'DRIVER_STATE_SERIALIZED_OK')
        # This only validates metadata; byte placeholders cannot pass native PT load.
        self.assertEqual(set(m.STEPS), {25, 50, 75, 100, 125, 150, 175, 200})

    def test_missing_rank_pt_rejected(self):
        (self.native / 'rank_7.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'native rank'):
            self.members()

    def test_extra_rank_and_fake_runtime_manifest_rejected(self):
        extra = self.native / 'rank_8.json'
        extra.write_text('{}')
        with self.assertRaises(ValueError):
            self.members()
        extra.unlink()
        (self.driver / 'manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'fabricated manifest'):
            self.members()

    def test_unknown_hf_member_or_missing_shard_rejected(self):
        extra = self.actor / 'unknown.pkl'
        extra.write_bytes(b'x')
        with self.assertRaises(ValueError):
            self.members()
        extra.unlink()
        (self.actor / 'model-00003-of-00003.safetensors').unlink()
        with self.assertRaises(ValueError):
            self.members()

    def test_index_path_escape_or_wrong_key_cardinality_rejected(self):
        indexpath = self.actor / 'model.safetensors.index.json'
        index = m.loads(indexpath.read_bytes())
        original = copy.deepcopy(index)
        first = next(iter(index['weight_map']))
        index['weight_map'][first] = '../evil.safetensors'
        indexpath.write_bytes(m.canonical(index))
        with self.assertRaises(ValueError):
            self.members()
        original['weight_map'].pop(first)
        indexpath.write_bytes(m.canonical(original))
        with self.assertRaisesRegex(ValueError, 'HF435'):
            self.members()

    def test_current_checkpoint_inode_alias_rejected(self):
        target = self.native / 'rank_1.pt'
        target.unlink()
        os.link(self.native / 'rank_0.pt', target)
        with self.assertRaisesRegex(ValueError, 'inode alias'):
            self.members()

    def test_new_report_step_scheduler_and_status_upgrade_rejected(self):
        path = self.native / 'rank_0.json'
        original = path.read_bytes()
        for field, value in (('step', 5), ('scheduler_last_epoch', 24),
                             ('status', 'EXACT_FULL_STATE_RELOAD_OK')):
            path.write_bytes(original)
            self.update_json(path, field, value)
            with self.assertRaisesRegex(ValueError, 'rank status/clock'):
                m.checkpoint_reports(self.members(), 25, self.context)

    def test_original_worker_pid_object_and_full_file_pin_rejected(self):
        path = self.native / 'rank_0.json'
        original = path.read_bytes()
        for field, value in (('pid', m.DRIVER_PID), ('object_ids', {'model': 1}),
                             ('state_sha256', '0' * 64)):
            path.write_bytes(original)
            self.update_json(path, field, value)
            with self.assertRaisesRegex(ValueError, 'same-object/file'):
                m.checkpoint_reports(self.members(), 25, self.context)

    def test_driver_status_step_and_pid_rejected(self):
        path = self.driver / 'driver.json'
        original = path.read_bytes()
        for field, value in (('step', 200), ('status', 'EXACT_DRIVER_STATE_RELOAD_OK'),
                             ('pid', m.DRIVER_PID + 1)):
            path.write_bytes(original)
            self.update_json(path, field, value)
            with self.assertRaisesRegex(ValueError, 'driver status/clock'):
                m.checkpoint_reports(self.members(), 25, self.context)

    def test_algorithm_policy_reward_or_distillation_change_rejected(self):
        self.update_json(self.algo, 'outcome_rewards_train_policy', True)
        with self.assertRaisesRegex(ValueError, 'algorithm/config'):
            m.checkpoint_reports(self.members(), 25, self.context)

    def test_step5_oldbudget_or201_rejected(self):
        for step in (5, 26, 201, 300, 500, True):
            with self.assertRaises(ValueError):
                m.checkpoint_files(self.run, step)


if __name__ == '__main__':
    unittest.main()
