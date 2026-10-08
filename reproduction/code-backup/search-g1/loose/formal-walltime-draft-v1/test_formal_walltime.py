"""UNEXECUTED offline contract/observer tests; new exact static PASS first.

Synthetic clocks, native strings, capture summaries and stand-in command
transport only. No real scheduler, RPC, /proc or trainer is contacted.
"""
import base64
import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import formal_walltime_contract as c
import formal_gpu_stream as stream
import observe_formal_same_job as observer


def native(duration=172800, state='RUNNING'):
    fields = {name: 'unchanged' for name in c.BIRTH_FIELDS}
    fields.update(JobId=c.JOB, SubmitTime='2026-10-06T21:50:19', StartTime='2026-10-06T21:50:20',
        TimeLimit=('%d-%02d:%02d:%02d' % (duration // 86400, duration // 3600 % 24, duration // 60 % 60, duration % 60)),
        EndTime=c.expected_end('2026-10-06T21:50:20', duration), JobState=state, ExitCode='0:0',
        Restarts='0', UserId='root(0)', NodeList='h100-8', BatchHost='h100-8', Partition='gpu',
        ReqNodeList='h100-8', NumNodes='1', MinMemoryNode='512G', TresPerNode='gres:gpu:h100:8',
        NumCPUs='64', NumTasks='1', **{'CPUs/Task': '64', 'TRES': 'cpu=64,mem=512G,gres/gpu=8'})
    return ' '.join(key + '=' + value for key, value in fields.items())


def anchor():
    return dict(job_id=c.JOB, run_id=c.RUN, base=c.BASE, original_native_fields=c.native_fields(native()),
        original_intent_sha256=c.INTENT_SHA, original_launch_sha256=c.LAUNCH_SHA, original_birth_sha256=c.BIRTH_SHA,
        original_allocation_sha256=c.ALLOCATION_SHA, original_claim_sha256=c.CLAIM_SHA,
        original_disposition_sha256=c.DISPOSITION_SHA, source_commit=c.COMMIT, source_count=372,
        source_map_sha256=c.MAP_SHA, method_sha256=c.METHOD_SHA, original_submitted_unix=1791294619.6805463,
        original_deadline_unix=1791294679.6805463)


def capture(when=10, duration=172800, prior=None, append=False):
    births = dict(trainer=c.TRAINER, sampler=c.SAMPLER)
    births.update({'rank%d' % rank: dict(pid=100 + rank, start_ticks=200 + rank, sid=c.TRAINER['sid'], uid=0)
                   for rank in range(8)})
    checks = {name: dict(saved_identity=value, actual_identity=value, evidence_state='observed',
        birth_matches=True, namespace_matches=True, process_state='S') for name, value in births.items()}
    prefixes = {name: dict(path=c.BASE + '/logs/' + name, dev=1, inode=index + 1, owner_uid=0, mode=33152,
        captured_bytes=3, complete_prefix_bytes=3, complete_prefix_sha256=c.sha(b'{}\n'), complete_rows=1)
        for index, name in enumerate(c.PREFIX_NAMES)}
    rechecks = {name: dict(prefix_bytes=row['complete_prefix_bytes'], sha256=row['complete_prefix_sha256'])
                for name, row in prior['prefixes'].items()} if prior else {}
    if append:
        prefixes['actual-attempts.jsonl'].update(captured_bytes=6, complete_prefix_bytes=6,
                                               complete_prefix_sha256=c.sha(b'{}\n{}\n'), complete_rows=2)
    return dict(role='same_job_readonly_capture', job_id=c.JOB, run_id=c.RUN, base=c.BASE,
        job_stdout=native(duration), checked_unix=when, observation_finished_unix=when + 1,
        source_hashes_match=True, source_count=372, source_commit=c.COMMIT, source_map_sha256=c.MAP_SHA,
        method_sha256=c.METHOD_SHA, allocation_sha256=c.ALLOCATION_SHA, original_claim_sha256=c.CLAIM_SHA,
        original_disposition_sha256=c.DISPOSITION_SHA, original_actual_launch_verified=True,
        process_births=births, current_birth_checks=checks, prefixes=prefixes, anchor_rechecks=rechecks)


def envelope(value):
    raw = c.canonical(value) + b'\n'
    return dict(raw_b64=base64.b64encode(raw).decode(), sha256=c.sha(raw))


def empty_chain():
    return dict(schema_version=1, role='same_job_exact_walltime_chain', links=[], unresolved_entered=[])


def chain_fixture(new_duration=259200):
    original = anchor(); planning = capture(10); before = capture(20, prior=planning)
    after = capture(40, new_duration, prior=before); continued = capture(50, new_duration, prior=after, append=True)
    new_slurm = '%d-00:00:00' % (new_duration // 86400)
    rows = (new_duration + 21600) // 2 + 1
    intent = dict(schema_version=1, role='prospective_same_job_walltime_intent', sequence=1,
        job_id=c.JOB, run_id=c.RUN, base=c.BASE, uid=0,
        numeric_value_attribution='agent_selected_necessary_implementation_choice', active_goal_reference='synthetic offline goal',
        original_anchor_sha256=c.sha(c.canonical(original)), previous_verified_chain_sha256=c.sha(c.canonical(original)),
        planning_before_capture_sha256=envelope(planning)['sha256'], old_time_limit_seconds=172800,
        old_time_limit_slurm='2-00:00:00', new_time_limit_seconds=new_duration, new_time_limit_slurm=new_slurm,
        old_expected_end_time=c.expected_end(original['original_native_fields']['StartTime'], 172800),
        new_expected_end_time=c.expected_end(original['original_native_fields']['StartTime'], new_duration),
        expected_update_argv=['scontrol', 'update', 'JobId=3309', 'TimeLimit=' + new_slurm],
        created_unix=12, expires_unix=100, reviewed_file_sha256={'synthetic.py': 'a' * 64},
        evidence_budget={'scope': 'synthetic offline cycle evidence'},
        final_phase_budget={'scope': 'synthetic offline estimated final phase'}, uncertainty_reserve_seconds=3600,
        status='FROZEN_EXACT_INTENT',
        capture_policy=dict(total_walltime_seconds=new_duration, capture_allowance_seconds=21600, max_rows=rows,
            max_bytes=rows * 4097 + 4096, max_line_bytes=4096, max_tail_bytes=4096, chunk_bytes=65536,
            read_seconds=120 if new_duration == 691200 else 90,
            summary_bytes=2097152))
    review = dict(approved=True, scope='staticonly', fork_turns='none', draft_code_executed=False,
        blocking_findings=[], intent_sha256=envelope(intent)['sha256'], external_file_sha256=intent['reviewed_file_sha256'])
    claim = dict(state='CLAIMED', intent_sha256=envelope(intent)['sha256'], review_sha256=envelope(review)['sha256'],
        previous_verified_chain_sha256=intent['previous_verified_chain_sha256'], created_unix=15,
        expected_update_argv=intent['expected_update_argv'])
    entered = dict(state='UPDATE_ENTERED', claim_sha256=envelope(claim)['sha256'], before_sha256=envelope(before)['sha256'],
        intent_sha256=envelope(intent)['sha256'], review_sha256=envelope(review)['sha256'],
        argv=intent['expected_update_argv'], entered_unix=30, deadline_unix=90)
    command = dict(state='COMMAND_RETURNED_OR_UNKNOWN', disposition_sha256=envelope(entered)['sha256'],
        argv=entered['argv'], call_count=1, returncode=0, outcome='returned', observed_unix=35)
    bundle = {name: envelope(value) for name, value in dict(intent=intent, planning_before=planning, review=review,
        claim=claim, before=before, disposition=entered, command=command, after=after, continued=continued).items()}
    return original, dict(empty_chain(), links=[bundle])


class WalltimeContract(unittest.TestCase):
    def test_original_only_exact_value_and_no_anchor_mutation(self):
        original = anchor(); frozen = c.canonical(original)
        tip = c.validate_chain(original, empty_chain())
        self.assertEqual(tip['time_limit_seconds'], 172800)
        self.assertEqual(c.current_native(native(), original, 172800)['JobId'], '3309')
        self.assertEqual(c.canonical(original), frozen)
        with self.assertRaises(ValueError):
            c.current_native(native(259200), original, 172800)
        with self.assertRaises(ValueError):
            c.original_anchor(b'{}', b'{}', b'{}')

    def test_exact_chain_and_unknown_single_call_reconciliation(self):
        original, chain = chain_fixture()
        self.assertEqual(c.validate_chain(original, chain)['time_limit_seconds'], 259200)
        command, _ = c.record(chain['links'][0], 'command')
        command.update(returncode=None, outcome='unknown_after_entered')
        chain['links'][0]['command'] = envelope(command)
        self.assertEqual(c.validate_chain(original, chain)['time_limit_seconds'], 259200)
        command['call_count'] = 2; chain['links'][0]['command'] = envelope(command)
        with self.assertRaises(ValueError):
            c.validate_chain(original, chain)

    def test_identity_restart_time_source_and_actual_birth_fail_closed(self):
        original = anchor()
        for token, changed in (('UserId=root(0)', 'UserId=root(1)'), ('Restarts=0', 'Restarts=1'),
            ('NodeList=h100-8', 'NodeList=h100-9'), ('NumCPUs=64', 'NumCPUs=63'),
            ('StartTime=2026-10-06T21:50:20', 'StartTime=2026-10-06T21:50:21')):
            with self.assertRaises(ValueError):
                c.current_native(native().replace(token, changed), original, 172800)
        for edit in ('method', 'birth', 'allocation'):
            value = capture()
            if edit == 'method': value['method_sha256'] = 'a' * 64
            if edit == 'allocation': value['allocation_sha256'] = 'b' * 64
            if edit == 'birth': value['current_birth_checks']['rank0']['birth_matches'] = False
            with self.assertRaises(ValueError): c.capture_identity(value, original, 172800)

    def test_prefix_growth_tail_completion_rewrite_replacement_and_no_append(self):
        before = capture(); after = capture(20, prior=before, append=True)
        self.assertTrue(c.continuity(before, after, require_append=True))
        for key, changed in (('inode', 999), ('complete_prefix_bytes', 0), ('owner_uid', 1)):
            invalid = copy.deepcopy(after); invalid['prefixes']['actual-attempts.jsonl'][key] = changed
            with self.assertRaises(ValueError): c.continuity(before, invalid)
        invalid = copy.deepcopy(after); invalid['anchor_rechecks']['actual-attempts.jsonl']['sha256'] = 'a' * 64
        with self.assertRaises(ValueError): c.continuity(before, invalid)
        with self.assertRaises(ValueError): c.continuity(before, capture(20, prior=before), require_append=True)

    def test_unresolved_forked_unreviewed_wrong_value_and_unbound_raw(self):
        original, chain = chain_fixture()
        for transform in ('unresolved', 'fork', 'review', 'raw', 'newvalue'):
            value = copy.deepcopy(chain)
            if transform == 'unresolved': value['unresolved_entered'] = ['entered-without-after']
            if transform == 'fork': value['links'].append(copy.deepcopy(value['links'][0]))
            if transform == 'review':
                row, _ = c.record(value['links'][0], 'review'); row['fork_turns'] = 'all'
                value['links'][0]['review'] = envelope(row)
            if transform == 'raw': value['links'][0]['before']['sha256'] = 'a' * 64
            if transform == 'newvalue':
                row, _ = c.record(value['links'][0], 'intent'); row['new_time_limit_seconds'] = 300000
                value['links'][0]['intent'] = envelope(row)
            with self.assertRaises((ValueError, KeyError)): c.validate_chain(original, value)

    def test_strict_json_duplicate_nonfinite_bool_duration_and_terminal_not_predicted_end(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'):
            with self.assertRaises(ValueError): c.strict_json(raw)
        with self.assertRaises(ValueError): c.current_native(native(), anchor(), True)
        terminal = native(state='COMPLETED').replace('EndTime=2026-10-08T21:50:20', 'EndTime=2026-10-08T20:50:20')
        self.assertEqual(c.current_native(terminal, anchor(), 172800)['JobState'], 'COMPLETED')
        with self.assertRaises(ValueError): c.current_native(terminal.replace('ExitCode=0:0', 'ExitCode=1:0'), anchor(), 172800)


class ReadonlyReachability(unittest.TestCase):
    def test_original_and_exact192h_candidate_policy_selection(self):
        original = dict(total_walltime_seconds=172800, capture_allowance_seconds=21600, max_rows=97201,
            max_bytes=398236593, max_line_bytes=4096, max_tail_bytes=4096, chunk_bytes=65536, read_seconds=90, summary_bytes=2097152)
        candidate = dict(original, total_walltime_seconds=691200, max_rows=356401, max_bytes=1460178993, read_seconds=120)
        document = dict(schema_version=1, role='reviewed_exact_known_job_reader_policies', scope='synthetic fixture',
                        policies={'172800': original, '691200': candidate})
        self.assertEqual(observer.select_policy(document, c.validate_chain(anchor(), empty_chain())), original)
        frozen_anchor, chain = chain_fixture(691200)
        exact = c.validate_chain(frozen_anchor, chain)
        self.assertEqual(c.current_native(native(691200), frozen_anchor, exact['time_limit_seconds'])['EndTime'],
                         '2026-10-14T21:50:20')
        self.assertEqual(observer.select_policy(document, exact), candidate)
        self.assertEqual(stream.validate_policy(candidate, 691200), candidate)
        for invalid in (dict(exact, link_count=0), dict(exact, capture_policy=None), dict(exact, time_limit_seconds=691201)):
            with self.assertRaises(ValueError): observer.select_policy(document, invalid)

    def test_fixed_read_command_allowlist_and_no_mutation(self):
        calls = []
        def capture(argv, timeout, out_cap, err_cap):
            calls.append(argv); return SimpleNamespace(returncode=0, stdout=native(), stderr='')
        protocol = SimpleNamespace(bounded_capture=capture)
        observer.read_command(protocol, ['scontrol', 'show', 'job', '-o', '3309'])
        for argv in (['scontrol', 'update', 'JobId=3309', 'TimeLimit=3-00:00:00'], ['sbatch'],
                     ['scontrol', 'show', 'job', '-o', '3310'], ['scancel', '3309']):
            with self.assertRaises(ValueError): observer.read_command(protocol, argv)
        self.assertEqual(len(calls), 1)

    def test_purge_early_return_does_not_infer_terminal_or_read_other_components(self):
        calls = []
        def capture(argv, timeout, out_cap, err_cap):
            calls.append(argv); return SimpleNamespace(returncode=1, stdout='', stderr='invalid job id')
        protocol = SimpleNamespace(bounded_capture=capture)
        selected = dict(total_walltime_seconds=172800, capture_allowance_seconds=21600, max_rows=97201,
            max_bytes=398236593, max_line_bytes=4096, max_tail_bytes=4096, chunk_bytes=65536,
            read_seconds=90, summary_bytes=2097152)
        payload = dict(base=c.BASE, run_id=c.RUN, job_id=c.JOB, policy=selected, chain=empty_chain(),
                       intent_b64='e30=', launch_b64='e30=', birth_b64='e30=')
        with (mock.patch.object(c, 'original_anchor', return_value=(anchor(), {}, {}, {})),
              mock.patch.object(observer.os, 'getuid', return_value=0)):
            result = observer.observe_remote(payload, protocol, SimpleNamespace(), c, stream, SimpleNamespace())
        self.assertEqual(result['capture_state'], 'unknown_native_missing_or_purged')
        self.assertFalse(result['native_terminal_authenticated'])
        self.assertEqual(result['terminal_success'], 'not_inferred')
        self.assertEqual(calls, [['scontrol', 'show', 'job', '-o', '3309']])


if __name__ == '__main__':
    unittest.main()
