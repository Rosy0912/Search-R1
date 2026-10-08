"""TEXT-ONLY, UNEXECUTED: pure exact-known-job and amendment validators.

No scheduler/transport/entry point. Original receipts remain separate bytes.
An accepted chain is observation provenance, never permission to execute update.
"""
import base64
import datetime
import hashlib
import json
import math
import re

BASE = '/mnt/data/shaoshitong/g1f-3ff5337a'
RUN = 'db5235708c354bffae6d0120c1cb5459'
JOB = '3309'
COMMIT = 'dec4e52b67042373348ef522ddc611d337eadf79'
MAP_SHA = '63c97052a028b4edafb4d2c4f5140f6aef40e0b0c835dd4fdcda04acb1cc5372'
METHOD_SHA = '5508dbc85e33e7de2dcf1099a6597ef8bceffb0bbec6d31243c22614051a2314'
ALLOCATION_SHA = '8c129513c8535f95616817b5da9b7a1a933532562d7f3bd6a6ba90d6e8adaa41'
INTENT_SHA = 'a9ccca418a9df7922730d96a19bcb239afb13590a8f8418bbce4e9773c040a36'
LAUNCH_SHA = 'ba9b0962cabf2ac322d1386a7e7ad32531770d46f67afe9502f93f00b4088a4e'
BIRTH_SHA = 'f05002ebcdd0a4410ffd1d495e7740d7587cc14e9d6de6af03fb6e43600dfe9d'
CLAIM_SHA = '3e6ee57795c5cbca501651769df2a798d101791c7af7e64b879287414fbcf0d8'
DISPOSITION_SHA = 'ec62d4f9761faee3f45ceff9be7bf4efb89589806f3ea3bb8b74de32b5ae1735'
TRAINER = dict(pid=2864716, start_ticks=2224036912, sid=2864716, uid=0)
SAMPLER = dict(pid=2836860, start_ticks=2223930773, sid=2836822, uid=0)
BIRTH_FIELDS = ('JobId', 'SubmitTime', 'StartTime', 'NodeList', 'BatchHost',
    'Command', 'WorkDir', 'StdOut', 'StdErr', 'Restarts', 'UserId', 'Partition',
    'ReqNodeList', 'NumNodes', 'MinMemoryNode', 'TresPerNode', 'JobName', 'TimeLimit')
RESOURCE_FIELDS = ('NumCPUs', 'NumTasks', 'CPUs/Task', 'TRES', 'AllocTRES',
                   'SuspendTime', 'SecsPreSuspend')
PREFIX_NAMES = ('actual-attempts.jsonl', 'actual-updates.jsonl') + tuple(
    'scheduler-rank-%d.jsonl' % i for i in range(8)) + ('gpu-samples-' + RUN + '.jsonl',)
INTENT_FIELDS = {'schema_version', 'role', 'sequence', 'job_id', 'run_id', 'base', 'uid',
    'numeric_value_attribution', 'active_goal_reference', 'original_anchor_sha256',
    'previous_verified_chain_sha256', 'planning_before_capture_sha256', 'old_time_limit_seconds',
    'old_time_limit_slurm', 'old_expected_end_time', 'new_time_limit_seconds', 'new_time_limit_slurm',
    'new_expected_end_time', 'expected_update_argv', 'created_unix', 'expires_unix',
    'reviewed_file_sha256', 'capture_policy', 'evidence_budget', 'final_phase_budget',
    'uncertainty_reserve_seconds', 'status'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    need(type(raw) is bytes, 'Raw bytes required')
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def strict_json(raw):
    need(type(raw) is bytes and len(raw) <= 2 * 1024 * 1024, 'Bounded JSON bytes required')
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    def bad(value):
        raise ValueError('Nonfinite JSON')
    def finite_float(value):
        number = float(value)
        need(math.isfinite(number), 'Nonfinite exponent')
        return number
    out = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad, parse_float=finite_float)
    canonical(out)
    return out


def digest(value):
    need(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None, 'SHA256 required')
    return value


def integer(value, minimum=0):
    need(type(value) is int and value >= minimum, 'Exact finite integer required')
    return value


def clock(value):
    need(type(value) in (int, float) and math.isfinite(value) and value > 0, 'Finite positive clock required')
    return value


def seconds(value):
    need(type(value) is str, 'Exact Slurm walltime string required')
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d\d):(\d\d)', value)
    need(match is not None, 'Finite Slurm walltime required')
    day, hour, minute, second = match.groups()
    need(int(minute) < 60 and int(second) < 60, 'Invalid walltime')
    return int(day or 0) * 86400 + int(hour) * 3600 + int(minute) * 60 + int(second)


def native_fields(raw):
    need(type(raw) is str and len(raw.encode()) <= 65536 and len(raw.strip().splitlines()) == 1,
         'One bounded actual native record required')
    out = {}
    for token in raw.split():
        need('=' in token, 'Invalid native token')
        key, value = token.split('=', 1)
        need(key and key not in out, 'Duplicate native field')
        out[key] = value
    need(set(BIRTH_FIELDS) | {'JobState', 'ExitCode', 'EndTime'} <= set(out), 'Missing native fields')
    return out


def native_time(value):
    need(type(value) is str and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d', value) is not None,
         'Actual scheduler timestamp required')
    return datetime.datetime.fromisoformat(value)


def expected_end(start, duration):
    return (native_time(start) + datetime.timedelta(seconds=integer(duration, 1))).isoformat()


def original_anchor(intent_raw, launch_raw, birth_raw):
    for raw, expected in ((intent_raw, INTENT_SHA), (launch_raw, LAUNCH_SHA), (birth_raw, BIRTH_SHA)):
        need(sha(raw) == expected, 'Original anchor bytes differ')
    intent, launch, birth = map(strict_json, (intent_raw, launch_raw, birth_raw))
    born = native_fields(birth['job_stdout'])
    need(intent['base'] == launch['base'] == birth['base'] == BASE and launch['run_id'] == RUN and
         launch['job_id'] == born['JobId'] == JOB and launch['uid'] == 0 and
         launch['source_commit'] == intent['source_commit'] == COMMIT and
         launch['source_sha256'] == intent['source_sha256'] and launch['source_count'] == intent['source_count'] == 372 and
         launch['source_map_sha256'] == intent['source_map_sha256'] == MAP_SHA and
         launch['method_sha256'] == intent['method_sha256'] == METHOD_SHA and
         launch['claim_sha256'] == CLAIM_SHA and launch['disposition_sha256'] == DISPOSITION_SHA and
         launch['submitted_unix'] == 1791294619.6805463 and
         launch['deadline_unix'] == 1791294679.6805463,
         'Original same-run admission identity differs')
    need(born['SubmitTime'] == '2026-10-06T21:50:19' and born['StartTime'] == '2026-10-06T21:50:20' and
         born['JobState'] == 'RUNNING' and born['ExitCode'] == '0:0' and born['Restarts'] == '0' and
         seconds(born['TimeLimit']) == 172800 and born['EndTime'] == expected_end(born['StartTime'], 172800),
         'Original actual48h birth differs')
    anchor = dict(job_id=JOB, run_id=RUN, base=BASE, original_intent_sha256=INTENT_SHA,
        original_launch_sha256=LAUNCH_SHA, original_birth_sha256=BIRTH_SHA,
        original_allocation_sha256=ALLOCATION_SHA, original_claim_sha256=CLAIM_SHA,
        original_disposition_sha256=DISPOSITION_SHA, original_native_fields=born,
        source_commit=COMMIT, source_count=372, source_map_sha256=MAP_SHA, method_sha256=METHOD_SHA,
        original_submitted_unix=launch['submitted_unix'], original_deadline_unix=launch['deadline_unix'])
    return anchor, intent, launch, birth


def current_native(raw, anchor, duration, *, state=None):
    fields = native_fields(raw)
    born = anchor['original_native_fields']
    need(all(fields[name] == born[name] for name in BIRTH_FIELDS if name != 'TimeLimit'),
         'Native non-time birth identity changed')
    need(fields['JobId'] == JOB and fields['Restarts'] == '0' and
         seconds(fields['TimeLimit']) == integer(duration, 1), 'Unexplained current walltime/restart')
    need(all(fields.get(key) == value for key, value in born.items() if key in RESOURCE_FIELDS),
         'Original resource or suspension facts changed')
    if state is not None:
        need(fields['JobState'] == state, 'Unexpected actual native state')
    if fields['JobState'] == 'RUNNING':
        need(fields['ExitCode'] == '0:0' and fields['EndTime'] == expected_end(fields['StartTime'], duration),
             'Running exact walltime/end mismatch')
    elif fields['JobState'] == 'COMPLETED':
        need(fields['ExitCode'] == '0:0' and native_time(fields['SubmitTime']) <= native_time(fields['StartTime']) <=
             native_time(fields['EndTime']), 'Actual terminal chronology/exit mismatch')
        need(native_time(fields['EndTime']) <= native_time(fields['StartTime']) + datetime.timedelta(seconds=duration),
             'Terminal exceeds bound; overtime/suspension requires separately reviewed contract')
    return fields


def capture_identity(capture, anchor, duration):
    need(type(capture) is dict and capture.get('role') == 'same_job_readonly_capture' and
         capture.get('job_id') == JOB and capture.get('run_id') == RUN and capture.get('base') == BASE,
         'Wrong capture handle')
    fields = current_native(capture['job_stdout'], anchor, duration)
    need(capture['source_hashes_match'] is True and capture['source_count'] == 372 and
         capture['source_map_sha256'] == MAP_SHA and capture['source_commit'] == COMMIT and
         capture['method_sha256'] == METHOD_SHA and capture['allocation_sha256'] == ALLOCATION_SHA and
         capture['original_actual_launch_verified'] is True and
         capture['original_claim_sha256'] == CLAIM_SHA and capture['original_disposition_sha256'] == DISPOSITION_SHA,
         'Capture immutable/source admission differs')
    if fields['JobState'] == 'RUNNING':
        names = {'trainer', 'sampler'} | {'rank%d' % i for i in range(8)}
        need(set(capture['process_births']) == set(capture['current_birth_checks']) == names and
             capture['process_births']['trainer'] == TRAINER and capture['process_births']['sampler'] == SAMPLER,
             'Actual original producer set differs')
        for name in names:
            identity = capture['process_births'][name]; checked = capture['current_birth_checks'][name]
            need(type(identity) is dict and set(identity) == {'pid', 'start_ticks', 'sid', 'uid'} and
                 all(type(identity[key]) is int and identity[key] > 0 for key in ('pid', 'start_ticks', 'sid')) and
                 type(identity['uid']) is int and identity['uid'] == 0 and
                 checked.get('evidence_state') == 'observed' and checked.get('birth_matches') is True and
                 checked.get('namespace_matches') is True and checked.get('process_state') != 'Z' and
                 checked.get('saved_identity') == checked.get('actual_identity') == identity,
                 'Actual original process birth/namespace not verified')
        need(len({(capture['process_births'][name]['pid'], capture['process_births'][name]['start_ticks'])
                  for name in names if name.startswith('rank')}) == 8, 'Rank producer births duplicate')
    clock(capture['checked_unix']); clock(capture['observation_finished_unix'])
    need(capture['checked_unix'] <= capture['observation_finished_unix'], 'Capture chronology differs')
    return capture


def continuity(before, after, *, require_append=False):
    need(before['observation_finished_unix'] <= after['checked_unix'], 'Continuity acquisition reordered')
    before_native, after_native = map(native_fields, (before['job_stdout'], after['job_stdout']))
    need(all(before_native.get(name) == after_native.get(name) for name in RESOURCE_FIELDS),
         'Same-job current resource facts changed')
    need(set(before['prefixes']) == set(after['prefixes']) == set(PREFIX_NAMES), 'All append anchors required')
    need(set(before['process_births']) == set(after['process_births']) ==
         {'trainer', 'sampler'} | {'rank%d' % i for i in range(8)}, 'All admitted producer births required')
    need(before['process_births'] == after['process_births'], 'Actual producer birth changed')
    progressed = False
    for name, old in before['prefixes'].items():
        new = after['prefixes'][name]
        need(all(new[key] == old[key] for key in ('path', 'dev', 'inode', 'owner_uid', 'mode')) and
             new['captured_bytes'] >= old['complete_prefix_bytes'] and
             new['complete_prefix_bytes'] >= old['complete_prefix_bytes'], 'Append inode/owner/extent changed')
        old_hash = digest(old['complete_prefix_sha256'])
        proof = after['anchor_rechecks'][name]
        need(proof == dict(prefix_bytes=old['complete_prefix_bytes'], sha256=old_hash),
             'Old complete prefix bytes were not rehashed exactly')
        if name != 'gpu-samples-' + RUN + '.jsonl':
            progressed |= new['complete_prefix_bytes'] > old['complete_prefix_bytes']
    need(not require_append or progressed, 'Later complete driver/rank append not observed')
    return True


def record(bundle, name):
    item = bundle[name]
    need(type(item) is dict and set(item) == {'raw_b64', 'sha256'}, 'Exact raw record envelope required')
    raw = base64.b64decode(item['raw_b64'], validate=True)
    need(sha(raw) == digest(item['sha256']), 'Record raw SHA differs')
    return strict_json(raw), item['sha256']


def validate_chain(anchor, chain):
    """Complete exact one-call records only; unresolved entered states fail closed.

    No command result can supply scheduler facts. Recovery after unknown is
    accepted only with the same durable entered reservation and actual captures.
    """
    need(type(chain) is dict and set(chain) == {'schema_version', 'role', 'links', 'unresolved_entered'} and
         chain['schema_version'] == 1 and type(chain['schema_version']) is int and
         chain['role'] == 'same_job_exact_walltime_chain' and chain['unresolved_entered'] == [] and
         type(chain['links']) is list and len(chain['links']) <= 8, 'Unresolved/unbounded amendment chain')
    tip, previous, previous_finished = 172800, sha(canonical(anchor)), 0
    names = {'intent', 'planning_before', 'review', 'claim', 'before', 'disposition', 'command', 'after', 'continued'}
    for sequence, bundle in enumerate(chain['links'], 1):
        need(type(bundle) is dict and set(bundle) == names, 'Complete exact amendment records required')
        records = {name: record(bundle, name) for name in names}
        intent, intent_sha = records['intent']; review, review_sha = records['review']
        claim, claim_sha = records['claim']; before, before_sha = records['before']
        entered, entered_sha = records['disposition']; command, command_sha = records['command']
        after, after_sha = records['after']; continued, continued_sha = records['continued']
        planning, planning_sha = records['planning_before']
        need(planning['checked_unix'] >= previous_finished, 'Amendment chain capture chronology reordered')
        need(type(intent) is dict and set(intent) == INTENT_FIELDS and intent['status'] == 'FROZEN_EXACT_INTENT' and
             type(intent['evidence_budget']) is dict and bool(intent['evidence_budget']) and
             type(intent['final_phase_budget']) is dict and bool(intent['final_phase_budget']) and
             type(intent['uncertainty_reserve_seconds']) is int and intent['uncertainty_reserve_seconds'] > 0,
             'Unknown/incomplete exact intent schema or planning attribution')
        need(intent['schema_version'] == 1 and intent['role'] == 'prospective_same_job_walltime_intent' and
             type(intent['sequence']) is int and intent['sequence'] == sequence and intent['job_id'] == JOB and
             intent['run_id'] == RUN and intent['base'] == BASE and intent['uid'] == 0 and
             intent['previous_verified_chain_sha256'] == previous and intent['old_time_limit_seconds'] == tip and
             seconds(intent['old_time_limit_slurm']) == tip and
             intent['original_anchor_sha256'] == sha(canonical(anchor)) and
             intent['planning_before_capture_sha256'] == planning_sha and
             intent['numeric_value_attribution'] == 'agent_selected_necessary_implementation_choice' and
             type(intent['active_goal_reference']) is str and intent['active_goal_reference'], 'Exact intent predecessor/identity differs')
        new = integer(intent['new_time_limit_seconds'], 1)
        policy = intent['capture_policy']
        need(type(policy) is dict and set(policy) == {'total_walltime_seconds', 'capture_allowance_seconds', 'max_rows',
             'max_bytes', 'max_line_bytes', 'max_tail_bytes', 'chunk_bytes', 'read_seconds', 'summary_bytes'} and
             all(type(value) is int and value > 0 for value in policy.values()) and
             policy['total_walltime_seconds'] == new and policy['max_rows'] == (new + policy['capture_allowance_seconds']) // 2 + 1 and
             policy['max_bytes'] == policy['max_rows'] * (policy['max_line_bytes'] + 1) + policy['max_tail_bytes'] and
             policy['max_tail_bytes'] <= policy['max_line_bytes'] <= 256 * 1024 and policy['chunk_bytes'] <= 65536 and
             policy['read_seconds'] <= 120 and policy['summary_bytes'] <= 2 * 1024 * 1024,
             'Exact future intent must seal finite longrun reader budget')
        need(new > tip and seconds(intent['new_time_limit_slurm']) == new and
             intent['new_expected_end_time'] == expected_end(anchor['original_native_fields']['StartTime'], new) and
             intent['old_expected_end_time'] == expected_end(anchor['original_native_fields']['StartTime'], tip) and
             intent['expected_update_argv'] == ['scontrol', 'update', 'JobId=' + JOB,
                                               'TimeLimit=' + intent['new_time_limit_slurm']], 'Exact update argv/value differs')
        need(review.get('approved') is True and review.get('scope') == 'staticonly' and
             review.get('fork_turns') == 'none' and review.get('draft_code_executed') is False and
             review.get('blocking_findings') == [] and review['intent_sha256'] == intent_sha and
             review['external_file_sha256'] == intent['reviewed_file_sha256'], 'Fresh exact-byte independent review missing')
        need(claim['state'] == 'CLAIMED' and claim['intent_sha256'] == intent_sha and claim['review_sha256'] == review_sha and
             claim['previous_verified_chain_sha256'] == previous and claim['expected_update_argv'] == intent['expected_update_argv'],
             'Exclusive amendment claim binding differs')
        capture_identity(planning, anchor, tip); capture_identity(before, anchor, tip)
        capture_identity(after, anchor, new); capture_identity(continued, anchor, new)
        need(native_fields(before['job_stdout'])['JobState'] == native_fields(after['job_stdout'])['JobState'] == 'RUNNING',
             'Amendment before/after not same running job')
        continuity(planning, before); continuity(before, after); continuity(after, continued, require_append=True)
        need(entered['state'] == 'UPDATE_ENTERED' and entered['claim_sha256'] == claim_sha and
             entered['before_sha256'] == before_sha and entered['intent_sha256'] == intent_sha and
             entered['review_sha256'] == review_sha and entered['argv'] == intent['expected_update_argv'] and
             before['observation_finished_unix'] <= clock(entered['entered_unix']) <= clock(intent['expires_unix']) and
             clock(intent['created_unix']) <= clock(claim['created_unix']) <= before['checked_unix'] and
             entered['entered_unix'] < clock(entered['deadline_unix']) <= entered['entered_unix'] + 60 and
             entered['entered_unix'] <= after['checked_unix'], 'Entered one-call chronology/hash binding differs')
        need(command['state'] == 'COMMAND_RETURNED_OR_UNKNOWN' and command['disposition_sha256'] == entered_sha and
             command['argv'] == entered['argv'] and command['call_count'] == 1 and type(command['call_count']) is int and
             (command['returncode'] is None or type(command['returncode']) is int) and
             command['outcome'] in ('returned', 'unknown_after_entered') and
             ((command['outcome'] == 'unknown_after_entered') == (command['returncode'] is None)) and
             entered['entered_unix'] <= clock(command['observed_unix']) <= after['checked_unix'], 'Actual single command receipt differs')
        previous, tip, previous_finished = sha(canonical(bundle)), new, continued['observation_finished_unix']
    return dict(time_limit_seconds=tip, chain_tip_sha256=previous, link_count=len(chain['links']),
                chain_sha256=sha(canonical(chain)),
                capture_policy=policy if chain['links'] else None)
