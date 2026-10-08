"""TEXT-ONLY, UNEXECUTED: independent read-only known3309 CLI/API.

Run ONLY after NEW fork_turns=none static exact-byte PASS. No update, new
submission, adoption, recovery writer, restart, requeue or training edit path.
"""
import argparse
import base64
import hashlib
import os
import stat
import time
import types
import uuid
from pathlib import Path

PREP = Path('/Users/shaoshitong/Documents/Search-G1/preparation')
DRAFT = PREP / 'formal-walltime-draft-v1'
V8 = PREP / 'formal-progress-draft-v8'
SUBMITTER_SHA = '091e9288163ccdf874fa042a87862ac9fd6ce40479d35830ea3c9c43dcf3ed92'
PROTOCOL_SHA = 'b4f80b83ad68b7aae40a8ebeabff963b0f0d5c1080e2d94c428c2ac1e6e4eecf'
CORE_SHA = '45c0a4660df4e4bd00d32ab20bcc5b124bc4154237bacea952d824524d948b47'
INTENT_SHA = 'a9ccca418a9df7922730d96a19bcb239afb13590a8f8418bbce4e9773c040a36'
LAUNCH_SHA = 'ba9b0962cabf2ac322d1386a7e7ad32531770d46f67afe9502f93f00b4088a4e'
BIRTH_SHA = 'f05002ebcdd0a4410ffd1d495e7740d7587cc14e9d6de6af03fb6e43600dfe9d'
FILES = ('observe_formal_same_job.py', 'formal_walltime_contract.py', 'formal_gpu_stream.py',
         'test_formal_walltime.py', 'test_formal_gpu_stream.py', 'observer-policy.json',
         'future-amendment-intent.template.json', 'README.md', 'NOEXEC-HANDOFF.md')
LIMIT = 2 * 1024 * 1024

REMOTE_BOOTSTRAP = r'''
import base64,hashlib,json,math,sys,types
raw=sys.stdin.buffer.read(2*1024*1024+1)
if len(raw)>2*1024*1024:raise ValueError('RPC input bound')
def pairs(items):
 out={}
 for key,value in items:
  if key in out:raise ValueError('Duplicate RPC key')
  out[key]=value
 return out
def bad(value):raise ValueError('Nonfinite RPC JSON')
def number(value):
 v=float(value)
 if not math.isfinite(v):raise ValueError('Nonfinite exponent')
 return v
p=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad,parse_float=number)
def loaded(name,encoded,expected):
 raw=base64.b64decode(encoded,validate=True)
 if len(raw)>2*1024*1024 or hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Reviewed bytes differ')
 m=types.ModuleType(name);m.__file__='<reviewed known-job '+name+'>'
 exec(compile(raw,m.__file__,'exec'),m.__dict__)
 return m
if p['protocol_sha256']!='b4f80b83ad68b7aae40a8ebeabff963b0f0d5c1080e2d94c428c2ac1e6e4eecf':raise ValueError('Protocol seal')
if p['core_sha256']!='45c0a4660df4e4bd00d32ab20bcc5b124bc4154237bacea952d824524d948b47':raise ValueError('Core seal')
protocol=loaded('protocol',p.pop('protocol_b64'),p['protocol_sha256'])
core=loaded('v8_core',p.pop('core_b64'),p['core_sha256'])
if p['submitter_sha256']!='091e9288163ccdf874fa042a87862ac9fd6ce40479d35830ea3c9c43dcf3ed92':raise ValueError('Submitter seal')
legacy=loaded('sealed_science_validator',p.pop('submitter_b64'),p['submitter_sha256'])
contract=loaded('contract',p.pop('contract_b64'),p['contract_sha256'])
stream=loaded('stream',p.pop('stream_b64'),p['stream_sha256'])
observer=loaded('observer',p.pop('observer_b64'),p['observer_sha256'])
result=observer.observe_remote(p,protocol,core,contract,stream,legacy)
result['collection_nonce']=p['collection_nonce']
result['observer_sha256']=p['observer_sha256']
result['review_sha256']=p['review_sha256']
raw=contract.canonical(result)
if len(raw)>2*1024*1024:raise ValueError('RPC summary bound')
sys.stdout.buffer.write(raw+b'\n')
'''


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sealed(path, expected):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink(), 'Canonical proof path required')
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_size <= LIMIT,
         'Owned bounded local ordinary proof required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        opened = os.fstat(fd)
        need((before.st_dev, before.st_ino, before.st_size) == (opened.st_dev, opened.st_ino, opened.st_size),
             'Local proof inode/size changed')
        chunks = []; size = 0
        while size <= LIMIT:
            block = os.read(fd, min(65536, LIMIT + 1 - size))
            if not block:
                break
            chunks.append(block); size += len(block)
        after, current = os.fstat(fd), path.lstat()
        signature = lambda item: (item.st_dev, item.st_ino, item.st_uid, item.st_mode,
                                  item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        need(size == before.st_size and signature(before) == signature(after) == signature(current), 'Local proof changed during capture')
        raw = b''.join(chunks)
    finally:
        os.close(fd)
    need(type(expected) is str and hashlib.sha256(raw).hexdigest() == expected, 'Exact local bytes/SHA differ')
    return raw


def loaded(name, path, raw):
    module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def review_gate(path, expected):
    # This tiny stdlib-only strict parser checks the externally supplied proof
    # bytes and review assertions before loading any legacy or draft dependency.
    import json
    import math
    raw = sealed(path, expected)
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, 'Duplicate review key'); out[key] = value
        return out
    def bad(value):
        raise ValueError('Nonfinite review')
    def number(value):
        out = float(value); need(math.isfinite(out), 'Nonfinite review exponent'); return out
    review = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad, parse_float=number)
    need(review.get('role') == 'same_job_walltime_independent_static_review' and review.get('approved') is True and
         review.get('scope') == 'staticonly' and review.get('fork_turns') == 'none' and
         review.get('draft_code_executed') is False and review.get('blocking_findings') == [],
         'NEW context-free strict source-only PASS required')
    mapping = review['external_file_sha256']
    need(type(mapping) is dict and set(mapping) == set(FILES) and
         Path(__file__).resolve() == DRAFT / 'observe_formal_same_job.py', 'All exact draft/spec/test bytes required')
    need(review['sealed_dependency_sha256'] == dict(submitter=SUBMITTER_SHA, protocol=PROTOCOL_SHA, v8_core=CORE_SHA,
         original_intent=INTENT_SHA, original_launch=LAUNCH_SHA, original_birth=BIRTH_SHA), 'Reviewed original dependencies differ')
    return review, {name: sealed(DRAFT / name, mapping[name]) for name in FILES}


def read_command(protocol, argv, timeout=15):
    need(argv in (['scontrol', 'show', 'job', '-o', '3309'], ['squeue', '-h', '-j', '3309', '-o', '%i|%T|%j']),
         'Only fixed known-job read argv allowed')
    result = protocol.bounded_capture(argv, timeout, 65536, 65536)
    need(type(result.returncode) is int and type(result.stdout) is str and type(result.stderr) is str,
         'Typed actual command receipt required')
    return result


def select_policy(document, verified_tip):
    need(type(document) is dict and set(document) == {'schema_version', 'role', 'scope', 'policies'} and
         type(document['schema_version']) is int and document['schema_version'] == 1 and
         document['role'] == 'reviewed_exact_known_job_reader_policies' and
         set(document['policies']) == {'172800', '691200'}, 'Exactly reviewed original/candidate reader policies required')
    duration = verified_tip['time_limit_seconds']
    need(type(duration) is int and str(duration) in document['policies'] and
         (duration == 172800 or (duration == 691200 and verified_tip['link_count'] > 0 and
                                verified_tip['capture_policy'] is not None)),
         'Candidate capacity requires complete exact691200 amendment chain')
    policy = document['policies'][str(duration)]
    need(verified_tip['capture_policy'] is None or policy == verified_tip['capture_policy'],
         'Selected capacity differs from verified exact amendment intent')
    return policy


def actual_namespace(core, deadline):
    """Bounded original namespace census, no process control."""
    active = []; count = 0
    marker = ('SEARCH_G1_BASE=' + core.BASE).encode(); prefix = core.BASE.encode()
    for proc in Path('/proc').iterdir():
        core.deadline_check(deadline); count += 1
        need(count <= 65536, 'Proc census entry bound')
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != 0:
                continue
            def bytes_at(name, bound):
                fd = os.open(proc / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                try:
                    raw = os.read(fd, bound + 1); need(len(raw) <= bound, 'Proc census byte bound'); return raw
                finally:
                    os.close(fd)
            fields = bytes_at('stat', 4096).rsplit(b')', 1)[1].split()
            if fields[0] == b'Z':
                continue
            env, argv = bytes_at('environ', 65536), bytes_at('cmdline', 65536).split(b'\0')
            if marker in env.split(b'\0') or any(arg == prefix or arg.startswith(prefix + b'/') for arg in argv):
                active.append(dict(pid=int(proc.name), start_ticks=int(fields[19]), sid=int(fields[3]), uid=0))
                need(len(active) <= core.MAX_PROCESS_BIRTHS, 'Namespace birth bound')
        except (FileNotFoundError, ProcessLookupError):
            pass
    return active


def observe_remote(payload, protocol, core, contract, stream, legacy):
    """Fixed handle; native-first summary retained even if another component fails."""
    anchor, intent, launch, birth = contract.original_anchor(*(
        base64.b64decode(payload[name], validate=True) for name in ('intent_b64', 'launch_b64', 'birth_b64')))
    chain = contract.validate_chain(anchor, payload['chain'])
    policy = stream.validate_policy(payload['policy'], chain['time_limit_seconds'])
    need(chain['capture_policy'] is None or policy == chain['capture_policy'], 'Current reader budget differs from exact amendment intent')
    need(os.getuid() == 0 and payload['base'] == contract.BASE and payload['job_id'] == contract.JOB and
         payload['run_id'] == contract.RUN, 'Fixed actual remote UID/job/run required')
    deadline = time.monotonic() + policy['read_seconds']
    out = dict(schema_version=1, role='same_job_readonly_capture', job_id=contract.JOB, run_id=contract.RUN,
        base=contract.BASE, checked_unix=time.time(), observation_incomplete=True,
        native_state='unknown', native_terminal_authenticated=False, terminal_success='not_inferred',
        chain=chain, prefixes={}, anchor_rechecks={}, process_births={}, current_birth_checks={},
        scope='Known original job only; actual disk/process/prefix facts; no health or goal-completion inference')
    job = read_command(protocol, ['scontrol', 'show', 'job', '-o', contract.JOB])
    out.update(job_returncode=job.returncode, job_stdout=job.stdout,
               job_stderr_sha256=contract.sha(job.stderr.encode()), native_acquired_unix=time.time())
    if job.returncode != 0:
        out.update(capture_state='unknown_native_missing_or_purged', observation_finished_unix=time.time())
        return out
    try:
        fields = contract.current_native(job.stdout, anchor, chain['time_limit_seconds'])
        out['native_state'] = fields['JobState']; out['native_terminal_authenticated'] = fields['JobState'] == 'COMPLETED'
        queue = read_command(protocol, ['squeue', '-h', '-j', contract.JOB, '-o', '%i|%T|%j'])
        out.update(queue_returncode=queue.returncode, queue_stdout=queue.stdout,
            queue_stderr_sha256=contract.sha(queue.stderr.encode()))
        base = Path(contract.BASE)
        need(base.resolve() == base and base.is_dir() and not base.is_symlink(), 'Canonical fixed remote base required')
        mapping, inventory_binding = core.bind_runtime_inventory(protocol.runtime_map(base / 'repo'), intent['source_sha256'])
        core.deadline_check(deadline)
        out.update(source_hashes_match=True, source_count=len(mapping), source_commit=contract.COMMIT,
            source_map_sha256=contract.MAP_SHA, observed_runtime_inventory=inventory_binding)
        method, method_prov = core.capture_json(base / 'preparation/formal-inline-method.json', deadline)
        need(method_prov['captured_sha256'] == contract.METHOD_SHA and not method_prov['observation_incomplete'], 'Original method bytes differ')
        out['method_sha256'] = method_prov['captured_sha256']
        claim, claim_prov = core.capture_json(base / 'preparation/native-submit-claim.json', deadline)
        entered, entered_prov = core.capture_json(base / 'preparation/native-submit-disposition.json', deadline)
        receipt, receipt_prov = core.capture_json(base / 'preparation/native-submit-receipt.json', deadline)
        need(claim_prov['captured_sha256'] == contract.CLAIM_SHA and entered_prov['captured_sha256'] == contract.DISPOSITION_SHA and
             receipt['submission_authentication'] == 'actual_sbatch_receipt' and type(receipt['returncode']) is int and
             receipt['returncode'] == 0 and receipt['job_id'] == contract.JOB and receipt['stdout'].strip().split(';')[0] == contract.JOB and
             receipt['claim_sha256'] == contract.CLAIM_SHA and receipt['disposition_sha256'] == contract.DISPOSITION_SHA and
             receipt['submitted_unix'] == entered['submitted_unix'] == launch['submitted_unix'] and
             receipt['deadline_unix'] == entered['deadline_unix'] == launch['deadline_unix'] and
             receipt['claim_created_unix'] == claim['created_unix'] == launch['claim_created_unix'] and
             all(receipt[key] == launch[key] for key in receipt) and
             not any(prov['observation_incomplete'] for prov in (claim_prov, entered_prov, receipt_prov)),
             'Exact actual original submission/reservation receipt differs')
        out.update(original_claim_sha256=contract.CLAIM_SHA, original_disposition_sha256=contract.DISPOSITION_SHA,
                   original_submission_receipt_sha256=receipt_prov['captured_sha256'])
        run, run_prov = core.capture_json(base / 'logs/run-manifest.json', deadline)
        # Reuse old exact science/source validation with actual original receipt;
        # no legacy live walltime validator is called or altered.
        out['run-manifest.json'] = run; out['submission'] = receipt
        out['submission_outcome_known'] = True
        need(run['run_id'] == contract.RUN and run['source_sha256'] == mapping and run['source_commit'] == contract.COMMIT,
             'Original actual manifest differs')
        need(legacy.actual_launch(intent, out) == launch, 'Exact original actual launch/science recipe differs')
        out['original_actual_launch_verified'] = True
        out['run_manifest_sha256'] = run_prov['captured_sha256']
        run_started = core.finite(run['started_unix'], 'run start', 0)
        training, training_prov = core.capture_json(base / 'logs/training-launch-claim.json', deadline)
        ledger, ledger_prov = core.capture_json(base / 'logs/process-ledger.json', deadline, maximum=core.MAX_PROCESS_LEDGER)
        boundary, admitted = core.validate_training_binding(training, ledger, run_started_unix=run_started, now=time.time())
        need(boundary['trainer_leader_birth'] == contract.TRAINER, 'Original actual trainer admission changed')
        out['training_boundary'] = boundary
        old = payload.get('previous_capture')
        if old is not None:
            contract.capture_identity(old, anchor, chain['time_limit_seconds'])
        out['process_births'].update(trainer=contract.TRAINER, sampler=contract.SAMPLER)
        attempts, attempt_prov = core.capture_jsonl(base / 'logs/actual-attempts.jsonl', core.MAX_ATTEMPTS,
                                                   core.MAX_ATTEMPT_ROWS, deadline)
        out['progress'], complete_attempts = core.validate_attempts(attempts)
        updates, update_prov = core.capture_jsonl(base / 'logs/actual-updates.jsonl', core.MAX_UPDATES, 200, deadline)
        out['metrics'] = core.validate_updates(updates, complete_attempts)
        # Rehash prior anchors using the bounded scanner; v8 driver/rank parsing
        # remains finite and unchanged. Hash equality joins the two captures.
        def prefix(name, known, maximum, rows):
            provenance, recheck = stream.scan_prefix(base / 'logs' / name, owner_uid=0, max_bytes=maximum,
                max_rows=rows, max_line_bytes=core.MAX_LINE, max_tail_bytes=core.MAX_LINE, chunk_bytes=65536,
                deadline=deadline, consume=lambda row: None, parse=core.strict_json,
                previous=old['prefixes'][name] if old is not None else None)
            need(provenance['complete_prefix_sha256'] == known['complete_prefix_sha256'] and
                 provenance['complete_prefix_bytes'] == known['complete_prefix_bytes'], 'Append raced between parse and prefix join')
            out['prefixes'][name] = provenance
            if recheck is not None:
                out['anchor_rechecks'][name] = recheck
        prefix('actual-attempts.jsonl', attempt_prov, core.MAX_ATTEMPTS, core.MAX_ATTEMPT_ROWS)
        prefix('actual-updates.jsonl', update_prov, core.MAX_UPDATES, 200)
        contract_sha = core.scheduler_contract_sha256(run['budget'])
        summaries, terminals = [], []
        for rank in range(8):
            name = 'scheduler-rank-%d.jsonl' % rank
            rows, prov = core.capture_jsonl(base / 'logs' / name, core.MAX_SCHEDULER, core.MAX_SCHEDULER_ROWS, deadline)
            summary, terminal = core.inspect_scheduler_rows(rows, rank, complete_attempts, contract_sha256=contract_sha,
                admitted_births=admitted, run_started_unix=run_started, training_claim_unix=boundary['claimed_unix'])
            need(summary['historical_birth_binding_state'] == 'observed' and not summary['unavailable_driver_attempts'],
                 'Rank original header/admitted birth/driver prefix unavailable')
            out['process_births']['rank%d' % rank] = summary['historical_process_birth']
            summaries.append(summary); terminals.append(terminal)
            prefix(name, prov, core.MAX_SCHEDULER, core.MAX_SCHEDULER_ROWS)
        need(len({(row['pid'], row['start_ticks']) for name, row in out['process_births'].items() if name.startswith('rank')}) == 8,
             'Rank producer births duplicated')
        out['scheduler_crosscheck'] = core.summarize_scheduler_crosscheck(summaries, terminals)
        allocation, allocation_prov = core.capture_json(base / 'logs/native-allocation-admission.json', deadline)
        need(allocation_prov['captured_sha256'] == contract.ALLOCATION_SHA and not allocation_prov['observation_incomplete'],
             'Original48h allocation receipt changed')
        inventory = core.allocation_inventory(allocation)
        out['allocation_sha256'] = allocation_prov['captured_sha256']
        name = 'gpu-samples-' + contract.RUN + '.jsonl'
        try:
            gpu, prov, recheck = stream.scan_gpu(base / 'logs' / name, core, inventory, policy,
                duration=chain['time_limit_seconds'], now=time.time(), run_started_unix=run_started,
                training_claim_unix=boundary['claimed_unix'], expected_sampler=contract.SAMPLER, deadline=deadline,
                previous=old['prefixes'][name] if old is not None else None)
            out['gpu_samples'], out['prefixes'][name] = gpu, prov
            if recheck is not None:
                out['anchor_rechecks'][name] = recheck
        except (ValueError, OSError, core.ObservationError) as error:
            out['gpu_samples'] = dict(evidence_state='unavailable', sample_state='unavailable', cumulative_sampled_peak_mib=None,
                                      observation_incomplete=True, reason_code=type(error).__name__)
        for name, identity in out['process_births'].items():
            current = core.actual_birth(identity, deadline)
            out['current_birth_checks'][name] = current
            if fields['JobState'] == 'RUNNING':
                need(current.get('evidence_state') == 'observed' and current.get('birth_matches') is True and
                     current.get('namespace_matches') is True and current.get('process_state') != 'Z',
                     'Actual original producer no longer live in exact namespace')
        out['observation_finished_unix'] = time.time()
        contract.capture_identity(out, anchor, chain['time_limit_seconds'])
        if old is not None:
            contract.continuity(old, out)
        if fields['JobState'] == 'COMPLETED':
            status, _ = core.capture_json(base / 'logs/pipeline-status.json', deadline)
            cleanup, _ = core.capture_json(base / 'logs/cleanup-status.json', deadline)
            active = actual_namespace(core, deadline)
            out['terminal_cleanup_authenticated'] = (status.get('run_id') == cleanup.get('run_id') == contract.RUN and
                status.get('stage') == 'FINISHED_FULL200_FINAL51713_AND_OWNED_CLEANUP' and cleanup.get('cleaned') is True and
                type(ledger.get('jobs')) is dict and bool(ledger['jobs']) and
                all(row.get('state') == 'STOPPED' for row in ledger['jobs'].values()) and active == [] and
                queue.returncode == 0 and not queue.stdout.strip())
            out['actual_active_namespace_processes'] = active
        # Purged/missing/failed record cannot enter this branch. Finite200 and
        # final51713 acceptance remain the parent's independently reviewed work.
        out['capture_state'] = 'observed'
        out['observation_incomplete'] = (set(out['prefixes']) != set(contract.PREFIX_NAMES) or
            any(row['observation_incomplete'] for row in out['prefixes'].values()) or
            out.get('gpu_samples', {}).get('observation_incomplete', True) or queue.returncode != 0)
    except (ValueError, OSError, KeyError, TypeError, core.ObservationError) as error:
        out.update(capture_state='unknown_or_invalid_component', component_error_type=type(error).__name__,
                   observation_incomplete=True)
    # Manifest/receipt were local validation inputs; raw training material stays
    # on target. Native scheduler stdout and bounded scalar/hash summaries stay.
    out.pop('run-manifest.json', None); out.pop('submission', None)
    out['observation_finished_unix'] = time.time()
    return out


def publish(contract, receipt):
    path = DRAFT / ('same-job-observation-' + str(time.time_ns()) + '-' + uuid.uuid4().hex + '.json')
    raw = contract.canonical(receipt) + b'\n'
    need(len(raw) <= LIMIT, 'Local summary bound')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            need(stream.write(raw) == len(raw), 'Short exclusive receipt write'); stream.flush(); os.fsync(stream.fileno())
    finally:
        pass
    directory = os.open(DRAFT, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return dict(receipt_path=str(path), receipt_sha256=contract.sha(raw), capture_state=receipt['capture']['capture_state'],
                observation_incomplete=receipt['capture']['observation_incomplete'])


def observe_same_job(review_path, review_sha, *, chain_path=None, chain_sha=None,
                     previous_path=None, previous_sha=None):
    review, contents = review_gate(review_path, review_sha)
    contract = loaded('walltime_contract', DRAFT / 'formal_walltime_contract.py', contents['formal_walltime_contract.py'])
    intent_raw = sealed(PREP / 'formal-inline-native-submit-intent.json', INTENT_SHA)
    launch_raw = sealed(PREP / 'formal-inline-native-launch.json', LAUNCH_SHA)
    birth_raw = sealed(PREP / 'formal-inline-native-birth-observation.json', BIRTH_SHA)
    anchor, intent, launch, birth = contract.original_anchor(intent_raw, launch_raw, birth_raw)
    submitter = loaded('sealed_submitter_transport', PREP / 'submit_formal_inline_native.py',
                       sealed(PREP / 'submit_formal_inline_native.py', SUBMITTER_SHA))
    # Only original captured48h birth and original saved handle use old rules.
    submitter.native_record(birth, launch, 'RUNNING', '0:0')
    chain = dict(schema_version=1, role='same_job_exact_walltime_chain', links=[], unresolved_entered=[])
    if chain_path is not None:
        chain = contract.strict_json(sealed(chain_path, chain_sha))
        need(review['chain_sha256'] == chain_sha, 'Observed exact chain must be review-bound')
    else:
        need(review['chain_sha256'] == contract.sha(contract.canonical(chain)), 'Original-only chain review binding required')
    tip = contract.validate_chain(anchor, chain)
    policy = select_policy(contract.strict_json(contents['observer-policy.json']), tip)
    stream = loaded('gpu_stream', DRAFT / 'formal_gpu_stream.py', contents['formal_gpu_stream.py'])
    stream.validate_policy(policy, tip['time_limit_seconds'])
    need(tip['capture_policy'] is None or policy == tip['capture_policy'], 'Reviewed reader policy differs from exact amendment chain')
    previous = None
    if previous_path is not None:
        previous_receipt = contract.strict_json(sealed(previous_path, previous_sha))
        previous = previous_receipt['capture']
        contract.capture_identity(previous, anchor, tip['time_limit_seconds'])
        need(previous_receipt['original_anchor_sha256'] == contract.sha(contract.canonical(anchor)) and
             previous['capture_state'] == 'observed' and previous['observation_incomplete'] is False,
             'Only full authenticated prior same-job capture supplies append anchors')
    payload = dict(base=contract.BASE, run_id=contract.RUN, job_id=contract.JOB, chain=chain, policy=policy,
        previous_capture=previous, collection_nonce=uuid.uuid4().hex, review_sha256=review_sha,
        intent_b64=base64.b64encode(intent_raw).decode(), launch_b64=base64.b64encode(launch_raw).decode(),
        birth_b64=base64.b64encode(birth_raw).decode(), protocol_sha256=PROTOCOL_SHA, core_sha256=CORE_SHA,
        submitter_sha256=SUBMITTER_SHA,
        submitter_b64=base64.b64encode(sealed(PREP / 'submit_formal_inline_native.py', SUBMITTER_SHA)).decode(),
        protocol_b64=base64.b64encode(sealed(PREP / 'test_formal_inline_target_cpu.py', PROTOCOL_SHA)).decode(),
        core_b64=base64.b64encode(sealed(V8 / 'formal_progress_core.py', CORE_SHA)).decode())
    for name, file in (('observer', 'observe_formal_same_job.py'), ('contract', 'formal_walltime_contract.py'), ('stream', 'formal_gpu_stream.py')):
        payload[name + '_b64'] = base64.b64encode(contents[file]).decode()
        payload[name + '_sha256'] = contract.sha(contents[file])
    need(len(contract.canonical(payload)) <= LIMIT, 'Known-job RPC input bound exceeded')
    try:
        capture = submitter.remote('h100-8', REMOTE_BOOTSTRAP, payload, timeout=policy['read_seconds'] + 30,
                                   producer_sha256=PROTOCOL_SHA)
        need(capture['collection_nonce'] == payload['collection_nonce'] and capture['observer_sha256'] == payload['observer_sha256'] and
             capture['review_sha256'] == review_sha and capture['job_id'] == contract.JOB and capture['run_id'] == contract.RUN,
             'Actual returned capture provenance differs')
        if capture['capture_state'] == 'observed':
            contract.capture_identity(capture, anchor, tip['time_limit_seconds'])
    except Exception as error:
        capture = dict(capture_state='unknown_transport', observation_incomplete=True,
            error_type=type(error).__name__, terminal_success='not_inferred', native_terminal_authenticated=False)
    receipt = dict(schema_version=1, role='same_job_readonly_observation_receipt', job_id=contract.JOB, run_id=contract.RUN,
        original_anchor_sha256=contract.sha(contract.canonical(anchor)), original_birth_sha256=BIRTH_SHA,
        original_intent_sha256=INTENT_SHA, original_launch_sha256=LAUNCH_SHA, original_allocation_sha256=contract.ALLOCATION_SHA,
        exact_chain=tip, independent_review_sha256=review_sha, external_file_sha256=review['external_file_sha256'],
        previous_receipt_sha256=previous_sha, capture=capture, source_snapshot_semantics='disk bytes; not proof of loaded modules')
    need(sealed(PREP / 'formal-inline-native-submit-intent.json', INTENT_SHA) == intent_raw and
         sealed(PREP / 'formal-inline-native-launch.json', LAUNCH_SHA) == launch_raw and
         sealed(PREP / 'formal-inline-native-birth-observation.json', BIRTH_SHA) == birth_raw,
         'Original local anchors changed across read')
    return publish(contract, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observe', action='store_true', required=True)
    parser.add_argument('--review-proof', type=Path, required=True); parser.add_argument('--review-sha256', required=True)
    parser.add_argument('--chain', type=Path); parser.add_argument('--chain-sha256')
    parser.add_argument('--previous-receipt', type=Path); parser.add_argument('--previous-sha256')
    args = parser.parse_args()
    need((args.chain is None) == (args.chain_sha256 is None) and
         (args.previous_receipt is None) == (args.previous_sha256 is None), 'Each optional receipt requires its exact SHA')
    import json
    print(json.dumps(observe_same_job(args.review_proof, args.review_sha256, chain_path=args.chain,
        chain_sha=args.chain_sha256, previous_path=args.previous_receipt, previous_sha=args.previous_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
