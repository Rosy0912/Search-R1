"""Stop this already-loaded R1 runtime after a requested durable update boundary.

No hot recipe changes. PID/start-ticks plus private session bind every signal.
The coordinator is suspended after the completed-update log, so an in-flight
next rollout or remote work may already be dispatched; only the completed-update
log is authoritative for whether the requested boundary was preserved.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time


def identity(pid):
    p = Path('/proc') / str(pid)
    stat = (p / 'stat').read_text()
    fields = stat[stat.rfind(')') + 2:].split()
    return dict(pid=pid, start_ticks=int(fields[19]), sid=int(fields[3]),
                uid=p.stat().st_uid, state=fields[0],
                command=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'))


def matches(record):
    try:
        now = identity(record['pid'])
        return all(now[k] == record[k] for k in ('start_ticks', 'sid', 'uid'))
    except (FileNotFoundError, ProcessLookupError):
        return False


def send(record, sig):
    # pidfd is mandatory on the target; no racy numerical-PID fallback.
    fd = os.pidfd_open(record['pid'])
    try:
        if not matches(record):
            raise RuntimeError('Process identity changed before signal')
        signal.pidfd_send_signal(fd, sig)
    finally:
        os.close(fd)


def logged_updates(text):
    ids = []
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    for line in text.splitlines():
        m = re.search(r'\bstep:(\d+) - (.*)', line)
        if m and 'critic/rewards/mean:' in m[2]:
            ids.append(int(m[1]))
    assert len(ids) == len(set(ids)), 'Duplicate completed-update records'
    return ids


def owned_session_records(sid, uid):
    records = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                r = identity(int(p.name))
                if r['sid'] == sid and r['uid'] == uid and r['state'] != 'Z':
                    records.append(r)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
    return records


def atomic(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=300)
    ap.add_argument('--base', type=Path, default=Path('/data/search-r1'))
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--start-ticks', type=int, required=True)
    ap.add_argument('--driver', type=int, required=True)
    args = ap.parse_args()
    logs = args.base / 'logs'
    lock = (logs / 'stop-at-checkpoint.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((logs / 'a800-continuation-manifest.json').read_text())
    assert manifest['run_id'] == args.run_id
    task, driver = identity(args.pid), identity(args.driver)
    assert task['start_ticks'] == args.start_ticks and task['command'].startswith('ray::main_task')
    assert task['sid'] == driver['sid'] and task['uid'] == driver['uid'] == os.getuid()
    assert '-m verl.trainer.main_ppo ' in driver['command']
    assert str(args.base / 'checkpoints') in driver['command']
    assert os.getsid(0) != driver['sid'], 'Stop monitor must have independent session'
    leader = identity(driver['sid'])
    assert str(args.base / 'repo/reproduction/a800/run_continuation.sh') in leader['command']
    initial_members = owned_session_records(driver['sid'], driver['uid'])
    result = dict(run_id=args.run_id, target_update=args.target, task_identity=task,
                  driver_identity=driver, monitor_identity=identity(os.getpid()),
                  requested_semantics='user-requested model-weight checkpoint stop',
                  state='WAITING_FOR_SAVED_BOUNDARY', started_unix=time.time(),
                  session_leader_identity=leader, initial_owned_session_members=initial_members)
    status = logs / 'r1-stop300-status.json'
    atomic(status, result)
    log = Path(manifest['train_log'])
    while True:
        assert matches(task) and matches(driver), 'Owned training process ended before requested boundary'
        text = log.read_text(errors='replace')
        assert text.splitlines()[0] == 'SEARCH_R1_SEGMENT ' + args.run_id
        steps = logged_updates(text)
        if steps and max(steps) >= args.target:
            break
        time.sleep(.1)
    send(task, signal.SIGSTOP)
    for _ in range(100):
        if identity(task['pid'])['state'] == 'T':
            break
        time.sleep(.01)
    assert identity(task['pid'])['state'] == 'T'
    result.update(state='COORDINATOR_SUSPENDED_VERIFYING_CHECKPOINT', suspended_unix=time.time())
    atomic(status, result)
    try:
        steps = logged_updates(log.read_text(errors='replace'))
        assert steps == list(range(manifest['completed_updates'] + 1, args.target + 1)), 'Requested boundary exceeded or incomplete; coordinator left suspended'
        checkpoint = Path(manifest['checkpoint_dir']) / 'actor' / ('global_step_' + str(args.target))
        from verl.utils.checkpoint_integrity import validate_saved_checkpoint
        tensors = validate_saved_checkpoint(checkpoint, Path(manifest['reference_model']), manifest['run_started'])
        hashes = {}
        index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
        files = set(index['weight_map'].values()) | {'config.json', 'model.safetensors.index.json'}
        for name in sorted(files):
            h = hashlib.sha256()
            with (checkpoint / name).open('rb') as reader:
                for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b''):
                    h.update(chunk)
            hashes[name] = h.hexdigest()
        result.update(state='CHECKPOINT_ACCEPTED_STOPPING_OWNED_RUNTIME',
                      checkpoint=str(checkpoint), validated_tensor_count=tensors,
                      finite_all_saved_tensors=True, file_sha256=hashes,
                      completed_updates=steps, next_rollout_may_have_started=True,
                      full_state_resume=False, checkpoint_verified_unix=time.time())
        atomic(status, result)
    except Exception as exc:
        result.update(state='CHECKPOINT_REJECTED_COORDINATOR_LEFT_SUSPENDED', error=repr(exc))
        atomic(status, result)
        raise
    # The suspended coordinator must never resume into301. Kill only its pidfd.
    send(task, signal.SIGKILL)
    records = owned_session_records(driver['sid'], driver['uid'])
    result['owned_process_identities'] = records
    atomic(status, result)
    if matches(driver):
        send(driver, signal.SIGINT)  # driver finally shuts down its private Ray
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if not owned_session_records(driver['sid'], driver['uid']):
            break
        time.sleep(.5)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        # Snapshot each remaining member; no other session is eligible.
        for record in owned_session_records(driver['sid'], driver['uid']):
            try:
                send(record, sig)
            except ProcessLookupError:
                pass
        time.sleep(2)
    remaining = owned_session_records(driver['sid'], driver['uid'])
    assert not remaining, 'Owned session has surviving processes'
    assert logged_updates(log.read_text(errors='replace')) == steps
    result.update(state='STOPPED_AFTER_VERIFIED_CHECKPOINT', stopped_unix=time.time(),
                  remaining_owned_processes=remaining, latest_completed_update=args.target)
    atomic(status, result)
    atomic(logs / 'r1-update300-acceptance.json', result)
    (logs / 'a800-pipeline-stage.txt').write_text('USER_REQUESTED_STOP_AFTER_VERIFIED_UPDATE_' + str(args.target) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        path = Path('/data/search-r1/logs/r1-stop300-status.json')
        if path.exists():
            result = json.loads(path.read_text())
            if result['state'] != 'CHECKPOINT_REJECTED_COORDINATOR_LEFT_SUSPENDED':
                result['state'] = 'MONITOR_FAILED_REQUIRES_INSPECTION'
                result['error'] = repr(exc)
                driver = result['driver_identity']
                result['remaining_owned_processes'] = owned_session_records(driver['sid'], driver['uid'])
                atomic(path, result)
        raise
