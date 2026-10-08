"""Latest human-directed stop of this one owned K1 evaluation; no broad kill."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

BASE = Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007')
RUN = '914bfff786c84ede8d82414dff65f60e'
CONTROLLER = 2459676
CONTROLLER_TICK = 915357855
PARENT = 3960546
PARENT_TICK = 921600425
CANDIDATE = 2239529
CANDIDATE_TICK = 914664099


def read_process(pid):
    try:
        p = Path('/proc') / str(pid)
        st = (p / 'stat').read_text().rsplit(') ', 1)[1].split()
        uid = int(next(x for x in (p / 'status').read_text().splitlines() if x.startswith('Uid:')).split()[1])
        argv = [x.decode() for x in (p / 'cmdline').read_bytes().split(b'\0') if x]
        env = dict(x.split(b'=', 1) for x in (p / 'environ').read_bytes().split(b'\0') if b'=' in x)
        return dict(pid=pid, uid=uid, state=st[0], ppid=int(st[1]), pgid=int(st[2]),
                    starttick=int(st[19]), argv=argv, run_id=env.get(b'E1_RUN_ID', b'').decode())
    except (FileNotFoundError, ProcessLookupError):
        return None


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2)


def signal_exact(original, sig):
    current = read_process(original['pid'])
    if current is None or current['state'] == 'Z':
        return 'already_terminal'
    for key in ('uid', 'starttick', 'argv', 'pgid', 'run_id'):
        if current[key] != original[key]:
            raise RuntimeError('Identity changed; refuse signal to PID ' + str(original['pid']))
    fd = os.pidfd_open(original['pid'])
    try:
        after_open = read_process(original['pid'])
        if after_open is None or after_open['starttick'] != original['starttick']:
            raise RuntimeError('PID changed while opening pidfd')
        signal.pidfd_send_signal(fd, sig)
    finally:
        os.close(fd)
    return signal.Signals(sig).name


def main():
    if os.getuid() != 1005:
        raise RuntimeError('Only owned K1 UID1005 allowed')
    stage = json.loads((BASE / 'receipts/pipeline-stage.json').read_text())
    active = json.loads((BASE / 'receipts/preparation-3-active-child.json').read_text())
    if stage.get('run_id') != RUN or stage.get('phase') != 'OPSD100_PREPARATION3_FINAL_SEVEN_BENCHMARK_EVALUATION':
        raise RuntimeError('Current stage is no longer the authorized target')
    if active.get('run_id') != RUN or active.get('child_pid') != PARENT or active.get('child_pgid') != PARENT:
        raise RuntimeError('Active child binding changed')
    controller = read_process(CONTROLLER)
    parent = read_process(PARENT)
    candidate = read_process(CANDIDATE)
    if not controller or controller['uid'] != 1005 or controller['starttick'] != CONTROLLER_TICK:
        raise RuntimeError('Controller identity mismatch')
    if str(BASE / 'source/preparation-3-xformers/pipeline_preparation3_xformers.py') not in controller['argv']:
        raise RuntimeError('Controller source marker mismatch')
    if not parent or parent['uid'] != 1005 or parent['starttick'] != PARENT_TICK or parent['argv'] != active['argv']:
        raise RuntimeError('Parent identity mismatch')
    if not candidate or candidate['uid'] != 1005 or candidate['starttick'] != CANDIDATE_TICK:
        raise RuntimeError('Preserved private retriever identity mismatch')
    members = []
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():
            continue
        try:
            item = read_process(int(d.name))
        except PermissionError:
            continue
        if item and item['pgid'] == PARENT and item['state'] != 'Z':
            if item['uid'] != 1005 or item['run_id'] != RUN:
                raise RuntimeError('Unexpected evaluation process-group member; refuse broad stop')
            members.append(item)
    if {x['pid'] for x in members} != {PARENT,3960643,3960647,3960648,3960649,3960650,3960651,3960652,3960653}:
        raise RuntimeError('Evaluation membership changed; require fresh scope review')
    if any(x['pid'] == CANDIDATE for x in members):
        raise RuntimeError('Retriever must never be a stop target')
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    preservation = BASE / 'preserved' / ('user-stop-eight-evaluation-debug-' + stamp)
    preservation.mkdir()
    authorization = dict(observed_unix=time.time(), original_human_timestamp='Not supplied; latest live human message received after the 2026-10-08T03:34:43Z heartbeat',
        original_text='8卡的OPSD可以停掉，优先用GPU搜索；三种情况尽可能保留debug信息，比如teacher和student轨迹以及每tokenloss',
        supersedes='Previous automatic continuation from selected baseline full7 into raw epoch1 full7',
        scope='Stop only current K1 eight-GPU evaluation controller and this exact evaluation group. Full OPSD epoch was already completed. K4 archive and private retriever remain authorized.',
        later_human_resume='None at execution', run_id=RUN, controller=controller, parent=parent,
        owned_members=members, preserved_candidate=candidate, no_broad_goal_pause=True)
    # Persist the time/scope authorization before the first signal.
    write_new(preservation / 'authorization-before-signals.json', authorization)
    copied = {}
    for name in ['receipts/pipeline-stage.json','receipts/preparation-3-pipeline-stage.json',
        'receipts/preparation-3-active-child.json','receipts/preparation-3-child-history.jsonl',
        'receipts/preparation-3-run.json','receipts/preparation-3-config.json',
        'runs/opsd/history.json','runs/opsd/config.json','logs/preparation-3-pipeline.log',
        'logs/preparation-3-opsd.log','logs/preparation-3-final-test.log']:
        src = BASE / name
        if src.exists():
            dst = preservation / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied[name] = dict(preserved=str(dst), sha256=digest(dst))
    targets = [controller] + members
    actions = [dict(pid=x['pid'], action=signal_exact(x, signal.SIGTERM)) for x in targets]
    still = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        still = [x for x in targets if (now := read_process(x['pid'])) and now['state'] != 'Z' and now['starttick'] == x['starttick']]
        if not still:
            break
        time.sleep(1)
    for x in still:
        actions.append(dict(pid=x['pid'], action=signal_exact(x, signal.SIGKILL)))
    time.sleep(2)
    terminal = {str(x['pid']): read_process(x['pid']) for x in targets}
    if any(v and v['state'] != 'Z' and v['starttick'] == next(x['starttick'] for x in targets if x['pid'] == int(k)) for k,v in terminal.items()):
        raise RuntimeError('Owned stop is incomplete')
    kept = read_process(CANDIDATE)
    if not kept or kept['starttick'] != CANDIDATE_TICK or kept['state'] == 'Z':
        raise RuntimeError('Preserved retriever no longer live')
    shards = []
    for f in sorted((BASE / 'runs/preparation-3-final-test.jsonl.parts').glob('rollouts-*.jsonl')):
        with f.open('rb') as stream:
            rows = sum(line.endswith(b'\n') for line in stream)
        shards.append(dict(path=str(f), rows=rows, bytes=f.stat().st_size, sha256=digest(f)))
    evidence = dict(status='STOPPED_BY_LATEST_HUMAN_FOR_DEBUG', time=time.time(), run_id=RUN,
        original_stage=stage, authorization_path=str(preservation/'authorization-before-signals.json'),
        preserved_files=copied, actions=actions, targets_terminal=terminal, retriever_kept=kept,
        completed_opsd_updates=3156, final_seven_partial_shards=shards,
        partial_rows=sum(x['rows'] for x in shards), partial_is_not_complete_final_evaluation=True,
        compute_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'], text=True).splitlines(),
        actual_memory_free=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'], text=True).splitlines())
    write_new(BASE/'receipts/user-stop-eight-evaluation-for-debug-20261008.json', evidence)
    pointer = dict(phase='STOPPED_BY_USER_FOR_GPU_DEBUG', time=time.time(), run_id=RUN,
        source_controller_pid=CONTROLLER, user_stop_receipt=str(BASE/'receipts/user-stop-eight-evaluation-for-debug-20261008.json'),
        scope='Latest explicit human stop of old K1 evaluation; not a fault or pause of diagnostic work. Do not automatically restart old baseline/raw seven jobs.')
    temp = BASE/'receipts/pipeline-stage.user-stop.tmp'
    temp.write_text(json.dumps(pointer, indent=2)); temp.replace(BASE/'receipts/pipeline-stage.json')
    print(json.dumps(dict(status=evidence['status'], partial_rows=evidence['partial_rows'],
        retriever_pid=CANDIDATE, receipt=str(BASE/'receipts/user-stop-eight-evaluation-for-debug-20261008.json'))), flush=True)


if __name__ == '__main__':
    main()
