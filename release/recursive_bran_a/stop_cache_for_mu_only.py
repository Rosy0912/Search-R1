"""Pinned own-tree stop for the human-authorized omission of full old HF beta."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

B = Path('/data/shaoshitong/recursive-bran-a-8gpu-20261011')
ROOT, TICK, UID = 3488460, 953384787, 1005
RUN = 'd9e99cd7924a44c7aaa63c87c0deb271'
MANIFEST = B/'checkpoints/training_state/global_step_4/manifest.json'
SHA = '4d781f6920ce8ea8d56bff7c488ea960aa5b60e7b99a7635d19961bbfa06816f'
RECEIPT = B/'receipts/sparse-schedule/cache-for-mu-only-owned-stop.json'


def read(pid):
    p = Path('/proc', str(pid))
    stat = (p/'stat').read_text().rsplit(') ', 1)[1].split()
    return dict(pid=pid, uid=p.stat().st_uid, ppid=int(stat[1]), tick=int(stat[19]), state=stat[0],
                argv=(p/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'))


assert os.getuid() == UID and not RECEIPT.exists()
raw = MANIFEST.read_bytes()
assert hashlib.sha256(raw).hexdigest() == SHA
manifest = json.loads(raw)
assert manifest['status'] == 'COMPLETE_SAME_WORLD_STATE_SAVED'
assert manifest['identity']['guard_continue_accounting']['total_optimizer_calls_per_rank'] == {str(r): 80 for r in range(8)}
assert not (B/'checkpoints/training_state/global_step_5/manifest.json').exists(), 'A completed fifth boundary requires a new pinned continuation'
owner = read(ROOT)
assert owner['uid'] == UID and owner['tick'] == TICK
env = dict(item.split(b'=', 1) for item in Path('/proc', str(ROOT), 'environ').read_bytes().split(b'\0') if b'=' in item)
assert env[b'RECURSIVE_A_SOURCE'].decode() == str(B/'source-cache-b905987')
assert env[b'RECURSIVE_A_CHILD_CONFIG_RECEIPT'].decode() == str(B/'receipts/sparse-schedule/cache-configuration.json')
start = json.loads((B/'receipts/cache-start-from4.json').read_text())
assert (start['pid'], start['uid'], start['run_id'], start['source_commit']) == (
    ROOT, UID, RUN, 'b905987d75d3034d58d5e1dfb235c9931cbb773b')
rows = {}
for p in Path('/proc').iterdir():
    if p.name.isdigit():
        try:
            rows[int(p.name)] = read(int(p.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
scope = {ROOT}
while True:
    children = {pid for pid, row in rows.items() if row['ppid'] in scope}
    if children <= scope:
        break
    scope |= children
assert 2239529 not in scope and all(rows[pid]['uid'] == UID for pid in scope)
gpu_pids = {int(line) for line in subprocess.check_output(
    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()}
assert gpu_pids <= scope | {2239529}, 'Unknown GPU owner; no takeover'
handles = {}
for pid in scope:
    if rows[pid]['state'] == 'Z':
        continue
    try:
        fd = os.pidfd_open(pid)
        assert read(pid)['tick'] == rows[pid]['tick']
        handles[pid] = fd
    except (FileNotFoundError, ProcessLookupError):
        pass
proof = dict(time=time.time(), reason='human authorized stop/resume to omit diagnostic full old-actor HF pass',
             owner_run=RUN, scope=[rows[pid] for pid in sorted(scope)], checkpoint=str(MANIFEST), sha256=SHA, signals=[])
RECEIPT.write_text(json.dumps(proof, indent=2))
# Freeze only authenticated own processes before teardown; never a node-wide kill.
for pid, fd in handles.items():
    try:
        signal.pidfd_send_signal(fd, signal.SIGSTOP)
    except ProcessLookupError:
        pass
for pid, fd in handles.items():
    try:
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        signal.pidfd_send_signal(fd, signal.SIGCONT)
        proof['signals'].append([pid, 'TERM'])
    except ProcessLookupError:
        pass
time.sleep(5)
for pid, fd in handles.items():
    try:
        row = read(pid)
        if row['tick'] == rows[pid]['tick'] and row['state'] != 'Z':
            signal.pidfd_send_signal(fd, signal.SIGKILL)
            proof['signals'].append([pid, 'KILL'])
    except (FileNotFoundError, ProcessLookupError):
        pass
    finally:
        os.close(fd)
for _ in range(20):
    remaining = []
    for pid in scope:
        try:
            row = read(pid)
            if row['tick'] == rows[pid]['tick'] and row['state'] != 'Z':
                remaining.append(row)
        except (FileNotFoundError, ProcessLookupError):
            pass
    if not remaining:
        break
    time.sleep(1)
proof.update(finished_at=time.time(), remaining=remaining)
assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == SHA
RECEIPT.write_text(json.dumps(proof, indent=2))
assert not remaining, 'Wait for actual owned release; do not launch over live descendants'
print(json.dumps(dict(stopped=True, owned_processes=len(scope), checkpoint_sha256=SHA)))
