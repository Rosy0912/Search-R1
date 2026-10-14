"""Live small state reader; process identity and observable work, never signals."""
import json
from pathlib import Path
import shutil
import subprocess
import time

b = Path('/dataset/shaoshitong/recursive-ablation150-20261014')

def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None

def proc(pid):
    p = Path('/proc', str(pid))
    try:
        s = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, uid=p.stat().st_uid, tick=int(s[19]), state=s[0])
    except FileNotFoundError:
        return None

state = read(b / 'study-progress.json')
group = state.get('current') if state else None
record = dict(time=time.time(), deployment=read(b / 'deployment.json'),
              study=state, study_start=read(b / 'study-start.json'),
              launch=read(b / 'study-launch.json'), actual_exit=read(b / 'study-actual-exit.json'))
if record['study_start']:
    record['study_process'] = proc(record['study_start']['pid'])
if group:
    g = b / group
    entry = read(g / 'receipts/current-entry.json')
    record['entry'] = entry
    if entry:
        record['entry_process'] = proc(entry['pid'])
        output = Path(entry['output'])
        record['progress'] = read(output / 'progress.json')
        record['entry_exit'] = read(output / 'entry-exit.json')
        record['native_complete'] = read(output / 'training-complete.json')
    log = g / 'logs/training.log'
    if log.is_file():
        size = log.stat().st_size
        with log.open('rb') as f:
            f.seek(max(0, size-24000))
            lines = f.read().decode(errors='replace').splitlines()
        keys = ('step:', 'Traceback', 'Error', 'Exception', 'ABLATION_', 'rollout/',
                'TREE_', 'RECURSIVE_', 'timing', 'epoch ', 'CUDA', 'Starting')
        record['log'] = dict(bytes=size, mtime=log.stat().st_mtime,
            evidence=[s[:1600] for s in lines if any(k in s for k in keys)][-18:])
for name, args in [('gpu', ['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits']),
                   ('apps', ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'])]:
    r = subprocess.run(args, capture_output=True, text=True, timeout=15)
    record[name] = dict(rc=r.returncode, stdout=r.stdout, stderr=r.stderr)
record['disk_free'] = shutil.disk_usage(b if b.exists() else b.parent).free
print(json.dumps(record, allow_nan=False), flush=True)
