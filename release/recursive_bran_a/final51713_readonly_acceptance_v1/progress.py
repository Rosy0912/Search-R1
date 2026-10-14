"""Read final evaluation receipts and original worker timing; no model or node writes."""
import json
import subprocess
import time
from pathlib import Path

BASE = Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013')
CONTROL = BASE / 'final51713-supervisor-recovery-v1'


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def tail(path, size=131072):
    if not path.exists():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-size))
        return stream.read().decode(errors='replace')


def main():
    global CONTROL
    recovery = BASE/'final51713-shortpath-recovery-v1'
    if (recovery/'evaluation-launch.json').exists():
        CONTROL = recovery
    tokenizer_recovery = BASE/'final51713-tokenizer-recovery-v1'
    if (tokenizer_recovery/'evaluation-launch.json').exists():
        CONTROL = tokenizer_recovery
    worker_recovery = BASE/'final51713-worker-recovery-v1'
    if (worker_recovery/'evaluation-launch.json').exists():
        CONTROL = worker_recovery
    plan = load(CONTROL / 'execution-plan.json')
    launch = load(CONTROL / 'evaluation-launch.json')
    terminal = load(CONTROL / 'evaluation-actual-exit.json')
    process = None
    if launch:
        p = Path('/proc') / str(launch['pid'])
        if p.exists():
            fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
            process = dict(pid=launch['pid'], uid=p.stat().st_uid,
                           tick=int(fields[19]), state=fields[0])
    records = {}
    if plan:
        runtime = load(CONTROL/'runtime-recovery.json')
        ray_root = Path(runtime['ray_tmpdir']) if runtime else Path('/tmp/rba-final-'+plan['run_id'])
        logs = list(ray_root.glob('**/logs/worker-*.out'))
        for path in logs + [CONTROL/'evaluation.log']:
            for line in tail(path, 524288).splitlines():
                if 'FINAL51713_GPU=' in line:
                    row = json.JSONDecoder().raw_decode(line.split('FINAL51713_GPU=', 1)[1])[0]
                    records[(row['event'], row['rank'], row['pid'], row['call'])] = row
    workers = {}
    for row in records.values():
        rank = str(row['rank'])
        w = workers.setdefault(rank, dict(pid=row['pid'], gpu_uuid=row['gpu_uuid'],
                                          latest_started_call=-1, latest_completed_call=-1))
        field = 'latest_completed_call' if row['event']=='complete' else 'latest_started_call'
        w[field] = max(w[field], row['call'])
        if row['event']=='complete' and row['call']==w['latest_completed_call']:
            w['latest_complete'] = row
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    summary = load(Path(plan['output'])/'summary.json') if plan else None
    progress = load(Path(plan['output'])/'progress.json') if plan else None
    log = tail(CONTROL/'evaluation.log', 16384)
    actions = []
    for line in tail(CONTROL/'evaluation.log', 4194304).splitlines():
        if 'TREE_ACTION_TIMING=' in line:
            row = json.JSONDecoder().raw_decode(line.split('TREE_ACTION_TIMING=', 1)[1])[0]
            actions.append(row)
    utilization = {}
    samples = []
    for line in tail(CONTROL/'gpu-device-observer.jsonl', 524288).splitlines()[1:]:
        try:
            sample = json.loads(line)
            if sample['rc']==0: samples.append(sample)
        except ValueError:
            pass
    for sample in samples[-60:]:
        for line in sample['device_sample'].strip().splitlines():
            fields = line.split(', ')
            utilization.setdefault(fields[1], []).append(float(fields[2]))
    utilization = {uuid:dict(samples=len(values),mean=sum(values)/len(values),peak=max(values))
                   for uuid,values in utilization.items()}
    timing = dict(retained_actions=len(actions),
                  wall_seconds=sum(row['wall_seconds'] for row in actions),
                  driver_cpu_seconds=sum(row['driver_cpu_seconds'] for row in actions))
    for key in ('generation_rpc_wall_seconds','retrieval_http_json_wall_seconds',
                'semantic_prefix_scan_wall_seconds','generated_tokens'):
        timing[key] = sum(row.get('cost_delta',{}).get(key,0) for row in actions)
    # Print structural events/errors only, never prompts, document contents, or answers.
    selected = [line for line in log.splitlines() if any(word in line for word in
                ('FINAL51713_', 'Traceback', 'Error:', 'Exception:', 'ERROR', 'Loading', 'checkpoint'))]
    print(json.dumps(dict(time=time.time(), control=str(CONTROL), plan=plan, launch=launch, actual_exit=terminal,
                          process=process, workers=workers, gpu=dict(rc=gpu.returncode, stdout=gpu.stdout,
                          stderr=gpu.stderr), summary=summary, progress=progress, structural_log_tail=selected[-15:],
                          recent_device_utilization=utilization, retained_action_timing=timing,
                          scope='Latest retained worker events and current whole-device utilization; cohost work may contribute')))


if __name__ == '__main__':
    main()
