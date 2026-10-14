"""One-time replacement of two untrained CPU-retrieval launches, never resume.

Only the explicit owned jobs below qualify. Freeze, inspect the positive fresh
initialization receipt and absence of mandatory pre-optimizer stages, then cancel.
Any sign of training causes CONT and refusal. Preserve all files and wasted cost.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

JOBS = [('original', 'original', '138635'), ('curriculum', 'curriculum-ada-r2', '138639')]


def command(args):
    return subprocess.check_output(args, text=True).strip()


def inspect(base, arm, label, job):
    run = base/'runs'/arm
    start = json.loads((run/'receipts/train-start.json').read_text())
    fresh = json.loads((run/'output/fresh-initialization.json').read_text())
    assert str(start['job']) == fresh['job_id'] == job
    assert fresh['status'] == 'FRESH_OFFICIAL_INITIAL_STATE_OBSERVED'
    assert fresh['initialized_global_steps'] == fresh['driver_completed_outer'] == 0
    assert len(fresh['workers']) == 8 and {r['rank'] for r in fresh['workers']} == set(range(8))
    assert all(r['optimizer_committed'] == r['optimizer_state_entries'] == r['scheduler_step'] == 0
               for r in fresh['workers'])
    log = base/'logs'/f'{label}-{job}.log'
    text = log.read_text()
    assert 'TREE_TIMING=' in text, 'Need positive first-rollout evidence'
    # This flushed stage precedes any actor update in the pinned running source.
    assert 'TREE_DRIVER_STAGE=' not in text, 'Sampling already returned; use full state recovery'
    assert not list((run/'output/rank-audit').rglob('*.json'))
    assert not list((run/'output').rglob('attempt-*.json'))
    assert not list((run/'checkpoints').rglob('manifest.json'))
    audit = run/'output/audit/driver.jsonl'
    assert not audit.exists() or not audit.read_text().strip()
    if arm == 'curriculum':
        assert (run/'output/curriculum/screen-1.intent.json').is_file()
        assert not (run/'output/curriculum/screen-1.complete.json').exists()
        assert not (run/'output/curriculum/selected-batches.jsonl').exists()
    return dict(job=job, arm=arm, source=start['source'], run_id=start['run_id'],
        fresh_receipt_sha256=hashlib.sha256((run/'output/fresh-initialization.json').read_bytes()).hexdigest(),
        mandatory_pre_optimizer_stage_absent=True, rank_update_records=0, completed_rounds=0,
        reason='Slow CPU retrieval before first actor update; fresh retry on validated GPU retrieval',
        scope='Aborted untrained sampling and fresh initialization; not checkpoint resume or RNG continuation')


def main():
    base = Path(sys.argv[1]).resolve()
    assert str(base) == '/home/jmzhang/Workspace/Recursive-Tree-GRPO-20261014-v1' and os.getuid() == 1066
    gpu = json.loads((base/'receipts/gpu-retriever-ready.json').read_text())
    assert gpu['job'] == '138645' and gpu['validation']['top3_identical']
    health = json.loads(urllib.request.urlopen(gpu['url']+'/health', timeout=15).read())
    assert health == gpu
    receipt = base/'receipts/pre-optimizer-GPU-retrieval-switch.json'
    with receipt.open('x') as f:
        json.dump(dict(at=time.time(),status='INTENT', jobs=JOBS), f)
    records = []
    for arm, label, job in JOBS:
        assert command(['squeue','-h','-j',job,'--format=%i|%u|%j']) == f'{job}|jmzhang|rt14-{label}'
        inspect(base, arm, label, job)
        subprocess.run(['scancel','--signal=STOP','--full',job], check=True)
        cancelled = False
        try:
            time.sleep(2)
            proof = inspect(base, arm, label, job)
            subprocess.run(['scancel',job], check=True)
            cancelled = True
        finally:
            if not cancelled:
                subprocess.run(['scancel','--signal=CONT','--full',job], check=True)
        for _ in range(40):
            state = command(['sacct','-X','-j',job,'--format=State','-n','-P']).split('|')[0].strip()
            if state.startswith('CANCELLED'):
                break
            time.sleep(1)
        else:
            raise RuntimeError('Cancellation outcome incomplete; inspect before any new launch')
        proof.update(inspect(base, arm, label, job))
        old = base/'runs'/arm
        archive = base/'runs'/f'aborted-pre-optimizer-{arm}-{job}'
        assert not archive.exists()
        files = {str(p.relative_to(old)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in old.rglob('*') if p.is_file()}
        log = base/'logs'/f'{label}-{job}.log'
        proof.update(state=state,archive=str(archive),original_path=str(old),files=files,
            log=str(log),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
            scheduler_cost=command(['sacct','-X','-j',job,'--format=JobID,State,ElapsedRaw,AllocTRES','-n','-P']))
        old.rename(archive)
        records.append(proof)
        receipt.write_text(json.dumps(dict(at=time.time(),status='PRESERVING_ABORTED_PRE_OPTIMIZER_RUNS',
            records=records),indent=2))
    receipt.write_text(json.dumps(dict(at=time.time(),status='BOTH_UNTRAINED_RUNS_PRESERVED_FOR_FRESH_GPU_RETRIEVAL_RETRY',
        records=records, GPU_service=gpu, restart_is_resume=False,
        costs='All prior sampling tokens and allocated GPU time remain included as setup/retry overhead'),indent=2))
    print(json.dumps(dict(receipt=str(receipt),jobs=[r['job'] for r in records],status='PRESERVED')))


if __name__ == '__main__': main()
