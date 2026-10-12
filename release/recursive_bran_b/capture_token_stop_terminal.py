"""Capture the actual own outer9 programming failure, without sending signals."""
import hashlib,json,time
from pathlib import Path
from search_r1.baselines.recursive_b_completed_resume import ROOT,PARENT_RUN,PARENT_COMMIT,validate_terminal

def main():
    out=ROOT/('output-speed-resume3-'+PARENT_RUN)
    failure=out/'token-preservation-rejections/cecc08dd0453442f87a7db813708814d.json'
    f=json.loads(failure.read_text())
    if 'Native returned stopping text differs' not in f['error']:raise ValueError('Different failure')
    writes=list((out/'rank-audit/policy-safety').glob('step-000009*'))
    if writes:raise ValueError('Outer9 update evidence needs explicit accounting')
    manifest=out/'checkpoints/training_state/global_step_8/manifest.json'
    m=json.loads(manifest.read_text())
    if m['identity']['completed_step']!=8 or any(r['actual_total_optimizer_calls']!=160 for r in m['worker_states']):
        raise ValueError('Parent boundary changed')
    t=dict(protocol='recursive-b-token-stop-parent-terminal-v1',run_id=PARENT_RUN,source_commit=PARENT_COMMIT,
        status='PROGRAMMING_FAILURE_ALL_OWNED_KERNEL_EXITED',failed_outer=9,completed_native_outer=8,
        outer9_optimizer_calls=0,controller_pid=1024017,main_pid=1026405,
        worker_pids=[1026891,1027302,1027303,1027304],verified_at=time.time(),
        failure_sha256=hashlib.sha256(failure.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),signals_sent=False)
    validate_terminal(t)
    path=ROOT/'receipts/token-stop9-a177507c-terminal.json'
    if path.exists():
        previous=json.loads(path.read_text());validate_terminal(previous)
        if previous['failure_sha256']!=t['failure_sha256']:raise ValueError('Terminal collision')
    else:path.write_text(json.dumps(t,sort_keys=True))
    print(json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),receipt=json.loads(path.read_text()))))

if __name__=='__main__':main()
