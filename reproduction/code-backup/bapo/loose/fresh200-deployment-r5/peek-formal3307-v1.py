import json,os,subprocess
from pathlib import Path
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006')
RUN=R/'runs/formal200-6a29642035ff40b2b7787de206f886c0-3307'
assert os.uname().nodename=='h100-5'
assert (R/'source-expected.txt').read_text().strip()=='aec5e057093b4c491bfb0a47abd7605d98981b56'
def tail(p,n=7000):
 if not p.exists():return None
 assert p.resolve()==p and p.is_file() and not p.is_symlink()
 with p.open('rb') as f:
  f.seek(0,2);end=f.tell();f.seek(max(0,end-n));return f.read().decode(errors='replace')
files=['driver-time.log','phase.log','train.log','retriever.log','resource-watch.jsonl','resource-watch.json','driver-history.jsonl','original-exit-code.txt','own-process-cleanup.json']
record=dict(job_id=3307,source_commit='aec5e057093b4c491bfb0a47abd7605d98981b56',run_root=str(RUN),slurm=subprocess.check_output(['scontrol','show','job','3307','-o'],text=True),files={name:tail(RUN/name,200000 if name=="resource-watch.jsonl" else 22000) for name in files},logs=sorted(p.name for p in (RUN/'logs').iterdir()) if (RUN/'logs').exists() else [],root_out=tail(R/'logs/slurm-formal200-3307.out'),root_err=tail(R/'logs/slurm-formal200-3307.err'))
record['initial_actor_proofs']={p.name:json.loads(p.read_text()) for p in (RUN/'diagnostics/worker-events').glob('native-initial-rank_*.json') if p.is_file() and not p.is_symlink()};record['checkpoint_logs']={p.name:tail(p,15000) for p in (RUN/'checkpoints/logs').glob('*') if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl')};print(json.dumps(record))
