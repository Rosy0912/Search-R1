import json,os,subprocess
from pathlib import Path
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r4-20261006')
RUN=R/'runs/formal200-d156ac06a8d943759b860e6786894ba5-3306'
assert os.uname().nodename=='h100-5'
assert (R/'source-expected.txt').read_text().strip()=='5121a8d9349eadb5a81119cc4573165338e041bd'
def tail(p,n=7000):
 if not p.exists():return None
 assert p.resolve()==p and p.is_file() and not p.is_symlink()
 with p.open('rb') as f:
  f.seek(0,2);end=f.tell();f.seek(max(0,end-n));return f.read().decode(errors='replace')
files=['driver-time.log','phase.log','train.log','retriever.log','resource-watch.jsonl','resource-watch.json','driver-history.jsonl','original-exit-code.txt','own-process-cleanup.json']
record=dict(job_id=3306,source_commit='5121a8d9349eadb5a81119cc4573165338e041bd',run_root=str(RUN),slurm=subprocess.check_output(['scontrol','show','job','3306','-o'],text=True),files={name:tail(RUN/name,200000 if name=="resource-watch.jsonl" else 22000) for name in files},logs=sorted(p.name for p in (RUN/'logs').iterdir()) if (RUN/'logs').exists() else [],root_out=tail(R/'logs/slurm-formal200-3306.out'),root_err=tail(R/'logs/slurm-formal200-3306.err'))
record['initial_actor_proofs']={p.name:json.loads(p.read_text()) for p in (RUN/'diagnostics/worker-events').glob('native-initial-rank_*.json') if p.is_file() and not p.is_symlink()};record['checkpoint_logs']={p.name:tail(p,15000) for p in (RUN/'checkpoints/logs').glob('*') if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl')};print(json.dumps(record))
