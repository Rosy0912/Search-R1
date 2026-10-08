import json,os,subprocess
from pathlib import Path
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r3-20261006')
RUN=R/'runs/formal200-00008d53f5fb4ac6b2205e4b92fe4fa0-3305'
assert os.uname().nodename=='h100-5'
assert (R/'source-expected.txt').read_text().strip()=='379b14c65e127cd3a71524f7783deaebafbbdb41'
def tail(p,n=7000):
 if not p.exists():return None
 assert p.resolve()==p and p.is_file() and not p.is_symlink()
 with p.open('rb') as f:
  f.seek(0,2);end=f.tell();f.seek(max(0,end-n));return f.read().decode(errors='replace')
files=['driver-time.log','phase.log','train.log','retriever.log','resource-watch.jsonl','resource-watch.json','driver-history.jsonl','original-exit-code.txt','own-process-cleanup.json']
record=dict(job_id=3305,source_commit='379b14c65e127cd3a71524f7783deaebafbbdb41',run_root=str(RUN),slurm=subprocess.check_output(['scontrol','show','job','3305','-o'],text=True),files={name:tail(RUN/name,200000 if name=="resource-watch.jsonl" else 22000) for name in files},logs=sorted(p.name for p in (RUN/'logs').iterdir()) if (RUN/'logs').exists() else [],root_out=tail(R/'logs/slurm-formal200-3305.out'),root_err=tail(R/'logs/slurm-formal200-3305.err'))
print(json.dumps(record))
