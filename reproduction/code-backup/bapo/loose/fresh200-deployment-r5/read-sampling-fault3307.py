from pathlib import Path
import os,json,subprocess
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006');run=r/'runs/formal200-6a29642035ff40b2b7787de206f886c0-3307'
assert os.uname().nodename=='h100-5' and (r/'source-expected.txt').read_text().strip()=='aec5e057093b4c491bfb0a47abd7605d98981b56'
p=run/'checkpoints/diagnostics';files=[x for x in p.rglob('*') if x.is_file() and not x.is_symlink()]
print(json.dumps({'slurm':subprocess.check_output(['scontrol','show','job','3307','-o'],text=True),'files':[{'path':str(x),'bytes':x.stat().st_size} for x in files], 'json':{str(x):json.loads(x.read_text()) for x in files if x.suffix=='.json' and x.stat().st_size<100000}}))
