from pathlib import Path
import tarfile,json,hashlib,subprocess
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004'); run=root/'runs/bapo-instruct300-formal300-3177'; logs=run/'logs'
slurm=subprocess.check_output(['scontrol','show','job','-dd','3177'],text=True); assert 'JobState=CANCELLED' in slurm
(logs/'user-stopped-slurm-terminal-20261005.txt').write_text(slurm)
files=[run/'train.log',run/'resource-watch.jsonl',run/'source-commit.txt',logs/'user-authorized-stop-20261005.json',logs/'user-stopped-slurm-terminal-20261005.txt',root/'logs/slurm-formal300-3177.out',root/'logs/slurm-formal300-3177.err']
for p in run.iterdir():
 if p.is_file() and p.suffix in ('.yaml','.json','.log') and p not in files: files.append(p)
target=logs/'user-stopped-raw-20261005.tar.gz'
with tarfile.open(target,'w:gz') as tf:
 for p in files:
  if p.is_file(): tf.add(p,arcname=str(p.relative_to(root)),recursive=False)
print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),members=[str(p) for p in files if p.is_file()],checkpoints_preserved=True)))
