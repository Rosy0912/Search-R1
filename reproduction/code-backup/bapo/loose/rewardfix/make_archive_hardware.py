from pathlib import Path
import base64
terminal=base64.b64encode(Path('work/evidence/rewardfix-hardware3064-terminal-watch.log').read_bytes()).decode()
s='''import pathlib,tarfile,hashlib,json,base64
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097');run=root/'runs/bapo-hardware-3064'
assert (run/'HARDWARE_MEASUREMENTS_RETURNED_SUCCESSFULLY').exists()
cleanup=json.loads((run/'own-process-cleanup.json').read_text());assert cleanup['remaining']==[]
p=root/'logs/hardware3064-terminal-slurm.log';b=base64.b64decode(TERMINAL)
if p.exists():assert p.read_bytes()==b
else:p.write_bytes(b)
archive=root/'logs/bapo-rewardfix-hardware3064-raw.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for path in sorted(run.rglob('*')):
  if path.is_file():
   assert not path.is_symlink()
   t.add(path,arcname='bapo-hardware-3064/'+str(path.relative_to(run)),recursive=False)
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),scope='NEW_B097_HARDWARE_RAW')))
'''.replace('TERMINAL',repr(terminal))
Path('work/rewardfix/archive-hardware.py').write_text(s)
