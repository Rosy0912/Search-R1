"""Same-process entry preserves all eight GPU lease FDs and terminal failures."""
import json,os,hashlib,runpy,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')
run=root/'runs/author-native-warm-hf-format-diagnostic-j'
tools=Path(__file__).parent
assert not run.exists(),'Existing immutable diagnostic run must not be overwritten'
try:
 for name,expected in json.loads((tools/'approved-sha.json').read_text())['tools_sha256'].items():
  assert hashlib.sha256((tools/name).read_bytes()).hexdigest()==expected,name
 runpy.run_path(str(tools/'probe.py'),run_name='__main__')
except BaseException as error:
 run.mkdir(exist_ok=True)
 (run/'driver-exit.json').write_text(json.dumps(dict(returncode=1,job=os.environ.get('SLURM_JOB_ID'),finished_at=time.time(),real_rl_updates=0,error=str(error))))
 raise
