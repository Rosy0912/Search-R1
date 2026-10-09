"""Remove only own redundant failed-run tar; preserve original run and logs archive."""
import os,json,hashlib,subprocess,datetime
from pathlib import Path
assert os.uname().nodename=='h100-11'
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v2-20261009')
run=r/'runs/formal-aligned-em200-09f93f1e773f4419928c0662e4537f32-3352'
assert (run/'cleanup-exit-code.txt').read_text().strip()=='0'
assert (run/'original-exit-code.txt').read_text().strip()=='1'
logs=r/'evidence/formal3352-failure-logs.tar'
assert hashlib.sha256(logs.read_bytes()).hexdigest()=='16afa37743622c8f05fa771d5582c8a131f64eef70410bae2f49d9f2363af56d'
p=r/'evidence/formal3352-failure-full.tar'
assert p.is_file() and not p.is_symlink()
n=p.stat().st_size
p.unlink()
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'removed_redundant_archive':str(p),'bytes':n,'original_run_preserved':run.is_dir(),'verified_logs_archive_preserved':logs.is_file(),'df':subprocess.check_output(['df','-B1',str(r)],text=True)}
print(json.dumps(out))
