"""Own-allocation supervised job; never runs RL or changes native author checkout."""
import os,json,subprocess,time,hashlib
from pathlib import Path
base=Path('/dataset/shaoshitong/search-r1-rl/cosearch');root=base/'author-native'
assert os.environ.get('SLURM_JOB_NODELIST')=='h100-16' and os.environ.get('SEARCH_R1_GPU_PREFLIGHT')=='1'
leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);assert len(leases)==8
fd=tuple(int(v['fd']) for v in leases)
tools=Path(__file__).parent
manifest=json.loads((tools/'approved-sha.json').read_text())
for name,sha in manifest.items():assert hashlib.sha256((tools/name).read_bytes()).hexdigest()==sha
engine_manifest=json.loads((tools/'engine-source-manifest.json').read_text())
for path,sha in engine_manifest.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
run=root/'runs/author-native-grammar-actor-continue-v4'
assert not run.exists(),'Existing immutable warm run must not be overwritten'
started=time.time()
p=subprocess.run([str(base/'env/bin/python'),str(tools/'native_grammar_warm_start.py')],env=os.environ.copy(),pass_fds=fd)
run=root/'runs/author-native-grammar-actor-continue-v4'
record=dict(job=os.environ['SLURM_JOB_ID'],returncode=p.returncode,total_wall_seconds=time.time()-started,real_rl_updates=0,strict_joint_acceptance=False,approved_sha256=manifest)
(run/'driver-exit.json' if run.exists() else root/'logs/native-actor-continue-v4-failure.json').write_text(json.dumps(record,indent=2)+'\n')
raise SystemExit(p.returncode)
