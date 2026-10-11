"""Execute immutable B under inherited leases; never silently restart fresh."""
import json,os,hashlib,socket,time
from pathlib import Path
from omegaconf import OmegaConf
b=Path(os.environ['RECURSIVE_BRAN_BASE']).resolve();r=json.loads(Path(os.environ['RECURSIVE_BRAN_CONFIGURATION']).read_text())
assert socket.gethostname().split('.')[0]=='xie-lab-4' and os.getuid()==1001
assert hashlib.sha256(Path(r['config']).read_bytes()).hexdigest()==r['config_sha256']
if (b/'receipts/start.json').exists() or (b/'output/fresh-initialization.json').exists():
 raise RuntimeError('No blind restart; preserve and restore the actual full boundary')
repo=Path(__file__).parents[2].resolve()
assert str(repo)==r['source']
commit=(repo/'SOURCE_COMMIT').read_text().strip()
assert commit==r['source_commit']
os.environ.update(SEARCH_R1_BASELINE_RUN_ID=r['run_id'],E1_RUN_ID=r['run_id'],
 SEARCH_R1_EXECUTION_BACKEND='hkust_direct',SEARCH_R1_BASELINE_COMMIT=commit,
 SEARCH_R1_BASELINE_CONFIG_SHA256=r['config_sha256'],SEARCH_R1_BASELINE_DATA_IDENTITY=r['data_identity'],
 SEARCH_R1_MODEL_IDENTITY='Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1',
 SEARCH_R1_BASELINE_OUTPUT=str(b/'output'),SEARCH_R1_BASELINE_AUDIT_DIR=str(b/'output/audit'),
 SEARCH_R1_TREE_RANK_AUDIT_DIR=str(b/'output/rank-audit'),SEARCH_R1_TRAJECTORY_DIR=str(b/'output/trajectories'),
 TREE_COMMON_DATA=str(b/'common-data'),TREE_INSTRUCT_MODEL=r['model'],
 SLURM_JOB_ID='hkust-direct-'+r['run_id'],SLURM_JOB_NODELIST='xie-lab-4',RAY_ADDRESS='local')
# These compatibility keys describe provenance, not a Slurm allocation.
(b/'output').mkdir(exist_ok=True)
with (b/'receipts/start.json').open('x') as f:
 json.dump(dict(time=time.time(),pid=os.getpid(),uid=os.getuid(),run_id=r['run_id'],
               source_commit=commit,execution_backend='hkust_direct',slurm_allocation=False),f)
from search_r1.baselines.main import main
main.__wrapped__(OmegaConf.load(r['config']))
