"""Execute frozen direct-HKUST experiment A under inherited GPU leases."""
import json,os,hashlib,socket,subprocess,time
from pathlib import Path
from omegaconf import OmegaConf
b=Path(os.environ['RECURSIVE_BRAN_BASE']).resolve();r=json.loads((b/'receipts/configuration.json').read_text())
assert socket.gethostname().split('.')[0]=='xielab1' and os.getuid()==1005
assert hashlib.sha256(Path(r['config']).read_bytes()).hexdigest()==r['config_sha256']
if (b/'output/fresh-initialization.json').exists():raise RuntimeError('No blind fresh restart after initialization')
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
os.environ.update(SEARCH_R1_BASELINE_RUN_ID=r['run_id'],E1_RUN_ID=r['run_id'],
 SEARCH_R1_EXECUTION_BACKEND='hkust_direct',SEARCH_R1_BASELINE_COMMIT=commit,
 SEARCH_R1_BASELINE_CONFIG_SHA256=r['config_sha256'],SEARCH_R1_BASELINE_DATA_IDENTITY=r['data_identity'],
 SEARCH_R1_MODEL_IDENTITY='Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1',
 SEARCH_R1_BASELINE_OUTPUT=str(b/'output'),SEARCH_R1_BASELINE_AUDIT_DIR=str(b/'output/audit'),
 SEARCH_R1_TREE_RANK_AUDIT_DIR=str(b/'output/rank-audit'),SEARCH_R1_TRAJECTORY_DIR=str(b/'output/trajectories'),
 TREE_COMMON_DATA=str(b/'common-data-v2'),TREE_INSTRUCT_MODEL=r['model'],
 SLURM_JOB_ID='hkust-direct-'+r['run_id'],SLURM_JOB_NODELIST='xielab1',RAY_ADDRESS='local')
# SLURM_JOB_ID is a compatibility provenance key only, not a Slurm allocation claim.
(b/'output').mkdir(exist_ok=True)
(b/'receipts/start.json').write_text(json.dumps(dict(time=time.time(),pid=os.getpid(),uid=os.getuid(),
 run_id=r['run_id'],source_commit=commit,execution_backend='hkust_direct',slurm_allocation=False)))
from search_r1.baselines.main import main
main.__wrapped__(OmegaConf.load(r['config']))
