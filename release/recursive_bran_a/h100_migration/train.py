import json,os,hashlib,socket,time
from pathlib import Path
from omegaconf import OmegaConf
base=Path(os.environ['RECURSIVE_BRAN_BASE']);r=json.loads((base/'receipts/configuration.json').read_text())
assert socket.gethostname().split('.')[0]=='h100-11' and os.environ['SLURM_JOB_ID']==os.environ['RBA_H100_ALLOCATION_ID']
assert hashlib.sha256(Path(r['config']).read_bytes()).hexdigest()==r['config_sha256']
out=base/'output';out.mkdir(exist_ok=True)
assert not (out/'fresh-initialization.json').exists(),'Never blindly restart a fresh run'
os.environ.update(SEARCH_R1_BASELINE_RUN_ID=r['run_id'],E1_RUN_ID=r['run_id'],SEARCH_R1_EXECUTION_BACKEND='slurm_h100',SEARCH_R1_BASELINE_COMMIT=r['source_commit'],SEARCH_R1_BASELINE_CONFIG_SHA256=r['config_sha256'],SEARCH_R1_BASELINE_DATA_IDENTITY=r['data_identity'],SEARCH_R1_MODEL_IDENTITY='Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1',SEARCH_R1_BASELINE_OUTPUT=str(out),SEARCH_R1_BASELINE_AUDIT_DIR=str(out/'audit'),SEARCH_R1_TREE_RANK_AUDIT_DIR=str(out/'rank-audit'),SEARCH_R1_TRAJECTORY_DIR=str(out/'trajectories'),TREE_COMMON_DATA=r['data'],TREE_INSTRUCT_MODEL=r['model'],RECURSIVE_A_OMIT_FULL_BETA='1',RECURSIVE_A_GUARD_ROUND_PROGRESS='1',TREE_LIGHTWEIGHT_CANDIDATE_REWARD='1')
(base/'receipts/train-start.json').write_text(json.dumps(dict(time=time.time(),pid=os.getpid(),run_id=r['run_id'],job=os.environ['SLURM_JOB_ID'],fresh=True)))
from search_r1.baselines.main import main
main.__wrapped__(OmegaConf.load(r['config']))
