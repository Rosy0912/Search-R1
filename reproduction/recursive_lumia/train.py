"""Single real Slurm allocation, fresh official model, native recursive-A trainer."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

def main():
    from omegaconf import OmegaConf
    base=Path(sys.argv[1]).resolve();arm=sys.argv[2];run=base/'runs'/arm
    r=json.loads((run/'receipts/configuration.json').read_text())
    aliases={'node10':'gpu-l40s-1','node11':'gpu-l40s-2'}
    assert os.environ.get('SLURM_JOB_ID') and aliases.get(os.environ['SLURM_JOB_NODELIST'])==r['host']==socket.gethostname().split('.')[0]
    assert os.getuid()==1066
    assert hashlib.sha256(Path(r['config']).read_bytes()).hexdigest()==r['config_sha256']
    assert not (run/'output/fresh-initialization.json').exists()
    out=run/'output'
    os.environ.update(SEARCH_R1_BASELINE_RUN_ID=r['run_id'],E1_RUN_ID=r['run_id'],
        SEARCH_R1_EXECUTION_BACKEND='slurm_lumia',SEARCH_R1_BASELINE_COMMIT=r['source_commit'],
        SEARCH_R1_BASELINE_CONFIG_SHA256=r['config_sha256'],SEARCH_R1_BASELINE_DATA_IDENTITY=r['data_identity'],
        SEARCH_R1_MODEL_IDENTITY='Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1',
        SEARCH_R1_BASELINE_OUTPUT=str(out), SEARCH_R1_BASELINE_AUDIT_DIR=str(out/'audit'),
        SEARCH_R1_TREE_RANK_AUDIT_DIR=str(out/'rank-audit'), SEARCH_R1_TRAJECTORY_DIR=str(out/'trajectories'),
        TREE_COMMON_DATA=r['data'],TREE_INSTRUCT_MODEL=r['model'],RECURSIVE_A_LUMIA_FRESH_PROFILE='1',
        RECURSIVE_A_OMIT_FULL_BETA='1',RECURSIVE_A_GUARD_ROUND_PROGRESS='1',
        TREE_LIGHTWEIGHT_CANDIDATE_REWARD='1',RECURSIVE_A_SPARSE_DETAILS='1')
    from release.recursive_bran_a.h100_migration.ray_cli_compat import prepare_click
    prepare_click()
    import ray
    ctx=ray.init(address='local',num_gpus=8,num_cpus=64,include_dashboard=False,
        object_store_memory=8*1024**3,_node_ip_address='127.0.0.1',
        _temp_dir='/tmp/rt14-'+os.environ['SLURM_JOB_ID'])
    os.environ['RAY_ADDRESS']=ctx.address_info['gcs_address']
    start=dict(at=time.time(),job=os.environ['SLURM_JOB_ID'],run_id=r['run_id'],arm=arm,source=r['source_commit'])
    (run/'receipts/train-start.json').write_text(json.dumps(start,indent=2))
    from search_r1.baselines.main import main as train
    train.__wrapped__(OmegaConf.load(r['config']))
    (run/'receipts/training-returned.json').write_text(json.dumps(dict(**start,ended=time.time())))

if __name__=='__main__':main()
