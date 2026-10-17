"""Unique full-state PJ return entry; existing native algorithm modules unchanged."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid

from release.recursive_bran_a.ablation150_v1.contracts import read_receipt, sha
from .pj_return import load, validate_allocation as validate_return_allocation, validate_probe, worker_setup, ENV

def runtime_environment(record, output, receipt_path):
    return dict(RECURSIVE_ABLATION_RECEIPT=str(receipt_path),
        SEARCH_R1_BASELINE_RUN_ID=record['run_id'], E1_RUN_ID=record['run_id'],
        SEARCH_R1_EXECUTION_BACKEND='slurm_lumia',
        SEARCH_R1_BASELINE_COMMIT=record['source_commit'],
        SEARCH_R1_BASELINE_CONFIG_SHA256=record['config_sha256'],
        SEARCH_R1_BASELINE_DATA_IDENTITY=record['data_identity'],
        SEARCH_R1_MODEL_IDENTITY='Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1',
        SEARCH_R1_BASELINE_OUTPUT=str(output), SEARCH_R1_BASELINE_AUDIT_DIR=str(output/'audit'),
        SEARCH_R1_TREE_RANK_AUDIT_DIR=str(output/'rank-audit'),
        SEARCH_R1_TRAJECTORY_DIR=str(output/'trajectories'),
        TREE_COMMON_DATA=record['data'], TREE_INSTRUCT_MODEL=record['model'],
        RECURSIVE_A_LUMIA_FRESH_PROFILE='1', RECURSIVE_A_OMIT_FULL_BETA='1',
        RECURSIVE_A_GUARD_ROUND_PROGRESS='1', TREE_LIGHTWEIGHT_CANDIDATE_REWARD='1',
        RECURSIVE_A_SPARSE_DETAILS='1', RECURSIVE_LUMIA_PAIRED_EVALUATION='1')


def execute(args):
    receipt_path = Path(args.receipt).resolve()
    record = read_receipt(receipt_path)
    base = Path(record['base'])
    binding, bound_record, manifest = load(record['input_binding'])
    if record != bound_record:raise ValueError('Training receipt differs from bound projection')
    validate_return_allocation(binding, record)
    # Keep this lock open through Ray shutdown; independent launches cannot
    # enter the same run even if they race before current-entry is written.
    with (base/'receipts/entry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _execute_locked(args,receipt_path,record,base)


def _execute_locked(args, receipt_path, record, base):
    from omegaconf import OmegaConf
    from release.recursive_bran_a.ablation150_v1.progress import replace_json
    binding_path=Path(record['input_binding']); stage=binding_path.parent
    binding, _, parent_manifest=load(binding_path)
    job=os.environ['SLURM_JOB_ID']
    probe=json.loads((stage/('return-probe-'+job+'.json')).read_text())
    validate_probe(probe,binding,parent_manifest,job)
    previous_path=base/'receipts/current-entry.json'
    if previous_path.exists():
        if sha(previous_path)!=binding['previous_lumia_entry_sha256']:
            raise ValueError('Existing entry changed; no duplicate continuation')
        archive=stage/('previous-entry-before-'+job+'.json')
        with archive.open('xb') as f:f.write(previous_path.read_bytes())
    elif binding['previous_lumia_entry_sha256'] is not None:
        raise ValueError('Previous lineage disappeared')
    for name in os.environ:
        if name.startswith(('TREE_RECOVERY_','TREE_CURRENT_CHAIN_RESUME_',
                            'RECURSIVE_A_CONTINUATION_','RECURSIVE_ABLATION_','RBA_H100_')):
            raise ValueError('Inherited recovery environment forbidden: '+name)
    from .pj_return import dump_new
    dump_new(stage/'return-cutover.json',dict(at=time.time(),job=job,
        parent_job=binding['parent_job'],parent_manifest_sha256=binding['metadata_sha256']['parent-manifest.json'],
        input_binding_sha256=sha(binding_path),probe_sha256=sha(stage/('return-probe-'+job+'.json')),
        completed_outer=parent_manifest['identity']['completed_step'],next_outer=parent_manifest['driver']['next_step']))
    output = base/'output'/('attempt-'+uuid.uuid4().hex)
    output.mkdir(exist_ok=False)
    os.environ.update(runtime_environment(record,output,receipt_path))
    os.environ.update({ENV:str(binding_path),
        'RECURSIVE_ABLATION_RESUME_MANIFEST':str(stage/'parent-manifest.json'),
        'RECURSIVE_ABLATION_RESUME_SHA256':binding['metadata_sha256']['parent-manifest.json']})
    worker_setup()
    config = OmegaConf.load(record['config'])
    from search_r1.baselines.preflight import validate_gpu_lease
    validate_gpu_lease(config)
    entry = dict(time=time.time(),pid=os.getpid(),uid=os.getuid(),
        tick=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        host=socket.gethostname().split('.')[0],allocation=os.environ['SLURM_JOB_ID'],
        run_id=record['run_id'],group=record['group'],output=str(output),
        config_sha256=record['config_sha256'],source_commit=record['source_commit'],
        entry_source_commit=os.environ['RECURSIVE_SOURCE_COMMIT'],
        gpu_hardware=subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,name,uuid,memory.total,driver_version',
            '--format=csv,noheader,nounits'],text=True).splitlines(),
        execution_backend='slurm_lumia',resume_manifest=str(stage/'parent-manifest.json'),
        configuration_receipt=str(receipt_path),input_binding=str(binding_path),
        parent_job=binding['parent_job'],parent_config_sha256=record['parent_config_sha256'])
    replace_json(previous_path,entry)
    print('LUMIA_ABLATION_ENTRY='+json.dumps(entry),flush=True)
    from release.recursive_bran_a.h100_migration.ray_cli_compat import prepare_click
    prepare_click()
    import ray
    env = {k:v for k,v in os.environ.items()
           if k.startswith(('RECURSIVE_ABLATION_','RECURSIVE_A_','RECURSIVE_LUMIA_',
                            'SEARCH_R1_','TREE_'))}
    for key in ('SLURM_JOB_ID','SLURM_JOB_NODELIST','RECURSIVE_SOURCE_COMMIT','PYTHONPATH','CPATH',
                'C_INCLUDE_PATH','TRITON_LIBCUDA_PATH','LIBRARY_PATH'):
        if key in os.environ:env[key]=os.environ[key]
    env.update(TOKENIZERS_PARALLELISM='false',NCCL_DEBUG='WARN')
    error = None
    try:
        context = ray.init(address='local',num_gpus=8,num_cpus=64,include_dashboard=False,
            object_store_memory=8*1024**3,_node_ip_address='127.0.0.1',
            _temp_dir='/tmp/la-'+os.environ['SLURM_JOB_ID'],
            runtime_env={'env_vars':env,'worker_process_setup_hook':worker_setup})
        os.environ['RAY_ADDRESS']=context.address_info['gcs_address']
        from search_r1.baselines.main import main
        main.__wrapped__(config)
    except BaseException as failure:
        error=repr(failure)
        raise
    finally:
        ray.shutdown()
        replace_json(output/'entry-exit.json',dict(time=time.time(),pid=entry['pid'],
            uid=entry['uid'],tick=entry['tick'],run_id=record['run_id'],group=record['group'],
            rc=0 if error is None else 1,error=error,
            scope='Native entry return; scheduler state and actual parent wait are separate'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',required=True)
    execute(p.parse_args())


if __name__ == '__main__':
    main()
