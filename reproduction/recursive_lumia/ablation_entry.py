"""Task-owned Lumia entry using the unchanged native ablation bootstrap.

No H100 allocation is fabricated. Fresh runs and explicitly bound full-native
resumes use the actual Slurm allocation and the existing GPU lease checks.
"""
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
from .prepare_ablation import BASELINE_CONFIG_SHA, BASELINE_RUN_ID, condition_for

HOSTS = {'node9':'gpu-ada6000-2', 'node10':'gpu-l40s-1', 'node11':'gpu-l40s-2'}
TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL',
            'PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def validate_allocation(record, env, uid, host):
    devices = [x.strip() for x in env.get('CUDA_VISIBLE_DEVICES','').split(',') if x.strip()]
    if (record.get('execution_backend') != 'slurm_lumia' or uid != 1066
            or not env.get('SLURM_JOB_ID','').isdigit()
            or HOSTS.get(env.get('SLURM_JOB_NODELIST')) != host or record.get('host') != host
            or len(devices) != 8 or len(set(devices)) != 8
            or record['source_commit'] != env.get('RECURSIVE_SOURCE_COMMIT')):
        raise ValueError('The actual Lumia allocation, host, source and eight devices must match')
    if (record.get('template_sha256') != BASELINE_CONFIG_SHA
            or record.get('baseline_run_id') != BASELINE_RUN_ID
            or record['condition'] != condition_for(record['group'])):
        raise ValueError('Ablation is not bound to the actual original-A recipe')


def check_previous_scheduler(previous, current_job, query):
    job = str(previous['allocation'])
    if not job.isdigit() or job == current_job:
        raise ValueError('A previous entry needs a distinct, verified terminal allocation')
    rows = [line.split('|') for line in query(job).splitlines() if line.strip()]
    states = [row[1].split()[0] for row in rows if len(row)>1 and row[0] == job]
    if len(states) != 1 or states[0] not in TERMINAL:
        raise ValueError('Previous allocation is live or unknown; never duplicate/restart it')
    return states[0]


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
    validate_allocation(record,os.environ,os.getuid(),socket.gethostname().split('.')[0])
    # Keep this lock open through Ray shutdown; independent launches cannot
    # enter the same run even if they race before current-entry is written.
    with (base/'receipts/entry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _execute_locked(args,receipt_path,record,base)


def _execute_locked(args, receipt_path, record, base):
    from omegaconf import OmegaConf
    from release.recursive_bran_a.ablation150_v1.progress import replace_json
    previous_path = base/'receipts/current-entry.json'
    retry_proof = previous_state = None
    if bool(args.resume_manifest) != bool(args.resume_sha256):
        raise ValueError('Full native resume requires both manifest path and SHA')
    if args.retry_unstarted and args.resume_manifest:
        raise ValueError('Pre-initialization retry and native resume are different operations')
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        previous_state = check_previous_scheduler(previous,os.environ['SLURM_JOB_ID'],
            lambda job:subprocess.check_output(['sacct','-X','-n','-P','-j',job,
                                                '--format=JobIDRaw,State'],text=True))
        if args.retry_unstarted:
            from release.recursive_bran_a.ablation150_v1.unstarted_retry import check
            retry_proof = check(base,previous)
        elif not args.resume_manifest:
            raise ValueError('Existing training requires its full native state; no fresh reset')
    elif args.retry_unstarted or args.resume_manifest:
        raise ValueError('No previous entry exists for this condition')
    if not previous_path.exists() and any((base/'output').iterdir()):
        raise ValueError('Unexpected prior output; inspect it before fresh initialization')
    for name in os.environ:
        if name.startswith(('TREE_RECOVERY_','TREE_CURRENT_CHAIN_RESUME_',
                            'RECURSIVE_A_CONTINUATION_','RECURSIVE_ABLATION_','RBA_H100_')):
            raise ValueError('Inherited recovery/H100 environment is forbidden: '+name)
    output = base/'output'/('attempt-'+uuid.uuid4().hex)
    output.mkdir(exist_ok=False)
    os.environ.update(runtime_environment(record,output,receipt_path))
    if args.resume_manifest:
        os.environ.update(RECURSIVE_ABLATION_RESUME_MANIFEST=str(Path(args.resume_manifest).resolve()),
                          RECURSIVE_ABLATION_RESUME_SHA256=args.resume_sha256)
        from release.recursive_bran_a.ablation150_v1.native_resume import manifest
        manifest() # Exact own-group boundary admission before allocating a model.
    from release.recursive_bran_a.ablation150_v1.bootstrap import install,worker_setup
    install()
    config = OmegaConf.load(record['config'])
    from search_r1.baselines.preflight import validate_gpu_lease
    validate_gpu_lease(config)
    entry = dict(time=time.time(),pid=os.getpid(),uid=os.getuid(),
        tick=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        host=socket.gethostname().split('.')[0],allocation=os.environ['SLURM_JOB_ID'],
        run_id=record['run_id'],group=record['group'],output=str(output),
        config_sha256=record['config_sha256'],source_commit=record['source_commit'],
        entry_source_commit=os.environ['RECURSIVE_SOURCE_COMMIT'],
        execution_backend='slurm_lumia',resume_manifest=args.resume_manifest,
        previous_allocation_state=previous_state,retry_unstarted_proof=retry_proof)
    replace_json(previous_path,entry)
    print('LUMIA_ABLATION_ENTRY='+json.dumps(entry),flush=True)
    from release.recursive_bran_a.h100_migration.ray_cli_compat import prepare_click
    prepare_click()
    import ray
    env = {k:v for k,v in os.environ.items()
           if k.startswith(('RECURSIVE_ABLATION_','RECURSIVE_A_','RECURSIVE_LUMIA_',
                            'SEARCH_R1_','TREE_'))}
    for key in ('SLURM_JOB_ID','SLURM_JOB_NODELIST','PYTHONPATH','CPATH',
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
    p.add_argument('--resume-manifest');p.add_argument('--resume-sha256')
    p.add_argument('--retry-unstarted',action='store_true')
    execute(p.parse_args())


if __name__ == '__main__':
    main()
