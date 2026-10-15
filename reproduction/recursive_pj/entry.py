"""Fresh PT-native training entry using the unchanged recursive ablation hooks."""
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid
from release.recursive_bran_a.ablation150_v1.contracts import read_receipt
from reproduction.recursive_lumia.ablation_entry import runtime_environment
from .prepare import BASE,prepare


def train(receipt_path):
    from omegaconf import OmegaConf
    from release.recursive_bran_a.ablation150_v1.progress import replace_json
    record=read_receipt(receipt_path);base=Path(record['base']);job=os.environ['SEARCH_R1_JOB_ID']
    assert record['execution_backend']=='pj_pt' and record['allocation']==job and not os.environ.get('SLURM_JOB_ID')
    assert os.getuid()==0 and socket.gethostname().split('.')[0]==record['host']
    assert os.environ['RECURSIVE_SOURCE_COMMIT']==record['source_commit']
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==8
    with (base/'receipts/entry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (base/'receipts/current-entry.json').exists(),'Existing run requires native recovery; no automatic reset'
        assert not any((base/'output').iterdir())
        output=base/'output'/('attempt-'+uuid.uuid4().hex);output.mkdir()
        env=runtime_environment(record,output,receipt_path)
        env.pop('RECURSIVE_A_LUMIA_FRESH_PROFILE')
        env.update(SEARCH_R1_EXECUTION_BACKEND='pj_pt',SEARCH_R1_JOB_ID=job,
                   SEARCH_R1_PJ_NODE=record['host'],RECURSIVE_A_PJ_FRESH_PROFILE='1')
        os.environ.update(env)
        from release.recursive_bran_a.ablation150_v1.bootstrap import install,worker_setup
        install()
        config=OmegaConf.load(record['config'])
        from search_r1.baselines.preflight import validate_gpu_lease
        validate_gpu_lease(config)
        entry=dict(time=time.time(),pid=os.getpid(),uid=os.getuid(),tick=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
          host=record['host'],scheduler_node=record['scheduler_node'],allocation=job,run_id=record['run_id'],
          group=record['group'],output=str(output),config_sha256=record['config_sha256'],
          source_commit=record['source_commit'],entry_source_commit=record['source_commit'],execution_backend='pj_pt',
          gpu_hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,driver_version','--format=csv,noheader,nounits'],text=True).splitlines())
        replace_json(base/'receipts/current-entry.json',entry)
        print('PJ_ABLATION_ENTRY='+json.dumps(entry),flush=True)
        from release.recursive_bran_a.h100_migration.ray_cli_compat import prepare_click
        prepare_click()
        import ray
        inherited={k:v for k,v in os.environ.items() if k.startswith(('SEARCH_R1_','RECURSIVE_A_','RECURSIVE_ABLATION_','RECURSIVE_LUMIA_','TREE_'))}
        for k in ('PYTHONPATH','CPATH','C_INCLUDE_PATH','TRITON_LIBCUDA_PATH','LIBRARY_PATH'):
            if k in os.environ:inherited[k]=os.environ[k]
        inherited.update(TOKENIZERS_PARALLELISM='false',NCCL_DEBUG='WARN')
        error=None
        try:
            context=ray.init(address='local',num_gpus=8,num_cpus=64,include_dashboard=False,
              object_store_memory=8*1024**3,_node_ip_address='127.0.0.1',_temp_dir='/tmp/'+job,
              runtime_env={'env_vars':inherited,'worker_process_setup_hook':worker_setup})
            os.environ['RAY_ADDRESS']=context.address_info['gcs_address']
            from search_r1.baselines.main import main
            main.__wrapped__(config)
        except BaseException as exc:
            error=repr(exc);raise
        finally:
            ray.shutdown()
            replace_json(output/'entry-exit.json',dict(time=time.time(),pid=entry['pid'],uid=entry['uid'],tick=entry['tick'],
              run_id=record['run_id'],group=record['group'],rc=0 if error is None else 1,error=error))


def launch(group):
    # This executes within the actual PT container. PT admission and node allocation
    # are independently recorded by the submitter; hostname can be a pod hostname.
    assert os.environ['SEARCH_R1_EXECUTION_BACKEND']=='pj_pt' and not os.environ.get('SLURM_JOB_ID')
    job=os.environ['SEARCH_R1_JOB_ID'];node=os.environ['RECURSIVE_PJ_SCHEDULER_NODE']
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        # GPU device namespace is restricted by the PT container runtime.
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).splitlines()
        assert len(rows)==8
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(x.strip() for x in rows)
    receipt=prepare(group,os.environ['RECURSIVE_SOURCE_COMMIT'],job,node)
    os.environ['RECURSIVE_ARM']=group
    proxy=subprocess.Popen([sys.executable,'-m','reproduction.recursive_lumia.proxy',str(BASE)])
    try:
        import urllib.request
        # Health is checked against the remote immutable receipt; proxy startup
        # races are bounded before training initialization.
        r=json.loads(receipt.read_text())
        with urllib.request.urlopen(r['retriever_url']+'/health',timeout=30) as f:
            h=json.load(f);assert h['job']==json.loads(Path(r['retriever_receipt']).read_text())['job']
        import socket as network
        for _ in range(30):
            assert proxy.poll() is None
            try:
                with network.create_connection(('127.0.0.1',29146),timeout=1):break
            except OSError:time.sleep(1)
        else:raise RuntimeError('Loopback proxy failed to start')
        child=subprocess.Popen([sys.executable,'-m','search_r1.baselines.preflight','--',sys.executable,
                                '-m','reproduction.recursive_pj.entry','train',str(receipt)])
        rc=child.wait()
        value=dict(at=time.time(),pid=child.pid,rc=rc,group=group,allocation=job,source=os.environ['RECURSIVE_SOURCE_COMMIT'])
        with (Path(r['base'])/'receipts/actual-parent-wait.json').open('x') as f:json.dump(value,f,indent=2)
        raise SystemExit(rc if rc>=0 else 128-rc)
    finally:
        proxy.terminate();proxy.wait(timeout=30)

if __name__=='__main__':
    if sys.argv[1]=='train':train(Path(sys.argv[2]))
    else:launch(sys.argv[1])
