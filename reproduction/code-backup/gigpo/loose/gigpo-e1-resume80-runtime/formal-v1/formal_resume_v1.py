"""Explicit local-only GiGPO Slurm executor; never submits or resumes jobs."""
import argparse,fcntl,json,math,os,signal,socket,subprocess,sys,time
from pathlib import Path
from owned_runtime_resume_v1 import ROOT,require,sha,cgroup_scope,cgroup_pids,OwnedProcesses,process_sample,gpu_sample,available
REPO=Path(os.environ['GIGPO_RESUME_REPO'])
HELPERS=Path(__file__).resolve().parent
SOURCE=ROOT/'assets/qwen2.5-3b-instruct-aa8e725'
CONTENT='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'

def config(run,stage):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from search_r1.baselines.holdout import prepare_common_holdout
    from search_r1.baselines.controller import validate_config
    os.environ.update(GIGPO_E1_ROOT=str(ROOT),GIGPO_E1_RUN=str(run))
    with initialize_config_dir(config_dir=str(REPO/'verl/trainer/config'),version_base=None):
        value=compose(config_name='ppo_gigpo_instruct300_resume80')
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(value,resolve=True)),run/'input-config.yaml')
    split=prepare_common_holdout(value);validate_config(value)
    require(split['content_identity']['sha256']==CONTENT and split['train_rows']==169028 and split['holdout_rows']==512,'Common input identity drift')
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(value,resolve=True)),run/'resolved-config.yaml')
    return value,300

from resume_final_audit_v1 import audit_gate, audit_evaluations, raw_driver_snapshot

def interrupted(sig,frame):
    raise RuntimeError(f'Own executor interrupted by signal {sig}')

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['instruct300'],required=True)
    p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--previous-review',type=Path);p.add_argument('--previous-review-sha');args=p.parse_args()
    require(sha(args.binding)==args.binding_sha,'Deployment binding changed')
    binding=json.loads(args.binding.read_text());require(binding['status']=='REVIEWED','Need independent deployment review')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    require(commit==binding['code_commit'] and not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Dirty/unfrozen source')
    require(binding['restore_manifest_sha256']=='7e108f4d1de488e8484a25deb3d50d9f9a200f2b30e0c90ffc3f9bc68751ad03','Wrong full restore source')
    for item in binding['files']:
        require(sha(item['path'])==item['sha256'],'Frozen deployment file changed')
    for item in binding['evidence']:
        require(sha(item['path'])==item['sha256'],'Evidence changed')
    require(binding['fresh_model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1','Wrong official model')
    require(binding['common_manifest_sha256']==CONTENT and binding['native_status']=='PASS','Missing exact common inputs/native GPU acceptance')
    from search_r1.baselines.instruct300 import prepare_sampler
    job=os.environ.get('SLURM_JOB_ID','');require(job and os.environ.get('SLURM_JOB_GPUS'),'Need actual Slurm allocation')
    require(int(os.environ.get('SLURM_RESTART_COUNT','0'))==0,'Automatic Slurm restart forbidden; inspect latest checkpoint before a new owned continuation')
    require(socket.gethostname().split('.')[0]=='h100-8' and os.environ.get('SLURM_JOB_NODELIST')=='h100-8','Wrong node')
    require(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==8,'Need eight allocated GPUs')
    require(int(os.environ['SLURM_CPUS_PER_TASK'])>=128 and int(os.environ['SLURM_MEM_PER_NODE'])>=256*1024,'Allocation too small')
    lock=(ROOT/'gpu-job.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    validate_inherited_gpu_leases(8)  # Parent acquired all UUID leases before any service uses GPU.
    scope,relative=cgroup_scope(job);owned=OwnedProcesses(scope,relative)
    for port in (18081,16381):available(port)
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):os.environ.pop(key,None)
    os.environ.update(NO_PROXY='localhost,127.0.0.1,::1',no_proxy='localhost,127.0.0.1,::1')
    header_cpath='/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include:/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/python3.10:/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/x86_64-linux-gnu/python3.10'
    os.environ.update(CPATH=header_cpath,SEARCH_R1_PYTHON_HEADER_CPATH=header_cpath)
    run_id=f'gigpo-e1-instruct300-resume80-job{job}-{time.time_ns()}';run=ROOT/'runs'/run_id;run.mkdir()
    for name in ('audit','tmp','ray','cache'):(run/name).mkdir()
    alias=Path('/tmp')/f'gigpo-r{job}';require(not alias.exists() and not alias.is_symlink(),'Existing Ray alias');alias.symlink_to(run,target_is_directory=True)
    os.environ.update(AGENT_ROOT=str(ROOT),SEARCH_R1_ASSETS='/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root',SEARCH_R1_BASELINE_OUTPUT=str(run),SEARCH_R1_BASELINE_RUN_ROOT=str(alias),
      RAY_ADDRESS='127.0.0.1:16381',TMPDIR=str(alias/'tmp'),RAY_TMPDIR=str(alias/'ray'),XDG_CACHE_HOME=str(alias/'cache'),HF_HOME=str(alias/'cache/hf'),
      HF_DATASETS_CACHE='/mnt/data/shaoshitong/search-r1-rl/gigpo/cache/retriever-datasets',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
      HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',WANDB_MODE='offline',VLLM_NO_USAGE_STATS='1',NCCL_NVLS_ENABLE='0',VLLM_ATTENTION_BACKEND='XFORMERS')
    started=time.monotonic();result={'status':'FAILED','stage':args.stage,'run_id':run_id,'job_id':job}
    try:
        value,steps=config(run,args.stage);config_hash=sha(run/'resolved-config.yaml')
        os.environ.update(SEARCH_R1_BASELINE_AUDIT_DIR=str(run/'audit'),SEARCH_R1_BASELINE_RUN_ID=run_id,SEARCH_R1_BASELINE_COMMIT=commit,
          SEARCH_R1_BASELINE_CONFIG_SHA256=config_hash,SEARCH_R1_BASELINE_DATA_IDENTITY=CONTENT)
        prepare_sampler(value,169028)
        manifest=dict(stage=args.stage,run_id=run_id,code_commit=commit,config_sha256=config_hash,split_content_identity=CONTENT,
          slurm_job=job,cgroup=str(scope),allocation_cpu=int(os.environ['SLURM_CPUS_PER_TASK']),allocation_mem_mib=int(os.environ['SLURM_MEM_PER_NODE']),
          cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],retriever_url=value.retriever.url,ray_address=os.environ['RAY_ADDRESS'],
          runtime_constraint='ALL LOCAL h100-8: GPU E5, full 8-GPU sharded FP16 FAISS, model and scoring; loopback only; no A800 or remote RPC',
          retriever_threads=8,actor_threads=1,initial_gpu_snapshot=gpu_sample(),entry_input_config_sha256=sha(run/'input-config.yaml'),
          warmup_note='Restore scheduler80/LR1e-6 after absolute57 warmup; scheduler advances once/outer',
          fresh_model=str(SOURCE),fresh_revision=binding['fresh_model_revision'],
          start_logical=80,stop_logical=300,new_actual_budget=220,restore_checkpoint=value.trainer.instruct300.restore_checkpoint,restore_manifest_sha256=binding['restore_manifest_sha256'])
        (run/'launch-manifest.json').write_text(json.dumps(manifest,indent=2))
        from search_r1.baselines.preflight import gpu_snapshot,check_selected_gpus
        inventory,processes=gpu_snapshot();selected=check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inventory,processes)
        (ROOT/'tools/expected-gpu-uuids.json').write_text(json.dumps([g['uuid'] for g in selected]))
        guard=owned.spawn([sys.executable,str(ROOT/'tools/resource_guard.py'),'--output',str(run/'guard-resources.jsonl')],run/'guard.log')
        deadline=time.monotonic()+30
        while not (run/'guard-resources.ready').is_file():
            require(guard.poll() is None and time.monotonic()<deadline,'Resource guard failed');time.sleep(1)
        retriever=owned.spawn([sys.executable,'reproduction/h20/retriever_local.py','--index_path','/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root/assets/wiki/e5_Flat.index',
          '--corpus_path','/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root/assets/wiki/wiki-18.jsonl','--retriever_model','/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root/assets/e5-base-v2',
          '--retriever_name','e5','--topk','3','--faiss_gpu','--device','cuda','--threads','8','--host','127.0.0.1','--port','18081'],run/'retriever.log',
          dict(os.environ,PYTHONPATH='/mnt/data/shaoshitong/search-r1-rl/gigpo/retriever-gpu-lib'+os.pathsep+str(REPO),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'))
        import requests
        session=requests.Session();session.trust_env=False;deadline=time.monotonic()+900
        while True:
            require(retriever.poll() is None and guard.poll() is None and time.monotonic()<deadline,'Own local retriever failed')
            try:
                if session.get('http://127.0.0.1:18081/openapi.json',timeout=2).ok:break
            except requests.RequestException:pass
            time.sleep(2)
        subprocess.run([sys.executable,str(ROOT/'tools/check_http_retrieval.py')],check=True,
          env=dict(os.environ,SEARCH_R1_HTTP_RECEIPT=str(run/'local-http-retrieval-verification.json')))
        # GPU health probe has finished. Record coexistence; retain both15% eligibility checks.
        deadline=time.monotonic()+60
        while True:
            inventory,processes=gpu_snapshot()
            try:
                check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inventory,processes)
                break
            except RuntimeError:
                require(time.monotonic()<deadline,'Local GPU retriever exceeds existing launch eligibility; do not bypass')
                time.sleep(3)
        (run/'gpu-retriever-coexistence-before-train.json').write_text(json.dumps(dict(gpus=inventory,compute_gpu_uuids=sorted(processes),retriever_pid=retriever.pid,snapshot=gpu_sample()),indent=2))
        ray=owned.spawn([str(ROOT/'env/bin/ray'),'start','--head','--node-ip-address=127.0.0.1','--port=16381','--num-cpus=24','--num-gpus=8',
          '--include-dashboard=false','--object-store-memory=8589934592','--temp-dir='+str(alias/'ray'),'--min-worker-port=21081','--max-worker-port=22080',
          '--disable-usage-stats','--block'],run/'ray.log')
        deadline=time.monotonic()+120
        while True:
            require(ray.poll() is None and guard.poll() is None and time.monotonic()<deadline,'Own Ray failed')
            try:
                with socket.create_connection(('127.0.0.1',16381),timeout=1):break
            except OSError:time.sleep(2)
        subprocess.run([sys.executable,str(HELPERS/'ray_audit_resume_v1.py'),str(run/'ray-audit-environment-verification.json')],check=True)
        training=owned.spawn(['bash','reproduction/baselines/run.sh','gigpo','--config-dir',str(run),'--config-name','input-config'],run/'training.log',pass_fds=tuple(int(r['fd']) for r in json.loads(os.environ['SEARCH_R1_GPU_LEASES'])))
        pause_verifier=None
        with (run/'resources.jsonl').open('a') as stream:
            while training.poll() is None:
                require(all(proc.poll() is None for proc in (retriever,ray,guard)),'Own dependent service died')
                ready=run/'resume85-ready.json'
                if ready.is_file() and pause_verifier is None:
                    pause_verifier=owned.spawn([sys.executable,str(ROOT/'tools/verify-gigpo-checkpoint.py'),
                      '--base',str(value.trainer.instruct300.restore_checkpoint),'--checkpoint',str(run/'checkpoints/actor/global_step_85'),
                      '--output',str(run/'checkpoint85-cpu-verification.json')],run/'checkpoint85-cpu-verification.log',
                      env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'))
                if pause_verifier is not None and pause_verifier.poll() is not None:
                    require(pause_verifier.returncode==0,'Continued step85 full HF reload failed')
                if (run/'resume85-approval.json').exists():
                    require(pause_verifier is not None and pause_verifier.poll()==0,'Independent approval cannot precede successful full HF reload')
                stream.write(json.dumps(dict(elapsed_seconds=time.monotonic()-started,gpu=gpu_sample(),retriever=process_sample(retriever.pid),
                  processes=[process_sample(pid) for pid in sorted(cgroup_pids(scope)-owned.baseline)]))+'\n');stream.flush();time.sleep(10)
        require(training.returncode==0,'Actual training failed')
        require((run/'resume85-approval.json').is_file(),'No independently accepted first5 same-process continuation')
        checkpoint=audit_gate(run,steps,commit,config_hash)
        raw = raw_driver_snapshot(run)
        audit_evaluations(run, commit, config_hash, raw)
        result.update(status='ACTUAL_UPDATES_COMPLETED_UNREVIEWED',checkpoint=str(checkpoint),retained_original_updates=80,actual_new_updates=220,retained_total_updates=300)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        try:owned.cleanup()
        except BaseException as cleanup_error:
            result.update(status='FAILED',cleanup_error=repr(cleanup_error));raise
        finally:
            result['cleanup_residual_states']=getattr(owned,'cleanup_residual',{});alias.unlink(missing_ok=True);result['wall_seconds']=time.monotonic()-started
            result['final_gpu_snapshot']=gpu_sample();(run/'run-result.json').write_text(json.dumps(result,indent=2))
    # Real HF reload in a fresh CPU process, after own actor/Ray workers exit.
    reload_started=time.monotonic()
    try:
        subprocess.run([sys.executable,str(ROOT/'tools/verify-gigpo-checkpoint.py'),
        '--base',str(SOURCE),'--checkpoint',str(checkpoint),
        '--output',str(run/'checkpoint-cpu-verification.json')],check=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'))
    except BaseException as reload_error:
        result.update(status='FAILED',checkpoint_reload_error=repr(reload_error))
        (run/'run-result.json').write_text(json.dumps(result,indent=2));raise
    result.update(status='CHECKPOINT_RELOADED_UNREVIEWED',checkpoint_reload_wall_seconds=time.monotonic()-reload_started,
        complete_wall_seconds=time.monotonic()-started)
    (run/'run-result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);main()
