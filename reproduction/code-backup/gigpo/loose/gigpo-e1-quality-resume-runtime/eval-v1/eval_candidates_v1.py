"""Explicit local-only GiGPO Slurm executor; never submits or resumes jobs."""
import argparse,fcntl,json,math,os,signal,socket,subprocess,sys,time
from pathlib import Path
from owned_runtime_resume_v1 import ROOT,require,sha,cgroup_scope,cgroup_pids,OwnedProcesses,process_sample,gpu_sample,available
REPO=Path(os.environ['GIGPO_RESUME_REPO'])
HELPERS=Path(__file__).resolve().parent
SOURCE=ROOT/'assets/qwen2.5-3b-instruct-aa8e725'
CONTENT='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'

def finalize_snapshot(result, run):
    try:
        result['final_gpu_snapshot']=gpu_sample()
        if result['final_gpu_snapshot']['processes']:
            result.update(status='FAILED',gpu_cleanup_error='GPU compute remains after own cleanup')
    except BaseException as snapshot_error:
        result.update(status='FAILED',gpu_snapshot_error=repr(snapshot_error))
    (run/'run-result.json').write_text(json.dumps(result,indent=2))

def select_best(results):
    require(bool(results),'No complete candidate results')
    for r in results:
        require(r['status']=='GPU512_EVALUATED' and r['metrics']['val/qa_count/all']==512,'Incomplete or failed candidate')
        for k in ('val/f1/all','val/em/all'):
            require(math.isfinite(float(r['metrics'][k])) and 0<=float(r['metrics'][k])<=1,'Invalid candidate metric')
    return max(results,key=lambda r:(r['metrics']['val/f1/all'],r['candidate_step']))

def config(run,stage):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf,open_dict
    from search_r1.baselines.holdout import prepare_common_holdout
    from search_r1.baselines.controller import validate_config
    os.environ.update(GIGPO_E1_ROOT=str(ROOT),GIGPO_E1_RUN=str(run))
    with initialize_config_dir(config_dir=str(REPO/'verl/trainer/config'),version_base=None):
        value=compose(config_name='ppo_gigpo_instruct300')
    with open_dict(value):
        value.trainer.instruct300.enabled=False
        value.trainer.val_only=True
        value.trainer.val_before_train=True
        value.actor_rollout_ref.model.path=os.environ['GIGPO_EVAL_CANDIDATE']
        value.actor_rollout_ref.ref.model_path=str(SOURCE)
        value.trainer.experiment_name='gigpo-e1-quality512-candidate'+os.environ['GIGPO_EVAL_STEP']
    split=prepare_common_holdout(value);validate_config(value)
    require(split['content_identity']['sha256']==CONTENT and split['train_rows']==169028 and split['holdout_rows']==512,'Common512 identity drift')
    resolved=OmegaConf.create(OmegaConf.to_container(value,resolve=True))
    OmegaConf.save(resolved,run/'input-config.yaml');OmegaConf.save(resolved,run/'resolved-config.yaml')
    return value,0

def interrupted(sig,frame):
    raise RuntimeError(f'Own executor interrupted by signal {sig}')

def evaluate_candidate():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['quality512'],required=True)
    p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--previous-review',type=Path);p.add_argument('--previous-review-sha');args=p.parse_args()
    require(sha(args.binding)==args.binding_sha,'Deployment binding changed')
    binding=json.loads(args.binding.read_text());require(binding['status']=='REVIEWED','Need independent deployment review')
    require(binding.get('evaluation_only') is True,'This launcher authorizes only candidate evaluation')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    require(commit==binding['code_commit'] and not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Dirty/unfrozen source')
    for item in binding['files']:
        require(sha(item['path'])==item['sha256'],'Frozen deployment file changed')
    for item in binding['evidence']:
        require(sha(item['path'])==item['sha256'],'Evidence changed')
    require(binding['fresh_model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1','Wrong official model')
    require(binding['common_manifest_sha256']==CONTENT and binding['native_status']=='PASS','Missing exact common inputs/native GPU acceptance')
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
    run_id=f'gigpo-e1-quality512-step{os.environ["GIGPO_EVAL_STEP"]}-job{job}-{time.time_ns()}';run=ROOT/'runs'/run_id;run.mkdir()
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
        manifest=dict(stage=args.stage,run_id=run_id,code_commit=commit,config_sha256=config_hash,split_content_identity=CONTENT,
          slurm_job=job,cgroup=str(scope),allocation_cpu=int(os.environ['SLURM_CPUS_PER_TASK']),allocation_mem_mib=int(os.environ['SLURM_MEM_PER_NODE']),
          cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],retriever_url=value.retriever.url,ray_address=os.environ['RAY_ADDRESS'],
          runtime_constraint='ALL LOCAL h100-8: GPU E5, full 8-GPU sharded FP16 FAISS, model and scoring; loopback only; no A800 or remote RPC',
          retriever_threads=8,actor_threads=1,initial_gpu_snapshot=gpu_sample(),entry_input_config_sha256=sha(run/'input-config.yaml'),
          warmup_note='Evaluation only, no optimizer/scheduler or training updates',
          fresh_model=str(SOURCE),fresh_revision=binding['fresh_model_revision'],
          candidate_logical=int(os.environ['GIGPO_EVAL_STEP']),training_updates=0,actor_checkpoint=value.actor_rollout_ref.model.path)
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
        candidate=Path(os.environ['GIGPO_EVAL_CANDIDATE'])
        identity=json.loads((candidate/'training_state/manifest.json').read_text())
        require(sha(candidate/'training_state/manifest.json')==os.environ['GIGPO_EVAL_MANIFEST_SHA'],'Pinned candidate manifest changed')
        require(identity['logical_step']==int(os.environ['GIGPO_EVAL_STEP']),'Candidate step identity changed')
        origin=next(x for x in json.loads(Path(os.environ['GIGPO_EVAL_CANDIDATES_FILE']).read_text()) if x['step']==int(os.environ['GIGPO_EVAL_STEP']))
        for key in ('job_id','code_commit','config_sha256','data_identity'):
            require(identity[key]==origin[key],'Candidate source identity changed: '+key)
        hf_records=[x for x in identity['files'] if Path(x['path']).parent==candidate]
        require(len(hf_records)==12,'Expected12 complete HF files')
        for x in hf_records:require(sha(x['path'])==x['sha256'] and Path(x['path']).stat().st_size==x['bytes'],'Saved HF changed')
        evaluation=owned.spawn([sys.executable,str(HELPERS/'eval_entry_v1.py'),'--config-path',str(run),'--config-name','input-config'],run/'evaluation.log',pass_fds=tuple(int(r['fd']) for r in json.loads(os.environ['SEARCH_R1_GPU_LEASES'])))
        with (run/'resources.jsonl').open('x') as stream:
            while evaluation.poll() is None:
                require(all(proc.poll() is None for proc in (retriever,ray,guard)),'Own evaluation service died')
                stream.write(json.dumps(dict(elapsed_seconds=time.monotonic()-started,gpu=gpu_sample(),processes=[process_sample(pid) for pid in sorted(cgroup_pids(scope)-owned.baseline)]))+'\n');stream.flush();time.sleep(10)
        require(evaluation.returncode==0,'Actual GPU evaluation failed')
        receipt=json.loads((run/'evaluation-result.json').read_text())
        require(receipt['metrics']['val/qa_count/all']==512 and receipt['hf_directory']==str(candidate),'Wrong512 result identity')
        for key in ('val/f1/all','val/em/all'):
            require(math.isfinite(float(receipt['metrics'][key])) and 0<=float(receipt['metrics'][key])<=1,'Invalid F1/EM')
        require(json.loads((run/'trajectories/complete.json').read_text())['actual_scored_rows']==512,'Incomplete trajectory evidence')
        for x in hf_records:require(sha(x['path'])==x['sha256'],'HF changed during evaluation')
        # A validation-only task must not produce training update evidence.
        for p in (run/'audit').glob('*.jsonl'):
            for line in p.read_text().splitlines():require(json.loads(line).get('kind')!='actor_outer','Unexpected training during candidate evaluation')
        raw=list((run/'ray').glob('session_*/logs/worker-*-02000000-*.out'))
        result.update(status='GPU512_EVALUATED',candidate_step=int(os.environ['GIGPO_EVAL_STEP']),candidate=str(candidate),metrics=receipt['metrics'],hf_files=hf_records,config_sha256=config_hash,raw_workers=[dict(path=str(p),bytes=p.stat().st_size,inode=p.stat().st_ino,sha256=sha(p)) for p in raw])
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        try:owned.cleanup()
        except BaseException as cleanup_error:
            result.update(status='FAILED',cleanup_error=repr(cleanup_error));raise
        finally:
            result['cleanup_residual_states']=getattr(owned,'cleanup_residual',{});alias.unlink(missing_ok=True);result['wall_seconds']=time.monotonic()-started
            finalize_snapshot(result,run)
    require(result['status']=='GPU512_EVALUATED','Evaluation or terminal cleanup failed')
    print(json.dumps(result),flush=True)
    return dict(run=str(run),**result)

if __name__=='__main__':
    candidates=json.loads(Path(os.environ['GIGPO_EVAL_CANDIDATES_FILE']).read_text())
    require([x['step'] for x in candidates]==[5,25,50,75,80,85,100,125],'Candidate set changed')
    summary_path=Path(os.environ['GIGPO_EVAL_SUMMARY'])
    require(not summary_path.exists(),'Never repeat an existing quality selection')
    results=[]
    for item in candidates:
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        os.environ.update(GIGPO_EVAL_CANDIDATE=item['path'],GIGPO_EVAL_STEP=str(item['step']),GIGPO_EVAL_MANIFEST_SHA=item['manifest_sha256'])
        results.append(evaluate_candidate())
        temp=summary_path.with_suffix('.partial')
        temp.write_text(json.dumps(dict(status='IN_PROGRESS',completed_candidates=results),indent=2));temp.replace(summary_path)
    require(len(results)==len(candidates),'Incomplete candidate selection')
    selected=select_best(results)
    summary_path.write_text(json.dumps(dict(status='ALL_CANDIDATES_GPU512_COMPLETE_UNREVIEWED',completed_candidates=results,selected=selected,selection_rule='highest full512 F1; tie latest iteration; never finaltest'),indent=2))
