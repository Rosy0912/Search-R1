"""Own h100-9 Slurm execution: same-node GPU retrieval and user-approved micro64."""
import argparse, fcntl, json, os, signal, socket, subprocess, sys, time
from pathlib import Path
from tree_shared_owned_runtime import ROOT, require, sha, cgroup_scope, cgroup_pids, OwnedProcesses, process_sample, gpu_sample, available
COMMIT='4d133efeb1f675139d2558640f1e21d50780f133'
CONTENT='fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524'

def configuration(run,stage):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from search_r1.baselines.holdout import prepare_common_holdout
    steps={'two':2,'five':5,'formal':500,'gate':0}[stage]
    overrides=['baselines.method=tree_grpo','actor_rollout_ref.rollout.n_agent=2','actor_rollout_ref.actor.ppo_micro_batch_size=64',
      'baselines.local_runtime_only=true','baselines.local_runtime_host=h100-9','trainer.n_gpus_per_node=8','trainer.nnodes=1',
      f'trainer.total_training_steps={steps+1}',f'trainer.smoke_only={str(stage!="formal").lower()}',
      f'trainer.val_before_train={str(stage=="formal").lower()}','trainer.test_freq=50',
      'trainer.logger=[console,wandb]',f'baselines.holdout.output_dir={ROOT}/common-data',
      'retriever.url=http://127.0.0.1:18082/retrieve']
    with initialize_config_dir(config_dir=str(ROOT/'repo/verl/trainer/config'),version_base=None):
        value=compose(config_name='ppo_baselines',overrides=overrides)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(value,resolve=True)),run/'input-config.yaml')
    split=prepare_common_holdout(value)
    require(split['content_identity']['sha256']==CONTENT and split['holdout_rows']==3393,'Split changed')
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(value,resolve=True)),run/'resolved.yaml')
    require(value.data.train_batch_size==512 and value.actor_rollout_ref.actor.ppo_mini_batch_size==256,'Batch changed')
    require(value.actor_rollout_ref.actor.ppo_micro_batch_size==64 and value.data.val_batch_size==64,'Approved micro/validation changed')
    require(value.baselines.tree.roots==2 and value.baselines.tree.rounds==1 and value.baselines.tree.expansions==2 and value.baselines.tree.leaves==3,'Tree changed')
    require(value.max_turns==4 and value.force_final_answer is False and value.retriever.topk==3,'Search changed')
    return value,steps

def interrupted(sig,frame): raise RuntimeError('Own shared supervisor interrupted '+str(sig))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['gate','two','five','formal'],required=True)
    p.add_argument('--previous-review',type=Path); p.add_argument('--previous-review-sha')
    a=p.parse_args()
    require(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==COMMIT,'Unfrozen production checkout')
    require(not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Dirty production checkout')
    job=os.environ.get('SLURM_JOB_ID','')
    require(job.isdigit() and socket.gethostname().split('.')[0]=='h100-9' and os.environ.get('SLURM_JOB_NODELIST')=='h100-9','Wrong Slurm node/job')
    require(len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))==8 and os.environ.get('SLURM_JOB_GPUS'),'Need eight allocated GPUs')
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    validate_inherited_gpu_leases(8)
    if a.stage!='gate':
        require(a.previous_review and sha(a.previous_review)==a.previous_review_sha,'Independent prior gate receipt required')
        prior=json.loads(a.previous_review.read_text())
        require(prior['status']=='PASS' and prior['code_commit']==COMMIT and prior['stage']=={'two':'gate','five':'two','formal':'five'}[a.stage],'Prior stage not accepted for this source/protocol')
        for item in prior['evidence']: require(sha(item['path'])==item['sha256'],'Prior immutable evidence changed')
    lock=(ROOT/'shared-gpu-job.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    scope,relative=cgroup_scope(job); owned=OwnedProcesses(scope,relative)
    for port in (18082,16382): available(port)
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'): os.environ.pop(key,None)
    run=ROOT/'runs'/f'tree-{a.stage}-{job}'; run.mkdir()
    for name in ('output','runtime','cache'): (run/name).mkdir()
    for name in ('ray','tmp'): (run/'runtime'/name).mkdir()
    alias=Path('/tmp')/f'sst-tree-shared-{job}'
    require(not alias.exists() and not alias.is_symlink(),'Existing own runtime alias'); alias.symlink_to(run/'runtime',target_is_directory=True)
    os.environ['PATH']=str(ROOT/'env/bin')+':'+os.environ['PATH']
    os.environ.update(SEARCH_R1_ASSETS=str(ROOT/'assets-root'),SEARCH_R1_BASELINE_OUTPUT=str(run/'output'),SEARCH_R1_BASELINE_RUN_ROOT=str(alias),
      SEARCH_R1_TREE_AUDIT_DIR=str(run/'output/tree-audit'),SEARCH_R1_TREE_RANK_AUDIT_DIR=str(run/'output/tree-rank-audit'),RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER='0',RAY_ADDRESS='127.0.0.1:16382',TMPDIR=str(alias/'tmp'),RAY_TMPDIR=str(alias/'ray'),
      XDG_CACHE_HOME=str(run/'cache'),HF_HOME=str(run/'cache/hf'),HF_DATASETS_CACHE=str(ROOT/'retriever-cache/datasets'),
      OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',WANDB_MODE='offline',WANDB_DIR=str(run),
      PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT/'tools')+':'+str(ROOT/'repo'),TOKENIZERS_PARALLELISM='false',
      C_INCLUDE_PATH=str(ROOT/'python-dev/usr/include/python3.10')+':'+str(ROOT/'python-dev/usr/include'),
      VLLM_NO_USAGE_STATS='1',NCCL_NVLS_ENABLE='0',VLLM_ATTENTION_BACKEND='XFORMERS',NO_PROXY='127.0.0.1,localhost,::1',no_proxy='127.0.0.1,localhost,::1')
    os.environ.pop('WANDB_API_KEY',None)
    start=time.time(); result={'status':'FAILED','stage':a.stage,'job_id':job,'code_commit':COMMIT,'actual_training_accepted':False}
    try:
        value,steps=configuration(run,a.stage); config_hash=sha(run/'resolved.yaml')
        os.environ.update(SEARCH_R1_BASELINE_AUDIT_DIR=str(run/'output/entry-audit'),SEARCH_R1_BASELINE_RUN_ID=run.name,
          SEARCH_R1_BASELINE_COMMIT=COMMIT,SEARCH_R1_BASELINE_CONFIG_SHA256=config_hash,SEARCH_R1_BASELINE_DATA_IDENTITY=CONTENT)
        (run/'launch-manifest.json').write_text(json.dumps(dict(result,config_sha256=config_hash,split_content_identity=CONTENT,
          cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],gpu_leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']),cgroup=str(scope),
          runtime='same h100-9 allocation: GPU E5 and complete 8-GPU sharded FP16 index, trainer, Ray; loopback only; zero online remote RPC',
          supervisor_source_sha256=sha(__file__),owned_runtime_source_sha256=sha(ROOT/'tools/tree_shared_owned_runtime.py'),initial_gpu=gpu_sample()),indent=2)+'\n')
        guard=owned.spawn([sys.executable,str(ROOT/'tools/resource_guard.py'),'--output',str(run/'resource-samples.jsonl')],run/'guard.log')
        deadline=time.monotonic()+30
        while not (run/'resource-samples.jsonl').is_file():
            require(guard.poll() is None and time.monotonic()<deadline,'Own GPU guard failed'); time.sleep(.2)
        first_guard=json.loads((run/'resource-samples.jsonl').read_text().splitlines()[0])
        require(first_guard.get('job_id')==job and len(first_guard.get('gpus',[]))==8 and not first_guard.get('event'),'Guard did not establish healthy inventory')
        fds=tuple(int(r['fd']) for r in json.loads(os.environ['SEARCH_R1_GPU_LEASES']))
        retriever=owned.spawn([sys.executable,'reproduction/h20/retriever_local.py','--index_path',str(ROOT/'assets-root/assets/wiki/e5_Flat.index'),
          '--corpus_path',str(ROOT/'assets-root/assets/wiki/wiki-18.jsonl'),'--retriever_model',str(ROOT/'assets-root/assets/e5-base-v2'),
          '--retriever_name','e5','--topk','3','--faiss_gpu','--device','cuda','--threads','8','--host','127.0.0.1','--port','18082'],run/'retriever.log',
          dict(os.environ,PYTHONPATH=str(ROOT/'retriever-gpu-lib')+':'+str(ROOT/'repo'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'),pass_fds=fds)
        import requests
        session=requests.Session(); session.trust_env=False; deadline=time.monotonic()+900
        while True:
            require(retriever.poll() is None and guard.poll() is None and time.monotonic()<deadline,'Own GPU retrieval startup failed')
            try:
                if session.get('http://127.0.0.1:18082/openapi.json',timeout=2).ok: break
            except requests.RequestException: pass
            time.sleep(2)
        # Run the same reviewed, raw-result HTTP probe on loopback. This checks
        # the real service in this allocation, including large/internal batches.
        subprocess.run([sys.executable,str(ROOT/'tools/probe_tree_shared_retrieval.py'),'--output',str(run/'retrieval-gate.json')],check=True,
          env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        from search_r1.baselines.preflight import gpu_snapshot,check_selected_gpus
        deadline=time.monotonic()+60
        while True:
            inventory,processes=gpu_snapshot()
            try: check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inventory,processes); break
            except RuntimeError:
                require(time.monotonic()<deadline,'Resident GPU retrieval violates dual15%; do not bypass'); time.sleep(2)
        (run/'retriever-coexistence.json').write_text(json.dumps({'gpus':inventory,'snapshot':gpu_sample(),'retriever_pid':retriever.pid},indent=2)+'\n')
        if a.stage!='gate':
            ray=owned.spawn([str(ROOT/'env/bin/ray'),'start','--head','--node-ip-address=127.0.0.1','--port=16382','--num-cpus=24','--num-gpus=8',
              '--include-dashboard=false','--object-store-memory=8589934592','--temp-dir='+str(alias/'ray'),'--min-worker-port=22082','--max-worker-port=23081',
              '--disable-usage-stats','--block'],run/'ray.log')
            deadline=time.monotonic()+120
            while True:
                require(ray.poll() is None and guard.poll() is None and time.monotonic()<deadline,'Own Ray failed')
                try:
                    with socket.create_connection(('127.0.0.1',16382),timeout=1): break
                except OSError: time.sleep(1)
            training=owned.spawn(['bash','reproduction/baselines/run.sh','tree_grpo','--config-dir',str(run),'--config-name','input-config'],run/'stage.log',pass_fds=fds)
            with (run/'process-resources.jsonl').open('a') as stream:
                while training.poll() is None:
                    require(all(proc.poll() is None for proc in (retriever,ray,guard)),'Own dependent service died')
                    stream.write(json.dumps({'epoch':time.time(),'retriever':process_sample(retriever.pid),'processes':[process_sample(pid) for pid in sorted(cgroup_pids(scope)-owned.baseline)]})+'\n'); stream.flush(); time.sleep(5)
            require(training.returncode==0,'Actual Tree training failed')
            require(sha(run/'output/entry-audit/entry-config.yaml')==config_hash,'Actual entry config differs')
            result['status']='ACTUAL_STAGE_COMPLETED_UNREVIEWED'
        else: result['status']='REAL_LOCAL_GPU_RETRIEVAL_GATE_READY_FOR_REVIEW'
    except BaseException as err:
        result.update(error_type=type(err).__name__,error=str(err)); raise
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN); signal.signal(signal.SIGINT,signal.SIG_IGN)
        try:
            try:
                (run/'termination-cost.json').write_text(json.dumps({'job_id':job,'stage':a.stage,'start_epoch':start,'termination_epoch':time.time(),'allocated_gpus':8,'scope':'own supervisor before cleanup'})+'\n')
            except BaseException as err:
                result.update(status='FAILED',termination_record_error=repr(err)); raise
            finally:
                try: owned.cleanup()
                except BaseException as err:
                    result.update(status='FAILED',cleanup_error=repr(err)); raise
        finally:
            alias.unlink(missing_ok=True); result['wall_seconds']=time.time()-start
            (run/'run-result.json').write_text(json.dumps(result,indent=2)+'\n')
            code=0 if result['status'] in ('ACTUAL_STAGE_COMPLETED_UNREVIEWED','REAL_LOCAL_GPU_RETRIEVAL_GATE_READY_FOR_REVIEW') else 1
            (run/'cost.json').write_text(json.dumps({'job_id':job,'stage':a.stage,'start_epoch':start,'end_epoch':time.time(),'allocated_gpus':8,'exit_code':code,'scope':'entire own supervisor including service loading, probes, training and cleanup; scheduler boundaries separately'})+'\n')
    if a.stage!='gate': (ROOT/'logs'/f'{a.stage}-stage-completed.run').write_text(str(run)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted); main()
