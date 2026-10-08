"""Owned approved Instruct300, one allocation and live step5 acceptance gate."""
import fcntl,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
from tree_shared_owned_runtime_fixed import ROOT,require,sha,cgroup_scope,cgroup_pids,OwnedProcesses,process_sample,gpu_sample,available
TOOLS=Path(__file__).resolve().parent
PACKAGE=json.loads((TOOLS/'manifest.json').read_text())
COMMIT=PACKAGE['code_commit']
CAND=TOOLS/'source'
CONTENT='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
MODEL=ROOT/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
class CandidateOwnedProcesses(OwnedProcesses):
    def spawn(self,command,log,env=None,pass_fds=()):
        return super().spawn(['bash','-c','cd "$SEARCH_R1_REPO" && exec "$@"','candidate-cwd',*command],log,env,pass_fds)

def interrupted(sig,frame): raise RuntimeError('Own Instruct300 supervisor interrupted '+str(sig))

def configuration(run,stage):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from search_r1.baselines.holdout import prepare_common_holdout
    with initialize_config_dir(config_dir=str(CAND/'repo/verl/trainer/config'),version_base=None):
        value=compose(config_name='ppo_tree_instruct300')
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(value,resolve=True)),run/'input-config.yaml')
    split=prepare_common_holdout(value)
    require(split['content_identity']['sha256']==CONTENT and split['train_rows']==169028 and split['holdout_rows']==512,'Fixed split changed')
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(value,resolve=True)),run/'resolved.yaml')
    require(value.trainer.total_training_steps==301 and value.trainer.actual_update_limit==300 and value.actor_rollout_ref.actor.optim.lr_warmup_steps==57,'300/57 budget changed')
    require(value.data.train_batch_size==512 and value.actor_rollout_ref.actor.ppo_mini_batch_size==128 and value.actor_rollout_ref.actor.ppo_micro_batch_size==16,'Approved batch changed')
    require(value.actor_rollout_ref.rollout.log_prob_micro_batch_size==32 and value.actor_rollout_ref.ref.log_prob_micro_batch_size==32 and value.data.val_batch_size==32,'Scoring/development batch changed')
    require(value.baselines.tree.roots==2 and value.baselines.tree.rounds==1 and value.baselines.tree.expansions==2 and value.baselines.tree.leaves==3,'Tree changed')
    require(value.max_turns==4 and value.force_final_answer is False and value.retriever.topk==3,'Search changed')
    return value,300

def main():
    a=SimpleNamespace(stage='instruct300')
    for file,expected in PACKAGE['files'].items(): require(sha(TOOLS/file)==expected,'Frozen package changed: '+file)
    for file,expected in PACKAGE['external_files'].items(): require(sha(file)==expected,'Frozen external helper changed: '+file)
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'repo',text=True).strip()=='4d133efeb1f675139d2558640f1e21d50780f133','Old production source changed')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT/'repo',text=True).strip(),'Old production checkout is dirty')
    import search_r1
    require(Path(search_r1.__file__).resolve()==(CAND/'repo/search_r1/__init__.py').resolve(),'Supervisor imported other source')
    job=os.environ.get('SLURM_JOB_ID','')
    require(job.isdigit() and socket.gethostname().split('.')[0]=='h100-9' and os.environ.get('SLURM_JOB_NODELIST')=='h100-9','Wrong allocation')
    require(len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))==8 and os.environ.get('SLURM_JOB_GPUS'),'Eight reserved GPUs required')
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    validate_inherited_gpu_leases(8)
    from search_r1.baselines.select_teacher import model_manifest
    require(model_manifest(MODEL)==PACKAGE['model_identity'],'Official model changed')
    lock=(ROOT/'shared-gpu-job.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    scope,relative=cgroup_scope(job); owned=CandidateOwnedProcesses(scope,relative)
    for port in (18082,16382): available(port)
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'): os.environ.pop(key,None)
    allocation=subprocess.check_output(['scontrol','show','job',job],text=True)
    require('JobState=RUNNING' in allocation and 'OverSubscribe=NO' in allocation and 'TresPerNode=gres:gpu:8' in allocation and 'CPUs/Task=32' in allocation,'Proper exclusive8GPU allocation required')
    run=ROOT/'runs'/f'tree-instruct300-{job}'; run.mkdir()
    for name in ('output','runtime','cache'): (run/name).mkdir()
    for name in ('ray','tmp'): (run/'runtime'/name).mkdir()
    (run/'allocation.txt').write_text(allocation)
    alias=Path('/tmp')/f'sst-tree-instruct300-{job}'
    require(not alias.exists() and not alias.is_symlink(),'Existing own runtime alias'); alias.symlink_to(run/'runtime',target_is_directory=True)
    os.environ['PATH']=str(ROOT/'env/bin')+':'+os.environ['PATH']
    os.environ.update(SEARCH_R1_REPO=str(CAND/'repo'),SEARCH_R1_ASSETS=str(ROOT/'assets-root'),SEARCH_R1_BASELINE_OUTPUT=str(run/'output'),SEARCH_R1_BASELINE_RUN_ROOT=str(alias),
      SEARCH_R1_TREE_AUDIT_DIR=str(run/'output/tree-audit'),SEARCH_R1_TREE_RANK_AUDIT_DIR=str(run/'output/tree-rank-audit'),RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER='0',RAY_ADDRESS='127.0.0.1:16382',TMPDIR=str(alias/'tmp'),RAY_TMPDIR=str(alias/'ray'),
      XDG_CACHE_HOME=str(run/'cache'),HF_HOME=str(run/'cache/hf'),HF_DATASETS_CACHE=str(ROOT/'retriever-cache/datasets'),
      OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',WANDB_MODE='offline',WANDB_DIR=str(run),
      PYTHONUNBUFFERED='1',PYTHONPATH=str(TOOLS)+':'+str(ROOT/'tools')+':'+str(CAND/'repo'),TREE_FIXED_ACTOR_PROFILE='1',TOKENIZERS_PARALLELISM='false',
      C_INCLUDE_PATH=str(ROOT/'python-dev/usr/include/python3.10')+':'+str(ROOT/'python-dev/usr/include'),
      VLLM_NO_USAGE_STATS='1',NCCL_NVLS_ENABLE='0',VLLM_ATTENTION_BACKEND='XFORMERS',NO_PROXY='127.0.0.1,localhost,::1',no_proxy='127.0.0.1,localhost,::1')
    os.environ.pop('WANDB_API_KEY',None)
    os.environ.update(TREE_VALIDATION_TRAINER_SHA256=PACKAGE['files']['source/repo/verl/trainer/ppo/ray_trainer.py'],TREE_INSTRUCT_TOOLS=str(TOOLS),TREE_INSTRUCT_MODEL=str(MODEL),TREE_COMMON_DATA=str(ROOT/'instruct300/common-data'),SEARCH_R1_MODEL_IDENTITY=PACKAGE['model_identity']['sha256'])
    start=time.time(); result={'status':'FAILED','stage':a.stage,'job_id':job,'code_commit':COMMIT,'actual_training_accepted':False}
    try:
        value,steps=configuration(run,a.stage); config_hash=sha(run/'resolved.yaml')
        os.environ.update(SEARCH_R1_BASELINE_AUDIT_DIR=str(run/'output/entry-audit'),SEARCH_R1_BASELINE_RUN_ID=run.name,
          SEARCH_R1_BASELINE_COMMIT=COMMIT,SEARCH_R1_BASELINE_CONFIG_SHA256=config_hash,SEARCH_R1_BASELINE_DATA_IDENTITY=CONTENT)
        (run/'launch-manifest.json').write_text(json.dumps(dict(result,config_sha256=config_hash,split_content_identity=CONTENT,
          cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],gpu_leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']),cgroup=str(scope),
          runtime='same h100-9 allocation: GPU E5 and complete 8-GPU sharded FP16 index, trainer, Ray; loopback only; zero online remote RPC',
          supervisor_source_sha256=sha(__file__),guard_source_sha256=sha(ROOT/'tools/resource_guard_identity.py'),owned_runtime_source_sha256=sha(ROOT/'tools/tree_shared_owned_runtime_fixed.py'),initial_gpu=gpu_sample()),indent=2)+'\n')
        guard=owned.spawn([sys.executable,str(ROOT/'tools/resource_guard_identity.py'),'--output',str(run/'resource-samples.jsonl')],run/'guard.log')
        deadline=time.monotonic()+30
        while not (run/'resource-samples.jsonl').is_file():
            require(guard.poll() is None and time.monotonic()<deadline,'Own GPU guard failed'); time.sleep(.2)
        first_guard=json.loads((run/'resource-samples.jsonl').read_text().splitlines()[0])
        require(first_guard.get('job_id')==job and len(first_guard.get('gpus',[]))==8 and not first_guard.get('event'),'Guard did not establish healthy inventory')
        fds=tuple(int(r['fd']) for r in json.loads(os.environ['SEARCH_R1_GPU_LEASES']))
        os.environ['TREE_GPU_KL_OUTPUT']=str(run/'gpu-kl-proof')
        kl=owned.spawn([sys.executable,'-m','torch.distributed.run','--nproc_per_node=8','--master_addr=127.0.0.1','--master_port=26382',str(TOOLS/'gpu-kl-proof.py')],run/'gpu-kl.log')
        deadline=time.monotonic()+180
        while kl.poll() is None:
            require(guard.poll() is None and time.monotonic()<deadline,'GPU KL guard failed or timeout'); time.sleep(1)
        require(kl.returncode==0,'GPU KL backward proof failed')
        for rank in range(8):
            record=json.loads((run/f'gpu-kl-proof/rank-{rank}.json').read_text())
            require(record['status']=='PASS' and record['rank']==rank and len(record['records'])==3,'GPU KL proof coverage incomplete')
        result['gpu_kl_proof_passed']=True
        os.environ['TREE_NATIVE_STATE_OUTPUT']=str(run/'native-fullstate')
        native=owned.spawn([sys.executable,'-m','torch.distributed.run','--nproc_per_node=8','--master_addr=127.0.0.1','--master_port=26382',str(TOOLS/'native_fullstate_probe.py')],run/'native-fullstate.log')
        deadline=time.monotonic()+600
        while native.poll() is None:
            require(guard.poll() is None and time.monotonic()<deadline,'Native state guard failed or timeout');time.sleep(1)
        require(native.returncode==0,'Native fullstate replay failed before any3B training')
        for rank in range(8):
            record=json.loads((run/f'native-fullstate/rank-{rank:02d}.json').read_text())
            require(record['status']=='PASS' and record['same_world_sixth_outer_replay_equal'] is True,'Native state rank coverage failed')
        result['native_fullstate_replay_passed']=True
        retriever=owned.spawn([sys.executable,str(ROOT/'repo/reproduction/h20/retriever_local.py'),'--index_path',str(ROOT/'assets-root/assets/wiki/e5_Flat.index'),
          '--corpus_path',str(ROOT/'assets-root/assets/wiki/wiki-18.jsonl'),'--retriever_model',str(ROOT/'assets-root/assets/e5-base-v2'),
          '--retriever_name','e5','--topk','3','--faiss_gpu','--device','cuda','--threads','8','--host','127.0.0.1','--port','18082'],run/'retriever.log',
          dict(os.environ,PYTHONPATH=str(ROOT/'retriever-gpu-lib')+':'+str(CAND/'repo'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'),pass_fds=fds)
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
            training=owned.spawn(['bash',str(TOOLS/'run_tree.sh'),'tree_grpo','--config-path',str(run),'--config-name','input-config'],run/'stage.log',pass_fds=fds)
            hf5=state5=None
            with (run/'process-resources.jsonl').open('a') as stream:
                while training.poll() is None:
                    gate=run/'output/acceptance-step5'
                    if hf5 is None and (gate/'ready.json').is_file():
                        ready=json.loads((gate/'ready.json').read_text())
                        hf5=owned.spawn([sys.executable,str(ROOT/'tools/check_real_checkpoint.py'),'--base',str(MODEL),'--checkpoint',ready['hf_checkpoint'],'--output',str(gate/'hf5-full-reload.json')],gate/'hf5-full-reload.log',dict(os.environ,CUDA_VISIBLE_DEVICES=''))
                        state5=owned.spawn([sys.executable,str(TOOLS/'check_complete_state.py'),'--ready',str(gate/'ready.json'),'--output',str(gate/'state5-files-cpu.json')],gate/'state5-files-cpu.log',dict(os.environ,CUDA_VISIBLE_DEVICES=''))
                    if hf5 is not None and hf5.poll() is not None:
                        require(hf5.returncode==0,'Actual HF5 reload failed while paused; no6 permitted')
                    if state5 is not None and state5.poll() is not None:
                        require(state5.returncode==0,'Actual step5 complete state failed while paused; no6 permitted')
                    require(all(proc.poll() is None for proc in (retriever,ray,guard)),'Own dependent service died')
                    stream.write(json.dumps({'epoch':time.time(),'retriever':process_sample(retriever.pid),'processes':[process_sample(pid) for pid in sorted(cgroup_pids(scope)-owned.baseline)]})+'\n'); stream.flush(); time.sleep(5)
            require(training.returncode==0,'Actual Tree training failed')
            require(guard.poll() is None,'Guard died before training completion')
            samples=[json.loads(line) for line in (run/'resource-samples.jsonl').read_text().splitlines()]
            require(samples and all(not x.get('event') for x in samples) and time.time()-samples[-1]['epoch']<15,'Final guard missing/stale/STOP')
            result['final_guard_healthy']=True
            require(sha(run/'output/entry-audit/entry-config.yaml')==config_hash,'Actual entry config differs')
            result.update(status='ACTUAL_STAGE_COMPLETED_UNREVIEWED',training_body_exit_code=training.returncode,entry_config_verified=True)
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
                try:
                    owned.cleanup()
                    result['cleanup_residual']=owned.cleanup_residual
                    result['final_gpu']=gpu_sample()
                    background={1680782,1680783,1680784,1680785,1696443,1696444,1696445,1696446}
                    pids={int(line.split(',')[1].strip()) for line in result['final_gpu']['processes']}
                    require(pids==background,'Original8 must be preserved, no unexpected GPU remainder')
                except BaseException as err:
                    result.update(status='FAILED',cleanup_error=repr(err)); raise
        finally:
            alias.unlink(missing_ok=True); result['wall_seconds']=time.time()-start
            (run/'run-result.json').write_text(json.dumps(result,indent=2)+'\n')
            code=0 if result['status'] in ('ACTUAL_STAGE_COMPLETED_UNREVIEWED','REAL_LOCAL_GPU_RETRIEVAL_GATE_READY_FOR_REVIEW') else 1
            (run/'cost.json').write_text(json.dumps({'job_id':job,'stage':a.stage,'start_epoch':start,'end_epoch':time.time(),'allocated_gpus':8,'exit_code':code,'scope':'entire own supervisor including service loading, probes, training and cleanup; scheduler boundaries separately'})+'\n')
    print('INSTRUCT300_COMPLETED_RUN='+str(run),flush=True)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted); main()
