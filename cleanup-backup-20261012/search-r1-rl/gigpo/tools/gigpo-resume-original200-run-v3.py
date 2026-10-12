"""Explicit local-only GiGPO Slurm executor; never submits or resumes jobs."""
import argparse,fcntl,json,math,os,signal,socket,subprocess,sys,time
from pathlib import Path
from owned_runtime import ROOT,require,sha,cgroup_scope,cgroup_pids,OwnedProcesses,process_sample,gpu_sample,available
SOURCE=ROOT/'runs/gigpo-formal500-job2992-1791016340597739432/checkpoints/actor/global_step_200'
CONTENT='fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524'

def config(run,stage):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from search_r1.baselines.holdout import prepare_common_holdout
    steps=300
    overrides=['baselines.method=gigpo','actor_rollout_ref.actor.ppo_micro_batch_size=64',
      'baselines.local_runtime_only=true','baselines.local_runtime_host=h100-8',
      'trainer.total_training_steps=501','trainer.smoke_only=false',
      'trainer.resume.enabled=true',f'trainer.resume.source={SOURCE}',
      'trainer.resume.start_step=200','trainer.resume.stop_step=300','trainer.resume.acceptance_step=205',
      f'trainer.resume.approval_path={run}/resume205-approval.json','trainer.resume.scheduler_horizon=501',
      f'actor_rollout_ref.model.path={SOURCE}',f'+actor_rollout_ref.ref.model_path={ROOT}/assets-root/assets/qwen2.5-3b',
      'trainer.val_before_train=false',
      'trainer.test_freq=50','trainer.save_freq=25','trainer.n_gpus_per_node=8','trainer.nnodes=1',
      f'trainer.default_local_dir={run}/checkpoints',f'baselines.holdout.output_dir={ROOT}/common-data',
      'retriever.url=http://127.0.0.1:18081/retrieve']
    with initialize_config_dir(config_dir=str(ROOT/'repo/verl/trainer/config'),version_base=None):
        value=compose(config_name='ppo_baselines',overrides=overrides)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(value,resolve=True)),run/'input-config.yaml')
    split=prepare_common_holdout(value)
    require(split['content_identity']['sha256']==CONTENT and split['holdout_rows']==3393,'Split drift')
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(value,resolve=True)),run/'resolved-config.yaml')
    require(value.data.train_batch_size==512 and value.actor_rollout_ref.actor.ppo_mini_batch_size==256,'Budget drift')
    require(value.actor_rollout_ref.rollout.n_agent==5 and value.max_turns==4,'Algorithm drift')
    return value,steps

def audit_gate(run,steps,commit,config_hash):
    require(sha(run/'audit/entry-config.yaml')==config_hash,'Actual entry config differs')
    entry=[json.loads(x) for x in (run/'audit/entry.jsonl').read_text().splitlines()]
    require(len(entry)==1 and entry[0]['kind']=='config_boundary' and entry[0]['actual_config_sha256']==config_hash
        and entry[0]['actual_data_identity']==CONTENT and entry[0]['entry_total_training_steps']==501,'Actual config boundary missing/drifted')
    driver=[json.loads(x) for x in (run/'audit/driver.jsonl').read_text().splitlines()]
    require([r['outer_step'] for r in driver]==list(range(201,steps+1)),'Missing actual driver updates')
    for row in driver:
        require(row['rows']==2560 and row['questions']==512 and row['method']=='gigpo','Actual sampling budget differs')
        for item in row['tensors'].values():require(item['finite']==item['count'],'Nonfinite driver tensors')
    require(any(r['observation_tokens']>0 for r in driver),'No real retrieval observations')
    for rank in range(8):
        rows=[json.loads(x) for x in (run/'audit'/f'{rank}.jsonl').read_text().splitlines()]
        require([r['outer_step'] for r in rows]==list(range(201,steps+1)),'Missing/duplicate actual actor updates')
        for r in rows:
            require(r['commit']==commit and r['config_sha256']==config_hash and r['data_identity']==CONTENT,'Audit identity drift')
            require(r['rank']==rank and r['epochs']==1 and r['optimizer_steps']==10,'Actual actor optimizer budget differs')
            require(r['grad_finite']==r['grad_nonzero']==10 and r['skipped_empty_minibatches']==0,'Invalid/empty actual gradient')
            require(r['fixed_unchanged'] and r['shard_layout_unchanged'] and r['mask_binary'],'Fixed snapshot/mask drift')
            require(all(x is not None and math.isfinite(x) and x>0 for x in r['grad_norms']),'Invalid gradient norms')
            for stats in list(r['tensors'].values())+[r['shard_delta']]:
                require(stats['finite']==stats['count'],'Nonfinite actor tensors')
            for micro in r['micro_stats']:
                require('joint_action_ratios' in micro,'Missing actual joint GiGPO action ratios')
                for stats in micro.values():
                    require(stats['finite']==stats['count'],'Nonfinite logp/loss/ratio')
        require(all(r['shard_delta']['nonzero']>0 for r in rows),'Every restored outer including201 must change parameters')
    checkpoint=run/'checkpoints/actor'/f'global_step_{steps}'
    require((checkpoint/'config.json').is_file(),'No full HF checkpoint')
    state=json.loads((run/'checkpoints/algorithm'/f'global_step_{steps}'/'state.json').read_text())
    require(state['method']=='gigpo' and state['global_step']==steps and state['resume_supported'] is False,'Algorithm checkpoint mismatch')
    return checkpoint

def interrupted(sig,frame):
    raise RuntimeError(f'Own executor interrupted by signal {sig}')

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['resume200to300'],required=True)
    p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--previous-review',type=Path);p.add_argument('--previous-review-sha');args=p.parse_args()
    require(sha(args.binding)==args.binding_sha,'Deployment binding changed')
    binding=json.loads(args.binding.read_text());require(binding['status']=='REVIEWED','Need independent deployment review')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    require(commit==binding['code_commit'] and not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Dirty/unfrozen source')
    for item in binding['evidence']:
        require(sha(item['path'])==item['sha256'],'Evidence changed')
    require(binding['resume_source']==str(SOURCE) and binding['resume_stop_logical']==300,'Wrong requested continuation')
    require(binding['native_resume_review_status']=='PASS','Need real eight-rank recovery state review')
    inventory=json.loads(Path(binding['original_checkpoint_inventory']).read_text())
    require(inventory['checkpoint']==str(SOURCE) and not inventory['state_candidates'],'Expected approved weights-only recovery')
    for item in inventory['files']:
        require(sha(item['path'])==item['sha256'],'Original200 checkpoint changed')
    reference=json.loads(Path(binding['reference_base_inventory']).read_text())
    require(reference['base_path']==str(ROOT/'assets-root/assets/qwen2.5-3b'),'Reference must remain original base')
    for item in reference['files']:
        require(sha(item['path'])==item['sha256'],'Original reference base assets changed')
    job=os.environ.get('SLURM_JOB_ID','');require(job and os.environ.get('SLURM_JOB_GPUS'),'Need actual Slurm allocation')
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
    run_id=f'gigpo-{args.stage}-job{job}-{time.time_ns()}';run=ROOT/'runs'/run_id;run.mkdir()
    for name in ('audit','tmp','ray','cache'):(run/name).mkdir()
    alias=Path('/tmp')/f'gigpo-r{job}';require(not alias.exists() and not alias.is_symlink(),'Existing Ray alias');alias.symlink_to(run,target_is_directory=True)
    os.environ.update(AGENT_ROOT=str(ROOT),SEARCH_R1_ASSETS=str(ROOT/'assets-root'),SEARCH_R1_BASELINE_OUTPUT=str(run),SEARCH_R1_BASELINE_RUN_ROOT=str(alias),
      RAY_ADDRESS='127.0.0.1:16381',TMPDIR=str(alias/'tmp'),RAY_TMPDIR=str(alias/'ray'),XDG_CACHE_HOME=str(alias/'cache'),HF_HOME=str(alias/'cache/hf'),
      HF_DATASETS_CACHE=str(ROOT/'cache/retriever-datasets'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
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
          warmup_note='Rebuilt scheduler retains horizon501/warmup142/logical200; every new outer must change weights',
          resume_source=str(SOURCE),resume_mode='weights_only; freshAdam and RNG/data; old201-227 not recovered',
          start_logical=200,stop_logical=300,new_actual_budget=100)
        (run/'launch-manifest.json').write_text(json.dumps(manifest,indent=2))
        from search_r1.baselines.preflight import gpu_snapshot,check_selected_gpus
        inventory,processes=gpu_snapshot();selected=check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inventory,processes)
        (ROOT/'tools/expected-gpu-uuids.json').write_text(json.dumps([g['uuid'] for g in selected]))
        guard=owned.spawn([sys.executable,str(ROOT/'tools/resource_guard.py'),'--output',str(run/'guard-resources.jsonl')],run/'guard.log')
        deadline=time.monotonic()+30
        while not (run/'guard-resources.ready').is_file():
            require(guard.poll() is None and time.monotonic()<deadline,'Resource guard failed');time.sleep(1)
        retriever=owned.spawn([sys.executable,'reproduction/h20/retriever_local.py','--index_path',str(ROOT/'assets-root/assets/wiki/e5_Flat.index'),
          '--corpus_path',str(ROOT/'assets-root/assets/wiki/wiki-18.jsonl'),'--retriever_model',str(ROOT/'assets-root/assets/e5-base-v2'),
          '--retriever_name','e5','--topk','3','--faiss_gpu','--device','cuda','--threads','8','--host','127.0.0.1','--port','18081'],run/'retriever.log',
          dict(os.environ,PYTHONPATH=str(ROOT/'retriever-gpu-lib')+os.pathsep+str(ROOT/'repo'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'))
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
        subprocess.run([sys.executable,str(ROOT/'tools/ray-audit-environment-probe-v2.py'),str(run/'ray-audit-environment-verification.json')],check=True)
        training=owned.spawn(['bash','reproduction/baselines/run.sh','gigpo','--config-dir',str(run),'--config-name','input-config'],run/'training.log',pass_fds=tuple(int(r['fd']) for r in json.loads(os.environ['SEARCH_R1_GPU_LEASES'])))
        pause_verifier=None
        with (run/'resources.jsonl').open('a') as stream:
            while training.poll() is None:
                require(all(proc.poll() is None for proc in (retriever,ray,guard)),'Own dependent service died')
                ready=run/'resume205-ready.json'
                if ready.is_file() and pause_verifier is None:
                    pause_verifier=owned.spawn([sys.executable,str(ROOT/'tools/verify-gigpo-checkpoint.py'),
                      '--base',str(SOURCE),'--checkpoint',str(run/'checkpoints/actor/global_step_205'),
                      '--output',str(run/'checkpoint205-cpu-verification.json')],run/'checkpoint205-cpu-verification.log',
                      env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'))
                if pause_verifier is not None and pause_verifier.poll() is not None:
                    require(pause_verifier.returncode==0,'Restored205 full HF reload failed')
                stream.write(json.dumps(dict(elapsed_seconds=time.monotonic()-started,gpu=gpu_sample(),retriever=process_sample(retriever.pid),
                  processes=[process_sample(pid) for pid in sorted(cgroup_pids(scope)-owned.baseline)]))+'\n');stream.flush();time.sleep(10)
        require(training.returncode==0,'Actual training failed')
        require((run/'resume205-approval.json').is_file(),'No independently accepted205 continuation')
        checkpoint=audit_gate(run,steps,commit,config_hash)
        result.update(status='ACTUAL_UPDATES_COMPLETED_UNREVIEWED',checkpoint=str(checkpoint))
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        try:owned.cleanup()
        finally:
            result['cleanup_residual_states']=getattr(owned,'cleanup_residual',{});alias.unlink(missing_ok=True);result['wall_seconds']=time.monotonic()-started
            result['final_gpu_snapshot']=gpu_sample();(run/'run-result.json').write_text(json.dumps(result,indent=2))
    # Real HF reload in a fresh CPU process, after own actor/Ray workers exit.
    reload_started=time.monotonic()
    subprocess.run([sys.executable,str(ROOT/'tools/verify-gigpo-checkpoint.py'),
        '--base',str(SOURCE),'--checkpoint',str(checkpoint),
        '--output',str(run/'checkpoint-cpu-verification.json')],check=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'))
    result.update(status='CHECKPOINT_RELOADED_UNREVIEWED',checkpoint_reload_wall_seconds=time.monotonic()-reload_started,
        complete_wall_seconds=time.monotonic()-started)
    (run/'run-result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);main()
