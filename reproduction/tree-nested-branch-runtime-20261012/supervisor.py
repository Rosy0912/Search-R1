"""Human-authorized direct200, retaining same-h100-10 owned resource wrappers."""
import fcntl,json,os,re,signal,socket,subprocess,sys,time
if sys.flags.optimize:
    raise RuntimeError('Fresh supervisor requires unoptimized Python before any runtime import')
os.environ['PYTHONOPTIMIZE']='0'
from pathlib import Path
from types import SimpleNamespace
from calibration_contract import validate_fresh_start,protected_processes,forbid_recovery_environment,final_gpu_process_observation
from production_contract import production_inputs,read_bound,DIRECT_MODE,CHAIN_SCOPE
from slurm_identity import verify_runtime,install_local_environment,submitted_lineage,verify_runtime_cgroup,guarded_gpu_query,install_gpu_query_guard
from train_monitor import guard_health,verify_terminal,progress_evidence
from tree_shared_owned_runtime_fixed import ROOT,require,sha,cgroup_scope,cgroup_pids,OwnedProcesses,process_sample,gpu_sample,available
TOOLS=Path(__file__).resolve().parent
PACKAGE=json.loads((TOOLS/'manifest.json').read_text())
COMMIT=PACKAGE['code_commit']
gpu_sample=guarded_gpu_query(gpu_sample)
CAND=TOOLS/'source'
CONTENT='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
MODEL=ROOT/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
PRODUCTION_THRESHOLDS=None
class CandidateOwnedProcesses(OwnedProcesses):
    def spawn(self,command,log,env=None,pass_fds=()):
        require(sys.flags.optimize==0,'Owned children require unoptimized parent Python')
        child_environment=dict(os.environ if env is None else env)
        child_environment['PYTHONOPTIMIZE']='0'
        return super().spawn(['bash','-c','cd "$SEARCH_R1_REPO" && exec "$@"','candidate-cwd',*command],log,child_environment,pass_fds)

def interrupted(sig,frame): raise RuntimeError('Own Instruct300 supervisor interrupted '+str(sig))

def configuration(run,stage):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from search_r1.baselines.holdout import prepare_common_holdout
    with initialize_config_dir(config_dir=str(TOOLS/'config'),version_base=None):
        value=compose(config_name='ppo_tree_nested_branch_fresh200',overrides=["hydra.searchpath=[file://"+str(CAND/'repo/verl/trainer/config')+"]", *PACKAGE['configuration_overrides']])
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(value,resolve=True)),run/'input-config.yaml')
    split=prepare_common_holdout(value)
    require(split['content_identity']['sha256']==CONTENT and split['train_rows']==169028 and split['holdout_rows']==512,'Fixed split changed')
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(value,resolve=True)),run/'resolved.yaml')
    require(value.trainer.total_training_steps==201 and value.trainer.actual_update_limit==200,'Fresh200 budget changed')
    actor=value.actor_rollout_ref.actor; rollout=value.actor_rollout_ref.rollout; tree=value.baselines.tree
    require(value.baselines.training_protocol=='tree_fresh200_five_leaf_v1','New explicit fresh protocol missing')
    require(actor.guard_continue_protocol=='tree_guard_continue_fresh_rollout_v1','Latest explicit opportunity/attempt continuation protocol missing')
    require(value.data.train_batch_size==512 and actor.ppo_mini_batch_size==128 and actor.ppo_micro_batch_size==16 and actor.ppo_epochs==1,'Fresh batch/epochs changed')
    require(rollout.log_prob_micro_batch_size==32 and value.actor_rollout_ref.ref.log_prob_micro_batch_size==32 and value.data.val_batch_size==32,'Scoring/dev batch changed')
    require((tree.roots,tree.rounds,tree.expansions,tree.leaves)==(2,1,2,3) and list(tree.leaf_quotas)==[2,3],'True five-leaf quotas missing')
    require(value.max_turns==4 and value.force_final_answer is False and value.retriever.topk==3,'Search changed')
    require(value.trainer.acceptance_step==5 and value.trainer.acceptance_gate_enabled and value.trainer.test_freq==5 and value.trainer.save_freq==50,'Fresh gate/cadence changed')
    require(value.actor_rollout_ref.ref.model_path==str(MODEL) and value.actor_rollout_ref.model.path==str(MODEL),'Fresh actor/ref must use official base')
    require(actor.optim.lr==1e-6 and actor.optim.lr_warmup_steps==57 and actor.optim.lr_warmup_steps_ratio==0 and actor.optim.total_training_steps==200 and actor.optim.warmup_style=='constant','57-complete-outer LR contract changed')
    require(actor.entropy_coeff==0 and actor.kl_loss_coef==0 and value.algorithm.kl_ctrl.kl_coef==0 and actor.kl_loss_type=='low_var_kl' and actor.clip_ratio==.2 and actor.grad_clip==1,'PPO/reference coefficients changed')
    require((value.data.max_start_length,value.data.max_prompt_length,value.data.max_response_length,value.data.max_obs_length)==(2048,4096,500,500),'Lengths changed')
    safety=actor.policy_safety
    require(safety.enabled and safety.dual_clip==3 and safety.kl_log_bound==30 and safety.monitor_rows_per_rank==32,'Three guard configuration changed')
    correction=safety.sampling_probability_correction
    require(correction.enabled is True and correction.protocol=='tree_behavior_mu_centered_ppo_v1' and tree.behavior_mu.enabled is True and tree.behavior_mu.protocol==correction.protocol,'Explicit actual-mu profile missing')
    require(tree.token_preservation.enabled is True and tree.token_preservation.protocol=='tree_sampled_prefix_legacy_semantics_v1' and tree.token_preservation.tokenizer_sha256==PACKAGE['model_identity']['files']['tokenizer.json'] and rollout.tree_raw_emission is True,'Actual native-token binding missing')
    require(PRODUCTION_THRESHOLDS is not None and safety.target_old_policy_kl==PRODUCTION_THRESHOLDS['target_old_policy_kl'] and safety.negative_tail_limit==PRODUCTION_THRESHOLDS['negative_tail_limit'] and correction.initial_actor_repeat_max_abs_delta==PRODUCTION_THRESHOLDS['initial_actor_repeat_max_abs_delta'],'Configured prospective direct thresholds differ')
    require(Path(value.trainer.default_local_dir).resolve()==(run/'checkpoints').resolve(),'Checkpoints must stay inside this own run')
    sampling=rollout.sampling_consistency
    require(sampling.enabled is False and sampling.enforce is True and sampling.max_abs_delta==.01,'Old probability gate must remain explicit/unaccepted')
    require(tree.branch_sampling=='random_nested_backtrack_v1' and tree.sampling_objective=='reward_selected_nested_tree_grpo_surrogate_v1' and tree.continuation_budget_per_question==3,'Nested selected surrogate protocol changed')
    require(tree.action_positions=='valid_action_start' and tree.exclude_initial_from_backtrack is True and tree.fallback=='initial_fresh_once_unconditional' and tree.remaining_budget=='inherit_parent' and tree.reward_comparison=='absolute_tolerance' and tree.reward_difference_tolerance==0.0,'Nested candidate/fallback budget contract changed')
    require(value.trainer.val_before_train is False,'Initial baseline forbidden')
    require(value.trainer.experiment_name=='tree-nested-branch-officialfresh200','Run/checkpoint name changed')
    return value,200

def main():
    supervisor_started=time.time()
    from predecessor_gate import verify_predecessor
    verify_predecessor(PACKAGE)
    verify_runtime_cgroup()
    install_gpu_query_guard()
    allocation_identity=verify_runtime()
    removed_scheduler_environment=install_local_environment(os.environ)
    a=SimpleNamespace(stage=PACKAGE['stage'])
    global PRODUCTION_THRESHOLDS
    require(PACKAGE['scope']==CHAIN_SCOPE and PACKAGE['execution_mode']==DIRECT_MODE and PACKAGE['packet_revision']==1,'Explicit human-authorized direct200 branch required')
    forbid_recovery_environment(os.environ)
    require(PACKAGE['stage']=='fresh_mu_train200','Production stage differs')
    require(not any(name in os.environ for name in ('TREE_PPO_CALIBRATION_ONLY','TREE_FRESH_INITIALIZATION_ONLY')),'Production forbids all calibration-only environment')
    PRODUCTION_THRESHOLDS, input_pins=production_inputs(PACKAGE,manifest_sha256=os.environ['TREE_INSTRUCT_PACKAGE_SHA256'],dispatch_review_descriptor={'path':os.environ['TREE_PRODUCTION_RELEASE_PATH'],'sha256':os.environ['TREE_INSTRUCT_REVIEW_SHA256']})
    for file,expected in PACKAGE['files'].items(): require(sha(TOOLS/file)==expected,'Frozen package changed: '+file)
    for file,expected in PACKAGE['external_files'].items(): require(sha(file)==expected,'Frozen external helper changed: '+file)
    import search_r1
    require(Path(search_r1.__file__).resolve()==(CAND/'repo/search_r1/__init__.py').resolve(),'Supervisor imported other source')
    job=os.environ.get('SLURM_JOB_ID','')
    require(job.isdigit() and socket.gethostname().split('.')[0]=='h100-10' and os.environ.get('SLURM_JOB_NODELIST')=='h100-10','Wrong allocation')
    require(len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))==8 and os.environ.get('SLURM_JOB_GPUS'),'Eight reserved GPUs required')
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    validate_inherited_gpu_leases(8)
    from search_r1.baselines.select_teacher import model_manifest
    require(model_manifest(MODEL)==PACKAGE['model_identity'],'Official model changed')
    require(PACKAGE['official_model_path']==str(MODEL) and not any(k in PACKAGE for k in ('parent_manifest','healthy110_metadata')),'Fresh package must not include recovery state')
    lock=(ROOT/'shared-gpu-job.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_runtime_cgroup()
    scope,relative=cgroup_scope(job); owned=CandidateOwnedProcesses(scope,relative)
    for port in (18082,16382): available(port)
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'): os.environ.pop(key,None)
    require(allocation_identity['job_id']==int(job),'Queried allocation numeric identity differs')
    allocation=allocation_identity['allocation_stdout']
    allocated_cpus=allocation_identity['allocated_cpus']
    from production_contract import submitted_intent
    submitted_intent(PACKAGE['dispatch']['once_intent'],os.environ['TREE_INSTRUCT_PACKAGE_SHA256'],os.environ['TREE_INSTRUCT_REVIEW_SHA256'],job,
                     cluster_name=allocation_identity['cluster_name'],node_list=allocation_identity['node_list'],run_id='tree-nested-branch-fresh200-'+job)
    from storage_guard import observe_storage
    observe_storage(ROOT/'runs',PACKAGE['storage_guard'])
    run=ROOT/'runs'/f'tree-nested-branch-fresh200-{job}'; run.mkdir()
    for name in ('output','runtime','cache'): (run/name).mkdir()
    for name in ('ray','tmp'): (run/'runtime'/name).mkdir()
    (run/'allocation.txt').write_text(allocation)
    identity_path=run/'allocation-identity.json'
    with identity_path.open('x') as stream:json.dump(dict(allocation_identity,manifest_sha256=os.environ['TREE_INSTRUCT_PACKAGE_SHA256'],
       review_sha256=os.environ['TREE_INSTRUCT_REVIEW_SHA256'],run_id=run.name,removed_scheduler_environment=removed_scheduler_environment),stream,indent=2)
    alias=Path('/tmp')/f'sst-tree-nested-branch-fresh200-{job}'
    require(not alias.exists() and not alias.is_symlink(),'Existing own runtime alias'); alias.symlink_to(run/'runtime',target_is_directory=True)
    os.environ['PATH']=str(ROOT/'env/bin')+':'+os.environ['PATH']
    os.environ.update(SEARCH_R1_REPO=str(CAND/'repo'),SEARCH_R1_ASSETS=str(ROOT/'assets-root'),SEARCH_R1_BASELINE_OUTPUT=str(run/'output'),SEARCH_R1_BASELINE_RUN_ROOT=str(alias),
      SEARCH_R1_TREE_AUDIT_DIR=str(run/'output/tree-audit'),SEARCH_R1_TREE_RANK_AUDIT_DIR=str(run/'output/tree-rank-audit'),RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER='0',RAY_ADDRESS='127.0.0.1:16382',TMPDIR=str(alias/'tmp'),RAY_TMPDIR=str(alias/'ray'),
      XDG_CACHE_HOME=str(run/'cache'),HF_HOME=str(run/'cache/hf'),HF_DATASETS_CACHE=str(ROOT/'retriever-cache/datasets'),
      OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',WANDB_MODE='offline',WANDB_DIR=str(run),
      PYTHONUNBUFFERED='1',PYTHONPATH=str(TOOLS)+':'+str(ROOT/'tools')+':'+str(CAND/'repo'),TREE_FIXED_ACTOR_PROFILE='1',SEARCH_R1_TRAJECTORY_DIR=str(run/'output/trajectories'),TOKENIZERS_PARALLELISM='false',
      C_INCLUDE_PATH=str(ROOT/'python-dev/usr/include/python3.10')+':'+str(ROOT/'python-dev/usr/include'),
      VLLM_NO_USAGE_STATS='1',NCCL_NVLS_ENABLE='0',VLLM_ATTENTION_BACKEND='XFORMERS',NO_PROXY='127.0.0.1,localhost,::1',no_proxy='127.0.0.1,localhost,::1')
    os.environ.pop('WANDB_API_KEY',None)
    os.environ.update(TREE_VALIDATION_TRAINER_SHA256=PACKAGE['files']['source/repo/verl/trainer/ppo/ray_trainer.py'],TREE_FRESH_MAIN_PPO_SHA256=PACKAGE['files']['source/repo/verl/trainer/main_ppo.py'],TREE_INSTRUCT_TOOLS=str(TOOLS),TREE_INSTRUCT_MODEL=str(MODEL),TREE_COMMON_DATA=str(ROOT/'instruct300/common-data'),SEARCH_R1_MODEL_IDENTITY=PACKAGE['model_identity']['sha256'],SEARCH_R1_BASELINE_CHECKPOINTS=str(run/'checkpoints'))
    require(not any(name in os.environ for name in ('TREE_PPO_CALIBRATION_ONLY','TREE_FRESH_INITIALIZATION_ONLY')),'Calibration environment appeared before training')
    start=supervisor_started; result={'status':'FAILED','stage':a.stage,'execution_mode':DIRECT_MODE,'scope':CHAIN_SCOPE,'parent_completed_outer':0,'planned_new_completed_outer':200,'resumed_current_chain':False,'exact_executor_resume':False,'job_id':job,'code_commit':COMMIT,'actual_training_accepted':False,'independent_acceptance':False,'training_accepted':False,'optimizer_committed':None,'scheduler_advanced':None,'allocated_gpus':8,'requested_cpus':32,'allocated_cpus':allocated_cpus,'production_input_pins':input_pins,'bootstrap_thresholds_accepted':False,'actual_zero_ready':False,'actual_calibration_accepted':False,'production_thresholds_accepted':False,'configured_prospective_thresholds':PRODUCTION_THRESHOLDS,'cluster_name':allocation_identity['cluster_name'],'allocation_identity_sha256':sha(identity_path),'original_sampler_actor_probability_gate_accepted':False}
    try:
        value,steps=configuration(run,a.stage); config_hash=sha(run/'resolved.yaml')
        os.environ.update(SEARCH_R1_BASELINE_AUDIT_DIR=str(run/'output/entry-audit'),SEARCH_R1_BASELINE_RUN_ID=run.name,
          SEARCH_R1_BASELINE_COMMIT=COMMIT,SEARCH_R1_BASELINE_CONFIG_SHA256=config_hash,SEARCH_R1_BASELINE_DATA_IDENTITY=CONTENT)
        (run/'launch-manifest.json').write_text(json.dumps(dict(result,config_sha256=config_hash,split_content_identity=CONTENT,
          cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],gpu_leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']),cgroup=str(scope),
          runtime='same h100-10 allocation: GPU E5 and complete 8-GPU sharded FP16 index, trainer, Ray; loopback only; zero online remote RPC',
          supervisor_source_sha256=sha(__file__),guard_source_sha256=sha(TOOLS/'resource_guard_retained18.py'),owned_runtime_source_sha256=sha(ROOT/'tools/tree_shared_owned_runtime_fixed.py'),initial_gpu=gpu_sample()),indent=2)+'\n')
        guard=owned.spawn([sys.executable,str(TOOLS/'guard_entry.py'),'--source',str(TOOLS/'resource_guard_retained18.py'),
          '--source-sha',PACKAGE['files']['resource_guard_retained18.py'],
          '--identity',str(identity_path),'--identity-sha',sha(identity_path),'--output',str(run/'resource-samples.jsonl')],run/'guard.log')
        deadline=time.monotonic()+30
        while not ((run/'resource-samples.jsonl').is_file() and (run/'resource-samples.ready').is_file()):
            require(guard.poll() is None and time.monotonic()<deadline,'Own GPU guard failed'); time.sleep(.2)
        result['original_background_at_start']=protected_processes()
        first_guard=json.loads((run/'resource-samples.jsonl').read_text().splitlines()[0])
        require(first_guard.get('job_id')==job and len(first_guard.get('gpus',[]))==8 and not first_guard.get('event'),'Guard did not establish healthy inventory')
        fds=tuple(int(r['fd']) for r in json.loads(os.environ['SEARCH_R1_GPU_LEASES']))
        result['synthetic_gpu_probes_reexecuted']=False
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
        # Explicit direct-training authority omits the extra retrieval probe.
        # Service readiness and the existing dual15% coexistence guard remain.
        # Initial whole-card idle admission has already passed before own loads.
        # From here, own memory/SM is expected; held identity and the original
        # foreign15% guard determine continued eligibility.
        os.environ.update(TREE_DIRECT_OWNED_MODELS_LOADED='1',
            TREE_DIRECT_ALLOCATION_IDENTITY_PATH=str(identity_path),TREE_DIRECT_ALLOCATION_IDENTITY_SHA256=sha(identity_path))
        from owned_lease_identity import validate_owned_gpu_leases
        lease_identity=validate_owned_gpu_leases(8)
        (run/'retriever-coexistence.json').write_text(json.dumps({'held_lease_identity':lease_identity,'snapshot':gpu_sample(),'retriever_pid':retriever.pid},indent=2)+'\n')
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
        fresh_start_checked=False
        with (run/'process-resources.jsonl').open('a') as stream:
            while training.poll() is None:
                if not fresh_start_checked and (run/'output/fresh-initialization.json').is_file():
                    result['fresh_official_initialization']=validate_fresh_start(run,PACKAGE,config_hash); fresh_start_checked=True
                from storage_guard import observe_storage
                observe_storage(run, PACKAGE['storage_guard'])
                validate_owned_gpu_leases(8); protected_processes(); guard_health(run,job)
                require(not (run/'output/probability-calibration/ready.json').exists(),'Production entered a zero-write calibration branch')
                # Same fresh trainer continues past fifth credited outer under user authority.
                require(all(proc.poll() is None for proc in (retriever,ray,guard)),'Own dependent service died')
                stream.write(json.dumps({'epoch':time.time(),'retriever':process_sample(retriever.pid),'processes':[process_sample(pid) for pid in sorted(cgroup_pids(scope)-owned.baseline)]})+'\n'); stream.flush(); time.sleep(5)
        require(training.returncode==0,'Actual Tree training failed')
        result['fresh_official_initialization']=validate_fresh_start(run,PACKAGE,config_hash)
        terminal=verify_terminal(run,PACKAGE,config_hash)
        with (run/'output/train200-terminal.json').open('x') as terminal_stream:json.dump(terminal,terminal_stream,indent=2,allow_nan=False)
        require(guard.poll() is None,'Guard died before training completion')
        samples=[json.loads(line) for line in (run/'resource-samples.jsonl').read_text().splitlines()]
        require(samples and all(not x.get('event') for x in samples) and time.time()-samples[-1]['epoch']<15,'Final guard missing/stale/STOP')
        result['final_guard_healthy']=True
        require(sha(run/'output/entry-audit/entry-config.yaml')==config_hash,'Actual entry config differs')
        result.update(status='ACTUAL_HUMAN_AUTHORIZED_DIRECT_FRESH200_TRAINING_COMPLETE',training_body_exit_code=training.returncode,entry_config_verified=True,parent_completed_outer=0,new_completed_outer=200,actual_completed_outer=200,
            actual_rank_optimizer_returns=terminal['actual_rank_optimizer_returns'],opportunities_processed=terminal['opportunities_processed'],
            actual_calls_per_rank=terminal['actual_calls_per_rank'],training_attempts=terminal['training_attempts'],scheduler_steps_each_rank=200,
            exact_executor_resume=False,
            actual_training_accepted=False,final_effect_independently_accepted=False,original_sampler_actor_probability_gate_accepted=False,production_thresholds_accepted=False,terminal_evidence_sha256=sha(run/'output/train200-terminal.json'),human_authorized_same_instance_continuation=True)
    except BaseException as err:
        result.update(error_type=type(err).__name__,error=str(err)); raise
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN); signal.signal(signal.SIGINT,signal.SIG_IGN)
        try:
            try:
                (run/'termination-cost.json').write_text(json.dumps({'job_id':job,'stage':a.stage,'start_epoch':start,'termination_epoch':time.time(),'allocated_gpus':8,'requested_cpus':32,'allocated_cpus':allocated_cpus,'scope':'own supervisor before cleanup'})+'\n')
            except BaseException as err:
                result.update(status='FAILED',termination_record_error=repr(err)); raise
            finally:
                try:
                    owned.cleanup()
                    result['cleanup_residual']=owned.cleanup_residual
                    result['final_gpu']=gpu_sample()
                    background_observation=protected_processes()
                    result['original_background_after_own_cleanup']=background_observation
                    result['final_gpu_process_observation']=final_gpu_process_observation(result['final_gpu'],relative,background_observation)
                    result['original_eight_pid_start_times_preserved']=background_observation['original_eight_pid_start_times_preserved']
                    result['original_background_observation_complete']=True
                    result['current_foreign_processes_untouched']=True
                    # Own cgroup cleanup proves own descendants exited. Foreign
                    # process existence is observed separately and never requires
                    # the original8 to continue living indefinitely.
                    require(all(state=='Z' for state in owned.cleanup_residual.values()),'Live own cgroup residual after cleanup')
                    result['own_cgroup_live_residual_empty']=True
                except BaseException as err:
                    result.update(status='FAILED',cleanup_error=repr(err)); raise
        finally:
            alias.unlink(missing_ok=True); result['wall_seconds']=time.time()-start
            try:result['actual_progress_evidence']=progress_evidence(run)
            except Exception as evidence_error:result['actual_progress_evidence_error']=repr(evidence_error)
            result['failed_unrecorded_partial_optimizer_writes_unknown']=result['status']=='FAILED'
            (run/'run-result.json').write_text(json.dumps(result,indent=2)+'\n')
            code=0 if result['status']=='ACTUAL_HUMAN_AUTHORIZED_DIRECT_FRESH200_TRAINING_COMPLETE' else 1
            (run/'cost.json').write_text(json.dumps({'job_id':job,'stage':a.stage,'execution_mode':DIRECT_MODE,'training_scope':CHAIN_SCOPE,'parent_costs_included':False,'start_epoch':start,'end_epoch':time.time(),'allocated_gpus':8,'requested_cpus':32,'allocated_cpus':allocated_cpus,'exit_code':code,'scope':'entire own supervisor including identity checks, service loading, fresh official nested reward-selected surrogate training, periodic dev and final evaluation, JSON-only terminal accounting and cleanup; earlier calibration input costs remain separate; launcher/preflight and scheduler boundaries separately; CPU allocation is not consumed process CPU time'})+'\n')
    print('TREE_FRESH200_TRAIN_RUN='+str(run),flush=True)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted); main()
