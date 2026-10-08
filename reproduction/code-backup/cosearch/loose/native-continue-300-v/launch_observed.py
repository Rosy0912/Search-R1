"""Native author execution plus external stop/measurement; no source-file patches."""
import functools, hashlib, json, math, numbers, os, re, time
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')

def write(kind, record):
    directory=Path(os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT'])
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/f'controller-{os.getpid()}.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(kind=kind,pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID'),
          monotonic_ns=time.monotonic_ns(),time_ns=time.time_ns(),
          wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),**record),allow_nan=False)+'\n')

def finite_numbers(data):
    result={}
    for key,value in data.items():
        if isinstance(value,numbers.Real):
            result[key]=float(value)
            if not math.isfinite(result[key]):raise ValueError(f'Nonfinite metric {key}')
    return result

def _verify_live_ack(record,job):
    directory=Path('/proc')/str(int(record['pid']))
    stat=directory.joinpath('stat').read_text()
    ticks=int(stat.rsplit(')',1)[1].split()[19])
    assert ticks==record['process_start_ticks'],'ACK process identity has changed'
    cgroup=directory.joinpath('cgroup').read_text()
    assert cgroup==record['process_cgroup'] and re.search(r'(?:^|/)job_'+re.escape(str(job))+r'(?:/|$)',cgroup,re.M)

def validate_acks(directory,job):
    records=[]
    for path in (Path(directory)/'workers').glob('*.jsonl'):
        for line in path.read_text().splitlines():
            record=json.loads(line)
            if record.get('kind')=='ack':records.append(record)
    assert len(records)==8,('Expected precisely eight live training rank ACKs',len(records))
    assert {(r['role'],r['rank']) for r in records}=={(role,rank) for role in ['main','reranker'] for rank in range(4)}
    assert len({r['uuid'] for r in records})==8,'Training pools must occupy eight distinct physical UUIDs'
    leased=set(os.environ['COSEARCH_LEASED_UUIDS'].split(','))
    assert len(leased)==8 and {r['uuid'] for r in records}==leased,'Actor UUIDs must equal this job lease'
    manifest=json.loads(Path(__file__).with_name('source-manifest.json').read_text())
    hook_sha=hashlib.sha256(Path(__file__).with_name('observe_worker.py').read_bytes()).hexdigest()
    manifest_sha=hashlib.sha256(Path(__file__).with_name('source-manifest.json').read_bytes()).hexdigest()
    for record in records:
        assert record['hook_sha256']==hook_sha
        assert record['source_manifest_sha256']==manifest_sha
        assert record['original_actor_file_sha256']==manifest['sha256']['verl/verl/workers/actor/dp_actor.py']
        assert str(record['job'])==str(job)
        assert record['world_size']==4 and record['hostname'].split('.')[0]=='h100-16'
        assert re.search(r'(?:^|/)job_'+re.escape(str(job))+r'(?:/|$)',record['process_cgroup'],re.M)
        _verify_live_ack(record,job)
    return records

def reject_context_before_ppo(step,validate):
    refusals=list((Path(os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT'])/'context-refusals').glob('*'))
    if refusals:
        write('context_batch_rejected_before_ppo',dict(step=step,validation=validate,
            refusal_files=[str(path) for path in refusals]))
        raise RuntimeError('Native full-context refusal: discard entire diagnostic batch before either PPO RPC')

def to_json_value(value):
    import numpy as np
    if isinstance(value,np.ndarray):return to_json_value(value.tolist())
    if isinstance(value,np.generic):return to_json_value(value.item())
    if isinstance(value,dict):return {str(key):to_json_value(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [to_json_value(item) for item in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(f'Unsupported observation metadata type: {type(value).__name__}')

class ObservedPolicyGroup:
    """Same existing Ray group; read complete post-advantage batch before RPC."""
    def __init__(self,group,step,role):self.group,self.step,self.role=group,step,role
    def compute_log_prob(self,*args,**kwargs):return self.group.compute_log_prob(*args,**kwargs)
    def compute_ref_log_prob(self,*args,**kwargs):return self.group.compute_ref_log_prob(*args,**kwargs)
    def update_actor(self,batch):
        import torch
        uids=[str(x) for x in batch.non_tensor_batch['uid']]
        scores=batch.batch['token_level_rewards'].detach().sum(-1).cpu().tolist()
        mask=batch.batch['response_mask'].bool()
        advantages=batch.batch['advantages'].detach()
        nonzero=int(((advantages.abs()>1e-5)&mask).sum().cpu().item())
        groups={}
        for uid,score in zip(uids,scores):groups.setdefault(uid,[]).append(score)
        informative=sum(len(set(values))>1 for values in groups.values())
        write('global_pre_update',dict(step=int(self.step),role=self.role,uids=uids,
            rewards=scores,group_sizes={uid:len(values) for uid,values in groups.items()},
            informative_reward_groups=informative,masked_nonzero_advantage_tokens=nonzero,
            response_ids=batch.batch['responses'].detach().cpu().tolist(),
            response_masks=batch.batch['response_mask'].detach().cpu().tolist(),
            advantages=advantages.cpu().tolist(),
            score_sums={key:batch.batch[key].detach().sum(-1).cpu().tolist() for key in ['token_level_scores','rm_scores'] if key in batch.batch},
            reward_metadata=to_json_value({key:value for key,value in batch.non_tensor_batch.items() if key!='uid'}),
            tensor_keys=list(batch.batch.keys()),non_tensor_keys=list(batch.non_tensor_batch.keys())))
        if self.role=='reranker':
            write('ranker_learning_support_diagnostic',dict(step=int(self.step),informative_reward_groups=informative,masked_nonzero_advantage_tokens=nonzero,diagnostic_only=True))
        directory=Path(os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT'])/'active-batches'
        directory.mkdir(parents=True,exist_ok=True)
        destination=directory/(self.role+'.json')
        pending=destination.with_suffix('.pending')
        pending.write_text(json.dumps(dict(step=int(self.step),role=self.role,uids=sorted(set(uids)))))
        pending.replace(destination)
        result=self.group.update_actor(batch)
        write('global_update_completed',dict(step=int(self.step),role=self.role))
        return result

class ObservedRemoteStep:
    def __init__(self,original):self.original=original
    def remote(self,**kwargs):
        copied=dict(kwargs)
        copied['actor_rollout_wg']=ObservedPolicyGroup(copied['actor_rollout_wg'],copied['global_steps'],copied['agent_name'])
        return self.original.remote(**copied)

def install_controller_observations():
    from verl.utils.tracking import Tracking
    if getattr(Tracking,'_native_observed',False):return
    original=Tracking.log
    @functools.wraps(original)
    def log(tracker,data,step,*args,**kwargs):
        result=original(tracker,data,step,*args,**kwargs)
        if 'training/global_step' in data:
            write('outer',dict(step=int(step),metrics=finite_numbers(data)))
        return result
    Tracking.log=log
    Tracking._native_observed=True

def compose_launch_config():
    from hydra import initialize_config_dir, compose
    argv=json.loads(Path(__file__).with_name('author-cli.json').read_text())
    run_id=os.environ.get('COSEARCH_NATIVE_RUN_ID','author-native-raw-five')
    assert re.fullmatch(r'author-native-[a-z0-9-]+',run_id)
    argv=[item.replace('/runs/author-native-raw-five/',f'/runs/{run_id}/').replace('trainer.experiment_name=author-native-raw-five','trainer.experiment_name='+run_id) for item in argv]
    argv=[item.replace(str(ROOT/'tools/tool.yaml'),str(Path(__file__).with_name('tool.yaml'))) for item in argv]
    argv.append('trainer.resume_mode=disable')
    argv.append('+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=observe_worker.defer_install')
    with initialize_config_dir(config_dir=str(ROOT/'repo/config'),version_base=None):
        config=compose(config_name='co_search_trainer',overrides=argv)
    reload_only=os.environ.get('COSEARCH_NATIVE_RELOAD_ONLY')=='1'
    from continuation import target, source_checkpoint
    continue_to=target()
    if reload_only or continue_to:
        from omegaconf import open_dict
        source_id=os.environ['COSEARCH_NATIVE_SOURCE_RUN_ID']
        assert source_id!=run_id and re.fullmatch(r'author-native-[a-z0-9-]+',source_id)
        checkpoint=ROOT/'runs'/source_id/'checkpoints/global_step_5' if reload_only else source_checkpoint()[2]
        assert checkpoint.is_dir(),'Exact step5 checkpoint required for pure reload'
        with open_dict(config.trainer):
            config.trainer.resume_mode='resume_path'
            config.trainer.resume_from_path=str(checkpoint)
            config.trainer.del_local_ckpt_after_load=False
            if continue_to:
                config.trainer.test_freq=50
                config.trainer.save_freq=1
                config.trainer.max_actor_ckpt_to_keep=2
    else:assert config.trainer.resume_mode=='disable'
    return config,reload_only

def run():
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    import ray
    import main_co_search_ppo as native
    import subprocess
    assert not ray.is_initialized(),'Native job must establish its own explicit worker hook runtime environment'
    assert subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()=='763bf8c4d7caa363ad845d39ddfd53b81ae377bd'
    assert not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip()
    config,reload_only=compose_launch_config()
    from continuation import target, after_original_load
    continue_to=target()
    assert config.trainer.test_freq>0 and config.trainer.save_freq>0
    assert config.reranker_actor_rollout_ref.trainable
    assert config.trainer.reranker_sampling_val_start_step>5
    assert not config.trainer.val_only
    assert not config.actor_rollout_ref.rollout.skip_rollout
    assert not config.transfer_queue.enable
    assert config.trainer.total_training_steps is None
    assert config.actor_rollout_ref.rollout.multi_turn.enable
    assert not config.reranker_actor_rollout_ref.rollout.multi_turn.enable
    write('launch',dict(config=OmegaConf.to_container(config,resolve=True),
        source_commit='763bf8c4d7caa363ad845d39ddfd53b81ae377bd',fresh_five=not reload_only and not continue_to,continue_to=continue_to,reload_only=reload_only,source_run=os.environ.get('COSEARCH_NATIVE_SOURCE_RUN_ID')))
    class ObservedRunner(native.DualAgentTaskRunner):
        def run(self,config):
            original_class=native.CoSearchRayTrainer
            class FiveTrainer(original_class):
                def _load_checkpoint(self):
                    result=super()._load_checkpoint()
                    if continue_to:after_original_load(self,write)
                    return result
                def fit(self):
                    assert self.total_training_steps>=5,'Natural scheduler horizon is smaller than acceptance'
                    assert len(self.val_dataset)==64,'Native periodic validation must contain exactly 64 questions'
                    acks=validate_acks(os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT'],os.environ['SLURM_JOB_ID'])
                    write('all_rank_ack_gate',dict(passed=True,records=acks))
                    write('natural_horizon',dict(total_steps=self.total_training_steps,
                        actor_horizon=self.config.actor_rollout_ref.actor.optim.total_training_steps,
                        ranker_horizon=self.config.reranker_actor_rollout_ref.actor.optim.total_training_steps,
                        workers_initialized=True))
                    if reload_only:
                        # Deliberately bypass the original training fit: it would train step6.
                        self._load_checkpoint()
                        assert self.global_steps==5
                        write('pure_reload_returned',dict(step=5,no_training_fit_called=True))
                        return
                    # Workers/schedulers already initialized; do not change their config/horizon.
                    assert self.total_training_steps >= (continue_to or 5)
                    self.total_training_steps=continue_to or 5
                    import verl.trainer.ppo.search_r1_reranker_ray_trainer as trainer_module
                    original_remote=trainer_module.process_single_agent_ppo_step
                    trainer_module.process_single_agent_ppo_step=ObservedRemoteStep(original_remote)
                    manager=self.async_rollout_manager
                    import empty_ranker
                    empty_ranker.install_controller(manager,original_class,write)
                    original_generate=manager.generate_sequences
                    import phase_sampling
                    original_aggregate, qualified_support=phase_sampling.install_aggregation(manager,write)
                    def generate(batch):
                        validate=bool(batch.meta_info.get('validate',False))
                        step=int(batch.meta_info.get('global_steps',-1))
                        write('generation_start',dict(step=step,validation=validate,input_rows=len(batch)))
                        from observe_worker import preserve_raw_payload
                        from bounded_sampling import generate_supported
                        result=generate_supported(original_generate,batch,qualified_support,
                            phase_sampling.require_training_support,write,preserve_raw_payload,
                            reject_context_before_ppo,step,validate)
                        if not validate:
                            phase_sampling.require_training_support(qualified_support)
                            assert isinstance(result,tuple) and len(result)==2
                            ranker_batch=result[1]
                            assert ranker_batch is not None and (len(ranker_batch)>0 or empty_ranker.is_empty(ranker_batch)), 'Unmarked empty ranker batch'
                            rewards=ranker_batch.batch['rm_scores'].detach().sum(-1).cpu().tolist()
                            uids=[str(value) for value in ranker_batch.non_tensor_batch['uid']]
                            groups={}
                            for uid,reward in zip(uids,rewards):groups.setdefault(uid,[]).append(reward)
                            informative=sum(len(set(values))>1 for values in groups.values())
                            write('whole_batch_reward_preview',dict(step=step,ranker_rows=len(ranker_batch),
                                groups=len(groups),informative_reward_groups=informative))
                            # Learning support is diagnostic; original fallback PPO continues.
                        write('generation_end',dict(step=step,validation=validate))
                        return result
                    manager.generate_sequences=generate
                    try:
                        return super().fit()
                    finally:
                        manager.generate_sequences=original_generate
                        manager._aggregate_reranker_metrics=original_aggregate
                        trainer_module.process_single_agent_ppo_step=original_remote
            native.CoSearchRayTrainer=FiveTrainer
            try:
                install_controller_observations()
                return super().run(config)
            finally:
                native.CoSearchRayTrainer=original_class
    native.run_dual_agent_ppo(config,task_runner_class=ray.remote(num_cpus=1)(ObservedRunner))

if __name__=='__main__':run()
