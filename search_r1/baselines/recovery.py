"""Fail-closed, manifest-bound cross-job BAPO training-state recovery.

The historical v2 payload cannot provide generation-RNG continuity. This
module never manufactures missing historical state or an approval receipt.
"""
import copy,json,math,os,subprocess
from pathlib import Path
from .checkpoint_gate import digest,validate_gate_config

GUARD_JOURNAL_SHA='16e14271d0f7710d7c86ce00e2a5e10bd0bdd754900be7951b5a0224e9574555'
RESOURCE_WATCH_SHA='53ed622dcce4aa5475d80f6489683860a3f43d7d1b6e4c6aa5657649d2a75a80'
OLD_SOURCE='39cdcdd1404213b897bb0de99cc52d72707682f0'
OLD_ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005')
OLD_RUN=OLD_ROOT/'runs/bapo-instruct300-formal300-3231'
HF_FILES={'added_tokens.json','config.json','generation_config.json','merges.txt','model-00001-of-00003.safetensors','model-00002-of-00003.safetensors','model-00003-of-00003.safetensors','model.safetensors.index.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json'}

def require(ok,message):
    if not ok:raise ValueError(message)
def _unique(items):
    out={}
    for key,value in items:
        require(key not in out,'Duplicate JSON proof field '+key);out[key]=value
    return out
def _bad_constant(value):raise ValueError('Nonfinite JSON proof constant '+value)
def decoder():return json.JSONDecoder(object_pairs_hook=_unique,parse_constant=_bad_constant)
def read(path):return decoder().decode(Path(path).read_text())
def exact_file(item,root):
    p=Path(item['path'])
    require(p.is_absolute() and p==p.resolve() and root in p.parents and not p.is_symlink() and p.is_file(),'Unowned/nonregular input')
    require(type(item.get('bytes'))is int and item['bytes']>0 and p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],'Input changed '+str(p))
    return p

def config_identity(config):
    value=copy.deepcopy(config)
    for k in ('default_local_dir','default_hdfs_dir','e1_gate_receipt','resume_full_state','resume_permission','resume_witness_dir','resume_verify_only','project_name','experiment_name','logger'):
        value.get('trainer',{}).pop(k,None)
    value.get('baselines',{}).pop('algorithm_source_commit',None)
    value.get('baselines',{}).get('holdout',{}).pop('output_dir',None)
    # The startup YAML precedes the trainer's resolved total-step injection.
    for section in (value.get('actor_rollout_ref',{}).get('actor',{}),value.get('critic',{})):
        if 'optim'in section:section['optim']['total_training_steps']=value['trainer']['total_training_steps']
    return value

def validate_manifest(path,config,target_source,target_run,permission):
    path=Path(path);m=read(path);run=Path(target_run).resolve();root=run.parent.parent
    require(path.is_absolute() and path==path.resolve() and root in path.parents and not path.is_symlink(),'Owned new recovery manifest required')
    require(m.get('version')==1 and m.get('old_source_commit')==OLD_SOURCE and m.get('old_run_root')==str(OLD_RUN),'Wrong historical provenance')
    for key,value in [('step',25),('next_step',26),('actual_outer_updates',25),('world_size',8)]:
        require(type(m.get(key))is int and m[key]==value,'Wrong complete25 counter '+key)
    require(type(m.get('target_job_id'))is int and m['target_job_id']>0 and os.environ.get('SLURM_JOB_ID')==str(m['target_job_id']) and os.environ.get('BAPO_RUN_ID')==str(run),'Actual allocated job/run mismatch')
    require(m.get('target_source_commit')==target_source and m.get('target_run_root')==str(run),'Wrong new source/run')
    require(validate_gate_config(config)==(300,114),'Changed 300/114 budget')
    from omegaconf import OmegaConf
    config_path=exact_file(m['resolved_config'],OLD_RUN)
    require(config_path==OLD_RUN/'resolved-config.yaml','Actual run-start config file required')
    old_config=OmegaConf.to_container(OmegaConf.load(config_path),resolve=True)
    require(config_identity(config)==config_identity(old_config),'Training/data/algorithm/runtime config changed')
    require(old_config['algorithm']['adv_estimator']=='grpo'and old_config['baselines']['method']=='bapo'and old_config['algorithm']['kl_ctrl']['type']=='fixed','Unsupported unsaved mutable controller')
    ranks=m.get('ranks',[]);require(len(ranks)==8 and {r.get('rank')for r in ranks}==set(range(8)),'Eight unique rank inputs required')
    for r in ranks:
        require(type(r['rank'])is int,'Rank must be integer')
        pt=exact_file(r['state'],OLD_RUN);rp=exact_file(r['report'],OLD_RUN);report=read(rp)
        require(pt==OLD_RUN/'checkpoints/actor/global_step_25/training_state'/f"rank_{r['rank']}.pt" and rp==pt.with_suffix('.json'),'Wrong rank path/step')
        require(type(report.get('rank'))is int and type(report.get('world_size'))is int and type(report.get('step'))is int and report.get('rank')==r['rank']and report.get('world_size')==8 and report.get('step')==25 and report.get('path')==str(pt)and report.get('sha256')==r['state']['sha256']and report.get('status')=='FULL_STATE_SERIALIZED_OK'and report.get('scheduler_last_epoch')==25 and type(report.get('pid'))is int and report['pid']>0 and report.get('optimizer_state_entries',0)>0 and report.get('model_dtensor_entries',0)>0,'Historical rank identity incomplete')
    driver=exact_file(m['driver'],OLD_RUN);dr=read(exact_file(m['driver_report'],OLD_RUN));algorithm=read(exact_file(m['algorithm'],OLD_RUN))
    require(Path(m['driver_report']['path'])==driver.parent/'driver-reload.json'and Path(m['algorithm']['path'])==OLD_RUN/'checkpoints/algorithm/global_step_25/state.json'and driver==OLD_RUN/'checkpoints/full_state/global_step_25/driver.pt'and dr.get('path')==str(driver)and dr.get('sha256')==m['driver']['sha256']and dr.get('status')=='DRIVER_SERIALIZED_OK'and dr.get('step')==25 and dr.get('sampler_cursor')==12800,'Historical driver incomplete')
    require(algorithm.get('method')=='bapo'and algorithm.get('global_step')==25 and algorithm.get('actual_outer_updates')==25 and algorithm.get('algorithm_source_commit')==OLD_SOURCE,'Historical algorithm identity mismatch')
    inputs=m.get('shared_inputs',[]);require(inputs and len({x['path']for x in inputs})==len(inputs),'Shared input inventory required')
    bindings={x['path']:x for x in inputs}
    for item in inputs:exact_file(item,OLD_ROOT)
    def paths(value):return value if isinstance(value,list)else [value]
    required=set(paths(old_config['data']['train_files'])+paths(old_config['data']['val_files'])+paths(old_config['data']['final_val_files'])+paths(old_config['baselines']['bapo']['stage_validation_files']))
    data_manifest=old_config['baselines']['holdout']['manifest_path'];required.add(data_manifest)
    model=Path(old_config['actor_rollout_ref']['model']['path']);model_manifest=model/'VERIFIED-MODEL-MANIFEST.json';required.add(str(model_manifest))
    require(required<=set(bindings),'Actual train/dev/stage/final/model manifests not bound')
    require(bindings[data_manifest]['sha256']==old_config['baselines']['holdout']['manifest_sha256'],'Shared data manifest differs from actual config')
    official=read(model_manifest)
    require(official.get('repo')=='Qwen/Qwen2.5-3B-Instruct'and official.get('revision')=='aa8e72537993ba99e69dfaafa59ed015b17504d1','Official base model identity mismatch')
    for item in official['files']:
        name=str(model/item['file']);require(name in bindings and bindings[name]['sha256']==item['sha256']and bindings[name]['bytes']==item['bytes'],'Official base model file missing/changed')
    hf=m.get('hf_files',[]);require(len(hf)==12 and {Path(x['path']).name for x in hf}==HF_FILES and len({x['path']for x in hf})==12,'Full HF12 inventory required')
    for item in hf:require(exact_file(item,OLD_RUN).parent==OLD_RUN/'checkpoints/actor/global_step_25','Wrong HF checkpoint')
    release=read(exact_file(m['historical_first5_receipt'],OLD_RUN));require(release.get('status')=='PASS_INDEPENDENT_E1_STEP5_REVIEW'and release.get('run_root')==str(OLD_RUN)and release.get('source_commit')==OLD_SOURCE and release.get('actual_outer_updates')==5,'Historical first5 not accepted')
    prefix=exact_file(m['train_prefix'],root);live=m['train_prefix_live']
    require(live.get('path')==str(OLD_RUN/'train.log')and type(live.get('bytes'))is int and live['bytes']==prefix.stat().st_size and live.get('sha256')==m['train_prefix']['sha256'],'Historical prefix source identity mismatch')
    import hashlib
    h=hashlib.sha256();remaining=live['bytes']
    with (OLD_RUN/'train.log').open('rb')as handle:
        while remaining:
            block=handle.read(min(1048576,remaining));require(block,'Historical live log shorter than prefix');h.update(block);remaining-=len(block)
    require(h.hexdigest()==live['sha256'],'Historical live prefix differs')
    events=[decoder().raw_decode(s.split('BAPO_TRAIN_METRICS_JSON=',1)[1].lstrip())[0]for s in prefix.read_text().splitlines()if 'BAPO_TRAIN_METRICS_JSON='in s]
    require([e['step']for e in events]==list(range(1,26))and all(type(e['step'])is int and type(e['metrics']['bapo/actual_outer_updates'])is int and e['metrics']['bapo/actual_outer_updates']==e['step']for e in events),'Historical exact25 prefix required')
    permit_path=Path(permission);require(permit_path.is_absolute()and permit_path==permit_path.resolve()and root in permit_path.parents and not permit_path.is_symlink(),'Owned new independent permission required')
    permit=read(permit_path)
    verify_only=config['trainer'].get('resume_verify_only',False)
    require(type(verify_only)is bool,'verify-only must be boolean')
    expected_status='PASS_INDEPENDENT_BAPO_RESUME25_GPU_VERIFY_PERMISSION' if verify_only else 'PASS_INDEPENDENT_BAPO_RESUME25_REVIEW'
    require(permit.get('status')==expected_status and permit.get('manifest_sha256')==digest(path)and permit.get('target_source_commit')==target_source and permit.get('target_run_root')==str(run)and permit.get('target_job_id')==m['target_job_id'],'New independent resume permission missing')
    require(permit.get('sampling_continuity')=='LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT','Historical generation RNG gap must be acknowledged')
    exact_file(permit['independent_review'],root)
    proof=permit.get('code_review_evidence' if verify_only else 'gpu_witnesses',[]);require(proof,'Independent code/GPU witnesses required')
    for item in proof:exact_file(item,root)
    if not verify_only:validate_gpu_proof(permit,m,old_config,root)
    return m

def validate_driver(payload,m):
    from .full_state import state_digest
    exact_file(m['driver'],OLD_RUN)
    report=read(exact_file(m['driver_report'],OLD_RUN));algorithm=read(exact_file(m['algorithm'],OLD_RUN))
    require(state_digest(payload)==report['state_digest'],'Driver content digest changed')
    require(payload['global_step']==25 and payload['next_global_step']==26 and payload['actual_outer_updates']==25 and payload['epoch']==0 and payload['driver_pid']==report['pid'],'Driver counters/PID mismatch')
    require(payload['sampler']['cursor']==12800 and payload['sampler']['epoch']==0 and payload['rng']['cuda']==[] and payload['bapo']==algorithm['bapo'],'Driver sampler/RNG/BAPO mismatch')

def validate_native(payload,report,rank,world):
    from .full_state import state_digest
    require(type(rank)is int and type(world)is int and world==8 and payload.get('version')==2 and payload.get('rank')==rank and payload.get('world_size')==world and payload.get('step')==25 and payload.get('model_state_dict_type')=='SHARDED_STATE_DICT','Native rank/world/type mismatch')
    require(type(payload['rank'])is int and type(payload['world_size'])is int and type(payload['step'])is int and state_digest(payload)==report['state_digest']and payload['scheduler']['last_epoch']==25,'Native payload/scheduler digest mismatch')
    for value in payload['model'].values():
        local=value.to_local()if hasattr(value,'to_local')else value
        require(__import__('torch').isfinite(local).all().item(),'Nonfinite model state')
    state=payload['optimizer']['state'];require(state and len(state)==report['optimizer_state_entries'],'Native Adam empty/incomplete')
    import torch
    for s in state.values():
        require({'step','exp_avg','exp_avg_sq'}<=set(s),'Adam moments incomplete')
        step=s['step']
        require(torch.is_tensor(step) and step.ndim==0 and step.numel()==1 and step.dtype in (torch.float32,torch.float64) and torch.isfinite(step).item() and step.item()==500.0,'Adam must retain exact finite scalar 25*20 updates')
        for key in ('exp_avg','exp_avg_sq'):require(torch.isfinite(s[key]).all().item(),'Nonfinite Adam state')
    expected=1e-6*25/114
    groups=payload['optimizer']['param_groups'];rates=payload['scheduler']['_last_lr']
    require(isinstance(groups,list)and groups and isinstance(rates,list)and rates and len(groups)==len(rates),'Nonempty matching native Adam/scheduler groups required')
    require(all(isinstance(g,dict)and type(g.get('lr'))in (int,float)and math.isfinite(g['lr'])and math.isclose(g['lr'],expected,rel_tol=1e-12,abs_tol=1e-18)for g in groups),'Adam LR25/114 mismatch')
    require(all(type(x)in (int,float)and math.isfinite(x)and math.isclose(x,expected,rel_tol=1e-12,abs_tol=1e-18)for x in rates),'Scheduler LR mismatch')

def write_witness(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(value,f,indent=2,allow_nan=False)

def process_identity():
    stat=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()
    require(stat[0]not in ('Z','X'),'Recovery process exited')
    return dict(pid=os.getpid(),start_ticks=int(stat[19]),state=stat[0],
                actual_env={k:os.environ.get(k)for k in ('SLURM_JOB_ID','BAPO_RUN_ID','CUDA_VISIBLE_DEVICES')})

def _finite(value):
    if type(value)is float:require(math.isfinite(value),'Nonfinite GPU proof field')
    elif isinstance(value,dict):
        for v in value.values():_finite(v)
    elif isinstance(value,list):
        for v in value:_finite(v)

def _process_witness(w,job,run):
    require(type(w.get('old_pid'))is int and w['old_pid']>0,'Positive integer historical PID required')
    p=w.get('process',{});require(type(w.get('new_pid'))is int and w['new_pid']>0 and type(p.get('pid'))is int and p.get('pid')==w['new_pid']and type(p.get('start_ticks'))is int and p['start_ticks']>0 and p.get('state')in {'R','S','D','T','t','W','I','P'},'Actual fresh process witness missing')
    require(p.get('actual_env',{}).get('SLURM_JOB_ID')==str(job)and p['actual_env'].get('BAPO_RUN_ID')==str(run),'GPU process allocation identity mismatch')

def validate_gpu_proof(permit,m,old_config,root,*,hardware_self_review=False):
    """A nonempty JSON/hash list is never a production recovery proof."""
    g=permit.get('gpu_validation',{});job=g.get('job_id');run=Path(g.get('run_root',''))
    require(type(hardware_self_review)is bool,'Readonly hardware review mode must be boolean')
    require(type(job)is int and job>0 and run.is_absolute()and run==run.resolve()and run==root/'runs'/f'bapo-instruct300-hardware-{job}','Actual GPU verification allocation required')
    if hardware_self_review:
        require(permit.get('status')=='PASS_INDEPENDENT_E1_HARDWARE_REVIEW'and type(permit.get('job_id'))is int and permit.get('job_id')==job and permit.get('run_root')==str(run)and permit.get('root')==str(root)and permit.get('source_commit')==m['target_source_commit']and m['target_job_id']==job and m['target_run_root']==str(run),'Readonly hardware self-review identity mismatch')
    else:
        require(job!=m['target_job_id'],'Separate actual GPU verification allocation required')
    gm=read(exact_file(g['manifest'],root))
    require(gm.get('target_job_id')==job and gm.get('target_run_root')==str(run)and gm.get('target_source_commit')==m['target_source_commit'],'GPU input manifest target mismatch')
    for key in ('old_source_commit','old_run_root','step','next_step','actual_outer_updates','world_size','ranks','driver','driver_report','algorithm','resolved_config','shared_inputs','hf_files','historical_first5_receipt','train_prefix','train_prefix_live'):
        require(gm.get(key)==m.get(key),'GPU verification used different training input '+key)
    bindings=permit.get('gpu_witnesses',[])
    require(len(bindings)==9 and {Path(b['path']).name for b in bindings}=={f'rank_{i}.json'for i in range(8)}|{'driver.json'},'Exact eight rank and driver witnesses required')
    records={}
    for b in bindings:
        p=exact_file(b,run);require(p.parent==run/'recovery-witness','Actual GPU witness directory required');records[p.name]=read(p);_finite(records[p.name])
    ranks=[records[f'rank_{i}.json']for i in range(8)];d=records['driver.json'];expected_lr=1e-6*25/114
    expected_model=read(Path(old_config['actor_rollout_ref']['model']['path'])/'config.json')
    for i,w in enumerate(ranks):
        old=read(next(b['report']['path']for b in m['ranks']if b['rank']==i));entry=next(b for b in m['ranks']if b['rank']==i)
        for key,value in [('rank',i),('world_size',8),('step',25),('next_step',26),('scheduler_last_epoch',25)]:require(type(w.get(key))is int and w[key]==value,'GPU native counter/type mismatch '+key)
        require(w.get('status')=='EXACT_CROSS_JOB_FULL_STATE_RELOAD_OK'and w.get('distributed_backend')=='nccl'and w.get('model_state_dict_type')=='SHARDED_STATE_DICT'and w.get('model_dtensor_entries')==old['model_dtensor_entries'],'Real NCCL8/FSDP model identity missing')
        require(w.get('model_class')in expected_model['architectures'],'Actual Qwen model class differs')
        require(w.get('model_config')=={k:expected_model[k]for k in ('model_type','hidden_size','num_hidden_layers','vocab_size','architectures')},'Real Qwen model identity differs')
        require(w.get('old_pid')==old['pid']and w.get('new_pid')!=old['pid']and w.get('old_state_sha256')==entry['state']['sha256']and w.get('state_digest')==old['state_digest']and w.get('optimizer_state_entries')==old['optimizer_state_entries'],'GPU native state/Adam/old PID mismatch')
        require(isinstance(w.get('lr'),list)and w['lr']and all(type(x)in (int,float)and math.isclose(x,expected_lr,rel_tol=1e-12,abs_tol=1e-18)for x in w['lr']),'GPU scheduler LR mismatch')
        require(w.get('actual_env',{}).get('SLURM_JOB_ID')==str(job)and w['actual_env'].get('BAPO_RUN_ID')==str(run),'Native GPU environment mismatch')
    old_driver=read(m['driver_report']['path'])
    require(d.get('status')=='EXACT_CROSS_JOB_DRIVER_RELOAD_OK'and d.get('verify_only')is True and d.get('no_generation_or_optimizer_update')is True,'Real verification-only nontraining driver required')
    for key,value in [('step',25),('next_step',26),('actual_outer_updates',25),('epoch',0),('sampler_cursor',12800),('target_job_id',job)]:require(type(d.get(key))is int and d[key]==value,'GPU driver counter/type mismatch '+key)
    require(d.get('old_pid')==old_driver['pid']and d.get('new_pid')!=old_driver['pid']and d.get('loaded_state_digest')==old_driver['state_digest']and d.get('bapo')==read(m['algorithm']['path'])['bapo'],'GPU driver content differs from25')
    for name in ('sampler','rng'):
        require(isinstance(d.get('saved_'+name+'_digest'),str)and len(d['saved_'+name+'_digest'])==64 and d['saved_'+name+'_digest']==d.get('restored_'+name+'_digest'),'Driver restored '+name+' differs')
    for w in ranks+[d]:
        require(w.get('target_source_commit')==m['target_source_commit']and w.get('target_run_root')==str(run)and w.get('sampling_continuity')=='LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT','GPU proof source/run/sampling mismatch')
        _process_witness(w,job,run)
        rel='verl/trainer/ppo/ray_trainer.py'if w is d else 'verl/workers/fsdp_workers.py';module=root/'repo'/rel
        require(w.get('runtime_module_path')==str(module)and w.get('runtime_module_sha256')==digest(module),'GPU runtime source module changed/shadowed')
    require(len({w['new_pid']for w in ranks+[d]})==9 and d.get('rank_witnesses')==ranks,'Nine distinct actual processes/driver eight rank binding missing')
    raw=g.get('raw_artifacts',{});needed={'slurm_terminal','source_start','resolved_config','tools_manifest','cpu_gate','cpu_log','cpu_source','launch_approval','guard_summary','guard_helper','resource_watcher','resource_watch','cleanup','train_log'}
    require(set(raw)==needed,'Complete actual GPU raw evidence set required')
    paths={k:exact_file(b,root)for k,b in raw.items()}
    import re
    text=paths['slurm_terminal'].read_text()
    for field,value in [('JobId',str(job)),('JobState','COMPLETED'),('ExitCode','0:0'),('DerivedExitCode','0:0'),('Restarts','0'),('NodeList','h100-5')]:require(re.search(r'(?:^|\s)'+field+'='+re.escape(value)+r'(?:\s|$)',text),'GPU terminal record not completed0/restarter0')
    require(paths['source_start']==run/'source-commit.txt'and paths['source_start'].read_text().strip()==m['target_source_commit'],'GPU source-start not bound')
    from omegaconf import OmegaConf
    cfg=OmegaConf.to_container(OmegaConf.load(paths['resolved_config']),resolve=True)
    require(cfg['trainer'].get('resume_verify_only')is True and config_identity(cfg)==config_identity(old_config),'GPU verification config changed/no verify-only')
    tools=read(paths['tools_manifest']);require(tools.get('source_commit')==m['target_source_commit']and isinstance(tools.get('files'),dict)and type(approval_count:=g.get('tools_file_count'))is int and approval_count>=22 and len(tools['files'])==approval_count,'New bound22 tools required')
    for name,sha in tools['files'].items():
        p=root/'tools'/name;require(p.parent==root/'tools'and not p.is_symlink()and p==p.resolve()and digest(p)==sha,'New GPU tool changed')
    cpu=read(paths['cpu_gate']);approval=read(paths['launch_approval'])
    require(all(type(cpu.get(k))is int for k in ('exit_code','failed','skipped','passed'))and cpu.get('source_commit')==m['target_source_commit']and cpu.get('exit_code')==0 and cpu.get('failed')==0 and cpu.get('skipped')==0 and cpu.get('scope')=='ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE'and cpu.get('sys_prefix')==str(root/'env'),'Actual target LinuxCPU proof missing')
    require(type(approval.get('cpu_expected_passed'))is int and approval['cpu_expected_passed']>=327 and cpu.get('passed')==approval['cpu_expected_passed']and approval.get('source_commit')==m['target_source_commit']and approval.get('tools_file_count')==approval_count,'Actual complete LinuxCPU suite binding missing')
    require(cpu.get('log_sha256')==digest(paths['cpu_log'])and paths['cpu_source'].read_text().strip()==m['target_source_commit']and approval.get('tools_manifest_sha256')==digest(paths['tools_manifest'])and approval.get('cpu_gate_record_sha256')==digest(paths['cpu_gate']),'New tools/LinuxCPU marker hashes differ')
    require(read(paths['cleanup']).get('remaining')==[],'GPU verification cleanup incomplete')
    summary=read(paths['guard_summary']);_finite(summary)
    require(paths['guard_helper']==root/'tools/guard_journal.py'and paths['resource_watcher']==root/'tools/resource_watch.py','Bound exact guard replay/watcher modules required')
    require(digest(paths['guard_helper'])==GUARD_JOURNAL_SHA and digest(paths['resource_watcher'])==RESOURCE_WATCH_SHA,'Independently reviewed guard code changed')
    import importlib.util,sys
    spec=importlib.util.spec_from_file_location('bapo_bound_guard_journal_'+GUARD_JOURNAL_SHA[:12],paths['guard_helper']);helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
    replayed=helper.replay(paths['resource_watch'],job,str(run),m['target_source_commit'],paths['resource_watcher'])
    require(digest(paths['guard_helper'])==GUARD_JOURNAL_SHA and digest(paths['resource_watcher'])==RESOURCE_WATCH_SHA,'Guard code changed during replay')
    require(summary.get('status')=='PASS_ACTUAL_GUARD_JOURNAL_REPLAY'and summary.get('job_id')==job and summary.get('source_commit')==m['target_source_commit']and summary.get('run_root')==str(run)and summary.get('raw_sha256')==digest(paths['resource_watch']),'Guard summary raw binding missing')
    require(summary.get('helper_sha256')==GUARD_JOURNAL_SHA and summary.get('watcher_sha256')==RESOURCE_WATCH_SHA,'Guard replay module origin binding missing')
    require(summary==replayed,'Guard summary does not match actual bound journal replay')
    v3=g.get('native_v3_roundtrip',[])
    require(isinstance(v3,list)and len(v3)==8 and {x.get('rank')for x in v3}==set(range(8)),'Actual production eight-rank v3 roundtrip required')
    v3_reports=[]
    for i in range(8):
        entry=next(x for x in v3 if x['rank']==i);require(type(entry['rank'])is int,'V3 rank must be integer')
        pt=exact_file(entry['state'],run);rp=exact_file(entry['report'],run)
        require(pt==run/'recovery-native-v3-roundtrip'/f'rank_{i}.pt'and rp==pt.with_suffix('.json'),'Exact fresh production v3 directory required')
        report=read(rp);_finite(report);top=report.get('generation_rng_topology',{})
        require(report.get('status')=='EXACT_FULL_STATE_RELOAD_OK'and type(report.get('native_state_version'))is int and report.get('native_state_version')==3 and report.get('roundtrip')is True and type(report.get('rank'))is int and report['rank']==i and type(report.get('world_size'))is int and report['world_size']==8 and type(report.get('step'))is int and report['step']==25 and report.get('path')==str(pt)and report.get('sha256')==entry['state']['sha256'],'V3 actual rank/state identity mismatch')
        require(all(type(report.get(k))is int for k in ('pid','scheduler_last_epoch','optimizer_state_entries','model_dtensor_entries'))and report.get('pid')==ranks[i]['new_pid']and report.get('scheduler_last_epoch')==25 and report.get('optimizer_state_entries')==ranks[i]['optimizer_state_entries']and report.get('model_state_dict_type')=='SHARDED_STATE_DICT'and report.get('model_dtensor_entries')==ranks[i]['model_dtensor_entries'],'V3 production state topology changed')
        for before,after in [('state_digest','reloaded_state_digest'),('generation_rng_before_digest','generation_rng_after_digest')]:
            require(isinstance(report.get(before),str)and re.fullmatch('[0-9a-f]{64}',report[before])and report[before]==report.get(after),'Actual v3 reload digest mismatch')
        require(type(top.get('version'))is int and top.get('version')==1 and top.get('device_mesh')is not None,'Production generation RNG mesh missing')
        for name in ('torch_random_states','gen_random_states'):
            size=top.get(name+'_numel');require(type(size)is int and size>0 and top.get(name+'_dtype')=='torch.uint8'and top.get(name+'_shape')==[size],'V3 generation RNG tensor topology invalid')
        mesh=top['device_mesh'];require(isinstance(mesh.get('shape'),list)and all(type(x)is int for x in mesh['shape'])and isinstance(mesh.get('ranks'),list)and all(isinstance(row,list)and all(type(x)is int for x in row)for row in mesh['ranks'])and mesh.get('shape')==[8,1]and mesh.get('names')==['dp','infer_tp']and mesh.get('ranks')==[[i]for i in range(8)],'Production vLLM DP8/TP1 generation mesh changed')
        v3_reports.append(report)
    text=paths['train_log'].read_text()
    require('BAPO_TRAIN_METRICS_JSON='not in text and 'E1_STEP5_PAUSED_JSON='not in text,'GPU verify-only unexpectedly trained/paused historical5')
    markers=[decoder().raw_decode(x.split('BAPO_RECOVERY_RANK_JSON=',1)[1].lstrip())[0]for x in text.splitlines()if 'BAPO_RECOVERY_RANK_JSON='in x]
    drivers=[decoder().raw_decode(x.split('BAPO_RECOVERY_DRIVER_JSON=',1)[1].lstrip())[0]for x in text.splitlines()if 'BAPO_RECOVERY_DRIVER_JSON='in x]
    require(len(markers)==8 and sorted(markers,key=lambda w:w['rank'])==ranks and drivers==[d],'Actual raw GPU train log does not bind exactly9 recovery producers')
    v3_markers=[decoder().raw_decode(x.split('BAPO_RECOVERY_V3_RANK_JSON=',1)[1].lstrip())[0]for x in text.splitlines()if 'BAPO_RECOVERY_V3_RANK_JSON='in x]
    require(len(v3_markers)==8 and sorted(v3_markers,key=lambda w:w['rank'])==v3_reports,'Raw GPU log does not bind exact8 production v3 roundtrips')
