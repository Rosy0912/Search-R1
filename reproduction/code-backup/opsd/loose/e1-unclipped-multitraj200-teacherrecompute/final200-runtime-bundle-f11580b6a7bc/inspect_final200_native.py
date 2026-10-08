"""NativeTorch2.4 stored200 content inspection, never live/new-process restore."""
import argparse
import math
from pathlib import Path
import time

# Only unchanged mathematical helpers are executed. Their step5 entrypoint,
# fixed LR Adam/scheduler functions and first5 archive consumer are not called.
SCOPE='NATIVE_CPU_FINAL200_STORED_CHECKPOINT_CONTENT_INSPECTION_NOT_LIVE_RESTORE'


def load_common(expected_sha):
    import hashlib,os,re,stat,types
    p=Path(__file__).resolve().parent/'common.py';before=p.stat()
    if not(re.fullmatch('[0-9a-f]{64}',expected_sha) and p.resolve(strict=True)==p and before.st_uid==os.getuid()):raise ValueError('Exact owned common source required')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        opened=os.fstat(fd);raw=b''
        while True:
            chunk=os.read(fd,1024**2)
            if not chunk:break
            raw+=chunk
        after=os.fstat(fd)
    finally:os.close(fd)
    names=('st_dev','st_ino','st_size','st_uid','st_mode','st_mtime_ns','st_ctime_ns')
    if not stat.S_ISREG(opened.st_mode) or any(tuple(getattr(x,k) for k in names)!=tuple(getattr(before,k) for k in names) for x in (opened,after,p.stat())) or hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('Common changed before exact-byte compilation')
    module=types.ModuleType('final200_common');module.__file__=str(p);exec(compile(raw,str(p),'exec',dont_inherit=True,optimize=0),module.__dict__)
    module.read_file(p,expected_sha);return module


def adam200(value,helper,c,expected_numel):
    c.require(type(value) is dict and set(value)=={'state','param_groups'} and type(value['state']) is dict and type(value['param_groups']) is list and value['state'] and value['param_groups'],'Adam200 schema missing')
    params=[];lrs=[]
    for group in value['param_groups']:
        c.require(type(group) is dict and group['betas']==(0.9,0.999) and group['eps']==1e-8 and group['weight_decay']==.01 and group.get('amsgrad') is False and group.get('maximize') is False,'Adam200 algorithm/hyperparameters differ')
        c.require(type(group['lr']) is float and math.isfinite(group['lr']) and group['lr']==1e-6,'Adam200 saved constant-after-warmup LR differs')
        c.require(type(group['params']) is list and all(type(v) is int for v in group['params']),'Adam200 parameter IDs not typed')
        params.extend(group['params']);lrs.append(group['lr'])
    c.require(len(set(params))==len(params) and set(params)==set(value['state']) and all(type(k) is int for k in value['state']),'Adam200 ID/state binding differs')
    import torch
    entries=[];numel=0
    for param in params:
        item=value['state'][param];c.require(type(item) is dict and set(item)=={'step','exp_avg','exp_avg_sq'},'Adam200 moment schema differs')
        step=item['step'];c.require(torch.is_tensor(step) and step.device.type=='cpu' and step.numel()==1 and step.is_floating_point() and math.isfinite(float(step)) and float(step)==2000,'Adam200 actual step differs')
        a,b=item['exp_avg'],item['exp_avg_sq']
        c.require(torch.is_tensor(a) and torch.is_tensor(b) and a.dtype==b.dtype==torch.float32 and a.shape==b.shape and a.ndim==1 and a.numel()>0,'Adam200 moments layout differs')
        helper.finite(a);helper.finite(b)
        for part in b.split(1024**2):c.require(bool((part>=0).all()),'Negative Adam200 second moment')
        numel+=a.numel();entries.append(dict(parameter_id=param,step=2000,shape=list(a.shape),dtype=str(a.dtype),exp_avg_sha256=helper.tensor_hash(a),exp_avg_sq_sha256=helper.tensor_hash(b)))
    c.require(numel==expected_numel,'Adam200 complete local flattened numel differs')
    return dict(entries=entries,local_parameter_numel=numel,lrs=lrs,steps=[2000])


def scheduler200(value,c):
    for key,target in [('last_epoch',200),('_step_count',201)]:c.typed(value[key],target,'Scheduler200 clock differs: '+key)
    for key in ('base_lrs','_last_lr'):
        c.require(type(value[key]) is list and len(value[key])==1 and type(value[key][0]) is float and value[key][0]==1e-6,'Scheduler200 LR differs')
    c.require(value['lr_lambdas']==[None],'Scheduler200 lambda state differs')
    return dict(last_epoch=200,step_count=201,base_lrs=value['base_lrs'],last_lr=value['_last_lr'])


def archive_progress200(algorithm,inventory,c):
    value=algorithm.get('trajectory_archive')
    c.require(type(value) is dict and set(value)=={'version','completed_outer','last_manifest','student_per_outer','teacher_per_outer','scope'},'Algorithm200 archive schema differs')
    for key,target in dict(version=1,completed_outer=200,student_per_outer=2560,teacher_per_outer=512,scope='ARCHIVE_PROGRESS_ONLY_NO_GENERATED_TARGETS').items():c.typed(value[key],target,'Algorithm200 archive progress differs: '+key)
    record=inventory['archive_files']['trajectory_outer_000200/paired-manifest.json'];last=value['last_manifest']
    c.require(type(last) is dict and set(last)=={'path','sha256','bytes'} and type(last['bytes']) is int and last==dict(path=record['physical_path'],sha256=record['sha256'],bytes=record['bytes']),'Algorithm200 last raw paired manifest differs')
    c.verify_record(record,c.RUN)
    return value


def teacher200(value,report,base,assets,base_digest,helper,state,c):
    c.require(helper.teacher_report_exact({k:v for k,v in value.items() if k!='buffers'},report),'Raw200 teacher/JSON exact projection differs')
    c.require(type(value['version']) is int and value['version']==1 and type(value['parameter_count']) is int and value['parameter_count']==c.NUMEL and value['storage_device']=='cpu' and value['model_content_digest']==base_digest,'Fixed200 FP32 teacher identity differs')
    import re
    c.require(type(value['execution_device']) is str and re.fullmatch(r'cuda(?::[0-9]+)?',value['execution_device']),'Teacher200 execution device differs')
    c.require(Path(value['asset_directory'])==base.directory and value['asset_files']==assets,'Teacher200 official assets differ')
    counters=value['counters'];c.require(set(counters)==set(helper.COUNTERS) and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in counters.values()),'Teacher200 counters invalid')
    c.typed(counters['cuda_update_entries'],200,'Teacher200 update clock differs')
    c.require(type(counters['scored_tokens']) is int and counters['scored_tokens']>0,'Teacher200 scored tokens missing')
    for key in ('model_type','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','vocab_size','tie_word_embeddings'):c.typed(value['model_config'][key],base.config[key],'Teacher200 architecture differs')
    for name,cache in value['caches'].items():
        c.require(type(name) is str and type(cache) is dict and cache and set(cache)<={'max_seq_len_cached','original_max_seq_len'} and all(type(v) is int and v>=0 for v in cache.values()),'Teacher200 history cache metadata invalid')
    c.require(type(value['buffers']) is dict and all(type(k) is str for k in value['buffers']),'Teacher200 saved buffers missing')
    for tensor in value['buffers'].values():helper.finite(tensor)
    c.require(state.state_digest(value['buffers'])==value['buffer_content_digest'] and type(value['model_bytes']) is int and value['model_bytes']==c.NUMEL*4+sum(v.numel()*v.element_size() for v in value['buffers'].values()),'Teacher200 saved buffer bytes differ')
    return dict(content_digest=base_digest,cuda_update_entries=200,counters=counters,buffer_content_digest=value['buffer_content_digest'],history_caches=value['caches'],teacher_parameters_anchored_to_fixed_official_assets_not_raw_arrays=True)


def inspect(a,c):
    own=c.read_file(Path(__file__).resolve())[1]
    execution=c.execution_identity(a.execution_job_id,a.execution_step_id);started=time.monotonic()
    inventory,irec=c.read_json(a.inventory,a.inventory_sha256);c.validate_inventory(inventory)
    c.typed(inventory['accepted5']['receipt_sha256'],a.accepted5_receipt_sha256,'Observed historical receipt differs')
    context=c.accepted_context(a.accepted5_receipt_sha256);c.require(inventory['accepted5']==context['accepted5'],'Observed historical context differs')
    modules,records,source_map,import_audit=c.cpu_modules(context);records[irec['physical_path']]=irec
    state,profile=modules['full_state'],modules['e1_profile']
    import torch
    from omegaconf import OmegaConf
    c.require(torch.__version__.split('+')[0]=='2.4.0' and not torch.cuda.is_initialized(),'NativeTorch2.4 CPU only')
    config=context['config'];c.require(profile.validate_profile(OmegaConf.create(config))==config,'Actual fixed pureOPSD profile differs')
    helper,hrec=c.exact_module(a.math_helper,c.MATH_HELPER,'final200_unchanged_storage_math');records[hrec['physical_path']]=hrec
    paths=c.verified_members(inventory,False);records.update({rec['physical_path']:rec for rec in inventory['checkpoint_files'].values()})
    official_rec=context['manifest']['files']['model_manifest'];official,orec=c.read_json(official_rec['path'],c.MODEL,c.RUN);records[orec['physical_path']]=orec
    basepath=Path(config['actor_rollout_ref']['model']['path']);c.typed(config['baselines']['distillation']['teacher_path'],str(basepath),'Fixed teacher official path differs')
    for item in official['files']:
        _,rec=c.read_file(basepath/item['file'],item['sha256']);c.typed(rec['bytes'],item['bytes'],'Official asset size differs');records[rec['physical_path']]=rec
    legacy_pins={};base=helper.Base(basepath,official,legacy_pins);base_digest=base.named_parameter_digest()
    original_manifest=basepath/'VERIFIED-MODEL-MANIFEST.json'
    _,original_record=c.read_file(original_manifest,c.MODEL);records[str(original_manifest)]=original_record
    assets=helper.base_asset_inventory(basepath,{item['file'] for item in official['files']},original_manifest,legacy_pins)
    for name,pin in assets.items():
        _,rec=c.read_file(basepath/name,pin);records[rec['physical_path']]=rec
    hf=helper.Export(paths,set(base.headers)|{'lm_head.weight'});c.require(len(hf.shards)==3 and len(hf.headers)==435,'Complete HF200 export required')
    ranks=[];actor_configs=set()
    for rank,w in enumerate(context['workers']):
        report,rrec=c.read_json(paths[f'actor/training_state/rank_{rank}.json'],inventory['checkpoint_files'][f'actor/training_state/rank_{rank}.json']['sha256'],c.RUN)
        c.require(c.canonical(report)==c.canonical(inventory['rank_reports'][rank]),'Observed/original rank200 report differs');c.rank_report(report,rank,200,'FULL_STATE_SERIALIZED_OK',w['pid'])
        pt=paths[f'actor/training_state/rank_{rank}.pt'];raw=torch.load(pt,map_location='cpu',weights_only=False)
        c.require(state.state_digest(raw)==report['state_digest'],'Raw200 full-state content digest differs')
        for key,target in dict(version=3,algorithm='pure_opsd',model_state_dict_type='SHARDED_STATE_DICT',rank=rank,world_size=8,step=200,pid=w['pid']).items():c.typed(raw[key],target,'Raw200 rank header differs: '+key)
        c.object_ids(raw['object_ids']);c.require(raw['object_ids']==report['object_ids']==w['object_ids'],'Stored200 objects changed since signed5')
        recovery=raw['recovery_identity'];c.require(recovery['source_closure']==dict(version=1,git_source_commit=c.SOURCE,files=source_map) and recovery['worker_source_sha256']==records[str(c.REPO/'verl/workers/fsdp_workers.py')]['sha256'],'Raw200 recovery source closure differs')
        prior=context['prior']['ranks'][rank];c.typed(recovery['actor_config_digest'],prior['actor_config_digest'],'Actor config changed since signed5');actor_configs.add(recovery['actor_config_digest'])
        model=raw['model'];c.require(set(model)==set(base.headers)|{'lm_head.weight'},'Native200 complete435 model keys differ')
        layouts={};changes={};numel=0
        for key,tensor in model.items():
            layouts[key]=helper.layout(tensor,rank);local=tensor.to_local();target=base.local(key,rank);original='model.embed_tokens.weight' if key=='lm_head.weight' else key
            c.require(tuple(tensor.shape)==tuple(base.headers[original]['shape']) and hf.headers[key]['shape']==list(tensor.shape) and local.shape==target.shape,'Native200/HF/official geometry differs')
            hf.equal_slice(key,rank,local);changes[key]=int(torch.count_nonzero(local!=target))
            if key!='lm_head.weight':numel+=local.numel()
        c.require(numel==c.NUMEL//8 and torch.equal(model['lm_head.weight'].to_local(),model['model.embed_tokens.weight'].to_local()) and sum(v for key,v in changes.items() if key!='lm_head.weight')>0,'Native200 fullparameter/tied/delta evidence differs')
        opt=adam200(raw['optimizer'],helper,c,c.NUMEL//8);c.require(len(opt['entries'])==37,'Native200 all37 Adam entries required')
        c.require([(v['parameter_id'],v['shape'],v['dtype']) for v in opt['entries']]==[(v['parameter_id'],v['shape'],v['dtype']) for v in prior['optimizer_moments']],'Native200 Adam IDs/flat geometry changed since5')
        sched=scheduler200(raw['scheduler'],c);rng=helper.rng(raw['rng'],True)
        c.require([v['device'] for v in rng['cuda']]==[v['device'] for v in prior['rng_content']['cuda']],'Native200 CUDA RNG topology differs')
        teacher=teacher200(raw['teacher'],report['teacher'],base,assets,base_digest,helper,state,c)
        c.require(helper.teacher_report_exact(raw['teacher']['model_config'],w['report']['teacher']['model_config']) and raw['teacher']['execution_device']==w['report']['teacher']['execution_device'],'Teacher200 fixed config/device changed since5')
        ranks.append(dict(rank=rank,pid=raw['pid'],object_ids=raw['object_ids'],raw_saved_status=report['status'],state_sha256=inventory['checkpoint_files'][f'actor/training_state/rank_{rank}.pt']['sha256'],state_digest=report['state_digest'],actor_config_digest=recovery['actor_config_digest'],component_digests={k:state.state_digest(raw[k]) for k in ('model','optimizer','scheduler','rng','teacher')},model_dtensor_entries=435,model_hf_slice_exact=True,model_hf_slice_keys=sorted(model),model_layouts=layouts,unique_local_model_numel=numel,changed_elements_vs_original_fp32=changes,optimizer_state_entries=37,optimizer_step_values=opt['steps'],optimizer_moments=opt['entries'],scheduler=sched,rng_content=rng,teacher=teacher))
        del raw,model
    c.require(len(actor_configs)==1,'Native200 actor configurations differ by rank')
    drec=inventory['checkpoint_files']['driver/driver.json'];report,_=c.read_json(drec['physical_path'],drec['sha256'],c.RUN);driver=torch.load(paths['driver/driver.pt'],map_location='cpu',weights_only=False)
    c.require(c.canonical(report)==c.canonical(inventory['driver_report']) and report['digest']==state.state_digest(driver) and report['state_sha256']==inventory['checkpoint_files']['driver/driver.pt']['sha256'],'Raw200 driver/report digest differs')
    for key,target in dict(version=1,method='pure_opsd',global_step=200,actual_outer_updates=200,next_global_step=201,epoch=0,pid=context['manifest']['driver_pid'],uid=execution['uid'],job_id='3304',source_commit=c.SOURCE,config_sha256=c.CONFIG).items():c.typed(driver[key],target,'Raw200 driver identity differs: '+key)
    prior_rec=context['manifest']['files']['driver/driver.pt'];_,original=c.read_file(prior_rec['path'],prior_rec['sha256'],c.RUN);records[original['physical_path']]=original
    old=torch.load(original['physical_path'],map_location='cpu',weights_only=False);c.require(state.state_digest(old)==context['prior']['driver']['state_digest'],'Actual signed5 driver digest changed')
    sampler=helper.sampler_content(driver['sampler'],config,state,size=169028,cursor=102400)
    for key in ('order','generator'):
        c.require(state.state_digest(driver['sampler'][key])==state.state_digest(old['sampler'][key]),'Stored200 epoch0 sampler drift since5: '+key)
    rng=helper.rng(driver['rng'],False);algorithm,arec=c.read_json(paths['algorithm'],inventory['checkpoint_files']['algorithm']['sha256'],c.RUN)
    expected=dict(version=1,method='pure_opsd',global_step=200,distillation=config['baselines']['distillation'],resume_scope='same_live_objects_same_topology',outcome_rewards_train_policy=False,trajectory_archive=archive_progress200(algorithm,inventory,c))
    c.require(c.canonical(algorithm)==c.canonical(expected)==c.canonical(driver['algorithm']),'Raw200 driver/OPSD algorithm state differs')
    c.require(not torch.cuda.is_initialized(),'CPU inspection initialized CUDA')
    records[own['physical_path']]=own;common=c.read_file(Path(c.__file__),a.common_sha256)[1];records[common['physical_path']]=common
    c.recheck(records);c.source_files(c.REPO,context['closure'])
    return dict(version=1,scope=SCOPE,passed=True,source_commit=c.SOURCE,config_sha256=c.CONFIG,job_id='3304',run_root=str(c.RUN),checkpoint_step=200,inventory_sha256=a.inventory_sha256,inventory_path=irec['physical_path'],inspector_source_sha256=own['sha256'],common_source_sha256=common['sha256'],math_helper_sha256=c.MATH_HELPER,execution_identity=execution,cpu_runtime=dict(torch_version=torch.__version__,torch_file=torch.__file__),import_audit=import_audit,clock=c.clock(),original_raw_statuses=inventory['raw_statuses'],accepted5=context['accepted5'],ranks=ranks,driver=dict(pid=driver['pid'],raw_saved_status=report['status'],state_digest=report['digest'],sampler=sampler,rng_content=rng,algorithm_digest=state.state_digest(algorithm),trajectory_archive=algorithm['trajectory_archive']),base_named_parameter_digest=base_digest,hf_all_rank_slices_exact=True,hf_files={str(p):inventory['checkpoint_files'][role]['sha256'] for role,p in paths.items() if role.startswith('actor/') and '/training_state/' not in role},verified_input_records=records,cuda_is_initialized=False,seconds=time.monotonic()-started,limitations=['Stored200 contents only, no training-object or new-process restore','Historical signed5 is a continuity anchor, not the200 manifest/receipt','Archive last progress only; all200 raw archives have their separate full-content consumer','No logs/terminal/resource release or effect qualification','Fixed teacher raw state stores digest and history buffers, not all parameter arrays'])


def main():
    p=argparse.ArgumentParser()
    for name in ('common-sha256','inventory','inventory-sha256','accepted5-receipt-sha256','math-helper','execution-job-id','execution-step-id','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();c=load_common(a.common_sha256);value=inspect(a,c);record=c.write_exclusive(a.output,value)
    print(c.canonical(dict(scope=SCOPE,passed=True,output=record,stored_training_job_id='3304',execution_identity=value['execution_identity'])).decode())


if __name__=='__main__':main()
