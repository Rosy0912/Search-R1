"""Terminal-only five-checkpoint validation; no native DTensor CPU reload claim."""
import hashlib,json,math,os,time
from pathlib import Path
SAVE_OUTERS=(5,50,100,150,200)
def need(ok,message):
    if not ok:raise ValueError(message)
def integer(value,want,message):need(type(value)is int and value==want,message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def regular(path):
    p=Path(path);need(p.is_file()and not p.is_symlink()and p.resolve()==p,'Canonical regular file required: '+str(p));return p
def read(path):
    from final_json import strict_json
    return strict_json(regular(path).read_bytes())
def horizon(config):
    # Full production validator plus explicit typed checks: bool/float cannot stand for counts.
    t=config['trainer'];o=config['actor_rollout_ref']['actor']['optim']
    for k,n in [('total_training_steps',201),('acceptance_step',5),('save_freq',50)]:integer(t.get(k),n,'Invalid horizon/save schedule '+k)
    integer(o.get('num_warmup_steps'),57,'Warmup must be explicit57; never zero')
    from search_r1.baselines.fresh200_contract import validate_fresh_config
    need(validate_fresh_config(config)==(200,57),'Production horizon changed')
def checkpoint_prefixes(attempts):
    result={}
    for a in attempts:
        if a['accepted_complete_outer'] and a['candidate_outer'] in SAVE_OUTERS:
            need(a['candidate_outer'] not in result,'Duplicate milestone acceptance')
            result[a['candidate_outer']]=a
    need(set(result)==set(SAVE_OUTERS),'Missing accepted milestone');return result
def inventories(run,step):
    run=Path(run);groups={'native_state':run/'checkpoints/fresh200_state'/('outer_%d_COMPLETE'%step),'hf_export':run/'checkpoints/actor'/('global_step_%d'%step),'algorithm':run/'checkpoints/algorithm'/('global_step_%d'%step)}
    files=[]
    for role,root in groups.items():
        need(root.is_dir()and not root.is_symlink()and root.resolve()==root,'Missing/canonical milestone directory '+str(root))
        local=[]
        for p in sorted(root.rglob('*')):
            need(not p.is_symlink()and p.resolve()==p,'Milestone member symlink')
            if p.is_file():
                st=p.stat();digest=sha(p);after=p.stat()
                need((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Checkpoint changed during hash')
                local.append(dict(role=role,path=str(p),relative=str(p.relative_to(root)),bytes=st.st_size,sha256=digest))
        need(local,'Empty milestone directory '+role);files.extend(local)
    export=groups['hf_export'];need((export/'config.json').is_file()and(export/'tokenizer_config.json').is_file()and(export/'tokenizer.json').is_file(),'Incomplete actor/tokenizer export')
    need(any(export.glob('*.safetensors')),'Missing actor safetensors')
    index=export/'model.safetensors.index.json'
    if index.exists():
        weights=read(index).get('weight_map');need(isinstance(weights,dict)and weights,'Incomplete weight index')
        for name in set(weights.values()):need(type(name)is str and Path(name).name==name and regular(export/name).stat().st_size>0,'Missing indexed checkpoint shard')
    return groups,files

def metadata(step,status,algorithm,reload,ranks,driver,attempt,pid,source,state):
    from search_r1.baselines.full_state import state_digest
    from search_r1.baselines.fresh200_contract import outer_contract,validate_rank_updates
    integer(step,step,'Step must be int');need(step in SAVE_OUTERS,'Unknown milestone')
    for k in ('actual_outer','completed_outer_updates'):integer(status.get(k),step,'Saved accepted counter differs')
    need(status.get('status')=='COMPLETE'and status.get('progress')==attempt['progress'],'Saved milestone progress differs')
    integer(status.get('attempt_id'),attempt['attempt_id'],'Saved attempt differs')
    need(algorithm.get('method')=='bapo'and algorithm.get('algorithm_source_commit')==source,'Algorithm/source identity differs')
    for k in ('global_step','actual_outer_updates'):integer(algorithm.get(k),step,'Algorithm accepted counter differs')
    need(algorithm.get('resume_supported')is False and algorithm.get('algorithm_json_resume_supported')is False,'No legacy resume evidence allowed')
    need(driver.get('source_commit')==source and driver.get('protocol')=='BAPO_FRESH200_V1'and driver.get('status')=='COMPLETE','Driver saved source/status differs')
    for k,n in [('global_step',step),('next_global_step',step+1),('actual_outer_updates',step),('driver_pid',pid),('attempt_id',attempt['attempt_id']),('attempted_outer',step),('previous_completed_outer',step-1)]:integer(driver.get(k),n,'Driver saved '+k+' differs')
    need(driver.get('infrastructure_fault')is False and driver.get('progress')==attempt['progress']and driver.get('bapo')==algorithm.get('bapo'),'Driver/algorithm/progress differs')
    need(driver.get('outer_contract')==outer_contract(step)and driver.get('rank_write_ledger')==attempt['rank_write_ledger'],'Driver contract/actualwrites differs')
    need(validate_rank_updates(driver['rank_write_ledger'],step),'Saved update cannot advance scheduler')
    need(reload.get('status')==('EXACT_DRIVER_RELOAD_OK'if step==5 else'DRIVER_SERIALIZED_OK')and reload.get('state_digest')==state_digest(driver),'Driver serialized/reload digest differs')
    integer(reload.get('step'),step,'Driver reload step differs');integer(reload.get('pid'),pid,'Driver reload PID differs')
    need(reload.get('path')==str(state/'driver.pt')and reload.get('sha256')==sha(state/'driver.pt'),'Driver reload raw file differs')
    need(len(ranks)==8,'Eight rank reports required');pids=[]
    events={e['rank']:e for e in attempt['rank_write_ledger']}
    for rank,r in enumerate(ranks):
        for k,n in [('rank',rank),('world_size',8),('step',step),('scheduler_last_epoch',step),('native_state_version',3),('attempt_id',attempt['attempt_id'])]:integer(r.get(k),n,'Native rank '+k+' differs')
        need(type(r.get('pid'))is int and r['pid']>0,'Native PID required');pids.append(r['pid'])
        need(r.get('status')==('EXACT_FULL_STATE_RELOAD_OK'if step==5 else'FULL_STATE_SERIALIZED_OK')and r.get('roundtrip')is(step==5),'Only5 actual native roundtrip; other saves serialized')
        need(r.get('model_state_dict_type')=='SHARDED_STATE_DICT'and type(r.get('model_dtensor_entries'))is int and r['model_dtensor_entries']>0 and type(r.get('optimizer_state_entries'))is int and r['optimizer_state_entries']>0,'Real native nonempty model/Adam required')
        need(r.get('latest_rank_event')==events[rank]and r.get('reloaded_state_digest')==r.get('state_digest')and r.get('generation_rng_before_digest')==r.get('generation_rng_after_digest'),'Native event/state/RNG differs')
        path=regular(state/'ranks'/('rank_%d.pt'%rank));need(r.get('path')==str(path)and r.get('sha256')==sha(path),'Native full file hash differs')
    need(len(set(pids))==8,'Distinct worker PIDs required')
    return pids

def collect_milestones(run,attempts,pid,source,hf_steps,hf_fn,base,config):
    import torch
    horizon(config);need(set(hf_steps).issubset(SAVE_OUTERS)and 200 in hf_steps,'Final200 CPU HF required; middle optional')
    need(not torch.cuda.is_initialized(),'No CUDA/native reload in collector')
    prefixes=checkpoint_prefixes(attempts);out=[];prior_pids=None
    for step in SAVE_OUTERS:
        wall=time.perf_counter();cpu=time.process_time();groups,files=inventories(run,step);state=groups['native_state']
        status=read(state/'status.json');algorithm=read(groups['algorithm']/'state.json');reload=read(state/'driver-reload.json');ranks=[read(state/'ranks'/('rank_%d.json'%r))for r in range(8)]
        driver=torch.load(regular(state/'driver.pt'),map_location='cpu',weights_only=False)
        need(read(state/'all-rank-write-ledger.json')==prefixes[step]['rank_write_ledger'],'Native saved allrank ledger differs')
        pids=metadata(step,status,algorithm,reload,ranks,driver,prefixes[step],pid,source,state)
        need(prior_pids is None or prior_pids==pids,'Native same-rank PID changed across milestones');prior_pids=pids
        entry=dict(actual_outer=step,actor_hf=str(groups['hf_export']),algorithm=str(groups['algorithm']),native_state=str(state),files=files,progress=status['progress'],ranks=ranks,driver_reload=reload,expected_opportunities=20 if step<50 else 10,native_gpu_reload_verified=step==5,scope='Saved actual milestone, not synthetic native CPU processgroup reload')
        if step in hf_steps:entry['cpu_hf']=hf_fn(base,groups['hf_export'])
        else:entry['cpu_hf']=dict(status='NOT_EXECUTED_OPTIONAL_MIDDLE_MILESTONE',native_acceptance=False)
        # HF is read-only; verify the entire inventory remains identical afterward.
        need(inventories(run,step)[1]==files,'Milestone changed during CPU validation')
        entry['cost']=dict(wall_seconds=time.perf_counter()-wall,driver_cpu_seconds=time.process_time()-cpu,scope='CPU collector hashing/loading/HF; excludes training GPU cost');out.append(entry)
    return out

def require_all_milestone_hf(entries):
    """Evidence coverage check only; never independent/native production acceptance."""
    need([e.get('actual_outer')for e in entries]==list(SAVE_OUTERS),'All five milestone HF entries required')
    for e in entries:
        proof=e.get('cpu_hf',{});hf=proof.get('hf',{});token=proof.get('tokenizer',{});delta=proof.get('parameter_delta',{})
        need(hf.get('status')=='REAL_CPU_HF_FINITE_FORWARD'and hf.get('device')=='cpu'and type(hf.get('parameters'))is int and hf['parameters']==3085938688 and hf.get('nonfinite_parameters')==[]and hf.get('tied_input_output_storage')is True and hf.get('forward',{}).get('logits_finite')is True,'Every milestone requires actual full CPU HF/tied/finite evidence')
        flags=('full_vocabulary_equal','bos_equal','eos_equal','pad_equal','special_tokens_equal','added_token_metadata_equal','chat_template_equal','real_chat_template_tokenization_equal')
        need(all(token.get(k)is True for k in flags),'Every actual tokenizer equality flag required')
        forward=hf.get('forward',{});shape=forward.get('logits_shape');info=hf.get('loading_info',{})
        need(forward.get('device')=='cpu'and isinstance(shape,list)and len(shape)==3 and all(type(n)is int and n>0 for n in shape)and shape[0]==1 and shape[2]==151936 and type(forward.get('next_token_id'))is int and 0<=forward['next_token_id']<151936,'Strict finite CPU forward shape/token evidence required')
        need(all(info.get(k)==[]for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),'Complete HF loading diagnostics required')
        need(delta.get('changed')is True and delta.get('comparison')=='base_cast_to_checkpoint_dtype_then_float64'and type(delta.get('max_abs_delta'))in(int,float)and math.isfinite(delta['max_abs_delta'])and delta['max_abs_delta']>0,'Every milestone dtype-aligned actual weight change required')
    return True
