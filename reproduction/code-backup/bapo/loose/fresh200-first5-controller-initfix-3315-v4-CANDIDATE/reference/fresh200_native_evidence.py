"""Diagnostic-only raw tensor collection and independent numerical replay.

No PASS status is generated here. Formal training must independently admit replayed
native evidence; synthetic CPU fixtures cannot establish CUDA provenance.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


_FILE_HASH_CACHE={}
_INITIAL_PROOF_CACHE={}
_ALIGNMENT_CACHE={}
_ROLLOUT_CACHE={}


def file_signature(path):
    st=Path(path).stat()
    return (str(Path(path).resolve()),st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)


def file_sha(path):
    key=file_signature(path)
    if key in _FILE_HASH_CACHE:return _FILE_HASH_CACHE[key]
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1048576),b''):h.update(block)
    value=h.hexdigest();_FILE_HASH_CACHE[key]=value
    return value


def require(ok,message):
    if not ok: raise ValueError('Fresh200 native evidence: '+message)


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def native_process(rank,world):
    import torch
    require(torch.cuda.is_available(), 'native CUDA device required')
    job=os.environ.get('SLURM_JOB_ID');require(job and job.isdigit(),'actual allocated Slurm job required')
    stat=Path('/proc/self/stat').read_text();fields=stat[stat.rfind(')')+2:].split()
    allocation=subprocess.check_output(['scontrol','show','job','--oneliner',job],text=True)
    nvidia=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,pci.bus_id','--format=csv,noheader'],text=True)
    import csv
    inventory=[list(map(str.strip,row)) for row in csv.reader(nvidia.splitlines())]
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');logical=torch.cuda.current_device()
    require(1<=len(visible)<=8 and 0<=logical<len(visible),'explicit Ray scheduler-visible GPU mapping required')
    token=visible[logical].strip();matches=[row for row in inventory if row[0]==token or row[1]==token]
    require(len(matches)==1,'actual scheduler-visible GPU not uniquely bound to NVML inventory')
    leases=json.loads(os.environ.get('SEARCH_R1_GPU_LEASES','[]'))
    require(len(leases)==8 and len({x.get('uuid') for x in leases})==8 and matches[0][1] in {x.get('uuid') for x in leases},'actual inherited eight-UUID lease inventory missing/mismatch')
    guard=json.loads(os.environ.get('SEARCH_R1_GPU_RESOURCE_GUARD','{}'))
    guard_pid=int(guard.get('pid',0));require(guard_pid>1,'actual resource guardian required')
    guard_stat=Path(f'/proc/{guard_pid}/stat').read_text()
    guard_cmd=Path(f'/proc/{guard_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    root=Path(__file__).resolve().parents[2]
    return dict(pid=os.getpid(), rank=rank,world_size=world, hostname=__import__('socket').gethostname(),
        job_id=job,run_root=os.environ.get('BAPO_RUN_ID'),cuda_runtime=torch.version.cuda,
        torch_version=str(torch.__version__),device=str(torch.cuda.current_device()),gpu_name=torch.cuda.get_device_name(),
        gpu_uuid=matches[0][1],gpu_pci_bus_id=matches[0][3],cuda_visible_devices=visible,
        proc_start_ticks=fields[19],proc_stat_raw=stat,gpu_leases=leases,resource_guard=guard,resource_guard_stat_raw=guard_stat,resource_guard_cmdline=guard_cmd,proc_cgroup_raw=Path('/proc/self/cgroup').read_text(),
        proc_maps_raw=Path('/proc/self/maps').read_text(),allocation_raw=allocation,nvidia_smi_raw=nvidia,
        source_head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip())


def verify_native_process(process,source):
    require(process.get('source_head')==source,'actual startup source HEAD mismatch')
    stat=process.get('proc_stat_raw','');require(stat.startswith(str(process.get('pid'))+' ('),'native PID raw stat mismatch')
    fields=stat[stat.rfind(')')+2:].split()
    require(len(fields)>19 and fields[19]==process.get('proc_start_ticks'),'native PID birth mismatch')
    maps=process.get('proc_maps_raw','')
    require('libcuda' in maps and ('libtorch_cuda' in maps or 'libcudart' in maps),'native CUDA runtime mappings missing')
    leases=process.get('gpu_leases',[]);guard=process.get('resource_guard',{});guard_stat=process.get('resource_guard_stat_raw','')
    require(len(leases)==8 and len({x.get('uuid') for x in leases})==8 and process.get('gpu_uuid') in {x.get('uuid') for x in leases},'native UUID not held in eight-card guardian lease inventory')
    guard_fields=guard_stat[guard_stat.rfind(')')+2:].split()
    require(guard_stat.startswith(str(guard.get('pid'))+' (') and len(guard_fields)>19 and guard_fields[19]==str(guard.get('start_ticks'))
            and guard.get('run')==process.get('run_root') and 'resource_watch.py' in process.get('resource_guard_cmdline','')
            and process.get('run_root') in process.get('resource_guard_cmdline',''),'actual guardian PID birth/run binding invalid')
    job=process.get('job_id');allocation=process.get('allocation_raw','')
    require('JobId='+str(job)+' ' in allocation and 'JobState=RUNNING' in allocation,'actual running Slurm allocation mismatch')
    require('gres/gpu=8' in allocation or 'gres/gpu:' in allocation and ':8' in allocation,'actual eight-GPU allocation not established')
    require(str(job) in process.get('proc_cgroup_raw',''),'native process not bound to actual Slurm job cgroup')
    import csv
    inventory=[list(map(str.strip,row)) for row in csv.reader(process.get('nvidia_smi_raw','').splitlines())]
    visible=process.get('cuda_visible_devices',[]);logical=int(process.get('device','-1'))
    require(1<=len(visible)<=8 and 0<=logical<len(visible),'native logical GPU mapping invalid')
    token=visible[logical];matches=[row for row in inventory if row[0]==token or row[1]==token]
    require(len(matches)==1 and matches[0][1]==process.get('gpu_uuid') and matches[0][3]==process.get('gpu_pci_bus_id'),'native scheduler-visible UUID/PCI mapping mismatch')
    require(process.get('gpu_uuid') in process.get('nvidia_smi_raw','') and len(process.get('nvidia_smi_raw','').splitlines())>=8,'actual GPU UUID inventory mismatch')


def replay_matrices(payload,expected_rows,kl_log_bound=30.):
    """Independent float64 sequence weighting and k3/tail recomputation."""
    import torch
    require(payload.get('denominator_role')=='actual_native_stochastic_sampling_policy' and payload.get('hf_probability_role')=='unchanged_hf_recomputation_diagnostic', 'explicit behavior/HF diagnostic roles missing')
    names=('old_log_probs','current_log_probs','advantages','mask','sampling_log_probs')
    require(all(isinstance(payload.get(n),torch.Tensor) for n in names), 'raw probability/advantage/mask matrices required')
    old,current,adv,mask,sampled=[payload[n].double() for n in names]
    require(old.ndim==2 and old.shape[0]==expected_rows and all(x.shape==old.shape for x in (current,adv,mask,sampled)), 'raw matrix scope/shape mismatch')
    require(bool(((mask==0)|(mask==1)).all()), 'mask must be binary')
    valid=mask.bool()
    require(bool(valid.any()) and all(bool(torch.isfinite(x[valid]).all()) for x in (old,current,adv,sampled)), 'nonfinite/empty native policy samples')
    require(bool((old[valid]<=0).all()) and bool((current[valid]<=0).all()) and bool((sampled[valid]<=0).all()), 'invalid probability logs')
    require(bool(torch.equal(old[valid],sampled[valid])), 'actual captured behavior/optimizer denominator transport mismatch')
    hf=payload.get('frozen_hf_log_probs')
    require(isinstance(hf,torch.Tensor) and hf.is_floating_point() and hf.shape==old.shape and bool(torch.isfinite(hf[valid]).all()), 'frozen HF diagnostic matrix missing/nonfinite')
    weights=mask/mask.sum(-1,keepdim=True).clamp_min(1)
    if 'weights' in payload:
        require(torch.equal(payload['weights'].double(),weights), 'raw monitor weights differ from sequence weighting')
    delta=torch.where(valid,current-old,0.)
    require(bool(torch.isfinite(delta[valid]).all()) and math.isfinite(kl_log_bound) and 0<kl_log_bound<=30 and bool((delta[valid]<=kl_log_bound).all()), 'nonfinite/overflow old-policy delta')
    negative=valid&(adv<0);tail=negative&(delta>math.log(3.))
    mass=float(weights.sum());numerator=float(((torch.expm1(delta)-delta)*weights).sum())
    negative_mass=float(weights[negative].sum());tail_mass=float(weights[tail].sum())
    return dict(effective_weight_mass=mass,weighted_k3_numerator=numerator,
        negative_effective_weight_mass=negative_mass,tail_effective_weight_mass=tail_mass,
        effective_token_count=int(valid.sum()),sampled_old_policy_kl=numerator/mass,
        negative_tail_weight_fraction=tail_mass/negative_mass if negative_mass else 0.)


def replay_context(payload):
    """Replay actual native token journals without importing the training driver."""
    from search_r1.llm_agent.sampling_trace import audit_row
    import torch
    tensors=payload['context_tensors'];journals=payload['journals'];ids=payload['row_ids'];snapshot=payload['policy_snapshot_id']
    width=tensors['prompts'].shape[1]
    require(len(ids)==len(journals)==tensors['input_ids'].shape[0] and len(set(ids))==len(ids),'native row identity duplication')
    for i,turns in enumerate(journals):
        row=tensors['input_ids'][i].tolist();att=tensors['attention_mask'][i].tolist()
        require(att[:width]==sorted(att[:width]) and att[width:]==sorted(att[width:],reverse=True) and all(a in(0,1) for a in att),'actual padding topology')
        require(tensors['prompts'][i].tolist()==row[:width] and tensors['responses'][i].tolist()==row[width:],'actual token context mismatch')
        model=payload['mask'][i].tolist();lp=payload['sampling_log_probs'][i].tolist()
        require(len(model)==len(lp)==len(att)-width and not any(m and not a for m,a in zip(model,att[width:])),'raw model mask covers padding or wrong response width')
        require(turns and all(turn.get('effective_stop_provenance',{}).get('probability_role')=='stochastic_sampling_policy' for turn in turns),'actual stochastic native effective sampler/stop proof missing or greedy evaluation journal supplied')
        require(model==tensors['info_mask'][i,width:].tolist(), 'actual model/observation mask mismatch')
        cumulative=0;positions=[]
        for a in att: cumulative+=a;positions.append((cumulative-1)*a)
        require(tensors['position_ids'][i].tolist()==positions,'actual position IDs mismatch')
        audit_row([v for v,a in zip(row[:width],att[:width]) if a], [v for v,a in zip(row[width:],att[width:]) if a],
            [m for m,a in zip(model,att[width:]) if a], [p for p,a in zip(lp,att[width:]) if a],turns,ids[i],snapshot)
    return snapshot


def verify_independent_review(review,safety_config,expected_bindings):
    """Read the separate review, anchored by the task's fixed SHA and nonce.

    Evidence producers never create a PASS review or choose its trusted hash.
    """
    path=Path(review.get('path',''))
    require(path.is_absolute() and path.is_file(),'independent native review required')
    actual_sha=file_sha(path)
    require(actual_sha==review.get('sha256')==safety_config.get('evidence_review_sha256'),
            'independent review does not match task-anchored report SHA')
    value=json.loads(path.read_text())
    nonce=safety_config.get('evidence_review_nonce')
    require(isinstance(nonce,str) and len(nonce)>=32 and value.get('receipt_nonce')==nonce,
            'independent native review receipt nonce mismatch')
    require(value.get('decision')=='PASS_BAPO_NATIVE_REPLAY' and value.get('bindings')==expected_bindings,
            'independent review does not bind actual native replay/process/source evidence')
    require(value.get('review_origin')=='INDEPENDENT_TASK_REVIEWER' and value.get('reviewer_task_id')
            and value.get('producer_task_id') and value['reviewer_task_id']!=value['producer_task_id'],
            'independent reviewer identity missing or same producer')
    return value


def verify_native_sample(sample,identity,role,layout_run_ids):
    import torch
    scope=sample['scope'];mini=sample['global_mini_batch_size'];rows=320 if scope=='full_frozen_rollout' else mini//8
    shards=sample.get('native_shards',[])
    require(len(shards)==8 and {s.get('rank') for s in shards}==set(range(8)), 'all eight native probability shards required')
    totals=[];proof=[]
    for entry in sorted(shards,key=lambda s:s['rank']):
        path=Path(entry.get('path',''))
        require(path.is_absolute() and path.is_file() and file_sha(path)==entry.get('sha256'),'native tensor shard missing/changed')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        require(payload.get('schema')=='BAPO_NATIVE_DIAGNOSTIC_BEHAVIOR_V2' and payload.get('identity')==identity
                and payload.get('role')==role and payload.get('scope')==scope
                and payload.get('global_mini_batch_size')==mini and payload.get('phase')==sample.get('phase')
                and payload.get('phase') in ('pre_adam','post_adam')
                and payload.get('diagnostic_write_index')==sample.get('diagnostic_write_index')
                and payload.get('completed_diagnostic_writes')==sample['diagnostic_write_index']+(sample['phase']=='post_adam')
                and payload.get('actual_optimizer_lr')==[identity['optimizer_contract']['diagnostic_probe_lr']]
                and payload.get('scheduler_cached_last_lr')==[identity['optimizer_contract']['initial_lr']],'native sample identity mismatch')
        shared_inputs=payload.get('shared_input_bindings',{})
        require(bind_shared_inputs(shared_inputs.get('manifest_path',''))==shared_inputs,'native actual data split/file identity mismatch')
        process=payload.get('native_process',{})
        require(process.get('rank')==entry['rank'] and process.get('world_size')==8
                and type(process.get('pid')) is int and process['pid']>0
                and process.get('cuda_runtime') and process.get('gpu_uuid') and process.get('job_id')
                and process.get('proc_start_ticks') and process.get('run_root')==layout_run_ids[mini],
                'native process/rank/GPU/run provenance incomplete')
        verify_native_process(process,identity['source_commit'])
        ledger_entry=entry.get('diagnostic_ledger',{})
        ledger_path=Path(ledger_entry.get('path',''))
        require(ledger_path.is_absolute() and ledger_path.is_file() and file_sha(ledger_path)==ledger_entry.get('sha256'),'actual rank diagnostic write ledger missing/changed')
        ledger=json.loads(ledger_path.read_text());expected=2560//mini
        validate_diagnostic_ledger(ledger,mini,role,entry['rank'],process)
        actual_events=ledger['events']
        bound=[e for e in actual_events if e.get('path')==str(path)]
        require(len(bound)==1 and bound[0].get('sha256')==entry['sha256'] and bound[0].get('diagnostic_write_index')==sample['diagnostic_write_index']
                and bound[0].get('phase')==sample['phase'] and bound[0].get('scope')==scope,'matrix sample not bound to actual rank write ledger')
        write=ledger['write_ledger'][sample['diagnostic_write_index']]
        require(write.get('actual_adam_write') is True and write.get('finite_loss_gradient_parameters_adam') is True
                and math.isfinite(write.get('grad_norm',float('nan'))) and write.get('lr')==identity['optimizer_contract']['diagnostic_probe_lr']
                and write.get('scheduler_cached_last_lr')==[identity['optimizer_contract']['initial_lr']] and str(path) in write.get('pre_paths' if sample['phase']=='pre_adam' else 'post_paths',[]),'native write transaction evidence incomplete')
        require(payload.get('frozen_rollout_tensor_sha256')==ledger.get('frozen_rollout_tensor_sha256'),'native frozen rollout identity changed')
        frozen_events=[e for e in actual_events if e.get('scope')=='full_frozen_rollout' and e.get('phase')=='pre_adam' and e.get('diagnostic_write_index')==0]
        require(len(frozen_events)==1,'unique full frozen initial rollout required')
        baseline=Path(frozen_events[0]['path']);require(file_sha(baseline)==frozen_events[0]['sha256'],'initial frozen rollout bytes changed')
        baseline_key=file_signature(baseline)
        if baseline_key not in _ROLLOUT_CACHE:
            base=torch.load(baseline,map_location='cpu',weights_only=True)
            combined=dict(base['context_tensors'],old_log_probs=base['old_log_probs'],frozen_hf_log_probs=base['frozen_hf_log_probs'],advantages=base['advantages'],loss_mask=base['mask'],sampling_log_probs=base['sampling_log_probs'])
            full_hashes={name:tensor_digest(tensor) for name,tensor in combined.items()}
            require(full_hashes==ledger['frozen_rollout_tensor_sha256'],'actual initial rollout tensor digest mismatch')
            subsets={i:{name:tensor_digest(tensor[i*(mini//8):(i+1)*(mini//8)]) for name,tensor in combined.items()} for i in range(expected)}
            _ROLLOUT_CACHE[baseline_key]=dict(full=full_hashes,minis=subsets)
            del base,combined
        combined=dict(payload['context_tensors'],old_log_probs=payload['old_log_probs'],frozen_hf_log_probs=payload['frozen_hf_log_probs'],advantages=payload['advantages'],loss_mask=payload['mask'],sampling_log_probs=payload['sampling_log_probs'])
        actual_hashes={name:tensor_digest(tensor) for name,tensor in combined.items()}
        expected_hashes=_ROLLOUT_CACHE[baseline_key]['full'] if scope=='full_frozen_rollout' else _ROLLOUT_CACHE[baseline_key]['minis'][sample['diagnostic_write_index']]
        require(actual_hashes==expected_hashes,'actual rollout or current mini changed across diagnostic writes')
        initial=payload.get('initial_state',{})
        require(initial.get('official_model_manifest_sha256')=='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
                and initial.get('scheduler_last_epoch')==0 and initial.get('optimizer_state_entries')==0
                and isinstance(initial.get('model_tensor_sha256'),dict) and initial['model_tensor_sha256'],
                'actual fresh model/empty Adam/initial scheduler proof missing')
        alignment=initial.get('official_alignment',{});alignment_path=Path(alignment.get('path',''))
        require(alignment_path.is_absolute() and alignment_path.is_file() and file_sha(alignment_path)==alignment.get('sha256'),'actual native official-alignment tensors missing/changed')
        alignment_key=(file_signature(alignment_path),alignment['sha256'])
        if alignment_key not in _ALIGNMENT_CACHE:
            alignment_payload=torch.load(alignment_path,map_location='cpu',weights_only=True)
            actual_alignment=compare_official_parameters(alignment_payload['model_parameters'],alignment_payload['official_model_root'])
            _ALIGNMENT_CACHE[alignment_key]=actual_alignment
            del alignment_payload
        require(_ALIGNMENT_CACHE[alignment_key]==alignment.get('tensor_sha256'),'independent native/official initial tensor replay mismatch')
        initial_path=Path(initial.get('path',''))
        require(initial_path.is_absolute() and initial_path.is_file() and file_sha(initial_path)==initial.get('sha256'),'native initial state bytes missing/changed')
        initial_key=(file_signature(initial_path),initial['sha256'])
        if initial_key not in _INITIAL_PROOF_CACHE:
            initial_payload=torch.load(initial_path,map_location='cpu',weights_only=True)
            require(not initial_payload['optimizer']['state'] and initial_payload['scheduler']['last_epoch']==0,'initial native Adam/scheduler not fresh')
            groups=initial_payload['optimizer']['param_groups'];contract=identity['optimizer_contract']
            require(groups and all(list(g['betas'])==contract['betas'] and g['weight_decay']==contract['weight_decay']
                and g['lr']==contract['initial_lr'] and g.get('initial_lr')==contract['base_lr'] for g in groups), 'native initial Adam hyperparameters differ from actual production contract')
            _INITIAL_PROOF_CACHE[initial_key]=dict(native_process=initial_payload.get('native_process'),source_sha256=initial_payload.get('source_sha256'),
                model_tensor_sha256={name:tensor_digest(tensor) for name,tensor in initial_payload['model_parameters'].items()})
            del initial_payload
        initial_proof=_INITIAL_PROOF_CACHE[initial_key]
        require(initial_proof['native_process']==process and initial_proof['source_sha256']==identity['source_sha256'],'initial native process/source binding mismatch')
        require(initial_proof['model_tensor_sha256']==initial['model_tensor_sha256'],'initial native model tensor bytes differ')
        require(payload.get('source_sha256')==identity['source_sha256'], 'native runtime source bytes mismatch')
        replay_context(payload)
        totals.append(replay_matrices(payload,rows,kl_log_bound=identity['kl_log_bound']));proof.append(dict(path=str(path),sha256=entry['sha256'],rank=entry['rank'],initial_state=initial,native_process=process,diagnostic_ledger=ledger_entry,shared_input_bindings=shared_inputs))
    require(all({x['uuid'] for x in p['native_process']['gpu_leases']}=={q['native_process']['gpu_uuid'] for q in proof} for p in proof),'native all-rank UUID set differs from guardian leased hardware')
    require(len({p['native_process']['pid'] for p in proof})==8 and len({p['native_process']['job_id'] for p in proof})==1 and len({p['native_process']['hostname'] for p in proof})==1,'native all-rank process/allocation identity mismatch')
    require(len({p['native_process']['gpu_uuid'] for p in proof})==8,'native ranks do not cover eight distinct allocated GPU UUIDs')
    summed={key:math.fsum(x[key] for x in totals) for key in ('effective_weight_mass','weighted_k3_numerator','negative_effective_weight_mass','tail_effective_weight_mass','effective_token_count')}
    summed['sampled_old_policy_kl']=summed['weighted_k3_numerator']/summed['effective_weight_mass']
    summed['negative_tail_weight_fraction']=summed['tail_effective_weight_mass']/summed['negative_effective_weight_mass'] if summed['negative_effective_weight_mass'] else 0.
    for field in ('sampled_old_policy_kl','negative_tail_weight_fraction'):
        require(math.isclose(sample.get(field,float('nan')),summed[field],rel_tol=1e-10,abs_tol=1e-12),'claimed monitor differs from independent raw recomputation')
    return summed,proof


def diagnostic_identity(safety):
    from .fresh200_contract import critical_source_hashes, PROTOCOL
    return dict(protocol=PROTOCOL,algorithm='bapo',objective='sequence_mean',
        model_repo='Qwen/Qwen2.5-3B-Instruct',model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
        data_manifest_sha256='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',
        source_commit=safety['source_commit'],source_sha256=critical_source_hashes(),dual_clip=3,
        kl_log_bound=float(safety['kl_log_bound']),target_old_policy_kl=float(safety['target_old_policy_kl']),
        monitor_scopes=['full_frozen_rollout','full_current_minibatch'],trajectory_count=2560,world_size=8,
        optimizer_contract=dict(base_lr=1e-6,initial_lr=1e-6/57,diagnostic_probe_lr=1e-6,betas=[.9,.999],weight_decay=.01,grad_clip=1.,ppo_epochs=1,clip_ratio=.2))


def run_diagnostic_update(worker,data,mini,role,directory):
    """Bounded native diagnostics, formal outer=0/scheduler=0, no PASS receipts."""
    import torch
    import torch.distributed as dist
    from .policy_safety import consensus,agree_document,dual_clip_surrogate,finite_inputs
    config=worker.config.actor;cap=config.get('diagnostic_max_writes');admission_error=None
    try:
        require(config.get('fresh200_diagnostic_only') is True, 'explicit diagnostic-only mode required')
        require(role in ('discovery','confirmation') and mini in (128,256), 'diagnostic role/layout invalid')
        expected=2560//mini
        require(config.get('diagnostic_probe_lr')==1e-6,'admitting native diagnostics require explicit production peak LR')
        require(type(cap) is int and cap in (1,expected) and getattr(worker,'_diagnostic_write_count',0)==0, 'diagnostic write cap invalid/exceeded')
        require(worker.world_size==8 and len(data)==320 and config.ppo_micro_batch_size==2,'native diagnostic requires actual eight-rank/fixed 2560-row input')
        require(worker.actor_lr_scheduler.last_epoch==0 and not worker.actor_optimizer.state,'fresh diagnostic Adam/scheduler required')
        safety=config.policy_safety;identity=diagnostic_identity(safety)
        shared_inputs=data.meta_info.get('fresh200_shared_input_bindings',{})
        require(bind_shared_inputs(shared_inputs.get('manifest_path',''))==shared_inputs,'actual diagnostic input bytes/rows do not match bound driver selectors')
        initial=getattr(worker,'_fresh200_initial_state',None)
        require(initial is not None and initial['optimizer_state_entries']==0 and initial['scheduler_last_epoch']==0, 'startup fresh native state capture missing')
        require({name:tensor_digest(p) for name,p in worker.actor_module_fsdp.named_parameters()}==initial['model_tensor_sha256'],'live diagnostic model differs from actual startup snapshot')
        process=worker._fresh200_native_process
        verify_native_process(process,identity['source_commit'])
        root=Path(directory);root.mkdir(parents=True,exist_ok=True)
        require(root.is_absolute() and Path(process['run_root']).resolve() in root.resolve().parents, 'owned diagnostic directory required')
        from .fresh200_contract import verify_sampling_policy_binding
        verify_sampling_policy_binding(data,safety)
    except Exception as error: admission_error=str(error)
    require(consensus(admission_error is None),'collective diagnostic configuration/fresh-state/context admission rejected: '+str(admission_error))
    require(agree_document(dict(identity=identity,role=role,global_mini=mini,cap=cap)),'diagnostic identity differs across ranks')
    # Explicit diagnostic peak probe; scheduler stays fresh/counter0. Its cached
    # last LR remains 1/57 and must never be reported as the LR actually used.
    for group in worker.actor_optimizer.param_groups:group['lr']=float(config.diagnostic_probe_lr)
    batch=data.batch;events=[];ledger=[];writes=0;stop_reason=None
    frozen_names=('old_log_probs','frozen_hf_log_probs','advantages','loss_mask','sampling_log_probs','prompts','responses','input_ids','attention_mask','info_mask','position_ids')
    frozen_identity={name:tensor_digest(batch[name]) for name in frozen_names}
    def capture(scope,part,phase,index):
        rows=len(part);training=worker.actor_module_fsdp.training;worker.actor_module_fsdp.eval()
        try:
            with torch.no_grad():
                current=torch.cat([worker.actor._forward_micro_batch(micro.cuda(),data.meta_info['temperature'])[1].cpu() for micro in part.split(2)])
        except Exception as error:
            try:(root/f'rank_{worker.rank}_{index}_{scope}_{phase}.forward-fault.log').write_text(repr(error))
            except OSError:pass
            raise
        finally:worker.actor_module_fsdp.train(training)
        start=index*(mini//8) if scope=='full_current_minibatch' else 0
        idx=list(range(start,start+rows))
        tensors={name:part[name].detach().cpu() for name in ('prompts','responses','input_ids','attention_mask','info_mask','position_ids')}
        payload=dict(schema='BAPO_NATIVE_DIAGNOSTIC_BEHAVIOR_V2',identity=identity,role=role,scope=scope,phase=phase,
            global_mini_batch_size=mini,diagnostic_write_index=index,completed_diagnostic_writes=writes,frozen_rollout_tensor_sha256=frozen_identity,
            native_process=process,initial_state=initial,source_sha256=identity['source_sha256'],shared_input_bindings=shared_inputs,
            actual_optimizer_lr=[g['lr'] for g in worker.actor_optimizer.param_groups],scheduler_cached_last_lr=worker.actor_lr_scheduler.get_last_lr(),
            policy_snapshot_id=data.meta_info['policy_snapshot_id'],
            row_ids=[str(data.non_tensor_batch['fresh200_row_id'][i]) for i in idx],
            journals=[data.non_tensor_batch['fresh200_turns'][i] for i in idx],context_tensors=tensors,
            old_log_probs=part['old_log_probs'].detach().cpu(),current_log_probs=current,
            frozen_hf_log_probs=part['frozen_hf_log_probs'].detach().cpu(),
            denominator_role='actual_native_stochastic_sampling_policy',
            hf_probability_role='unchanged_hf_recomputation_diagnostic',
            advantages=part['advantages'].detach().cpu(),mask=part['loss_mask'].detach().cpu(),
            sampling_log_probs=part['sampling_log_probs'].detach().cpu())
        numerical=None;capture_error=None;captured_sha=None
        path=root/f'rank_{worker.rank}_{mini}_{index}_{scope}_{phase}.pt'
        try:
            replay_context(payload); numerical=replay_matrices(payload,rows,kl_log_bound=float(safety.kl_log_bound))
            require(not path.exists(),'diagnostic evidence is immutable; refusing overwrite')
            torch.save(payload,path);captured_sha=file_sha(path)
        except Exception as error:
            capture_error=str(error)
            try:(root/f'rank_{worker.rank}_{index}_{scope}_{phase}.fault.log').write_text(capture_error)
            except OSError:pass
        require(consensus(capture_error is None),'collective native capture/replay rejected; raw fault retained')
        vector=torch.tensor([numerical[k] for k in ('effective_weight_mass','weighted_k3_numerator','negative_effective_weight_mass','tail_effective_weight_mass')],dtype=torch.float64,device='cuda')
        dist.all_reduce(vector)
        mass,numerator,negative,tail=vector.tolist()
        global_numerical=dict(sampled_old_policy_kl=numerator/mass,negative_tail_weight_fraction=tail/negative if negative else 0.)
        return dict(rank=worker.rank,path=str(path.resolve()),sha256=captured_sha,scope=scope,phase=phase,
                    diagnostic_write_index=index,global_mini_batch_size=mini,numerical=numerical,global_numerical=global_numerical)
    for index in range(cap):
        selected=batch[index*(mini//8):(index+1)*(mini//8)]
        before=[]
        for scope,part in (('full_frozen_rollout',batch),('full_current_minibatch',selected)):
            before.append(capture(scope,part,'pre_adam',index))
        events.extend(before)
        if any(e['global_numerical']['sampled_old_policy_kl']>float(safety.target_old_policy_kl) for e in before):
            stop_reason='candidate_old_policy_kl_before_write';break
        worker.actor_optimizer.zero_grad(set_to_none=True)
        for micro in selected.split(2):
            micro=micro.cuda();_,current=worker.actor._forward_micro_batch(micro,data.meta_info['temperature'])
            mask=micro['loss_mask'];adv=micro['advantages'];old=micro['old_log_probs']
            weights=mask.double()/mask.sum(-1,keepdim=True).clamp_min(1).double()
            delta=finite_inputs(old,current,adv,mask,weights)
            require(consensus(bool((delta[mask.bool()]<=float(safety.kl_log_bound)).all())),'diagnostic subtraction exceeds absolute KL log bound')
            surrogate=dual_clip_surrogate(delta,adv,mask,epsilon=.2,c=3.)
            loss=(surrogate*weights).sum()/(mini//8)
            require(consensus(bool(torch.isfinite(loss))),'diagnostic nonfinite loss')
            loss.backward()
        from .gradient_norm import clip_grad_norm64
        norm=clip_grad_norm64(worker.actor_module_fsdp,1.)
        require(consensus(bool(torch.isfinite(norm)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in worker.actor_module_fsdp.parameters())),'diagnostic nonfinite gradient')
        worker.actor_optimizer.step();writes+=1;worker._diagnostic_write_count=writes;worker.actor_optimizer.zero_grad(set_to_none=True)
        ledger.append(dict(write_index=index,actual_adam_write=True,grad_norm=float(norm),
            finite_loss_gradient_parameters_adam=False,lr=worker.actor_optimizer.param_groups[0]['lr'],
            scheduler_cached_last_lr=worker.actor_lr_scheduler.get_last_lr(),
            pre_paths=[e['path'] for e in before],post_paths=[]))
        worker._diagnostic_last_report=dict(status='DIAGNOSTIC_PARTIAL_OR_BOOTSTRAP',role=role,global_mini_batch_size=mini,
            expected_optimizer_steps=expected,optimizer_writes=writes,formal_outer_updates=0,scheduler_outer_steps=0,
            rank=worker.rank,stop_reason='postwrite_checks_pending',write_ledger=ledger,events=events,native_process=process,
            frozen_rollout_tensor_sha256=frozen_identity)
        persistence_error=None
        try:(root/f'rank_{worker.rank}_diagnostic_ledger.json').write_text(json.dumps(worker._diagnostic_last_report,indent=2,allow_nan=False))
        except OSError as error:persistence_error=str(error)
        require(consensus(persistence_error is None),'collective postwrite ledger persistence failed; native current-state backup required')
        finite_state=all(bool(torch.isfinite(p).all()) for p in worker.actor_module_fsdp.parameters()) and all(not isinstance(v,torch.Tensor) or bool(torch.isfinite(v).all()) for st in worker.actor_optimizer.state.values() for v in st.values())
        require(consensus(finite_state),'diagnostic postwrite nonfinite state; retain fault backup')
        after=[]
        for scope,part in (('full_frozen_rollout',batch),('full_current_minibatch',selected)):
            after.append(capture(scope,part,'post_adam',index))
        events.extend(after)
        ledger[-1].update(finite_loss_gradient_parameters_adam=True,post_paths=[e['path'] for e in after])
        if any(e['global_numerical']['sampled_old_policy_kl']>float(safety.target_old_policy_kl) for e in after):
            stop_reason='candidate_old_policy_kl_after_write';break
    require(worker.actor_lr_scheduler.last_epoch==0,'diagnostic scheduler advanced unexpectedly')
    status='DIAGNOSTIC_COMPLETE_SEQUENCE' if writes==expected and stop_reason is None else 'DIAGNOSTIC_PARTIAL_OR_BOOTSTRAP'
    report=dict(status=status,role=role,global_mini_batch_size=mini,expected_optimizer_steps=expected,
        optimizer_writes=writes,formal_outer_updates=0,scheduler_outer_steps=0,rank=worker.rank,
        stop_reason=stop_reason,write_ledger=ledger,events=events,native_process=process,frozen_rollout_tensor_sha256=frozen_identity,
        actual_optimizer_lr=[g['lr'] for g in worker.actor_optimizer.param_groups],scheduler_cached_last_lr=worker.actor_lr_scheduler.get_last_lr())
    worker._diagnostic_last_report=report
    persistence_error=None
    try:(root/f'rank_{worker.rank}_diagnostic_ledger.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    except OSError as error:persistence_error=str(error)
    require(consensus(persistence_error is None),'collective final diagnostic ledger persistence failed')
    return report


def compare_official_parameters(parameters,model_root):
    """Compare actual native initial HF tensors to every pinned official tensor."""
    import torch
    from safetensors import safe_open
    from .fresh200_contract import verify_official_model_assets
    require(verify_official_model_assets(model_root)=='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44','official origin manifest changed')
    root=Path(model_root);keys=set()
    for file in sorted(root.glob('*.safetensors')):
        with safe_open(str(file),framework='pt',device='cpu') as handle:
            for name in handle.keys():
                expected=handle.get_tensor(name);actual=parameters.get(name)
                require(isinstance(actual,torch.Tensor) and actual.shape==expected.shape
                    and torch.equal(actual.detach().cpu(),expected.to(actual.dtype)),'actual native initial tensor differs from official '+name)
                keys.add(name)
    extras=set(parameters)-keys
    if extras=={'lm_head.weight'} and 'model.embed_tokens.weight' in parameters:
        require(torch.equal(parameters['lm_head.weight'],parameters['model.embed_tokens.weight']),'official tied embedding alias changed');extras=set()
    require(not extras and keys,'native initial official parameter inventory differs')
    return {name:tensor_digest(value) for name,value in parameters.items()}


def capture_official_alignment(worker,directory):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,FullStateDictConfig
    from .policy_safety import consensus
    root=Path(directory);directory_error=None
    try:root.mkdir(parents=True,exist_ok=True)
    except OSError as error:directory_error=str(error)
    require(consensus(directory_error is None),'collective initial official capture directory rejected')
    with FSDP.state_dict_type(worker.actor_module_fsdp,StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
        parameters=worker.actor_module_fsdp.state_dict()
    error=None;report=None
    if worker.rank==0:
        try:
            hashes=compare_official_parameters(parameters,worker.config.model.path)
            path=root/'native-initial-official-hf.pt';require(not path.exists(),'refuse initial official tensor overwrite')
            torch.save(dict(model_parameters=parameters,official_model_root=str(Path(worker.config.model.path).resolve())),path)
            report=dict(path=str(path.resolve()),sha256=file_sha(path),official_model_root=str(Path(worker.config.model.path).resolve()),tensor_sha256=hashes)
            (root/'native-initial-official-alignment.json').write_text(json.dumps(report,indent=2))
        except Exception as caught:
            error=str(caught)
            try:(root/'initial-official-alignment.fault.log').write_text(error)
            except OSError:pass
    require(consensus(error is None),'actual native initial model differs from pinned official tensors')
    objects=[report];dist.broadcast_object_list(objects,src=0)
    return objects[0]


def validate_diagnostic_ledger(ledger,mini,role,rank,process):
    expected=2560//mini
    require(ledger.get('status')=='DIAGNOSTIC_COMPLETE_SEQUENCE' and ledger.get('role')==role
        and ledger.get('global_mini_batch_size')==mini and ledger.get('optimizer_writes')==expected
        and ledger.get('expected_optimizer_steps')==expected and ledger.get('formal_outer_updates')==0
        and ledger.get('scheduler_outer_steps')==0 and ledger.get('rank')==rank
        and ledger.get('native_process')==process and ledger.get('stop_reason') is None
        and ledger.get('actual_optimizer_lr')==[1e-6] and ledger.get('scheduler_cached_last_lr')==[1e-6/57],
        'incomplete/bootstrap/partial native sequence cannot admit production')
    require(len(ledger.get('write_ledger',[]))==expected and [w.get('write_index') for w in ledger['write_ledger']]==list(range(expected)), 'actual native Adam write index coverage invalid')
    events=ledger.get('events',[])
    require(len(events)==expected*4 and {(e.get('diagnostic_write_index'),e.get('scope'),e.get('phase')) for e in events}==
        {(i,s,p) for i in range(expected) for s in ('full_frozen_rollout','full_current_minibatch') for p in ('pre_adam','post_adam')}, 'native raw pre/post scope coverage incomplete')


def assemble_native_candidate(specifications,destination,producer_task_id):
    """Replay four complete fresh runs and write an UNRELEASED candidate.

    No independent PASS report, receipt nonce, release or approval is fabricated.
    """
    import torch
    require(set(specifications)=={(r,m) for r in ('discovery','confirmation') for m in (128,256)},'four independent fresh native runs required')
    output=Path(destination);require(output.is_absolute(),'absolute candidate output directory required')
    output.mkdir(parents=True,exist_ok=True)
    require(not any(output.iterdir()),'candidate output must be new/empty; no evidence overwrite')
    identity=None;roles={};native_bindings={};all_runs=[]
    for role in ('discovery','confirmation'):
        records=[];layout_roots={}
        for mini in (128,256):
            directory=Path(specifications[(role,mini)]);ledgers=[]
            for rank in range(8):
                path=directory/f'rank_{rank}_diagnostic_ledger.json';ledger=json.loads(path.read_text())
                validate_diagnostic_ledger(ledger,mini,role,rank,ledger['native_process'])
                ledgers.append((path,ledger))
            run=ledgers[0][1]['native_process']['run_root'];layout_roots[str(mini)]=run;all_runs.append(run)
            for index in range(2560//mini):
                for scope in ('full_frozen_rollout','full_current_minibatch'):
                    for phase in ('pre_adam','post_adam'):
                        entries=[];totals=[]
                        for path,ledger in ledgers:
                            events=[e for e in ledger['events'] if (e['diagnostic_write_index'],e['scope'],e['phase'])==(index,scope,phase)]
                            require(len(events)==1,'unique native diagnostic event required');event=events[0]
                            payload=torch.load(event['path'],map_location='cpu',weights_only=True)
                            if identity is None:identity=payload['identity']
                            require(payload['identity']==identity,'diagnostic source/model/data/sampler candidate identities differ')
                            local=replay_matrices(payload,320 if scope=='full_frozen_rollout' else mini//8,kl_log_bound=identity['kl_log_bound'])
                            totals.append(local)
                            entries.append(dict(rank=ledger['rank'],path=event['path'],sha256=event['sha256'],
                                diagnostic_ledger=dict(path=str(path.resolve()),sha256=file_sha(path))))
                        mass=math.fsum(x['effective_weight_mass'] for x in totals);negative=math.fsum(x['negative_effective_weight_mass'] for x in totals)
                        records.append(dict(global_mini_batch_size=mini,expected_optimizer_steps=2560//mini,
                            diagnostic_write_index=index,scope=scope,phase=phase,
                            trajectory_count=2560 if scope=='full_frozen_rollout' else mini,native_shards=entries,
                            sampled_old_policy_kl=math.fsum(x['weighted_k3_numerator'] for x in totals)/mass,
                            negative_tail_weight_fraction=math.fsum(x['tail_effective_weight_mass'] for x in totals)/negative if negative else 0.))
        value=dict(identity=identity,role=role,run_id=role+'-independent-fresh-layout-pair',layout_run_roots=layout_roots,pre_adam_monitors=records)
        path=output/f'{role}-native-raw.json';path.write_text(json.dumps(value,indent=2,allow_nan=False))
        roles[role]=dict(raw_path=str(path.resolve()),raw_sha256=file_sha(path))
        proofs=[]
        for sample in records:
            replay,proof=verify_native_sample(sample,identity,role,{int(k):v for k,v in layout_roots.items()})
            proofs.append(dict(scope=sample['scope'],phase=sample['phase'],diagnostic_write_index=sample['diagnostic_write_index'],
                global_mini_batch_size=sample['global_mini_batch_size'],replay=replay,proof=proof))
        native_bindings[role]=dict(raw_path=str(path.resolve()),raw_sha256=file_sha(path),samples=proofs)
    require(len(set(all_runs))==4,'diagnostic discovery/confirmation/layout states must be four independent fresh runs')
    tail=max(s['replay']['negative_tail_weight_fraction'] for s in native_bindings['discovery']['samples'])
    require(all(s['replay']['negative_tail_weight_fraction']<=tail and s['replay']['sampled_old_policy_kl']<=identity['target_old_policy_kl'] for s in native_bindings['confirmation']['samples']), 'independent confirmation rejects candidate calibration')
    candidate=dict(status='CANDIDATE_NOT_RELEASED',identity=identity,experiment_owner_id=producer_task_id,
        discovery=roles['discovery'],confirmation=roles['confirmation'],negative_tail_limit=tail,independent_review=None)
    (output/'candidate-manifest.json').write_text(json.dumps(candidate,indent=2,allow_nan=False))
    bindings=dict(identity=identity,discovery=native_bindings['discovery'],confirmation=native_bindings['confirmation'],negative_tail_limit=tail)
    (output/'independent-replay-bindings.json').write_text(json.dumps(bindings,indent=2,allow_nan=False))
    return output/'candidate-manifest.json'




SHARED_SPLITS={
    'train':dict(path='train.parquet',sha256='5d40180ba54d35513bb780408caa068d7d60123214a49b0739f835a2ae47c8a1',rows=169028),
    'validation':dict(path='validation.parquet',sha256='3d6318365b5bb98039f1e28ca6bc7d1de98eccc5cc833600e39ce1c2306f9ceb',rows=512),
    'stage':dict(path='stage.parquet',sha256='b529fe2a4b6424091aa4ca827a28337c978808ecde0e01e6fe7c867e14a9084e',rows=64),
    'final':dict(path='final.parquet',sha256='55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1',rows=51713),
}


def bind_shared_inputs(manifest_path,selectors=None):
    import pyarrow.parquet as pq
    path=Path(manifest_path)
    require(path.is_absolute() and path.is_file() and file_sha(path)=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24','actual shared input manifest differs')
    value=json.loads(path.read_text());result=dict(manifest_path=str(path.resolve()),manifest_sha256=file_sha(path),files={})
    for name,expected in SHARED_SPLITS.items():
        split=value['splits'][name]
        require(all(split[k]==v for k,v in expected.items()),'frozen data split identity changed '+name)
        member=path.parent/split['path']
        if selectors is not None:
            selector=selectors[name]
            require(isinstance(selector,str) and Path(selector).resolve()==member.resolve(),'actual configured input selector differs '+name)
        require(file_sha(member)==split['sha256'] and pq.ParquetFile(member).metadata.num_rows==split['rows'],'actual shared data bytes/rows changed '+name)
        result['files'][name]=dict(path=str(member.resolve()),sha256=split['sha256'],rows=split['rows'],bytes=member.stat().st_size)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Replay four native diagnostics; generate an unreleased candidate, never a PASS receipt')
    parser.add_argument('--run',action='append',required=True,help='role:global_mini:absolute_native_diagnostics_directory')
    parser.add_argument('--output',required=True);parser.add_argument('--producer-task-id',required=True)
    args=parser.parse_args();specs={}
    for entry in args.run:
        role,mini,path=entry.split(':',2);key=(role,int(mini));require(key not in specs,'duplicate native run specification');specs[key]=path
    print(assemble_native_candidate(specs,args.output,args.producer_task_id))
