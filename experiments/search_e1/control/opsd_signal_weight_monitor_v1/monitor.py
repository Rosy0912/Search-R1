"""Bound CPU read-only progress. Never launches, signals, imports Torch or recovers.

Counts committed native Adam receipts separately from complete outer/controller
evidence. Latest checkpoint payload SHA checks establish byte identity only;
this module never deserializes or claims independent optimizer tensor checks.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import types

VARIANTS=('G_POS','G_NEG','H_POS','H_NEG')
STUDY='/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010'
CHECKPOINT_NAME=re.compile(r'update-(\d{8})-outer-(\d{5})')
LOG_TERMS=('Traceback','RuntimeError','ValueError','CUDA error','illegal memory','out of memory','HOLD_','FAILED_')


class EvidenceError(ValueError):pass
class SnapshotRace(EvidenceError):pass


def require(ok,message):
    if not ok:raise EvidenceError(message)


def fingerprint(path):
    s=Path(path).stat()
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


def proof(path):
    path=Path(path).resolve();before=fingerprint(path);digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):digest.update(block)
    if before!=fingerprint(path):raise SnapshotRace('File changed during immutable read: '+str(path))
    return dict(path=str(path),sha256=digest.hexdigest(),bytes=before[2])


def checked(reference):
    require(isinstance(reference,dict) and isinstance(reference.get('path'),str)
            and isinstance(reference.get('sha256'),str) and len(reference['sha256'])==64,'Explicit path/SHA required')
    actual=proof(reference['path']);require(actual['sha256']==reference['sha256'],'File SHA differs: '+actual['path'])
    if 'bytes' in reference:require(actual['bytes']==reference['bytes'],'File length differs: '+actual['path'])
    return actual


def read(reference):
    actual=checked(reference);before=fingerprint(actual['path']);value=json.loads(Path(actual['path']).read_text())
    if before!=fingerprint(actual['path']):raise SnapshotRace('JSON changed during read: '+actual['path'])
    return value,actual


def current(path):return read(proof(path))


def same_ref(a,b):
    return isinstance(a,dict) and isinstance(b,dict) and Path(a.get('path','')).resolve()==Path(b.get('path','')).resolve() and a.get('sha256')==b.get('sha256')


def guarded(callable_):
    try:return dict(accepted=True,result=callable_())
    except (FileNotFoundError,SnapshotRace,json.JSONDecodeError) as error:
        return dict(accepted=False,state='UNKNOWN_OR_READ_RACE',error_type=type(error).__name__,message=str(error))
    except Exception as error:
        return dict(accepted=False,state='EVIDENCE_CHECK_FAILED',error_type=type(error).__name__,message=str(error))


def load_bound_module(helper,manifest,name):
    entry=manifest['files'][name+'.py'];digest=entry['sha256'] if isinstance(entry,dict) else entry
    path=Path(helper)/(name+'.py');checked(dict(path=str(path),sha256=digest))
    module=types.ModuleType('readonly_bound_signal_'+name);module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    require('torch' not in sys.modules,'CPU monitor unexpectedly imported Torch')
    return module


def source_identity(study,configs,publication_ref):
    publication,pub=read(publication_ref)
    require(same_ref(publication_ref,study['publication']),'Explicit source publication differs from study binding')
    require(publication.get('accepted') is True and publication.get('branch')=='search-e1'
        and publication.get('source_commit')==publication.get('remote_readback_sha')==study['source_commit'],
        'Missing actual matching source publication/readback')
    manifest,mref=read(study['helper_source_manifest']);helper=Path(study['helper_root']).resolve()
    require(publication['source_manifest_sha256']==mref['sha256'],'Publication names another helper manifest')
    found=set()
    for name,entry in manifest['files'].items():
        rel=Path(name);require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe helper source path')
        checked(dict(path=str(helper/rel),sha256=entry['sha256'] if isinstance(entry,dict) else entry));found.add(name)
    require({str(p.relative_to(helper)) for p in helper.rglob('*.py')}=={n for n in found if n.endswith('.py')},
        'Fresh helper Python inventory differs from frozen publication')
    first=configs['G_POS'][0];runtime,rref=read(first['runtime_manifest']);entries=runtime.get('files',runtime)
    require(len(entries)==305,'Original frozen305 runtime inventory differs')
    for name,entry in entries.items():
        rel=Path(name);require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe frozen runtime path')
        checked(dict(path=str(Path(first['paths']['runtime'])/rel),sha256=entry['sha256'] if isinstance(entry,dict) else entry))
    for config,_ in configs.values():
        require(config['study_id']==study['study_id'] and config['source_commit']==study['source_commit']
            and same_ref(config['helper_source_manifest'],mref) and config['paths']['helper_root']==str(helper)
            and same_ref(config['runtime_manifest'],rref),'Group source/runtime/study binding differs')
        for key in ('source_model_manifest','common_initial_adapter_manifest','common_cohort_manifest','train_identity','validation_identity'):
            require(same_ref(config[key],first[key]),'Groups do not share '+key)
    model,modelref=read(first['source_model_manifest']);require(model['accepted'] is True and model['source_grpo_step']==100 and len(model['files'])==12,'Source100 manifest differs')
    for entry in model['files']:
        stat=Path(entry['path']).stat();require(stat.st_size==entry['bytes'],'Source100 payload length differs')
    checked(first['train_identity']);checked(first['validation_identity'])
    return dict(source_commit=study['source_commit'],publication=pub,helper_manifest=mref,helper_files=len(found),
        runtime_manifest=rref,runtime_fresh_sha_files=305,source100_manifest=modelref,source100_payloads=12,
        source100_payload_scope='Current length checks only; no fresh full-weight SHA or tensor deserialization in this monitor',
        data_fresh_sha_files=2),manifest


def identity_state(expected,gates):
    if not expected:return dict(state='NOT_BOUND',wait_returncode='UNKNOWN')
    try:actual=gates.process(expected['pid'])
    except (FileNotFoundError,ProcessLookupError):return dict(state='ABSENT',expected=expected,wait_returncode='UNKNOWN')
    require(all(actual.get(k)==expected.get(k) for k in ('pid','uid','starttick','run_id')),'Expected process was reused/identity changed')
    return dict(state='LIVE',identity={k:actual.get(k) for k in ('pid','uid','starttick','ppid','pgid','run_id')},wait_returncode='UNKNOWN')


def verify_phase_exit(value,config,phase):
    require(value.get('accepted') is True,'Parent phase exit was not accepted')
    actual,_=read(value['actual_terminal']);exits=value['child_exit_evidence']
    require(actual['run_id']==config['run_id'] and actual['config_sha256']==config['_config_sha256']
        and actual['phase']==phase and actual['child_exit_evidence']==exits,'Actual wait evidence is outside this run/config/phase')
    require(exits and all(e['actual_wait_returncode']==0 and e['identity'].get('starttick') is not None
        and e['identity']['uid']==1001 and e['identity']['run_id']==config['run_id'] for e in exits),
        'Unknown/nonzero actual wait cannot establish phase success')
    require(value['gpu_release']['all_owned_release']['accepted'] is True,'Actual phase release is missing')
    return dict(accepted=True,actual_wait_returncodes=[e['actual_wait_returncode'] for e in exits],terminal=proof(value['actual_terminal']['path']))


def checkpoint_metadata(reference,config,cp_module):
    value,actual=read(reference)
    require(CHECKPOINT_NAME.fullmatch(Path(actual['path']).parent.name) is not None,'Pending checkpoint is not committed')
    name=CHECKPOINT_NAME.fullmatch(Path(actual['path']).parent.name)
    require(value['schema']=='opsd-online-checkpoint-v1' and value['accepted'] is True
        and value['run_id']==config['run_id'] and value['config_sha256']==config['_config_sha256']
        and value['contract']==cp_module.science_contract(config)
        and value['runtime_identity']==config['runtime_manifest']['sha256']
        and value['source_model_identity']==config['source_model_manifest']['sha256'],'Checkpoint source/config/contract differs')
    require(value['optimizer_updates']==int(name[1]) and value['cursor']['outer_index']==int(name[2]),'Committed checkpoint filename/cursor differs')
    require(value['complete_rank_count']==4 and value['equal_optimizer_steps'] is True
        and {r['rank'] for r in value['ranks']}==set(range(4)),'Four-rank checkpoint declaration missing')
    return value,actual


def committed_updates(checkpoint,previous_updates):
    refs=checkpoint['outer_update_receipts'];cursor=checkpoint['cursor']['next_pair_cursor'];K=checkpoint['cursor']['usable_pair_count']
    require(type(K) is int and type(cursor) is int and 0<=cursor<=K<=512,'Invalid usable/current pair cursor')
    require(cursor==K or cursor%32==0,'Incomplete mini was presented as a committed cursor')
    require(len(refs)==math.ceil(cursor/32),'Committed mini receipt count differs')
    exposed=0;counts=[]
    for number,ref in enumerate(refs,1):
        update,_=read(ref);count=min(32,cursor-exposed)
        require(update['schema']=='opsd-online-real-update-v1' and update['accepted'] is True
            and update['run_id']==checkpoint['run_id'] and update['outer_index']==checkpoint['cursor']['outer_index']
            and update['optimizer_updates']==previous_updates+number and update['global_actual_pairs']==count,
            'Committed actual update identity/tail count differs')
        ranks=update['all_rank_actual_token_metrics'];pairs=[p for r in ranks for p in r['pairs']]
        require(len(ranks)==4 and {r['rank'] for r in ranks}==set(range(4)) and len(pairs)==count,'Actual four-rank metrics/pair denominator differs')
        require(all(p['valid_tokens']>0 and all(math.isfinite(p[k]) for k in ('raw_loss','post_loss','weighted_post_loss','pair_weight')) for p in pairs),'Nonfinite actual pair scalar')
        counts.append(count);exposed+=count
    require(exposed==cursor and checkpoint['optimizer_updates']==previous_updates+len(refs),'Adam/exposure chain discontinuity')
    return dict(committed_minis=len(refs),committed_pairs=exposed,actual_tail_pairs=counts[-1] if counts else None)


def compact_train(summary,index,config):
    require(summary['accepted'] is True and summary['outer_index']==index and summary['run_id']==config['run_id']
        and summary['config_sha256']==config['_config_sha256'] and summary['variant']==config['variant']
        and summary['train_denominator']==2560 and summary['train_post_update'] is False,'Full2560 preupdate TRAIN EM identity differs')
    hits=summary['train_correct'];require(type(hits) is int and 0<=hits<=2560 and abs(summary['train_em']-hits/2560)<1e-12,'TRAIN EM denominator arithmetic differs')
    for key in ('actual_outer_unweighted_post_loss','actual_outer_weighted_post_loss'):
        require(summary[key] is None if summary['K']==0 else type(summary[key]) in (int,float) and math.isfinite(summary[key]),
            'Actual outer scalar is nonfinite or zero-K NA is missing')
    return {key:summary[key] for key in ('outer_index','displayed_iteration','train_correct','train_denominator','train_em','all5_wrong',
        'P','K','optimizer_updates','optimizer_updates_delta','actual_outer_unweighted_post_loss','actual_outer_weighted_post_loss','weight')}


def train_progress(config,cp_module):
    output=Path(config['paths']['output_root']);native=[];accepted=[];previous=None;updates=0;exposure=0
    for index in range(25):
        outer=output/'outers'/('outer-%05d'%index);path=outer/'training/outer-complete.json'
        if not path.exists():
            require(not any((output/'outers'/('outer-%05d'%j)/'training/outer-complete.json').exists() for j in range(index+1,25)),
                'Outer completion chain contains a gap')
            break
        completion,cproof=current(path);manifest,_=current(outer/'outer-manifest.json')
        require(completion['schema']=='opsd-online-outer-complete-v1' and completion['accepted'] is True
            and completion['run_id']==config['run_id'] and completion['outer_index']==index and type(completion['P']) is int
            and type(completion['K']) is int and 0<=completion['K']<=completion['P']<=512
            and manifest['outer_index']==index and manifest['run_id']==config['run_id'] and manifest['config_sha256']==config['_config_sha256']
            and manifest['P']==completion['P'],'Outer completion/input identity differs')
        require(same_ref(manifest['previous_complete_checkpoint'],previous) if previous else manifest['previous_complete_checkpoint'] is None,
            'Outer starts from a different previous complete checkpoint')
        require(manifest['cumulative_optimizer_updates_before']==updates,'Outer starts from another Adam count')
        cp,cpref=checkpoint_metadata(completion['checkpoint'],config,cp_module);minis=committed_updates(cp,updates)
        delta=math.ceil(completion['K']/32)
        require(completion['optimizer_updates_delta']==delta and completion['optimizer_updates']==updates+delta
            and completion['cumulative_pair_exposure']==exposure+completion['K'] and cp['cumulative_pair_exposure']==exposure+completion['K']
            and cp['cursor']['outer_index']==index and cp['cursor']['next_pair_cursor']==cp['cursor']['usable_pair_count']==completion['K'],
            'Outer completion/tail/raw state accounting differs')
        native.append(dict(outer_index=index,P=completion['P'],K=completion['K'],optimizer_updates=completion['optimizer_updates'],completion=cproof,raw_checkpoint=cpref))
        if (outer/'summary.json').exists():
            require(index==len(accepted),'Controller-accepted TRAIN summary chain contains a gap')
            summary,_=current(outer/'summary.json');row=compact_train(summary,index,config)
            require(row['P']==completion['P'] and row['K']==completion['K'] and row['optimizer_updates']==completion['optimizer_updates'],'TRAIN summary count differs')
            successful=[]
            for exit_path in sorted((outer/'trainer-logs').glob('attempt-*-exit.json')):
                value,_=current(exit_path)
                if value.get('accepted') is True:successful.append(verify_phase_exit(value,config,'training-running'))
            require(successful,'Controller summary lacks an actual successful trainer wait/release')
            accepted.append(row)
        updates=completion['optimizer_updates'];exposure+=completion['K'];previous=completion['checkpoint']
    candidates=[]
    for path in (output/'checkpoints').glob('*/manifest.json'):
        if CHECKPOINT_NAME.fullmatch(path.parent.name):
            value,ref=checkpoint_metadata(proof(path),config,cp_module)
            candidates.append((value['optimizer_updates'],value['cursor']['outer_index'],value['cursor']['next_pair_cursor'],value,ref))
    latest=None
    if candidates:
        _,outer_index,_,value,ref=max(candidates,key=lambda item:item[:3])
        require(outer_index<=len(native),'Checkpoint skipped an uncompleted prior outer')
        base=native[outer_index-1]['optimizer_updates'] if outer_index else 0
        minis=committed_updates(value,base)
        cp_module.check_checkpoint(ref,config,config['run_id'],value['optimizer_updates'],outer_index)
        latest=dict(manifest=ref,outer_index=outer_index,optimizer_updates=value['optimizer_updates'],
            next_pair_cursor=value['cursor']['next_pair_cursor'],usable_pairs=value['cursor']['usable_pair_count'],**minis,
            payload_fresh_sha=True,optimizer_tensor_numeric_deserialization=False)
    current_outer=None
    if len(native)<25:
        index=len(native);outer=output/'outers'/('outer-%05d'%index)
        if outer.exists():
            current_outer=dict(outer_index=index,displayed_iteration=index+1,P=None,K=None,stored_preupdate_train_em=None)
            if (outer/'outer-manifest.json').exists():
                value,ref=current(outer/'outer-manifest.json')
                require(value['run_id']==config['run_id'] and value['config_sha256']==config['_config_sha256'] and value['outer_index']==index,
                    'Active outer manifest is outside current run/config')
                current_outer.update(P=value['P'],manifest=ref,sampling_preupdate_model_sha256=value['sampling_model_identity']['sha256'])
            if (outer/'sampling-complete.json').exists():
                value,_=current(outer/'sampling-complete.json');metric=value['train_em']
                require(value['accepted'] is True and value['run_id']==config['run_id'] and metric['accepted'] is True
                    and metric['variant']==config['variant'] and metric['outer_index']==index
                    and metric['overall']['denominator']==2560 and abs(metric['overall']['em']-metric['overall']['strict_correct']/2560)<1e-12,
                    'Active unfiltered TRAIN EM identity/denominator differs')
                current_outer['stored_preupdate_train_em']=dict(strict_correct=metric['overall']['strict_correct'],denominator=2560,
                    em=metric['overall']['em'],post_training=False)
            if latest and latest['outer_index']==index:current_outer['K']=latest['usable_pairs']
    return dict(native_completed_outers=len(native),controller_accepted_completed_outers=len(accepted),completed_chain_optimizer_updates=updates,
        completed_chain_pair_exposures=exposure,completed_chain_P=sum(x['P'] for x in native),completed_chain_K=sum(x['K'] for x in native),
        latest_committed_checkpoint=latest,current_outer=current_outer,train_em_rows=accepted,
        train_em_scope='All2560 TRAIN slots sampled before this outer; different question cohort per outer, not heldout performance',
        numeric_optimizer_check_scope='Fresh latest payload SHA and declared four-rank identities; no Torch/tensor readback')


def partial_slots(directory,request):
    seen=set();completed=[];trailing=0;counts=dict(hits=0,final_invalid=0,valid_gold_mismatch=0,no_executed_search=0)
    for rank in range(8):
        path=directory/('worker-'+str(rank))/'results.jsonl'
        if path.exists():
            raw=path.read_bytes();lines=raw.splitlines(keepends=True)
            for line in lines:
                if not line.endswith(b'\n'):trailing+=1;continue
                row=json.loads(line);index=row['validation_index']
                require(type(index) is int and 0<=index<512 and index%8==rank and index not in seen
                    and row['evaluation_id']==request['evaluation_id'] and row['run_id']==request['run_id'],
                    'Partial native512 slot is duplicated/outside logical rank or request')
                hit=row['protocol_em'];valid=row['protocol_valid_final_action']
                require(row['status']=='complete' and row['primary'] is True and type(hit) is int and hit in (0,1)
                    and type(valid) is bool and (not hit or valid)
                    and row['failure_class']==(None if hit else 'final_invalid' if not valid else 'valid_gold_mismatch')
                    and type(row['search_calls']) is int and 0<=row['search_calls']<=4,
                    'Stored strict native512 hit/protocol/failure metadata differs')
                seen.add(index)
                counts['hits']+=hit;counts['final_invalid']+=row['failure_class']=='final_invalid'
                counts['valid_gold_mismatch']+=row['failure_class']=='valid_gold_mismatch';counts['no_executed_search']+=row['search_calls']==0
        complete=directory/('worker-'+str(rank))/'complete.json'
        if complete.exists():
            value,_=current(complete);require(value['accepted'] is True and value['evaluation_id']==request['evaluation_id'] and value['rank']==rank and value['primary_rows']==64,'Worker completion differs')
            completed.append(rank)
    return dict(saved_complete_primary_rows=len(seen),completed_logical_workers=completed,trailing_partial_lines=trailing,
        denominator_when_complete=512,partial_not_a_final_metric=True,stored_strict_counts=counts,
        score_scope='Stored strict scorer metadata only; no fresh model generation or trace rescoring')


def evaluation_progress(config,completed_outer):
    directory=Path(config['paths']['output_root'])/'evaluations'/('zero' if completed_outer==0 else 'outer-%02d'%completed_outer)
    parent_path=directory/'parent-complete.json';request_path=directory/'execution/request.json'
    if not request_path.exists():return dict(state='NOT_STARTED',completed_outer=completed_outer,saved_complete_primary_rows=0)
    request,rref=current(request_path);require(request['run_id']==config['run_id'] and request['config']['sha256']==config['_config_sha256']
        and request['completed_outer']==completed_outer,'Evaluation request config/run/boundary differs')
    partial=partial_slots(directory/'execution',request)
    result=dict(state='IN_PROGRESS_OR_PRESERVED_PARTIAL',completed_outer=completed_outer,request=rref,**partial)
    if parent_path.exists():
        parent,pref=current(parent_path);child,cref=read(parent['child'])
        require(parent['accepted'] is True and parent['run_id']==config['run_id'] and parent['config_sha256']==config['_config_sha256']
            and parent['completed_outer']==completed_outer and child['accepted'] is True and child['run_id']==config['run_id']
            and child['config']['sha256']==config['_config_sha256'] and child['variant']==config['variant']
            and child['evaluation_id']==request['evaluation_id'] and child['completed_outer']==completed_outer and child['primary_slots']==512
            and child['actual_child_exit_and_release'] is True,'Parent/child native512 completion differs')
        if completed_outer:
            require(same_ref(child['raw_checkpoint']['manifest'],parent['raw_checkpoint']),
                'Formal evaluation child did not use this explicit raw boundary checkpoint')
        actual=verify_phase_exit(parent['exit_proof'],config,'evaluation-running')
        require(len(child['child_exit_evidence'])==8 and all(x['actual_wait_returncode']==0 for x in child['child_exit_evidence']),'Eight logical native workers did not actually exit0')
        summary,sref=read(child['summary']);require(summary['accepted'] is True and summary['evaluation_id']==request['evaluation_id']
            and summary['completed_outer']==completed_outer and summary['primary_slots']==512 and partial['saved_complete_primary_rows']==512,
            'Formal512 summary/slot coverage differs')
        hits=summary['overall']['hits'];require(type(hits) is int and 0<=hits<=512 and summary['overall']['denominator']==512
            and abs(summary['overall']['em']-hits/512)<1e-12,'Formal512 EM arithmetic differs')
        require(all(summary['overall'][k]==v for k,v in partial['stored_strict_counts'].items()),'Summary strict counts differ from complete512 rows')
        checked(child['results'])
        result.update(state='ACCEPTED_512_WITH_ACTUAL_PARENT_WAIT_RELEASE',parent=pref,child=cref,summary=sref,
            actual_parent_wait=actual,overall=summary['overall'],datasets=summary['datasets'])
    return result


def log_exceptions(config,study_root):
    output=Path(config['paths']['output_root']);paths=list(output.rglob('*.log')) if output.exists() else []
    paths+=list((Path(study_root)/'sequence-attempts').glob(config['variant']+'-*/controller.log'))
    paths=sorted(paths,key=lambda p:p.stat().st_mtime_ns,reverse=True)[:8];matches=[]
    for path in paths:
        with path.open('rb') as stream:
            length=path.stat().st_size;stream.seek(max(0,length-(1<<20)));raw=stream.read(1<<20)
        for line in raw.splitlines():
            terms=[t for t in LOG_TERMS if t.encode().lower() in line.lower()]
            if terms:matches.append(dict(path=str(path),log_bytes=length,terms=terms,line_sha256=hashlib.sha256(line).hexdigest()))
    return dict(scanned_latest_logs=len(paths),maximum_tail_bytes_per_file=1<<20,exception_or_hold_lines=len(matches),
        recent_markers=matches[-16:],private_text_omitted=True,scope='Tail keyword markers are diagnostic hints, not process return codes or a causal conclusion')


def resource_snapshot(configs,statuses,gates):
    first=configs['G_POS'][0];receipt,_=read(first['gates']['retriever_admission']);oracle=gates.retriever_admission(first,receipt)
    all_cards,all_apps=gates.inventory();uuids=first['owner']['gpu_uuids'];cards=[c for c in all_cards if c['index'] in [4,5,6,7]]
    require([c['index'] for c in cards]==[4,5,6,7] and [c['uuid'] for c in cards]==uuids,'Actual owned-card UUID mapping differs')
    applications=[]
    for app in all_apps:
        if app['gpu_uuid'] not in uuids:continue
        if app['pid']==oracle['owner']['pid']:
            require(app['gpu_uuid']==uuids[-1] and app['used_mib']<=1024,'Original sharedGPU7 oracle exceeds gate')
            applications.append(dict(app,owner_kind='approved_original_oracle'));continue
        owners=[]
        for variant,status in statuses.items():
            if status is not None and status.get('owner'):
                cfg=configs[variant][0]
                try:actual=gates.owned_gpu_process(app['pid'],cfg['run_id'],status['owner'],status.get('known_children',[]))
                except Exception:continue
                owners.append((variant,actual))
        require(len(owners)==1,'Owned card has a foreign/unknown compute process; preserve work and do not signal it')
        variant,actual=owners[0];applications.append(dict(app,owner_kind='this_study',variant=variant,
            identity={k:actual.get(k) for k in ('pid','uid','starttick','run_id')}))
    require(all(c['actual_free_mib']>=6144 for c in cards),'Actual direct memory.free below owned running6GiB gate')
    return dict(devices=cards,compute_applications=applications,minimum_actual_free_mib=min(c['actual_free_mib'] for c in cards),
        other_cards_ignored=[0,1,2,3],oracle_owner={k:oracle['owner'][k] for k in ('pid','uid','starttick')},
        oracle_ready=oracle['health']['ready'],oracle_numeric_gate=oracle['numeric_gate'],no_signals_sent=True)


def snapshot(study_ref,config_refs,publication_ref):
    require('torch' not in sys.modules,'Run monitor in a fresh CPU process without Torch')
    require(os.getuid()==1001 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Only authorized HKUST4 UID1001/CVDempty CPU monitor')
    study,sref=read(study_ref);require(Path(study['study_base']).resolve()==Path(STUDY),'Wrong authorized remote study')
    require([g['variant'] for g in study['groups']]==list(VARIANTS) and set(config_refs)==set(VARIANTS),'Explicit four configs in predetermined study required')
    configs={}
    for group in study['groups']:
        variant=group['variant'];require(same_ref(group['config'],config_refs[variant]),'Explicit group ref differs from published study')
        config,cref=read(config_refs[variant]);config['_config_sha256']=cref['sha256']
        require(config['schema']=='opsd-signal-weight-config-v1' and config['variant']==variant and config['scientific']['outer_iterations']==25
            and config['owner']['uid']==1001 and config['owner']['physical_gpus']==[4,5,6,7],'Wrong config/25outer/fourcard owner')
        scientific=config['scientific']
        require(all(scientific[k]==v for k,v in dict(world_size=4,micro_batch=1,global_batch=32,
            gradient_accumulation_full_batch_per_rank=8,original_eight_rank_accumulation=4,learning_rate=1e-6,
            component_clip=10,loss_reduction='trajectory_mean').items()),'Fixed R science or doubled four-rank accumulation differs')
        configs[variant]=(config,cref)
    source,manifest=source_identity(study,configs,publication_ref)
    gates=load_bound_module(study['helper_root'],manifest,'gates');cp_module=load_bound_module(study['helper_root'],manifest,'checkpoint')
    statuses={};groups={};issues=[]
    for variant,(config,cref) in configs.items():
        path=Path(config['paths']['output_root'])/'status.json';status=None
        if path.exists():
            status,_=current(path);require(status['run_id']==config['run_id'] and status['config_sha256']==cref['sha256'],'Status belongs to another immutable run/config')
        statuses[variant]=status;output_exists=Path(config['paths']['output_root']).exists()
        group=dict(config=cref,run_id=config['run_id'],phase=status['phase'] if status else 'OUTPUT_CREATED_NO_STATUS' if output_exists else 'NOT_STARTED',
            controller=guarded(lambda s=status:identity_state(s.get('owner') if s else None,gates)),
            training=guarded(lambda c=config:train_progress(c,cp_module)),
            evaluations={str(i):guarded(lambda c=config,i=i:evaluation_progress(c,i)) for i in (15,25)},
            log_markers=guarded(lambda c=config:log_exceptions(c,study['study_base'])))
        if status and status['phase']=='TRAIN_COMPLETE':
            group['terminal_claim']=dict(status_complete25=status.get('outer_iterations')==25,
                wait_returncode='UNKNOWN unless actual study-driver terminal evidence exists')
        if status and ('FAILED' in status['phase'] or 'HOLD' in status['phase']):issues.append(dict(variant=variant,kind='ACTIVE_STATUS_REQUIRES_ATTENTION',phase=status['phase']))
        if status and group['controller']['accepted'] and group['controller']['result']['state']=='ABSENT' and status['phase']!='TRAIN_COMPLETE':
            issues.append(dict(variant=variant,kind='CONTROLLER_ABSENT_WAIT_UNKNOWN',phase=status['phase']))
        registry_path=Path(config['gates']['scoped_cpu_processes_path'])
        if registry_path.exists():
            def role_progress(c=config):
                roles,_=current(c['gates']['scoped_cpu_processes_path'])
                require(roles['run_id']==c['run_id'] and roles['config_sha256']==c['_config_sha256'],'CPU registry run/config differs')
                return {r['role']:identity_state(r['identity'],gates) for r in roles['roles']}
            group['external_cpu_roles']=guarded(role_progress)
        binding=Path(config['gates']['memory_binding_path'])
        if binding.exists():
            def memory_progress(c=config):
                terminal=Path(c['paths']['receipts'])/'memory-terminal.json'
                if terminal.exists():
                    saved,tref=current(terminal);require(saved['accepted'] is True and saved['run_id']==c['run_id'] and saved['config_sha256']==c['_config_sha256'],'Terminal memory identity differs')
                    return dict(state='NATURAL_TERMINAL_RECEIPT',terminal=tref,samples=saved['samples'],samples_below_limit=saved['samples_below_limit'],
                        observer=identity_state(saved['observer_identity'],gates),actual_wait_returncode='UNKNOWN; monitor never calls wait on another process')
                value=gates.memory_observer_admission(c);report,_=current(Path(c['paths']['receipts'])/'memory-live.json')
                return dict(state='LIVE_FRESH',age_seconds=value['age_seconds'],samples=report['samples'],samples_below_limit=report['samples_below_limit'],
                    minimum_actual_free_mib=report['minimum_actual_free_mib'],current_actual_free_mib={d['index']:d['actual_free_mib'] for d in value['current']['devices']})
            group['memory']=guarded(memory_progress)
        else:group['memory']=dict(accepted=True,result={'state':'NOT_STARTED_NO_OBSERVER_BINDING'})
        for name in ('controller','training','memory','external_cpu_roles'):
            if name in group and not group[name]['accepted']:issues.append(dict(variant=variant,kind=name,detail=group[name]))
        for boundary,value in group['evaluations'].items():
            if not value['accepted']:issues.append(dict(variant=variant,kind='evaluation_'+boundary,detail=value))
        groups[variant]=group
    baseline=guarded(lambda:evaluation_progress(configs['G_POS'][0],0))
    if not baseline['accepted']:issues.append(dict(kind='common_zero',detail=baseline))
    resources=guarded(lambda:resource_snapshot(configs,statuses,gates))
    if not resources['accepted']:issues.append(dict(kind='resources',detail=resources))
    sequence=None;seqpath=Path(study['study_base'])/'study-status.json'
    if seqpath.exists():
        def sequence_progress():
            seq,qref=current(seqpath);result=dict(phase=seq['phase'],status=qref,active_group=seq.get('active_group'),completed_groups=len(seq.get('finished_groups',[])))
            if seq.get('supervisor'):result['actual_supervisor']=identity_state(seq['supervisor'],gates)
            if seq.get('receipt'):
                terminal,tref=read(seq['receipt']);require(terminal['accepted'] is True and terminal['completed_groups']==4 and terminal['completed_outers']==100,'Four-group terminal claim differs')
                result.update(terminal=tref,completed_groups=4,completed_outers=100,
                    group_actual_wait_returncodes=[g['actual_wait_returncode'] for g in terminal['groups']],analysis_still_required=terminal['analysis_still_required'])
            return result
        sequence=guarded(sequence_progress)
        if not sequence['accepted']:issues.append(dict(kind='study_driver',detail=sequence))
    require('torch' not in sys.modules,'Monitor imported Torch unexpectedly')
    return dict(schema='opsd-signal-readonly-monitor-v1',accepted=not issues,checked_unix=time.time(),study_id=study['study_id'],study_manifest=sref,
        source=source,sequence=sequence,groups=groups,common_zero=baseline,resources=resources,issues=issues,
        units=dict(outer='512 TRAIN questions x5 fresh trajectories;25 pergroup',adam='ceil(K/32) committed minibatches perouter',
            train_em='unfiltered2560 sampled preupdate slots',heldout_em='fixed512 ordinary greedy only atzero/15/25'),
        no_signals_sent=True,no_training_started=True,no_torch_or_cuda_initialized=True,
        scope='Monitoring evidence only; an accepted monitor snapshot is not numerical Adam, experiment completion, causal or performance acceptance')


def write_audit(path,value):
    path=Path(path);require(path.is_absolute(),'Small immutable audit needs an explicit absolute path')
    raw=(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()
    require(len(raw)<2<<20,'Monitor audit exceeds small-summary2MiB cap')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return proof(path)


def main():
    p=argparse.ArgumentParser()
    for name in ('study-manifest','study-manifest-sha256','source-publication','source-publication-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--config',action='append',nargs=3,metavar=('VARIANT','PATH','SHA256'),required=True)
    p.add_argument('--audit-output');a=p.parse_args();require(len(a.config)==4 and {x[0] for x in a.config}==set(VARIANTS),'Exactly four explicit configrefs required')
    result=guarded(lambda:snapshot(dict(path=a.study_manifest,sha256=a.study_manifest_sha256),
        {v:dict(path=path,sha256=digest) for v,path,digest in a.config},dict(path=a.source_publication,sha256=a.source_publication_sha256)))
    value=result['result'] if result['accepted'] else dict(schema='opsd-signal-readonly-monitor-v1',accepted=False,input_read_failure=result,
        no_signals_sent=True,no_training_started=True,checked_unix=time.time())
    if a.audit_output:value['immutable_audit']=write_audit(a.audit_output,value)
    print(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False))
    return 0 if value['accepted'] else 2


if __name__=='__main__':raise SystemExit(main())
