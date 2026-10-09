"""SOURCE080 final hash/HF evidence candidate; no release or goal completion."""
import argparse
import datetime
import json
import os
from pathlib import Path
import resource
import stat
import sys
import time
import traceback

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from aligned_first5_cpu_evidence import ROOT,SOURCE,identity,regular,binding,write_new,verify_original_bindings,cpu_only,bind_source,official_model_path,full_model_files,cpu_hf
from aligned_first5_math import require
from aligned_final_evidence import terminal_observation,PENDING as PHASE1_PENDING

SAVE_OUTERS=(5,50,100,150,200)
HASH_TASKS=['official_complete_12_file_full_SHA256']+[f'outer_{outer}_{group}_full_SHA256_inventory'for outer in SAVE_OUTERS for group in ('hf','native_driver','algorithm')]
CPU_TASKS=['official_actual_CPU_HF_all_parameters_finite_tied_forward']+[f'outer_{outer}_actual_CPU_HF_tokenizer_all_parameters_finite_tied_forward_full_dtype_aligned_delta'for outer in SAVE_OUTERS]
PENDING=['native_DTENSOR_CPU_inspection_and_actual_driver_state200_semantic_validation',
    'complete_own_cleanup_GPU_exit_UUID_lease_and_resource_validation',
    'fault_or_resume_cross_job_identity_and_retained_partial_write_accounting','independent_final_review']
DEPENDENCIES=('aligned_final_phase2.py','aligned_final_evidence.py','aligned_first5_cpu_evidence.py',
    'aligned_first5_math.py','aligned-first5-math-provenance.json','collect_hardware_evidence.py')


def pending_tasks(progress):
    # Only work closed against original input bindings can leave pending.
    require(type(progress.get('hash_complete',False))is bool and type(progress.get('official_hf',False))is bool,'exact stage progress flags required')
    exports=progress.get('export_hf',[])
    require(isinstance(exports,list)and all(type(outer)is int and outer in SAVE_OUTERS for outer in exports)and len(set(exports))==len(exports),'exact completed export identities required')
    if not progress.get('hash_complete',False):return HASH_TASKS+CPU_TASKS+PENDING
    cpu=[]
    if not progress.get('official_hf',False):cpu.append(CPU_TASKS[0])
    cpu.extend(task for outer,task in zip(SAVE_OUTERS,CPU_TASKS[1:])if outer not in exports)
    return cpu+PENDING


def bound_input_records(bound):
    return bound['original_bindings']+bound.get('archive_bindings',[])+[bound['binding'],bound['actual_submission']]


def completion_inputs(bound,helpers,prior_hash_inputs,hash_data):
    verify_original_bindings(helpers+prior_hash_inputs)
    if bound is not None:verify_original_bindings(bound_input_records(bound))
    if hash_data is not None:
        location=hash_data['official_location']
        require(str(official_model_path(Path(location['logical'])))==location['resolved'],'official logical model mapping changed')
        for record in hash_data['official_records']:unchanged(record)
        for cp in hash_data['checkpoints']:
            for group in cp['groups'].values():check_inventory(group)


def invalidate_changed_completion(progress,bound,helpers,prior_hash_inputs,hash_data):
    try:completion_inputs(bound,helpers,prior_hash_inputs,hash_data)
    except Exception as error:
        return dict(hash_complete=False,official_hf=False,export_hf=[]),dict(error_type=type(error).__name__,error=str(error))
    return progress,None


def fingerprint(path):
    path=regular(path);s=path.stat()
    require(s.st_uid==os.getuid() and s.st_nlink==1,'own single-link original asset required')
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def hash_file(path):
    before=fingerprint(path);record=binding(path)
    require(fingerprint(path)==before,'original changed during full hash')
    return dict(record,fingerprint=before)


def unchanged(record):
    require(fingerprint(record['path'])==record['fingerprint'],'hashed original fingerprint changed; recollect full hash')


def inventory(folder,root):
    require(folder.is_absolute()and folder.resolve()==folder and folder.is_dir()and root in folder.parents,'canonical own checkpoint directory required')
    files=[]
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink()and path.resolve()==path,'symlink/noncanonical checkpoint inventory')
        if path.is_dir():continue
        require(stat.S_ISREG(path.stat().st_mode),'ordinary checkpoint inventory file required')
        files.append(path)
    require(files,'complete nonempty checkpoint inventory required')
    return files


def inventory_manifest(folder,root):
    paths=inventory(folder,root);records=[hash_file(p)for p in paths]
    require(inventory(folder,root)==paths,'checkpoint file membership changed during hash')
    for record in records:unchanged(record)
    return dict(directory=str(folder),files=records,scope='DERIVED_FULL_BYTE_HASH_INVENTORY_NOT_PRODUCER_RELOAD',
        files_total=len(records),bytes_total=sum(r['bytes']for r in records))


def check_inventory(manifest):
    folder=Path(manifest['directory'])
    require([str(p)for p in inventory(folder,ROOT)]==[r['path']for r in manifest['files']],'hashed checkpoint membership changed')
    for record in manifest['files']:unchanged(record)


def phase1_identity(report,actual_submission,phase1_binding):
    require(report.get('schema')=='aligned-final-evidence-candidate-phase1-v1'and report.get('source_commit')==SOURCE
        and report.get('root')==str(ROOT)and report.get('acceptance')is False and report.get('goal_complete')is False
        and report.get('release_receipt')is False
        and report.get('status')=='TERMINAL_RAW_200_LEDGER_EM_FINAL_AGGREGATE_COLLECTED_PARTIAL_NOT_ACCEPTANCE','bound completed phase1 report required')
    who=identity(actual_submission)
    require(report.get('actual_submission')==phase1_binding,'actual formal receipt changed since phase1')
    phase=report.get('phase1',{})
    require(phase.get('source_commit')==SOURCE and phase.get('run_root')==str(who['run'])and phase.get('job')==who['job']
        and phase.get('supported_scope')=='SINGLE_ORIGINAL_FORMAL_JOB_NO_INFRASTRUCTURE_FAULT_NO_RESUME_ONLY'
        and phase.get('accounting',{}).get('progress',{}).get('actual_adam_writes_per_rank')==[4000]*8
        and phase.get('accounting',{}).get('progress',{}).get('completed_outers')==200,'phase1 no-fault original job accounting differs')
    return who


def bind_phase1(path,output):
    path=regular(path,ROOT/'evidence');require(path.name=='phase1-evidence.json','original phase1 report file required')
    initial=binding(path);report=json.loads(path.read_text());verify_original_bindings([initial])
    receipt=regular(ROOT/'scheduler-head/formal-rayip-v2/formal-submit-receipt.json',ROOT);submission_binding=binding(receipt)
    submission=json.loads(receipt.read_text());who=phase1_identity(report,submission,submission_binding)
    write_new(output/'original-phase1-report.json',report)
    def query(argv):
        from collect_hardware_evidence import execute
        raw=execute(argv);write_new(output/'scheduler-original-query.json',raw);return raw
    terminal=terminal_observation(submission,query)
    write_new(output/'scheduler-terminal-observation.json',terminal)
    require(terminal.get('terminal_verified')is True and terminal.get('completed_successfully')is True,'actual formal completed terminal required now')
    originals=report['phase1']['original_bindings']+[report['actual_submission']]+report['formal_tools']+report['helper_files']
    verify_original_bindings(originals)
    # Archive files are also independent continuation inputs; keep their content bound.
    archives=[]
    for original in report['phase1']['original_bindings']:
        saved=regular(path.parent/'raw'/Path(original['path']).relative_to(ROOT),path.parent)
        actual=binding(saved)
        require(actual['sha256']==original['sha256']and actual['bytes']==original['bytes'],'immutable phase1 archive bytes changed')
        archives.append(actual)
    return dict(binding=initial,report=report,actual_submission=submission_binding,submission=submission,
        terminal=terminal,original_bindings=originals,archive_bindings=archives),who


def full_hash_stage(bound,who,output,progress=None):
    run=who['run'];config=bound['report']['phase1'].get('config')
    if config is None:
        config_path=run/'checkpoints/fresh200_state/outer_5_COMPLETE/resolved-config.json'
        config_binding=binding(regular(config_path,run))
        require(config_binding in bound['report']['phase1']['original_bindings'],'phase1 resolved config unbound')
        config=json.loads(config_path.read_text())
    logical=Path(config['actor_rollout_ref']['model']['path']);base=official_model_path(logical)
    manifest_path=base/'VERIFIED-MODEL-MANIFEST.json'
    manifest_doc=json.loads(regular(manifest_path).read_text())
    require(len(manifest_doc['files'])==12,'official complete12 manifest required')
    before={str(path):fingerprint(path)for path in [manifest_path]+[base/item['file']for item in manifest_doc['files']]}
    official=full_model_files(base)
    official_records=[dict(record,fingerprint=before[record['path']])for record in [official['manifest']]+official['files']]
    for record in official_records:unchanged(record)
    write_new(output/'official-model-full-sha256.json',dict(official=official,official_records=official_records))
    checkpoints=[]
    for outer in (5,50,100,150,200):
        groups={}
        for name,folder in (('hf',run/'checkpoints/actor'/f'global_step_{outer}'),
                ('native_driver',run/'checkpoints/fresh200_state'/f'outer_{outer}_COMPLETE'),
                ('algorithm',run/'checkpoints/algorithm'/f'global_step_{outer}')):
            groups[name]=inventory_manifest(folder,run)
            write_new(output/f'outer-{outer}-{name}-full-sha256.json',groups[name])
        hf=groups['hf']['files'];require(len(hf)==12,'complete HF12-file export required')
        names={Path(record['path']).name for record in hf}
        require({'config.json','tokenizer_config.json'}<=names and any(name.endswith('.safetensors')for name in names),'HF export model/tokenizer inventory incomplete')
        native_names={Path(record['path']).name for record in groups['native_driver']['files']}
        require({'driver.pt','driver-reload.json','status.json','all-rank-write-ledger.json'}<=native_names
            and {f'rank_{r}.pt'for r in range(8)}<=native_names and {f'rank_{r}.json'for r in range(8)}<=native_names,'eight native/driver originals missing')
        checkpoints.append(dict(actual_outer=outer,groups=groups))
    verify_original_bindings(bound_input_records(bound))
    for record in official_records:unchanged(record)
    for cp in checkpoints:
        for group in cp['groups'].values():check_inventory(group)
    require(official_model_path(logical)==base,'reviewed model link changed during hash')
    if progress is not None:progress['hash_complete']=True
    return dict(status='FULL_HASH_INVENTORIES_COLLECTED_CPU_HF_PENDING_NOT_ACCEPTANCE',official=official,
        official_records=official_records,official_location=dict(logical=str(logical),resolved=str(base)),checkpoints=checkpoints,
        provenance='Full byte inventories newly derived here; no fabricated producer manifest or later GPU roundtrip.')


def official_cpu_hf(base,official=True):
    import gc
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    cpu_only();start=time.monotonic();cpu=time.process_time()
    tokenizer=AutoTokenizer.from_pretrained(str(base),local_files_only=True,trust_remote_code=False)
    model,loading=AutoModelForCausalLM.from_pretrained(str(base),local_files_only=True,trust_remote_code=False,
        torch_dtype='auto',attn_implementation='eager',output_loading_info=True)
    require(all(not loading.get(key)for key in('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),'actual official HF loading differs')
    require(all(p.device.type=='cpu'and torch.isfinite(p).all()for p in model.parameters()),'all actual official parameters must be finite CPU')
    count=sum(p.numel()for p in model.parameters())
    require(model.config.tie_word_embeddings is True and model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr(),'actual official tied storage differs')
    if official:require(model.config.model_type=='qwen2'and model.config.vocab_size==151936 and model.config.hidden_size==2048 and count==3085938688,'official Qwen2.5-3B model dimensions differ')
    model.eval();inputs=tokenizer('The capital of France is',return_tensors='pt',truncation=True,max_length=16,return_token_type_ids=False)
    with torch.inference_mode():result=model(**inputs,use_cache=False)
    require(result.logits.device.type=='cpu'and torch.isfinite(result.logits).all(),'actual official CPU forward nonfinite')
    report=dict(status='ACTUAL_OFFICIAL_CPU_HF_CHECKED_NOT_ACCEPTANCE',scope='official'if official else'CPU_FIXTURE_ONLY',
        all_actual_parameters_finite=True,parameters=count,tied_input_output_storage=True,loading_info=loading,
        finite_cpu_forward=True,logits_shape=list(result.logits.shape),cost=dict(wall_seconds=time.monotonic()-start,
            cpu_seconds=time.process_time()-cpu,peak_rss_linux_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    del model,result;gc.collect();cpu_only();return report


def hf_stage(hash_report,bound,who,output,progress=None):
    require(hash_report.get('schema')=='aligned-final-evidence-candidate-phase2-v1'and hash_report.get('stage')=='hash'
        and hash_report.get('status')=='FULL_HASH_INVENTORIES_COLLECTED_CPU_HF_PENDING_NOT_ACCEPTANCE'
        and hash_report.get('source_commit')==SOURCE and hash_report.get('root')==str(ROOT)
        and hash_report.get('acceptance')is False and hash_report.get('goal_complete')is False
        and hash_report.get('phase1_binding')==bound['binding']and hash_report.get('actual_submission')==bound['actual_submission'],'immutable same-source/job hash-stage report required')
    data=hash_report['hash_stage'];location=data['official_location'];logical=Path(location['logical']);base=official_model_path(logical)
    require(str(base)==location['resolved'],'official borrowed directory changed')
    require([cp['actual_outer']for cp in data['checkpoints']]==[5,50,100,150,200],'complete five-checkpoint hash-stage coverage required')
    for record in data['official_records']:unchanged(record)
    # Validate the complete prior hash stage before claiming its work complete.
    for cp in data['checkpoints']:
        outer=cp['actual_outer']
        expected={'hf':who['run']/'checkpoints/actor'/f'global_step_{outer}',
            'native_driver':who['run']/'checkpoints/fresh200_state'/f'outer_{outer}_COMPLETE',
            'algorithm':who['run']/'checkpoints/algorithm'/f'global_step_{outer}'}
        require(set(cp['groups'])==set(expected)and all(cp['groups'][key]['directory']==str(folder)for key,folder in expected.items()),'actual job checkpoint directories differ')
        for group in cp['groups'].values():check_inventory(group)
    if progress is not None:progress['hash_complete']=True
    official_result=official_cpu_hf(base)
    for record in data['official_records']:unchanged(record)
    write_new(output/'official-actual-cpu-hf.json',official_result)
    if progress is not None:progress['official_hf']=True
    results=[]
    for cp in data['checkpoints']:
        outer=cp['actual_outer'];checkpoint=who['run']/'checkpoints/actor'/f'global_step_{outer}'
        expected={'hf':checkpoint,'native_driver':who['run']/'checkpoints/fresh200_state'/f'outer_{outer}_COMPLETE',
            'algorithm':who['run']/'checkpoints/algorithm'/f'global_step_{outer}'}
        require(set(cp['groups'])==set(expected)and all(cp['groups'][key]['directory']==str(folder)for key,folder in expected.items()),'actual job checkpoint directories differ')
        for group in cp['groups'].values():check_inventory(group)
        cpu_only();start=time.monotonic();cpu=time.process_time()
        result=cpu_hf(base,checkpoint)
        result.update(actual_outer=outer,source_commit=SOURCE,run_root=str(who['run']),job=who['job'],
            stage_cost=dict(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
                peak_rss_linux_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        for record in data['official_records']:unchanged(record)
        for group in cp['groups'].values():check_inventory(group)
        write_new(output/f'outer-{outer}-actual-cpu-hf.json',result);results.append(result)
        if progress is not None:progress.setdefault('export_hf',[]).append(outer)
    verify_original_bindings(bound_input_records(bound))
    for record in data['official_records']:unchanged(record)
    for cp in data['checkpoints']:
        for group in cp['groups'].values():check_inventory(group)
    require(official_model_path(logical)==base,'reviewed borrowed model changed during CPU work')
    return dict(status='FIVE_REAL_CPU_HF_EXPORTS_CHECKED_PARTIAL_NOT_ACCEPTANCE',official=official_result,results=results,
        continuation_integrity='Previously full-hashed originals retained identical device/inode/size/mtime/ctime and exact membership; no full hash repeated on HF continuation.',
        native_role='ORIGINAL_FULL_BYTES_HASHED_ONLY_NOT_CPU_DESERIALIZED_OR_GPU_RELOADED')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('hash','hf'))
    parser.add_argument('--phase1',type=Path,required=True);parser.add_argument('--hash-stage',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();output=args.output
    require(output.is_absolute()and output.parent==ROOT/'evidence'and output.parent.resolve()==output.parent,'fixed own new output required')
    require((args.stage=='hf')==(args.hash_stage is not None),'HF continuation requires exactly one original hash-stage report')
    output.mkdir();start=time.monotonic();cpu=time.process_time()
    progress=dict(hash_complete=False,official_hf=False,export_hf=[]);hash_data=None;bound=None;prior_hash_inputs=[]
    result=dict(schema='aligned-final-evidence-candidate-phase2-v1',stage=args.stage,candidate=True,acceptance=False,
        goal_complete=False,release_receipt=False,source_commit=SOURCE,root=str(ROOT),pending=pending_tasks(progress),
        status='CANDIDATE_UNEXECUTED_OR_UNBOUND_NOT_ACCEPTANCE',helper_files=[binding(HERE/name)for name in DEPENDENCIES])
    try:
        cpu_only();gate=bind_source()
        for name in ('aligned_final_evidence','aligned_first5_cpu_evidence','aligned_first5_math','collect_hardware_evidence'):
            require(Path(sys.modules[name].__file__).resolve()==HERE/(name+'.py'),'foreign phase2 dependency imported')
        bound,who=bind_phase1(args.phase1,output)
        result.update(phase1_binding=bound['binding'],actual_submission=bound['actual_submission'],run_root=str(who['run']),job=who['job'])
        config=json.loads((who['run']/'checkpoints/fresh200_state/outer_5_COMPLETE/resolved-config.json').read_text())
        require(gate.validate_gate_config(config)==(200,57),'source080 production profile required')
        if args.stage=='hash':
            result['hash_stage']=full_hash_stage(bound,who,output,progress);hash_data=result['hash_stage'];result['status']=result['hash_stage']['status']
        else:
            path=regular(args.hash_stage,ROOT/'evidence');require(path.name=='phase2-evidence.json','original hash-stage evidence filename required')
            original=binding(path);report=json.loads(path.read_text());prior_hash_inputs=[original]+report['helper_files']
            verify_original_bindings(prior_hash_inputs)
            write_new(output/'original-hash-stage-report.json',report);result['hash_stage_binding']=original;hash_data=report.get('hash_stage')
            result['hf_stage']=hf_stage(report,bound,who,output,progress);verify_original_bindings(prior_hash_inputs)
            result['status']=result['hf_stage']['status']
        completion_inputs(bound,result['helper_files'],prior_hash_inputs,hash_data);cpu_only()
    except Exception as error:
        # Every continuation original participates in both success and failure closure.
        progress,invalidation=invalidate_changed_completion(progress,bound,result['helper_files'],prior_hash_inputs,hash_data)
        if invalidation is not None:result['completion_invalidation']=invalidation
        result['status']='PHASE2_INCOMPLETE_OR_UNAVAILABLE_NOT_ACCEPTANCE'
        result['issue']=dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    result['completed_stage_tasks']=progress
    result['pending']=pending_tasks(progress)
    result['cost']=dict(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
        peak_rss_linux_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='actual own hash/CPU extraction only',monetary_price_assumed=False)
    write_new(output/'phase2-evidence.json',result)
    print(json.dumps(dict(status=result['status'],acceptance=False,goal_complete=False,path=str(output/'phase2-evidence.json'))))

if __name__=='__main__':main()
