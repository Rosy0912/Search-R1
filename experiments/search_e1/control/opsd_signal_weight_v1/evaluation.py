"""Isolated CPU evaluator launcher. Formal native512 only at zero/15/25."""
import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
try:
    from .checkpoint import require, object_sha, checked_json, checked_file, fileproof, exclusive_json, check_checkpoint, stat_identity
    from .training_metrics import rows_from_ref, protocol, VARIANTS
    from . import gates, prepare
except ImportError:
    from checkpoint import require, object_sha, checked_json, checked_file, fileproof, exclusive_json, check_checkpoint, stat_identity
    from training_metrics import rows_from_ref, protocol, VARIANTS
    import gates, prepare

PROTOCOL = dict(representation='HF_SearchAgent_unmerged', ordinary_prompt=True, base_dtype='bfloat16',
    adapter_dtype='float32', attention='sdpa', temperature=0., repetition_penalty=1.05,
    eos_token_id=[151645,151643], max_context=8192, tokens_per_turn=500, turns=5,
    search_calls=4, topk=3, observation_tokens=480, logical_workers=8, rank_seed=42,
    logical_slice='validation[rank::8]', scorer='frozen_authored_final_action', primary_rows=512)
INITIAL_ADAPTER_CONFIG_SHA='b30b7758c00a5010d0de1fe6aae667443406b676860c00b994b1f6cfe0974ccb'


def validate_adapter_config(value, config):
    scientific=config['scientific']
    require(type(value.get('r')) is int and value['r']==scientific['lora_rank']==16
            and type(value.get('lora_alpha')) is int and value['lora_alpha']==scientific['lora_alpha']==32
            and value.get('lora_dropout')==scientific['lora_dropout']==0
            and isinstance(value.get('target_modules'),list) and len(value['target_modules'])==7
            and set(value['target_modules'])==set(scientific['target_modules'])
            and value.get('task_type')=='CAUSAL_LM' and value.get('peft_type')=='LORA'
            and value.get('use_rslora',False) is False and value.get('use_dora',False) is False
            and value.get('bias')=='none' and value.get('fan_in_fan_out') is False
            and value.get('rank_pattern',{}) == {} and value.get('alpha_pattern',{}) == {},
            'Adapter configuration changes LoRA scaling/target/forward scientific contract')
    return dict(accepted=True,rank=16,alpha=32,dropout=0,target_modules=sorted(value['target_modules']),
                task_type='CAUSAL_LM',peft_type='LORA',use_rslora=False,use_dora=False,bias='none',fan_in_fan_out=False,
                rank_pattern={},alpha_pattern={},
                base_model_name_is_not_output_weight_identity=True)


def verify_metadata(proofs):
    for proof in proofs:
        now = stat_identity(proof['path'])
        require(now == {k: proof[k] for k in now}, 'Bound model/source changed during evaluation')


def input_identity(config, completed_outer, checkpoint_ref=None):
    require(type(completed_outer) is int and completed_outer in (0,15,25), 'Onlyzero/15/25 formal evaluation')
    prepare.validate_config(config, launch=True)
    require(config['evaluation']['completed_outers'] == [15,25] and config['evaluation']['common_zero'] is True,
            'Evaluation times changed')
    runtime = prepare.verify_runtime(config)
    model, model_manifest = checked_json(config['source_model_manifest'])
    files = model.get('files', model.get('source_files'))
    require(isinstance(files, list) and len(files) == 12, 'Source100 twelve-file inventory required')
    root = Path(config['paths']['model']).resolve(); source_files = [checked_file(f) for f in files]
    require(all(Path(f['path']).is_relative_to(root) for f in source_files), 'Source model escaped bound source100')
    data, validation = rows_from_ref(config['validation_identity'])
    require(len(data) == 512 and len({r['question_id'] for r in data}) == 512 and
            Counter(r['data_source'] for r in data) == {'nq':256,'hotpotqa':256}, 'Original fixed512 changed')
    if completed_outer == 0:
        require(checkpoint_ref is None, 'Zero must use immutable common initial, not another checkpoint')
        initial, initial_ref = checked_json(config['common_initial_adapter_manifest'])
        require(initial['adapter_tensor_sha256'] == config['common_initial_adapter_tensor_sha256'], 'Initial digest changed')
        adapter_files = [checked_file(f) for f in initial['payload_files']]
        adapter = dict(path=str(Path(initial['adapter_model']['path']).parent.resolve()), files=adapter_files)
        expected_digest = initial['adapter_tensor_sha256']; cp = None; updates = 0
        checkpoint_identity = dict(kind='common_zero_adapter', manifest=initial_ref, common_initial=config['common_initial_adapter_manifest'])
    else:
        require(checkpoint_ref is not None, 'Actual end-of-outer raw checkpoint required')
        cp, cp_proof = check_checkpoint(checkpoint_ref, config, config['run_id'], expected_outer=completed_outer-1)
        require(cp['cursor']['next_pair_cursor'] == cp['cursor']['usable_pair_count'], 'Raw outer has uncommitted tail')
        adapter = cp['adapter']; adapter_files = cp_proof['adapter_files']
        expected_digest = cp['all_rank_adapter_sha256'][0]; updates = cp['optimizer_updates']
        checkpoint_identity = dict(kind='raw_end_of_outer', manifest=cp_proof['manifest'], cursor=cp['cursor'], optimizer_updates=updates,
                                   rank_count=cp['complete_rank_count'], no_best_or_cache=True)
    adapter_root = Path(adapter['path']).resolve()
    require(all(Path(f['path']).parent.resolve() == adapter_root for f in adapter_files) and
            {'adapter_model.safetensors','adapter_config.json'} <= {Path(f['path']).name for f in adapter_files}, 'Adapter inventory incomplete/escaped')
    adapter_config_ref=next(f for f in adapter_files if Path(f['path']).name=='adapter_config.json')
    adapter_config, adapter_config_proof=checked_json(adapter_config_ref)
    config_contract=validate_adapter_config(adapter_config,config)
    if completed_outer==0:
        require(adapter_config_proof['sha256']==INITIAL_ADAPTER_CONFIG_SHA,'Common zero adapter config bytes differ from original')
    return dict(source_model_manifest=model_manifest, source_files=source_files, runtime=runtime,
        validation=validation, validation_semantic_sha256=[object_sha(r) for r in data],
        common_initial=config['common_initial_adapter_manifest'], adapter=adapter, adapter_files=adapter_files,
        expected_adapter_tensor_sha256=expected_digest, raw_checkpoint=checkpoint_identity,
        adapter_config=adapter_config_proof,adapter_config_contract=config_contract,
        optimizer_updates=updates, protocol=PROTOCOL), data


def verify_common_zero(config, shared_receipt):
    """Allow reuse across group config SHA, never across source/protocol/cohort."""
    value, proof = checked_json(shared_receipt)
    if value.get('schema') == 'opsd-signal-common-zero-reference-v1':
        require(value.get('accepted') is True and value['study_id'] == config['study_id']
                and value['source_model_identity'] == config['source_model_manifest']['sha256']
                and value['runtime_identity'] == config['runtime_manifest']['sha256']
                and value['validation_identity'] == config['validation_identity']
                and value['common_initial_adapter'] == config['common_initial_adapter_manifest'], 'Common-zero wrapper identity differs')
        parent, parent_proof = checked_json(value['parent'])
        require(parent.get('schema') == 'opsd-signal-evaluation-parent-v1' and parent.get('accepted') is True
                and parent['completed_outer'] == 0 and parent['raw_checkpoint'] is None
                and parent['child'] == value['child'] and parent['isolated_child_no_training_rng_consumed'] is True
                and parent['raw_state_unchanged'] is True and parent['parent_torch_imported'] is False,
                'Common-zero parent boundary not genuinely accepted')
        exits = parent['exit_proof']
        require(exits.get('accepted') is True and len(exits['child_exit_evidence']) == 1
                and exits['child_exit_evidence'][0]['actual_wait_returncode'] == 0
                and exits['child_exit_evidence'][0]['identity']['uid'] == config['owner']['uid']
                and type(exits['child_exit_evidence'][0]['identity']['starttick']) is int,
                'Common-zero evaluator CPU parent wait is nonzero/unknown')
        actual_terminal, _ = checked_json(exits['actual_terminal'])
        require(actual_terminal['child_exit_evidence'] == exits['child_exit_evidence']
                and exits['gpu_release']['all_owned_release']['accepted'] is True,
                'Common-zero actual terminal/release proof differs')
        value, proof = checked_json(value['child'])
        require(value['run_id'] == parent['run_id'] and value['config']['sha256'] == parent['config_sha256'],
                'Common-zero child differs from actually waited evaluator')
    if 'evaluation_receipt' in value:
        require(value.get('accepted') is True, 'Common-zero wrapper rejected')
        value, proof = checked_json(value['evaluation_receipt'])
    require(value.get('accepted') is True and value['completed_outer'] == 0 and value['version'] == 'source100'
            and value['primary_slots'] == 512 and value['actual_child_exit_and_release'] is True, 'Common-zero not actually complete')
    identity = value['input_identity']
    require(identity['protocol'] == PROTOCOL and identity['source_model_manifest']['sha256'] == config['source_model_manifest']['sha256']
            and identity['runtime']['manifest']['sha256'] == config['runtime_manifest']['sha256']
            and identity['validation']['sha256'] == config['validation_identity']['sha256']
            and identity['common_initial'] == config['common_initial_adapter_manifest']
            and identity['expected_adapter_tensor_sha256'] == config['common_initial_adapter_tensor_sha256'],
            'Common-zero source/runtime/validation/adapter/protocol differs')
    require(value['study_id'] == config['study_id'], 'Common-zero belongs to another study')
    # Reusing a prior result does not exempt the new group from fresh immutable
    # data/runtime/adapter byte checks. No model or RNG is initialized here.
    prepare.verify_runtime(config)
    source_manifest, _ = checked_json(config['source_model_manifest'])
    source_files = source_manifest.get('files', source_manifest.get('source_files'))
    require(isinstance(source_files,list) and len(source_files)==12, 'Common-zero fresh source100 inventory incomplete')
    for ref in source_files:
        checked_file(ref)
    initial, _ = checked_json(config['common_initial_adapter_manifest'])
    require(initial['adapter_tensor_sha256']==config['common_initial_adapter_tensor_sha256'], 'Common-zero initial digest changed')
    for ref in initial['payload_files']:
        checked_file(ref)
    data, _ = rows_from_ref(config['validation_identity'])
    summary, sp = checked_json(value['summary']); records, rp = rows_from_ref(value['results'])
    require(summary['accepted'] is True and summary['primary_slots'] == len(records) == 512 and
            summary['completed_outer'] == 0 and summary['overall']['denominator'] == 512, 'Common-zero rows incomplete')
    require({r['validation_index'] for r in records} == set(range(512)) and
            all(r['version'] == 'source100' and r['completed_outer'] == 0 and r['primary'] is True for r in records),
            'Common-zero formal slots invalid')
    require(len(data)==512 and all(r['question_id']==data[r['validation_index']]['question_id']
        and r['dataset']==data[r['validation_index']]['data_source']
        and r['row_semantic_sha256']==object_sha(data[r['validation_index']]) for r in records),
        'Common-zero actual validation row/semantic identity changed')
    return dict(accepted=True, evaluation_receipt=proof, summary=sp, results=rp, reused_common_zero=True,
                scope='Same source100/zero adapter/runtime/512/protocol/study; group config hashes may differ')


def boundary(config, owner, workers=(), released=False):
    """Fresh four-card actualfree and exact admitted descendants; never signals."""
    publication = gates.publication_admission(config, gates.bound(config['gates']['publication']))
    storage = gates.storage_admission(config, gates.bound(config['gates']['storage']))
    retriever = gates.retriever_admission(config, gates.bound(config['gates']['retriever_admission']))
    observer = gates.memory_observer_admission(config)
    require(gates.same_process(gates.process(owner['pid']), owner) and owner['uid'] == config['owner']['uid']
            and owner['run_id'] == config['run_id'], 'Evaluator process ownership changed')
    devices, apps = gates.inventory(); allowed = set(config['owner']['gpu_uuids'])
    selected = [d for d in devices if d['index'] in config['owner']['physical_gpus']]
    require([d['index'] for d in selected] == config['owner']['physical_gpus'] and
            [d['uuid'] for d in selected] == config['owner']['gpu_uuids'], 'Four-card UUID mapping changed')
    actual_owned = []
    for app in apps:
        if app['gpu_uuid'] not in allowed:
            continue
        require(app['pid'] != owner['pid'], 'CPU evaluation parent initialized CUDA')
        if app['pid'] == retriever['owner']['pid']:
            require(app['gpu_uuid'] == config['owner']['gpu_uuids'][-1] and app['used_mib'] <= 1024,
                    'Admitted GPU7 retriever sharing limit changed')
        else:
            actual_owned.append(gates.owned_gpu_process(app['pid'], config['run_id'], owner, list(workers)))
    require(not released or not actual_owned, 'Previous actual evaluation GPU descendants remain')
    floor = config['resources']['startup_minimum_free_mib'] if released else config['resources']['training_minimum_free_mib']
    require(all(d['actual_free_mib'] >= floor for d in selected), 'Actual four-GPU memory.free below boundary')
    return dict(accepted=True, devices=selected, owned=actual_owned, retriever=retriever, observer=observer,
                publication=publication, storage=storage, released=released, no_signals_sent=True, checked_unix=time.time())


def aggregate(records, data, config, request, verify_traces=True):
    require(len(records) == 512 and len(data) == 512, 'Formal512 incomplete')
    index = {}; core = qa = None
    if verify_traces:
        prepare.verify_runtime(config); core=importlib.import_module('search_e1.core'); qa=importlib.import_module('search_r1.qa_protocol')
    for row in records:
        i=row['validation_index']; require(type(i) is int and 0 <= i < 512 and i not in index, 'Duplicate/foreign formal slot')
        q=data[i]
        require(row['version']==request['version'] and row['completed_outer']==request['completed_outer']
            and row['evaluation_id']==request['evaluation_id'] and row['run_id']==config['run_id']
            and row['primary'] is True and row['status']=='complete'
            and row['rank']==i%8 and row['slot']==i//8 and row['question_id']==q['question_id']
            and row['dataset']==q['data_source'] and row['row_semantic_sha256']==object_sha(q),'Formal validation identity changed')
        require(type(row['protocol_em']) is int and row['protocol_em'] in (0,1) and type(row['protocol_valid_final_action']) is bool,
                'Invalid strict result types')
        require(not row['protocol_em'] or row['protocol_valid_final_action'], 'Hit without valid final')
        require(row['failure_class']==(None if row['protocol_em'] else 'final_invalid' if not row['protocol_valid_final_action'] else 'valid_gold_mismatch'),
                'Failure class changed')
        require(type(row['search_calls']) is int and 0 <= row['search_calls'] <= 4, 'Invalid search count')
        if row['slot']==0:
            require(row['hook_control']['accepted'] is True and row['hook_control']['excluded_from_primary_denominator'] is True,
                    'Actual hook null-effect control absent')
            if request['completed_outer']==0:
                require(row['zero_effect_control']['accepted'] is True,'Actual base/zero adapter control absent')
        if verify_traces:
            trace,_=checked_json(row['trace']); result=trace['native_result']; decoded=protocol(core,qa,result,q['answers'])
            require(trace['evaluation_id']==request['evaluation_id'] and trace['validation_index']==i and
                    trace['question_id']==q['question_id'] and trace['question']==q['question'] and trace['answers']==q['answers']
                    and trace['privileged_reference'] is False and trace['hooks_restored'] is True and not trace['logging_errors']
                    and object_sha(result)==row['native_result_sha256'] and decoded==trace['protocol']
                    and all(decoded[k]==row[k] for k in ('protocol_em','protocol_valid_final_action','failure_class'))
                    and result['termination']==row['termination'] and result['search_calls']==row['search_calls'], 'Actual trace/scoring changed')
        index[i]=row
    require(set(index)==set(range(512)), 'Missing fixed512 slots')
    def counts(items):
        hits=sum(r['protocol_em'] for r in items)
        return dict(denominator=len(items),hits=hits,em=hits/len(items),
            final_invalid=sum(r['failure_class']=='final_invalid' for r in items),
            valid_gold_mismatch=sum(r['failure_class']=='valid_gold_mismatch' for r in items),
            no_executed_search=sum(r['search_calls']==0 for r in items), termination=dict(Counter(r['termination'] for r in items)))
    return dict(schema='opsd-signal-native512-summary-v1',accepted=True,evaluation_id=request['evaluation_id'],
        run_id=config['run_id'],variant=config['variant'],version=request['version'],completed_outer=request['completed_outer'],
        primary_slots=512,overall=counts(records),datasets={d:counts([r for r in records if r['dataset']==d]) for d in ('nq','hotpotqa')},
        hook_controls_excluded=8,zero_effect_controls_excluded=8 if request['completed_outer']==0 else 0,
        previous_source239_is_not_acceptance_gate=True, protocol=PROTOCOL, no_best_selection=True)


def run(config_ref, completed_outer, checkpoint_ref, output_dir, receipt_path):
    config, cp = checked_json(config_ref); config['_config_sha256']=cp['sha256']
    require(os.getuid()==config['owner']['uid'] and os.environ.get('E1_RUN_ID')==config['run_id']
            and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Evaluation CPU parent run/UID/CVD changed')
    directory=Path(output_dir).resolve(); require(not directory.exists() and directory.is_relative_to(Path(config['paths']['output_root']).resolve()),
                                                'Evaluation attempt must be new under remote group output')
    directory.mkdir(parents=True)
    owner=gates.process(os.getpid()); initial_gate=boundary(config,owner,released=True)
    identity,data=input_identity(config,completed_outer,checkpoint_ref)
    request=dict(schema='opsd-signal-native512-request-v1',evaluation_id=uuid.uuid4().hex,run_id=config['run_id'],study_id=config['study_id'],
        config=cp,variant=config['variant'],version='source100' if completed_outer==0 else config['variant'],completed_outer=completed_outer,
        input_identity=identity,controller=owner,output=str(directory),protocol=PROTOCOL,created_unix=time.time(),parent_admission=initial_gate)
    request_ref=exclusive_json(directory/'request.json',request)
    exit_evidence=[]; complete_refs=[]; records=[]
    for wave_number,wave in enumerate(config['sampling']['waves']):
        require(wave==list(range(wave_number*4,wave_number*4+4)), 'Logical rank waves changed')
        boundary(config,owner,released=True); jobs=[]
        for rank in wave:
            physical=config['owner']['physical_gpus'][config['sampling']['logical_rank_to_slot'][rank]]
            command=[config['paths']['python'],str(Path(config['paths']['helper_root'])/'evaluation_worker.py'),
                '--request',request_ref['path'],'--request-sha256',request_ref['sha256'],'--rank',str(rank)]
            log=(directory/f'worker-{rank}.log').open('xb'); env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(physical),PYTHONDONTWRITEBYTECODE='1')
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            expected=None
            while process.poll() is None and expected is None:
                try:expected=gates.process(process.pid)
                except (FileNotFoundError,ProcessLookupError):time.sleep(.02)
            require(expected is not None,'Worker exited before live PID/starttick binding; cannot accept unknown identity')
            jobs.append((process,log,expected));exclusive_json(directory/f'worker-{rank}-launch.json',dict(identity=expected,command=command,physical_gpu=physical))
        while any(p.poll() is None for p,_,_ in jobs):
            try:
                boundary(config,owner,[e for _,_,e in jobs])
            except (ValueError,FileNotFoundError,ProcessLookupError) as error:
                # Retain the CPU owner/parent leases while genuine GPU children
                # run. A transient /proc exit race never relaunches or signals.
                print(json.dumps(dict(status='HOLD_RUNNING_EVALUATION_GATE',error=str(error))),flush=True)
            time.sleep(1)
        codes=[]
        for process,log,expected in jobs:
            code=process.wait();log.flush();os.fsync(log.fileno());log.close();codes.append(code)
            exit_evidence.append(dict(identity=expected,actual_wait_returncode=code))
        terminal_ref=exclusive_json(directory/f'wave-{wave_number}-actual-exit.json',dict(child_exit_evidence=exit_evidence[-4:]))
        # No new GPU phase until actual descendants have released. Do not kill any PID.
        while True:
            try:release=boundary(config,owner,[e for _,_,e in jobs],released=True);break
            except (ValueError,FileNotFoundError,ProcessLookupError):time.sleep(1)
        exclusive_json(directory/f'wave-{wave_number}-released.json',dict(actual_exit=terminal_ref,release=release))
        require(all(code==0 for code in codes),'Nonzero evaluation attempt preserved; do not count partial slots')
        for rank in wave:
            complete, proof=checked_json(fileproof(directory/f'worker-{rank}'/'complete.json'))
            require(complete['accepted'] is True and complete['evaluation_id']==request['evaluation_id'] and complete['primary_rows']==64
                and complete['rank']==rank and complete['actual_loaded_adapter']['tensor_sha256']==identity['expected_adapter_tensor_sha256'],
                'Actual worker model/completion identity differs')
            forward=complete['actual_loaded_forward']
            require(forward['accepted'] is True and forward['adapter_name']==request['version']
                and forward['layers']==252 and len(forward['layer_checks'])==252
                and forward['scaling']==2. and forward['rank']==16 and forward['alpha']==32
                and forward['dropout']==0. and forward['A_B_dtype']=='torch.float32',
                'Actual selected adapter forward scaling/rank/dtype proof missing')
            rr, rp=rows_from_ref(complete['results']);require(len(rr)==64,'Worker64 incomplete');records.extend(rr);complete_refs.extend([proof,rp])
    summary=aggregate(records,data,config,request)
    verify_metadata(identity['source_files']+identity['adapter_files']);prepare.verify_runtime(config)
    require([r['actual_wait_returncode'] for r in exit_evidence]==[0]*8,'All eight actual exits required')
    result_path=directory/'results.jsonl'
    with result_path.open('x') as sink:
        for row in sorted(records,key=lambda r:r['validation_index']):sink.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        sink.flush();os.fsync(sink.fileno())
    summary_ref=exclusive_json(directory/'summary.json',summary);final_release=boundary(config,owner,released=True)
    result=dict(schema='opsd-signal-native512-complete-v1',accepted=True,evaluation_id=request['evaluation_id'],run_id=config['run_id'],study_id=config['study_id'],
        config=cp,variant=config['variant'],version=request['version'],completed_outer=completed_outer,primary_slots=512,
        raw_checkpoint=identity['raw_checkpoint'],input_identity=identity,request=request_ref,summary=summary_ref,
        results=fileproof(result_path),actual_child_exit_and_release=True,child_exit_evidence=exit_evidence,
        worker_proofs=complete_refs,gpu_release=final_release,no_training=True,checkpoint_metadata_unchanged=True)
    return exclusive_json(receipt_path,result)


def main():
    p=argparse.ArgumentParser()
    for name in ('config','config-sha256','output-dir','receipt'):p.add_argument('--'+name,required=True)
    p.add_argument('--checkpoint');p.add_argument('--checkpoint-sha256');p.add_argument('--completed-outer',type=int,choices=(0,15,25),required=True)
    a=p.parse_args();require(bool(a.checkpoint)==bool(a.checkpoint_sha256),'Checkpoint path/SHA must appear together')
    ref=run(dict(path=a.config,sha256=a.config_sha256),a.completed_outer,
        dict(path=a.checkpoint,sha256=a.checkpoint_sha256) if a.checkpoint else None,a.output_dir,a.receipt)
    print(json.dumps(dict(accepted=True,receipt=ref)))


if __name__=='__main__':main()
