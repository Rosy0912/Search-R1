"""Readonly CPU diagnosis of the original failed3294 private journal.

V2 only permits the two actual Git-authenticated empty package source files.
The original V1 remains immutable; metadata and private journals stay nonempty.

No default remote action. Future --execute performs one readonly RPC after the
original FAILED/1:0 terminal/source/official-tokenizer bindings pass. Raw prompts,
IDs, generated text and selected LP values remain private on the native node.
The output is only scalar/hash/token-class evidence, never an accepted outer,
health decision, tensor audit, corrected trajectory, or retry authorization.
"""
import argparse
import ast
import base64
import copy
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
import re
import shlex
import stat
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
LAUNCH_SHA256='1d50b96ed51f916e21891c1db88af03cdc3dea035bdec8471aa66959877c041f'
BIRTH_SHA256='c57f8e39d081ff114d9ce9c2c3e8f532c0cb15ef54caf2039b44d65dbf0f4889'
TERMINAL_SHA256='681b38622bb4c32284dbd203fbede46f61efa46db6bde04de450797d64b2df70'
READER_SHA256='f10c3371a446a1816840e89ebb60aa3716a8e61157936733cb2b6a9f514297fd'
COLLECTOR_SHA256='aa93cca3ce61d60b427b7c34ebfa357632cbb5bd52881b9eb9a4c5103b93fd31'
SOURCE_COMMIT='e670dfacab4ff0146f7ea0ca60e3f49675a4d2f6'
SOURCE_MAP_SHA256='2bee41505c1c01bd87083b6e792d7e1c4233ad3998b77345c66357b16b134b8e'
RUN_ID='ee0d52bd4edb465993214a2a48f7521a'
PARENT_RUN_ID='00f18587fb4444faa62210eb72c84e98'
MAX_JOURNAL_BYTES=128*1024**2
MAX_RETURN_BYTES=1024**2
DETAIL_RECORD_LIMIT=64
DETAIL_POSITION_LIMIT=16
TOKENIZER_FILES=('config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','generation_config.json')
EMPTY_SOURCE_FILES=frozenset(('search_r1/__init__.py','search_r1/llm_agent/__init__.py'))
EMPTY_SOURCE_SHA256='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
PRIVATE_FIELDS=frozenset(('run_id','source_sha256','slurm_job_id','logical_update','policy_snapshot_id',
    'role','actual_sampling_records','failure_reason','complete_outer'))
RECORD_FIELDS=frozenset(('rollout_call_id','actual_request_token_ids','raw_generated_token_ids',
    'raw_generated_token_count','selected_logprobs','finish_reason','stop_reason','eos_token_id',
    'pad_token_id','sampling','policy_snapshot_id','schema_version','request_id','row_id','rollout_ordinal'))
EVENT_FIELDS=frozenset(('role','rollout_call_id','row_ids','observation_or_correction_token_ids','is_search','final_turn'))


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def strict_json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'Duplicate original private/metadata field');result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite untyped private JSON')))


def stable(st):
    return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_uid,st.st_mode,st.st_nlink)


def readonly_bytes(path,maximum,*,private=False,expected_sha256=None,empty_source_relative=None):
    path=Path(path);require(path.is_absolute() and path.resolve()==path and not path.is_symlink(),
        'Readonly bound path follows symlink');start=time.time();fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        zero_permitted=empty_source_relative is not None
        if zero_permitted:
            require(type(empty_source_relative) is str and empty_source_relative in EMPTY_SOURCE_FILES and
                str(path).endswith('/repo/'+empty_source_relative) and private is False and
                expected_sha256==EMPTY_SOURCE_SHA256,'Empty permission is only for exact authenticated package source')
        require(stat.S_ISREG(before.st_mode) and (0<=before.st_size if zero_permitted else 0<before.st_size) and
            before.st_size<=maximum and stable(before)==stable(path.lstat()),
            'Not regular bounded original evidence')
        if private:
            parent=path.parent
            require(before.st_uid==os.getuid() and before.st_nlink==1 and stat.S_IMODE(before.st_mode)==0o600 and
                parent.resolve()==parent and not parent.is_symlink() and parent.stat().st_uid==os.getuid() and
                stat.S_IMODE(parent.stat().st_mode)==0o700,'Original journal private ownership/mode changed')
        with os.fdopen(os.dup(fd),'rb') as stream:raw=stream.read(maximum+1)
        require(len(raw)==before.st_size and stable(before)==stable(os.fstat(fd))==stable(path.lstat()) and path.resolve()==path,
            'Readonly original changed during capture')
        digest=sha(raw);require(expected_sha256 is None or digest==expected_sha256,'Bound original SHA differs')
        return raw,dict(size_bytes=len(raw),sha256=digest,uid=before.st_uid,mode=stat.S_IMODE(before.st_mode),
            device=before.st_dev,inode=before.st_ino,mtime_ns=before.st_mtime_ns,ctime_ns=before.st_ctime_ns,
            read_started_unix=start,read_finished_unix=time.time())
    finally:os.close(fd)


def safe_error(error):
    if type(error).__name__=='BehaviorTraceError' and re.fullmatch('[A-Za-z0-9_:.-]{1,128}',str(error)):
        return str(error)
    return type(error).__name__+':message_sha256='+sha(str(error).encode())


def diagnostic_prefix(raw,decode,eos):
    """Observe the production boundary before its guard, without accepting it.

    This independently mirrors only the original character/token-prefix scan.
    It does not remove an internal token, retokenize, construct a valid turn,
    or confer context equality. Decoded strings never leave this function.
    """
    trailing=bool(raw and raw[-1]==eos);eligible=raw[:-1] if trailing else raw
    text=decode(eligible);require(isinstance(text,str),'Invalid deterministic decode')
    closing='</search>' if '</search>' in text else '</answer>' if '</answer>' in text else None
    count=len(eligible);retained=text;execution=text;suffix='';prefix_stable=True
    if closing:
        for count in range(1,len(eligible)+1):
            candidate=decode(eligible[:count]);require(isinstance(candidate,str),'Invalid deterministic prefix decode')
            if closing in candidate:
                end=candidate.find(closing)+len(closing)
                retained=candidate;execution=candidate[:end];suffix=candidate[end:]
                prefix_stable=text.startswith(candidate);break
        else:raise ValueError('Missing original raw-prefix boundary')
    return dict(retained_count=count,trailing_eos_excluded=trailing,
        decoded_prefix_stable=prefix_stable,closing_tag_kind='search' if closing=='</search>' else 'answer' if closing else 'none',
        suffix_class='none' if not suffix else 'whitespace' if suffix.isspace() else 'nonwhitespace',
        full_decode_char_count=len(text),retained_decode_char_count=len(retained),execution_char_count=len(execution),
        full_decode_sha256=sha(text.encode()),retained_decode_sha256=sha(retained.encode()),execution_sha256=sha(execution.encode()),
        full_replacement_count=text.count('\ufffd'),retained_replacement_count=retained.count('\ufffd'))


def token_class(token,eos,pad,special):
    if token==eos and token==pad:return 'eos_and_pad'
    if token==eos:return 'eos'
    if token==pad:return 'pad'
    if token in special:return 'other_special'
    return 'ordinary'


def generation_eos_ids(configuration):
    require(type(configuration) is dict,'Official generation config not a mapping')
    value=configuration.get('eos_token_id')
    if value is None:return set()
    values=value if type(value) is list else [value]
    require(bool(values) and all(type(item) is int and item>=0 for item in values),
        'Official generation EOS schema not int/list of ints')
    return set(values)


def journal_diagnosis(document,tokenizer,helper,*,run_id,source_sha256,job_id,generation_config=None):
    require(type(document) is dict and set(document)==PRIVATE_FIELDS and
        document['role']=='common8_failed_partial_generation' and document['complete_outer'] is False and
        document['run_id']==run_id and document['source_sha256']==source_sha256 and
        document['slurm_job_id']==job_id and type(document['logical_update']) is int and document['logical_update']==1 and
        isinstance(document['policy_snapshot_id'],str) and document['policy_snapshot_id'].startswith(run_id+':1:policy-') and
        type(document['actual_sampling_records']) is list and 0<len(document['actual_sampling_records'])<=12805,
        'Not this original failed partial outer1 private schema')
    eos,pad=tokenizer.eos_token_id,tokenizer.pad_token_id
    require(type(eos) is int and type(pad) is int,'Official tokenizer EOS/PAD not concrete')
    special=set(tokenizer.all_special_ids)|{eos,pad};require(all(type(x) is int for x in special),'Official special ID schema changed')
    generation_eos=generation_eos_ids(generation_config or {})
    decode=lambda values:tokenizer.decode(values,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    counts=Counter();errors=Counter();ordinals=Counter();classes=Counter();included_classes=Counter()
    lengths=dict(request=[],raw=[],selected_lp=[],retained=[]);details=[];call_ids=set();request_keys=set()
    lp_bad=lp_positive=raw_count_mismatch=metadata_bad=0;environment_tokens=environment_rows=0
    prefix_stable=prefix_unstable=0;snapshot=document['policy_snapshot_id'];record_counter=0;detail_candidates=0
    for index,record in enumerate(document['actual_sampling_records']):
        require(type(record) is dict,'Unknown stored record type')
        if record.get('role')=='actual_environment_event':
            require(set(record)==EVENT_FIELDS,'Unknown stored environment fields')
            arrays=(record['row_ids'],record['observation_or_correction_token_ids'],record['is_search'])
            require(all(type(x) is list for x in arrays) and len({len(x) for x in arrays})==1 and
                type(record['final_turn']) is bool and isinstance(record['rollout_call_id'],str),'Environment alignment changed')
            for ids,search in zip(record['observation_or_correction_token_ids'],record['is_search']):
                require(type(ids) is list and all(type(x) is int and x>=0 for x in ids) and type(search) is int and search in (0,1),
                    'Environment token/classification schema changed')
                environment_tokens+=len(ids);counts['observation_tokens' if search else 'correction_tokens']+=len(ids)
            counts['environment_events']+=1;environment_rows+=len(record['row_ids']);continue
        require(set(record)==RECORD_FIELDS,'Unknown raw sampling record/mask schema')
        record_counter+=1;ordinal=record['rollout_ordinal'];require(type(ordinal) is int and 0<=ordinal<5,'Invalid stored turn ordinal')
        ordinals[str(ordinal)]+=1;row_id=record['row_id'];require(isinstance(row_id,str),'Invalid stored row identity')
        padded=row_id.startswith('padding-');counts['padding_transport_records' if padded else 'nonpadding_row_records']+=1
        call=record['rollout_call_id'];require(isinstance(call,str) and call,'Missing actual stored call identity');call_ids.add(call)
        require(record['policy_snapshot_id']==snapshot and type(record['schema_version']) is int and record['schema_version']==1,
            'Stored snapshot/native schema differs')
        request_key=sha(canonical([call,record['request_id'],row_id]));request_keys.add(request_key)
        raw=record['raw_generated_token_ids'];request=record['actual_request_token_ids'];lp=record['selected_logprobs']
        require(type(raw) is list and type(request) is list and type(lp) is list and
            all(type(x) is int and x>=0 for x in raw+request) and len(raw)<=500 and len(request)<=4096,
            'Stored raw/request schema or original budget changed')
        lengths['request'].append(len(request));lengths['raw'].append(len(raw));lengths['selected_lp'].append(len(lp))
        raw_ok=type(record['raw_generated_token_count']) is int and record['raw_generated_token_count']==len(raw)==len(lp)
        raw_count_mismatch+=int(not raw_ok)
        finite=all(type(x) in (int,float) and math.isfinite(x) for x in lp)
        lp_bad+=sum(not(type(x) in (int,float) and math.isfinite(x)) for x in lp)
        lp_positive+=sum(type(x) in (int,float) and math.isfinite(x) and x>0 for x in lp)
        metadata_bad+=int(record['eos_token_id']!=eos or record['pad_token_id']!=pad)
        try:helper._sampling(record['sampling'])
        except Exception:counts['stored_sampler_metadata_rejected']+=1
        helper_error=None;cut=None
        try:
            cut=helper.raw_prefix_to_first_closing_tag(raw,decode,eos_token_id=eos,pad_token_id=pad,special_token_ids=sorted(special))
            counts['existing_helper_pass']+=1
        except Exception as error:
            helper_error=safe_error(error);errors[helper_error]+=1
        diagnostic=diagnostic_prefix(raw,decode,eos);retained=diagnostic['retained_count'];lengths['retained'].append(retained)
        genuine_stop=record['finish_reason']=='stop' and record['stop_reason'] in (None,eos)
        stop=record['stop_reason'];explicit_stops=record['sampling'].get('stop_token_ids',[])
        explicit_stop_set=set(explicit_stops) if type(explicit_stops) is list and all(type(x) is int for x in explicit_stops) else set()
        config_stop=type(stop) is int and stop in generation_eos
        counts['native_stop_reason_in_official_generation_eos']+=int(config_stop)
        counts['native_generation_eos_not_tokenizer_eos']+=int(config_stop and stop!=eos)
        counts['native_generation_eos_not_in_captured_explicit_stops']+=int(config_stop and stop not in explicit_stop_set)
        counts['finish_stop_and_final_raw_matches_official_generation_eos']+=int(
            config_stop and record['finish_reason']=='stop' and bool(raw) and raw[-1]==stop)
        counts['trailing_eos_without_native_stop']+=int(diagnostic['trailing_eos_excluded'] and not genuine_stop)
        prefix_stable+=int(diagnostic['decoded_prefix_stable']);prefix_unstable+=int(not diagnostic['decoded_prefix_stable'])
        if cut is not None:
            require(cut['kept_token_count']==retained and cut['retained_tokens_sha256']==sha(canonical(raw[:retained])) and
                cut['decoded_retained_prefix']==decode(raw[:retained]),'Diagnostic boundary differs from original helper pass')
        positions=[];included=0
        for position,token in enumerate(raw):
            category=token_class(token,eos,pad,special)
            if category=='ordinary':continue
            classes[category]+=1;is_included=position<retained
            if is_included:included_classes[category]+=1;included+=1
            if len(positions)<DETAIL_POSITION_LIMIT:
                positions.append(dict(raw_position=position,category=category,
                    captured_pre_padding_position=raw_ok and position<record['raw_generated_token_count'],
                    inside_original_retained_prefix=is_included,
                    is_final_raw_position=position==len(raw)-1,
                    in_official_generation_config_eos=token in generation_eos,
                    matches_actual_native_stop_reason=type(stop) is int and stop==token,
                    trailing_eos_excluded=(position==len(raw)-1 and diagnostic['trailing_eos_excluded']),
                    special_symbol_sha256=sha(decode([token]).encode())))
        counts['records_with_special']+=int(bool(positions));counts['records_with_included_special']+=int(included>0)
        counts['stored_raw_token_count']+=len(raw);counts['observed_diagnostic_retained_token_count']+=retained
        if helper_error or positions:detail_candidates+=1
        if (helper_error or positions) and len(details)<DETAIL_RECORD_LIMIT:
            details.append(dict(stored_record_index=index,sampling_record_index=record_counter-1,rollout_ordinal=ordinal,
                row_id_sha256=sha(row_id.encode()),request_identity_sha256=request_key,
                padding_transport_row=padded,request_token_count=len(request),raw_generated_token_count=len(raw),
                native_finish_reason_class=record['finish_reason'] if record['finish_reason'] in ('length','stop','abort') else 'other',
                native_stop_reason_class='none' if record['stop_reason'] is None else
                    token_class(record['stop_reason'],eos,pad,special) if type(record['stop_reason']) is int else 'string_or_other',
                trailing_eos_has_native_stop_evidence=genuine_stop,
                native_stop_reason_in_official_generation_eos=config_stop,
                native_stop_reason_in_captured_explicit_stop_ids=type(stop) is int and stop in explicit_stop_set,
                native_finish_stop_and_final_raw_matches_stop=record['finish_reason']=='stop' and
                    type(stop) is int and bool(raw) and raw[-1]==stop,
                stored_raw_count_matches_lp=raw_ok,selected_lp_all_finite=finite,selected_lp_nonpositive=finite and all(x<=0 for x in lp),
                selected_lp_vector_sha256=sha(canonical(lp)),raw_tokens_sha256=sha(canonical(raw)),
                retained_tokens_sha256=sha(canonical(raw[:retained])),existing_helper_error=helper_error,
                diagnostic_projection=diagnostic,special_positions=positions,
                special_positions_truncated=max(0,sum(x in special for x in raw)-len(positions)),
                included_special_position_count=included,context_audit_accepted=False))
    require(record_counter>0,'Journal lacks real stored sampling records')
    summary=lambda values:dict(count=len(values),minimum=min(values),maximum=max(values),sum=sum(values))
    return dict(role='failed3294_private_special_token_cpu_diagnosis',run_id=run_id,slurm_job_id=job_id,
        source_snapshot_kind='current_disk_hashes_not_loaded_module_proof',logical_update=1,
        private_record_count=len(document['actual_sampling_records']),sampling_record_count=record_counter,
        unique_call_count=len(call_ids),unique_record_identity_count=len(request_keys),ordinal_counts=dict(ordinals),
        counts=dict(counts),existing_helper_error_counts=dict(errors),raw_special_token_class_counts=dict(classes),
        retained_special_token_class_counts=dict(included_classes),tokenizer_eos_equals_pad=eos==pad,
        official_special_id_count=len(special),lengths={name:summary(values) for name,values in lengths.items()},
        official_generation_config_eos_count=len(generation_eos),
        official_generation_config_eos_not_tokenizer_eos_count=len(generation_eos-{eos}),
        official_generation_config_eos_ids_sha256=sha(canonical(sorted(generation_eos))),
        engine_effective_stop_set_authenticated=False,
        engine_eos_scope='Official file values and captured stop reason only; not a live engine stop-set proof or changed exclusion rule',
        raw_count_or_lp_length_mismatch_count=raw_count_mismatch,selected_lp_nonfinite_or_typed_marker_count=lp_bad,
        selected_lp_positive_count=lp_positive,eos_pad_metadata_mismatch_count=metadata_bad,
        diagnostic_prefix_stable_count=prefix_stable,diagnostic_prefix_unstable_count=prefix_unstable,
        environment_row_count=environment_rows,environment_token_count=environment_tokens,
        stored_final_actor_mask_count=0,stored_sampling_tensor_mask_count=0,
        mask_observation_scope='Partial sampling journal has no final actor or worker tensor masks; do not synthesize them',
        details=details,details_record_limit=DETAIL_RECORD_LIMIT,
        detail_records_omitted=detail_candidates-len(details),
        accepted=False,context_verified=False,complete_outer=False,production_admission=False,
        healthy_checkpoint_accepted=False,physical_execution_authenticated=False,
        actual_tensor_transport_reverification_pending=True,
        scope='Every stored raw sampling record replayed with original raw-prefix helper; no full context/tensor/PG audit or guard bypass')


def authentic_sources(base,mapping):
    require(type(mapping) is dict and len(mapping)==353,'Missing original full353 source map')
    for name,digest in mapping.items():
        require(isinstance(name,str) and not Path(name).is_absolute() and '..' not in Path(name).parts,'Invalid original source path')
        readonly_bytes(base/'repo'/name,4*1024**2,expected_sha256=digest,
            empty_source_relative=name if name in EMPTY_SOURCE_FILES else None)
    repo=base/'repo';actual=set()
    for path in repo.rglob('*'):
        if {'.git','__pycache__','evidence'}.intersection(path.relative_to(repo).parts):continue
        require(not path.is_symlink(),'Original native source tree now contains symlink')
        name=str(path.relative_to(repo))
        if path.is_file() and (path.suffix in ('.py','.sh','.yaml','.yml','.json') or
            name in ('verl/version/version','verl/single_controller/version/version')):actual.add(name)
    extras=actual-set(mapping)
    require(set(mapping)<=actual and len(extras)==3 and
        len({str(Path(name).parent) for name in extras})==1 and
        {Path(name).name for name in extras}=={'config.yaml','hydra.yaml','overrides.yaml'} and
        all(re.fullmatch(r'outputs/\d{4}-\d\d-\d\d/\d\d-\d\d-\d\d/\.hydra/(config|hydra|overrides)\.yaml',name) for name in extras),
        'Source set differs from original353 plus actual three Hydra snapshots')


def evaluation_aggregate(base,mapping):
    relative='reproduction/common8/calibration_evaluation.py'
    raw,_=readonly_bytes(base/'repo'/relative,4*1024**2,expected_sha256=mapping[relative])
    names={'_require','_integer','_actions','aggregate'};constants={'QUESTION_COUNT','SOURCE_COUNTS','_ROW_FIELDS'}
    selected=[node for node in ast.parse(raw).body if
        (isinstance(node,ast.FunctionDef) and node.name in names) or
        (isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in constants for target in node.targets))]
    namespace=dict(math=math,Counter=Counter,Mapping=Mapping,Sequence=Sequence)
    exec(compile(ast.Module(body=selected,type_ignores=[]),relative,'exec'),namespace)
    return namespace['aggregate']


def before512_projection(value,run,validation,aggregate):
    require(value['run_id']==run['run_id'] and value['parent_production_run_id']==run['parent_production_run_id'] and
        value['slurm_job_id']==run['policy_calibration_slurm_job_id'] and type(value['step']) is int and value['step']==0 and
        value['plan_sha256']==run['policy_calibration_plan_sha256'] and
        value['validation_sha256']==validation['validation_sha256'] and
        value['phase_role']==run['policy_calibration_role'] and value['production_admission'] is False,
        'Initial512 original run/step/source binding differs')
    calculated=aggregate(value['rows'],validation['question_ids'],validation['data_sources'])
    require(all(canonical(value[key])==canonical(item) for key,item in calculated.items()),'Initial512 arithmetic differs')
    hidden={'rows','question_ids','data_sources'}
    result={key:item for key,item in calculated.items() if key not in hidden}
    result.update(exists=True,step=0,ordered_question_ids_sha256=sha(canonical(calculated['question_ids'])),
        ordered_sources_sha256=sha(canonical(calculated['data_sources'])),
        validation_sha256=validation['validation_sha256'],health_evaluated=False,accepted=False)
    return result


def before512_diagnosis(base,run,mapping):
    path=base/'logs/policy-calibration-evaluation-before.json'
    if not os.path.lexists(path):return dict(exists=False,health_evaluated=False,accepted=False)
    raw,receipt=readonly_bytes(path,4*1024**2);value=strict_json(raw)
    manifest_raw,manifest_receipt=readonly_bytes(base/'data/manifest.json',4*1024**2)
    from reproduction.common8.verify_data import MANIFEST_SHA,probe_question_id
    require(sha(manifest_raw)==MANIFEST_SHA==run['input_sha256']['manifest.json'],'Shared manifest bytes differ')
    record=strict_json(manifest_raw)['splits']['validation']
    require(record['path']=='validation.parquet' and type(record['rows']) is int and record['rows']==512 and
        record['sha256']==run['input_sha256']['validation.parquet'],'Shared actual512 metadata differs')
    parquet_raw,parquet_receipt=readonly_bytes(base/'data/validation.parquet',64*1024**2,expected_sha256=record['sha256'])
    import pyarrow.parquet as pq
    table=pq.read_table(io.BytesIO(parquet_raw),columns=['prompt','data_source']);del parquet_raw
    require(table.num_rows==512,'Shared512 actual rows differ')
    validation=dict(validation_sha256=record['sha256'],question_ids=[probe_question_id(row) for row in table['prompt'].to_pylist()],
        data_sources=table['data_source'].to_pylist());del table
    result=before512_projection(value,run,validation,evaluation_aggregate(base,mapping));del value
    result.update(original_file=receipt,original_shared_manifest=manifest_receipt,original_validation_file=parquet_receipt,
        raw_qa_published=False,model_or_adam_read=False)
    return result


def native_diagnosis(payload,reader):
    started=time.time();launch=payload['launch'];terminal=payload['terminal'];base=Path(launch['base'])
    require(started>=terminal['observation_finished_unix'],'CPU diagnostic predates original failed capture')
    route=reader.current_completed_terminal(terminal)  # Exact reader supports FAILED as well as COMPLETED.
    authentic_sources(base,launch['source_sha256'])
    run_raw,run_stat=readonly_bytes(base/'logs/run-manifest.json',4*1024**2)
    run=strict_json(run_raw)
    require(run['run_id']==RUN_ID and run['parent_production_run_id']==PARENT_RUN_ID and
        run['policy_calibration_slurm_job_id']==launch['job_id']=='3294' and run['source_commit']==SOURCE_COMMIT and
        run['source_sha256']==launch['source_sha256'] and run['model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1',
        'Original failed run/source/official revision differs')
    journal=base/'artifacts/behavior-traces/outer-1.json'
    raw,journal_stat=readonly_bytes(journal,MAX_JOURNAL_BYTES,private=True)
    document=strict_json(raw);del raw
    model=Path(run['initial_model']);require(model.resolve()==model and model.is_dir(),'Official tokenizer root not canonical')
    tokenizer_receipts={};generation_config={}
    for name in TOKENIZER_FILES:
        content,receipt=readonly_bytes(model/name,32*1024**2,expected_sha256=run['initial_model_sha256'][name])
        tokenizer_receipts[name]=receipt
        if name=='generation_config.json':generation_config=strict_json(content)
        del content
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
    sys.path.insert(0,str(base/'repo'))
    from reproduction.common8 import behavior_trace as helper
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
    torch_module=sys.modules.get('torch')
    initialized=bool(torch_module and torch_module.cuda.is_initialized())
    require(not initialized,'Readonly tokenizer CPU diagnostic initialized CUDA')
    result=journal_diagnosis(document,tokenizer,helper,run_id=run['run_id'],source_sha256=run['source_sha256'],
        job_id=launch['job_id'],generation_config=generation_config)
    del document
    result['initial_before512']=before512_diagnosis(base,run,launch['source_sha256'])
    # File snapshots, not a GPU/tokenizer loaded-module proof. No weights opened.
    current=journal.stat();require((current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns)==
        (journal_stat['device'],journal_stat['inode'],journal_stat['size_bytes'],journal_stat['mtime_ns'],journal_stat['ctime_ns']),
        'Original private journal changed during CPU replay')
    result.update(original_journal=journal_stat,original_manifest=run_stat,official_tokenizer_file_snapshots=tokenizer_receipts,
        source_commit=SOURCE_COMMIT,source_map_sha256=sha(canonical(launch['source_sha256'])),
        original_failed_terminal_sha256=terminal['observation_sha256'],current_failed_terminal_route=route,
        checked_unix=started,observation_finished_unix=time.time(),cuda_visible_devices_disabled=True,cuda_initialized=initialized,
        model_or_adam_tensors_read=False,raw_text_or_ids_or_lp_values_published=False)
    require(len(canonical(result))<=MAX_RETURN_BYTES,'Diagnostic scalar response exceeds publication bound')
    return result


def frozen_reader():
    path=HERE/'submit_policy_confirmation_v5_native.py';require(sha(path.read_bytes())==READER_SHA256,'Frozen terminal reader changed')
    spec=importlib.util.spec_from_file_location('_failed3294_reader',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def input_contract(artifacts):
    require(type(artifacts) is dict and set(artifacts)=={'launch','birth','terminal'},'All three explicit original failure inputs required')
    originals={};hashes=dict(launch=LAUNCH_SHA256,birth=BIRTH_SHA256,terminal=TERMINAL_SHA256)
    for label,detail in artifacts.items():
        require(type(detail) is dict and set(detail)=={'path','sha256'} and detail['sha256']==hashes[label],
            'Explicit original failed3294 SHA required: '+label)
        originals[label],_=readonly_bytes(Path(detail['path']),4*1024**2,expected_sha256=hashes[label])
    reader=frozen_reader();launch=strict_json(originals['launch'])
    require(launch['source_commit']==SOURCE_COMMIT and sha(canonical(launch['source_sha256']))==SOURCE_MAP_SHA256,'Original sourcee670 differs')
    terminal=reader.read_terminal_observation(originals['terminal'],expected_sha256=TERMINAL_SHA256,
        launch_bytes=originals['launch'],expected_launch_sha256=LAUNCH_SHA256,expected_run_id=RUN_ID,
        expected_parent_run_id=PARENT_RUN_ID,expected_state='FAILED',expected_exit_code='1:0',
        birth_observation_bytes=originals['birth'],expected_birth_sha256=BIRTH_SHA256)
    return reader,launch,terminal


def remote_code(reader):
    prefix='import ast,copy,datetime,hashlib,io,json,math,os,re,stat,subprocess,sys,time\nfrom pathlib import Path\nfrom types import SimpleNamespace\nfrom collections import Counter\nfrom collections.abc import Mapping,Sequence\n'
    reader_body='\n'.join(inspect.getsource(getattr(reader,name)) for name in
        ('canonical','sha','require','strict_json','_digest','_number','_job_fields','verify_terminal_current','current_completed_terminal'))
    reader_body+='\ncommon=SimpleNamespace(require=require,sha=sha,canonical=canonical,strict_json=strict_json)\n'
    for name in ('INVALID_JOB','BIRTH_FIELDS'):reader_body+=name+'='+repr(getattr(reader,name))+'\n'
    code=prefix+'reader_ns={};exec('+repr(prefix+reader_body)+',reader_ns);reader=SimpleNamespace(**reader_ns)\n'
    for name in ('MAX_JOURNAL_BYTES','MAX_RETURN_BYTES','DETAIL_RECORD_LIMIT','DETAIL_POSITION_LIMIT',
        'TOKENIZER_FILES','PRIVATE_FIELDS','RECORD_FIELDS','EVENT_FIELDS','RUN_ID','PARENT_RUN_ID','SOURCE_COMMIT'):
        code+=name+'='+repr(globals()[name])+'\n'
    for name in ('EMPTY_SOURCE_FILES','EMPTY_SOURCE_SHA256'):code+=name+'='+repr(globals()[name])+'\n'
    code+='\n'.join(inspect.getsource(globals()[name]) for name in
        ('require','sha','canonical','strict_json','stable','readonly_bytes','safe_error','diagnostic_prefix','token_class',
        'generation_eos_ids','journal_diagnosis','authentic_sources','evaluation_aggregate','before512_projection',
        'before512_diagnosis','native_diagnosis'))
    code+='\np=json.load(sys.stdin);print(json.dumps(native_diagnosis(p,reader),allow_nan=False))\n'
    compile(code,'<readonly-failed3294-special-diagnostic>','exec');return code


def execute_once(artifacts,output,*,execute=False,rpc=None):
    require(execute is True,'Future explicit execute required; no automatic diagnostic RPC')
    reader,launch,terminal=input_contract(artifacts)
    output=Path(output);require(output.is_absolute() and output.parent.resolve()==output.parent and not output.exists() and
        not output.is_symlink(),'New exclusive local scalar output required')
    code=remote_code(reader);payload=dict(launch=launch,terminal=terminal)
    if rpc is None:
        sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
        def rpc(code,payload):
            command=shlex.quote(str(Path(launch['base'])/'env/bin/python'))+' -B -c '+shlex.quote(code)
            result=subprocess.run(control.route('h100-8',command),input=json.dumps(payload),capture_output=True,text=True,timeout=300)
            require(result.returncode==0,'Readonly CPU diagnosis failed; original evidence unchanged: '+sha(result.stderr.encode()))
            require(len(result.stdout.encode())<=MAX_RETURN_BYTES,'Unexpected diagnostic response size')
            return strict_json(result.stdout)
    result=rpc(code,payload)
    require(result['run_id']==RUN_ID and result['slurm_job_id']=='3294' and
        result['original_failed_terminal_sha256']==TERMINAL_SHA256 and result['source_commit']==SOURCE_COMMIT and
        all(result[name] is False for name in ('accepted','context_verified','complete_outer','production_admission',
            'healthy_checkpoint_accepted','physical_execution_authenticated','model_or_adam_tensors_read',
            'raw_text_or_ids_or_lp_values_published')),'Diagnostic scalar identity/boundaries differ')
    raw=canonical(result);require(len(raw)<=MAX_RETURN_BYTES,'Readonly scalar output exceeds bound')
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    directory=os.open(output.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
    return dict(output=str(output),sha256=sha(raw),remote_executed=True,accepted=False,production_admission=False,healthy_checkpoint_accepted=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for label in ('launch','birth','terminal'):
        parser.add_argument('--'+label,type=Path,required=True);parser.add_argument('--'+label+'-sha256',required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args();artifacts={label:dict(path=str(getattr(args,label)),sha256=getattr(args,label+'_sha256')) for label in ('launch','birth','terminal')}
    if args.execute:
        require(args.output is not None,'Future execute requires exclusive scalar output')
        result=execute_once(artifacts,args.output,execute=True)
    else:
        reader,_,_=input_contract(artifacts);remote_code(reader)
        result=dict(role='failed3294_diagnostic_local_structure_check',remote_executed=False,accepted=False,production_admission=False)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
