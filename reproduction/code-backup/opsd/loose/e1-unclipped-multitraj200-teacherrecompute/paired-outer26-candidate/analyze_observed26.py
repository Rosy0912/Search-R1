"""Local observed-outer adapter; immutable first5 scoring kernel stays unchanged."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import sys
import tarfile
import time
import types

KERNEL_SHA='fa1a3dd12d762e8d18e2e577f9232642f8232d8fc03001ab0ffe406a4be6920d'
OUTER1_REPORT_SHA='371d6f99f2c4751d4f2de620c3ae88c4246445ae710399b750a4a170fc3f6687'


def load(path,pin,name):
    path=Path(path).absolute()
    if path.resolve(strict=True)!=path:raise ValueError('Helper symlink rejected')
    with path.open('rb') as handle:
        before=os.fstat(handle.fileno());raw=handle.read();after=os.fstat(handle.fileno())
    fields=('st_dev','st_ino','st_uid','st_size','st_mtime_ns','st_ctime_ns')
    if (hashlib.sha256(raw).hexdigest()!=pin or
        any(tuple(getattr(s,f) for f in fields)!=tuple(getattr(before,f) for f in fields) for s in (after,path.stat()))):
        raise ValueError('Helper exact bytes/physical identity changed')
    module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    exec(compile(raw,str(path),'exec'),module.__dict__);return module


def extract_once(kernel,collector,gzip_path,output,wait_record,wait_sha,collector_sha):
    """Preserve partial originals on failure; no normalized payload rewriting."""
    wait,wrec=kernel.json_member(wait_record,wait_sha)
    kernel.require(wait['kind']=='ACTUAL_READ_ONLY_OUTER26_COLLECTION_WAIT' and wait.get('waited') is True and type(wait['returncode']) is int
        and wait['returncode']==0 and wait['runtime_sha256']==collector_sha
        and wait['authority_sha256']==collector.AUTHORITY_SHA and wait['remote_host']=='h100-10','Actual collection wait is not qualified')
    output=Path(output).absolute();output.mkdir(exist_ok=False)
    for directory in (output.parent,output):
        fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    compressed,crec=kernel.read(gzip_path,wait['compressed_stdout_sha256'])
    kernel.require(len(compressed)<=collector.MAX_RAW,'Compressed envelope exceeds1GiB')
    import io
    import gzip
    # Consume through actual gzip EOF/CRC, not merely tar's logical end marker.
    uncompressed_bytes=0
    with gzip.GzipFile(fileobj=io.BytesIO(compressed),mode='rb') as envelope:
        while True:
            block=envelope.read(8*1024**2)
            if not block:break
            uncompressed_bytes+=len(block)
            kernel.require(uncompressed_bytes<=collector.MAX_RAW+4*1024**2,'Gzip raw size exceedslimit')
    seen=set();records={};total=0
    with tarfile.open(fileobj=io.BytesIO(compressed),mode='r:gz') as archive:
        for entry in archive:
            kernel.require(entry.name in set(collector.NAMES)|{'observed-inventory.json'} and entry.name not in seen
                and entry.isfile() and entry.mode==0o444 and not entry.pax_headers
                and 0<entry.size<=collector.MAX_RAW,'Unknown/aliased/unsafe tar member')
            kernel.require(entry.name!='observed-inventory.json' or len(seen)==20,'Observed inventory must be last')
            total+=entry.size;kernel.require(total<=collector.MAX_RAW+2*1024**2,'Uncompressed envelope exceedslimit')
            source=archive.extractfile(entry);digest=hashlib.sha256();size=0
            with os.fdopen(os.open(output/entry.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444),'wb') as target:
                while True:
                    block=source.read(8*1024**2)
                    if not block:break
                    target.write(block);digest.update(block);size+=len(block)
                target.flush();os.fsync(target.fileno())
            kernel.require(size==entry.size,'Partial copied tar member');seen.add(entry.name)
            records[entry.name]=dict(sha256=digest.hexdigest(),bytes=size)
    kernel.require(seen==set(collector.NAMES)|{'observed-inventory.json'},'Exact21 tar payload members missing')
    inventory,irec=kernel.json_member(output/'observed-inventory.json')
    validate_inventory(kernel,collector,inventory)
    for name in collector.NAMES:
        kernel.require(records[name]['sha256']==inventory['members'][name]['sha256']
            and records[name]['bytes']==inventory['members'][name]['bytes'],'Extracted original SHA/bytes differs')
    kernel.recheck(crec)
    kernel.publish(output/'collection-copy-qualification.json',kernel.canonical(dict(kind='LOCAL_ORIGINAL_BYTE_COPY_ONLY',
        compressed_original=crec,actual_wait_original=wrec,observed_inventory=irec,members=records))+b'\n')
    return output


def validate_inventory(k,c,inventory):
    k.require(type(inventory['version']) is int and inventory['version']==1
        and inventory['kind']=='OBSERVED_COMPLETED_OUTER26_ARCHIVE_INVENTORY_NOT_TRAINING_CHECKPOINT_MANIFEST', 'Not observed26 inventory')
    c.identity_ok(dict(identity=inventory['identity'],student_count=2560,teacher_count=512,teacher_diagnostic_answers_used_as_targets=False))
    k.require(c.sha(c.canonical(inventory['authority']))==c.AUTHORITY_SHA and set(inventory['members'])==set(c.NAMES)
        and type(inventory['raw_bytes']) is int and 0<inventory['raw_bytes']<=c.MAX_RAW,'Observed inventory authority/roles/cap differs')
    k.require(sum(r['bytes'] for r in inventory['members'].values())==inventory['raw_bytes'],'Observed byte total differs')
    for name,record in inventory['members'].items():
        k.require(record['origin_path']==str(c.DIR/name) and type(record['bytes']) is int and record['bytes']>0
            and type(record['uid']) is int and record['uid']==0 and type(record['mode']) is int and record['mode']==0o444,'Observed original member identity differs')
    k.require(inventory['members']['paired-manifest.json']['sha256']==c.PAIR_SHA,'Observed pair pin differs')
    for context in (inventory['original_context_before'],inventory['original_context_after']):
        k.require(context['driver']['pid']==c.DRIVER[0] and context['driver']['start_ticks']==c.DRIVER[1]
            and [(r['pid'],r['start_ticks']) for r in context['workers']]==c.WORKERS
            and context['git']['head']['exit_code']==0 and context['git']['head']['stdout'].strip()==c.SOURCE
            and context['git']['clean']['exit_code']==0 and context['git']['clean']['stdout']=='','Observed process/source context differs')


def add_length_and_macro_metrics(k,cohorts,students):
    for source,cohort in cohorts.items():
        rows=students if source=='all' else [r for r in students if r['data_source']==source]
        cohort['student']['raw_sampled_tokens_mean']=k.mean(rows,'raw_sampled_token_count')
        cohort['student']['recorded_executor_legal_action_macro_mean']=sum(r['recorded_legal_actions']/r['tool_turn_count'] for r in rows)/len(rows)
    return cohorts


def validate_reward_and_macro(k,cohorts,authority):
    metrics=authority['outer26_log_metrics'];student=cohorts['all']['student']
    for field,key,tolerance in [('em','critic/rewards/mean',1e-7),('em','critic/score/mean',1e-7),
        ('recorded_executor_legal_action_macro_mean','env/ratio_of_valid_action',1e-10),('search_calls_mean','env/number_of_valid_search',1e-10),
        ('recorded_executor_done_rate','env/finish_ratio',1e-10)]:
        value=metrics[key];k.require(type(value) in (int,float) and math.isfinite(value)
            and abs(student[field]-value)<=tolerance,'Actual raw archive vs reward/executor log differs: '+key)


def analyze(k,c,raw26,inventory_sha,raw5,repo,outer1_path,output):
    started=time.monotonic();cpu=time.process_time();output=Path(output).absolute();output.mkdir(exist_ok=False)
    raw26=Path(raw26).absolute();raw5=Path(raw5).absolute()
    inventory,irec=k.json_member(raw26/'observed-inventory.json',inventory_sha);validate_inventory(k,c,inventory)
    old,oldrec=k.json_member(outer1_path,OUTER1_REPORT_SHA)
    k.require(old['identity']['outer_step']==1 and old['identity']['source_commit']==c.SOURCE
        and old['identity']['config_sha256']==c.CONFIG,'Previously qualified outer1 report differs')
    refs={'observed_inventory':irec,'outer1_report':oldrec}
    def member(name):
        value,record=k.json_member(raw26/name,inventory['members'][name]['sha256']);refs[name]=record;return value
    pair=member('paired-manifest.json');c.identity_ok(pair);identity=pair['identity']
    expected_identity=dict(old['identity'],outer_step=26,directory=str(c.DIR))
    k.require(identity==inventory['identity']==expected_identity,'Observed pair/official teacher identity differs from actual source/config cohort')
    question_member=member('questions.json');questions=question_member['questions'];student_meta=member('student-manifest.json')
    expected=c.expected_members(pair,student_meta)
    k.require(len(questions)==512 and len({q['question_uid'] for q in questions})==512
        and all(type(q['question_position']) is int and q['question_position']==i for i,q in enumerate(questions))
        and question_member['identity']==identity and student_meta['identity']==identity
        and student_meta['recorded_before_balancing'] is True and student_meta['response_count_per_question']==5,'Observed512x5 grouping differs')
    for name,record in expected.items():
        k.require(record['sha256']==inventory['members'][name]['sha256'] and record['bytes']==inventory['members'][name]['bytes'], 'Observed nested member references differ')
    model_manifest,mrec=k.json_member(raw5/'full_state/global_step_5/manifest.json',k.ACCEPTED5);refs['accepted5_manifest']=mrec
    files=model_manifest['files'];hf=raw5/'checkpoints/actor/global_step_5'
    for name in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json'):
        _,record=k.read(hf/name,files['actor/'+name]['sha256']);refs['tokenizer_'+name]=record
    model,record=k.json_member(raw5/'bindings/model_manifest.json',files['model_manifest']['sha256']);refs['model_manifest']=record
    k.require(model['revision']==k.REVISION,'Official teacher revision differs')
    official={r['file']:r for r in model['files']};k.require(refs['tokenizer_vocab.json']['sha256']==official['vocab.json']['sha256'],'Official vocabulary differs')
    merges,_=k.read(hf/'merges.txt',refs['tokenizer_merges.txt']['sha256']);k.official_merges_equivalence(merges,official['merges.txt']['sha256'])
    modules,source_records=k.source_modules(repo);refs.update({'source_'+name:r for name,r in source_records.items()})
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(hf,local_files_only=True,trust_remote_code=False)
    teachers={}
    for rank in range(8):
        meta=member(f'teacher-rank-{rank}.manifest.json')
        k.require(meta['identity']==identity and type(meta['rank']) is int and meta['rank']==rank and meta['pid']==c.WORKERS[rank][0]
            and meta['uid']==0 and meta['hostname']=='h100-10' and meta['execution_device']=='cuda'
            and meta['question_count']==64 and meta['diagnostic_answers_used_as_targets'] is False
            and meta['rng_restored'] is True and meta['model_state_restored'] is True,'Teacher diagnostic execution differs')
        name=f'teacher-rank-{rank}.jsonl';raw,refs[name]=k.read(raw26/name,expected[name]['sha256'])
        k.require(raw.endswith(b'\n') and len(raw.splitlines())==64,'Teacher raw EOF/rows differ')
        for line in raw.splitlines():
            row=k.parse(line);pos=row['question_position'];k.require(type(pos) is int and 0<=pos<512 and pos not in teachers,'Teacher cohort differs')
            q=questions[pos];ids=row['output_token_ids'];k.int_list(ids,'Teacher output IDs')
            k.require(row['question_uid']==q['question_uid'] and row['question']==q['question'] and row['data_source']==q['data_source']
                and row['accepted_answers']==q['accepted_answers'] and row['generated_by']=='fixed_official_teacher.model.generate'
                and row['privileged_training_answers'] is True and row['is_full_teacher_search_trajectory'] is False
                and row['not_used_as_student_or_distillation_or_evaluation_input'] is True and row['tool_turns']==[]
                and row['decoding']==dict(do_sample=False,max_new_tokens=128,num_beams=1,use_cache=False,temperature_applies=False,independent_of_student_history=True)
                and row['output_attention_mask']==[1]*len(ids) and row['diagnostic_loss_mask']==[0]*len(ids)
                and row['output_text']==tokenizer.decode(ids,skip_special_tokens=False),'Teacher privileged raw scope differs')
            teachers[pos]=dict(**k.answer_metrics([row['output_text']],q['accepted_answers'],modules),output_tokens=len(ids),rank=rank)
    k.require(len(teachers)==512,'Teacher512 incomplete')
    students=[];path,before,handle=k.open_physical(raw26/'student.jsonl');digest=hashlib.sha256();size=0
    with handle:
        for i,line in enumerate(handle):
            k.require(line.endswith(b'\n') and i<2560,'Student raw EOF/rows differ');digest.update(line);size+=len(line)
            row=k.parse(line);pos,ri=divmod(i,5);q=questions[pos]
            k.require(type(row['question_position']) is int and row['question_position']==pos and type(row['response_index']) is int
                and row['response_index']==ri and row['question_uid']==q['question_uid']
                and row['training_targets_are_teacher_q_on_student_history'] is True,'Student raw5 grouping differs')
            k.require(row['tensor_sha256']==k.sha(k.canonical(row['tensors'])),'Tensor digest differs')
            mask=k.authored_mask(row['tensors']);k.validate_turns(row,q,tokenizer);spans=modules['protocol'].decode_model_spans(tokenizer,row['tensors']['responses'],mask)
            turns=row['tool_turns'];students.append(dict(**k.answer_metrics(spans,q['accepted_answers'],modules),question_uid=q['question_uid'],question_position=pos,
                response_index=ri,trajectory_uid=row['trajectory_uid'],data_source=q['data_source'],tool_turn_count=len(turns),
                strict_legal_actions=sum(modules['protocol'].parse_model_action(t['text'],strict_prefix=True) is not None for t in turns),
                recorded_legal_actions=sum(t['valid_action'] for t in turns),recorded_valid_search_calls=sum(t['valid_search'] for t in turns),
                executor_done=turns[-1]['done'],generated_model_tokens=sum(mask),raw_sampled_token_count=sum(sum(t['raw_response_attention_mask']) for t in turns)))
        k.require(len(students)==2560 and len({s['trajectory_uid'] for s in students})==2560,'Student2560 incomplete/repeated')
        refs['student.jsonl']=k.finish_read(path,before,handle,digest.hexdigest(),size,expected['student.jsonl']['sha256'])
    summaries=[k.question_summary(q,students[i*5:(i+1)*5],teachers[i]) for i,q in enumerate(questions)]
    cohorts={'all':k.aggregate(summaries,students)}
    for source in sorted({q['data_source'] for q in questions}):cohorts[source]=k.aggregate([q for q in summaries if q['data_source']==source],[s for s in students if s['data_source']==source])
    outputs={'students':k.publish(output/'students-summary.jsonl',b''.join(k.canonical(s)+b'\n' for s in students)),
             'questions':k.publish(output/'questions-summary.jsonl',b''.join(k.canonical(q)+b'\n' for q in summaries))}
    add_length_and_macro_metrics(k,cohorts,students);validate_reward_and_macro(k,cohorts,inventory['authority'])
    old_students_raw,old_students_record=k.read(old['outputs']['students']['physical_path'],old['outputs']['students']['sha256'])
    old_students=[k.parse(line) for line in old_students_raw.splitlines()];k.require(len(old_students)==2560,'Outer1 previously qualified summary size differs')
    refs['outer1_students_summary']=old_students_record;add_length_and_macro_metrics(k,old['cohorts'],old_students)
    for record in refs.values():k.recheck(record)
    import torch
    k.require(not torch.cuda.is_initialized(),'Offline parser initialized CUDA')
    report=dict(version=1,kind='OBSERVED3304_OUTER26_CPU_BEHAVIOR_DIAGNOSTICS_NOT_CHECKPOINT_ACCEPTANCE',identity=identity,
        inputs=refs,outputs=outputs,cohorts=cohorts,comparison=dict(scope='DIFFERENT_TRAINING_COHORT_OBSERVATION_NOT_CAUSAL_OR_FAIR_PAIRED_EFFECT',
            outer1=old['cohorts'],outer26=cohorts),
        token_means=dict(retained_model_span=k.mean(students,'generated_model_tokens'),raw_sampled=k.mean(students,'raw_sampled_token_count')),
        execution=dict(pid=os.getpid(),uid=os.getuid(),hostname=socket.gethostname(),wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,
            cuda_initialized=False,max_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        limitations=old['limitations']+['Outer1 and26 use different training questions; source mixing and stochastic samples can change metrics without learning effects.'])
    k.publish(output/'report.json',k.canonical(report)+b'\n');print(json.dumps(dict(output=str(output),cohort=cohorts['all']),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--kernel',required=True);parser.add_argument('--collector',required=True);parser.add_argument('--collector-sha256',required=True)
    sub=parser.add_subparsers(dest='mode',required=True)
    extract=sub.add_parser('extract');extract.add_argument('--gzip',required=True);extract.add_argument('--output',required=True);extract.add_argument('--wait-record',required=True);extract.add_argument('--wait-sha256',required=True)
    run=sub.add_parser('analyze');run.add_argument('--raw26',required=True);run.add_argument('--inventory-sha256',required=True);run.add_argument('--raw5',required=True);run.add_argument('--source-repo',required=True);run.add_argument('--outer1-report',required=True);run.add_argument('--output',required=True)
    args=parser.parse_args();kernel=load(args.kernel,KERNEL_SHA,'outer1_exact_kernel');collector=load(args.collector,args.collector_sha256,'outer26_exact_collector')
    if args.mode=='extract':extract_once(kernel,collector,args.gzip,args.output,args.wait_record,args.wait_sha256,args.collector_sha256)
    else:analyze(kernel,collector,args.raw26,args.inventory_sha256,args.raw5,args.source_repo,args.outer1_report,args.output)
