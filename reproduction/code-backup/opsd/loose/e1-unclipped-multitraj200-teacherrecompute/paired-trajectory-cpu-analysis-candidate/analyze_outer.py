#!/usr/bin/env python3
"""Offline one-outer diagnostics. No model loading, inference, or remote calls."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import stat
import subprocess
import sys
import time
import types

SOURCE = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
ACCEPTED5 = '30b38d6d96abd56ff956baf8a4ba954319c62049aff8263e0c9700037e2c8577'
CONFIG = '0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990'
COMMON = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
RUN = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/runs/formal-3304'
ATTRS = ('st_dev','st_ino','st_uid','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def signature(value):
    return tuple(getattr(value, attr) for attr in ATTRS)


def open_physical(path):
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path, 'Symlink path rejected')
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_nlink == 1,
            'Local original type/owner/link count differs')
    handle = os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb')
    require(signature(os.fstat(handle.fileno())) == signature(before), 'Original changed before opening')
    return path, before, handle


def finish_read(path, before, handle, pin, size, expected=None):
    require(signature(os.fstat(handle.fileno())) == signature(before)
            == signature(path.stat()) and size == before.st_size, 'Original changed during read')
    require(path.resolve(strict=True) == path, 'Original path replaced')
    if expected is not None:
        require(pin == expected, 'Original full SHA differs')
    return dict(physical_path=str(path), sha256=pin, bytes=size, device=before.st_dev,
                inode=before.st_ino, local_copy_uid=before.st_uid, stable_read=True)


def read(path, expected=None):
    path, before, handle = open_physical(path)
    with handle:
        raw = handle.read()
        record = finish_read(path, before, handle, sha(raw), len(raw), expected)
    return raw, record


def json_member(path, expected=None):
    raw, record = read(path, expected)
    return parse(raw), record


def recheck(record):
    path,before,handle=open_physical(record['physical_path']);hashed=hashlib.sha256();size=0
    with handle:
        while True:
            block=handle.read(8*1024*1024)
            if not block:break
            hashed.update(block);size+=len(block)
        after=finish_read(path,before,handle,hashed.hexdigest(),size,record['sha256'])
    require(all(after[k]==record[k] for k in ('device','inode','local_copy_uid','bytes')), 'Original changed after analysis')


def source_modules(repo):
    """Execute exact whole source bytes, never stale .pyc or rewritten functions."""
    repo = Path(repo).absolute()
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip() == SOURCE,
            'Frozen source HEAD differs')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=repo), 'Frozen source is dirty')
    sys.path.insert(0, str(repo))
    modules, records = {}, {}
    for name, relative in [('protocol','search_r1/qa_protocol.py'),
                           ('em','verl/utils/reward_score/qa_em.py'),
                           ('bapo','search_r1/baselines/bapo.py')]:
        raw, record = read(repo / relative)
        require(raw == subprocess.check_output(['git','show', SOURCE+':'+relative],cwd=repo),
                'Source working bytes differ from fixed commit')
        full_name = 'search_r1.qa_protocol' if name == 'protocol' else 'offline_'+name
        module = types.ModuleType(full_name)
        module.__file__ = str(repo / relative)
        sys.modules[full_name] = module
        exec(compile(raw, module.__file__, 'exec'), module.__dict__)
        modules[name], records[name] = module, record
    import torch
    require(not torch.cuda.is_initialized(), 'Offline parser initialized CUDA')
    return modules, records


def bool_field(value, name):
    require(type(value) is bool, name+' is not an exact boolean')
    return value


def int_list(values, name, binary=False):
    require(type(values) is list and all(type(v) is int and (v in (0,1) if binary else v >= 0)
                                       for v in values), name+' differs')


def authored_mask(tensors):
    response = tensors['responses']
    int_list(response, 'responses')
    width = len(response)
    require(width > 0 and tensors['input_ids'] == tensors['prompts'] + response, 'Sequence layout differs')
    total = len(tensors['input_ids'])
    for name in ('attention_mask','info_mask'):
        int_list(tensors[name], name, True)
        require(len(tensors[name]) == total, 'Full sequence mask width differs')
    int_list(tensors['loss_mask'], 'loss_mask', True)
    require(len(tensors['loss_mask']) == width, 'Loss mask width differs')
    attention, info = tensors['attention_mask'][-width:], tensors['info_mask'][-width:]
    require(tensors['loss_mask'] == info and all(a >= b for a,b in zip(attention, info)),
            'Stored policy mask differs or includes padding')
    return [a & b for a,b in zip(attention, info)]


def validate_turns(row, question, tokenizer):
    """Independent reconstruction of the frozen writer's actual retained history."""
    rolling = question['original_student_input_ids']
    int_list(rolling, 'Original input IDs')
    response, mask, steps = [], [], []
    turns = row['tool_turns']
    require(type(turns) is list and 1 <= len(turns) <= 4, 'Tool budget differs')
    pad = tokenizer.pad_token_id
    for index, turn in enumerate(turns):
        require(type(turn['turn_index']) is int and turn['turn_index'] == index
                and turn['generation_context_token_ids'] == rolling, 'Actual history differs')
        for name in ('raw_response_token_ids','postprocessed_action_token_ids',
                     'observation_token_ids_before_truncation','retained_observation_token_ids'):
            int_list(turn[name], name)
        raw = turn['raw_response_token_ids']
        raw_mask = turn['raw_response_attention_mask']
        int_list(raw_mask, 'Raw attention', True)
        require(0 < len(raw) <= 500 and len(raw) == len(raw_mask), 'Action token budget differs')
        require(turn['raw_response_text'] == tokenizer.decode(raw, skip_special_tokens=False),
                'Raw response text differs from tokens')
        processed = tokenizer.decode(raw, skip_special_tokens=True)
        processed = (processed.split('</search>')[0]+'</search>' if '</search>' in processed else
                     processed.split('</answer>')[0]+'</answer>' if '</answer>' in processed else processed)
        action = turn['postprocessed_action_token_ids']
        full_obs = turn['observation_token_ids_before_truncation']
        obs = turn['retained_observation_token_ids']
        require(processed == turn['text'] and action == [v for v in tokenizer.encode(processed, add_special_tokens=False) if v != pad],
                'Postprocessing differs')
        require(full_obs == tokenizer.encode(turn['observation_text_before_truncation'], add_special_tokens=False)
                and obs == [v for v in full_obs[:500] if v != pad]
                and type(turn['action_tokens']) is int and turn['action_tokens'] == len(action)
                and type(turn['observation_tokens']) is int and turn['observation_tokens'] == len(obs), 'Observation provenance differs')
        for name in ('done','normal_search_round','scoring_context_matches_generation'):
            bool_field(turn[name], name)
        for name in ('valid_action','valid_search'):
            require(type(turn[name]) is int and turn[name] in (0,1), name+' is not an executor binary integer')
        require(turn['scoring_context_matches_generation'], 'Scoring context differs')
        response = (response+action+obs)[:4096]
        mask = (mask+[1]*len(action)+[0]*len(obs))[:4096]
        steps = (steps+[index]*len(action)+[-1]*len(obs))[:4096]
        rolling = (rolling+action+obs)[-4096:]
    tensors = row['tensors']
    require(tensors['responses'][:len(response)] == response and tensors['loss_mask'][:len(mask)] == mask
            and tensors['step_ids'][:len(steps)] == steps
            and tensors['loss_mask'][len(mask):] == [0]*(len(tensors['responses'])-len(mask))
            and tensors['step_ids'][len(steps):] == [-1]*(len(tensors['responses'])-len(steps)),
            'Stored response/masks/steps differ from actual turns')
    prompt = question['original_student_input_ids'][-2048:]
    require(len(tensors['prompts']) <= 2048 and tensors['prompts'][-len(prompt):] == prompt
            and tensors['prompts'][:-len(prompt)] == [pad]*(len(tensors['prompts'])-len(prompt))
            and tensors['responses'][len(response):] == [pad]*(len(tensors['responses'])-len(response)), 'Padding differs')


def answer_metrics(spans, golds, modules):
    answer = modules['protocol'].extract_final_answer(spans)
    em = modules['em'].em_check(answer, golds) if answer is not None else 0
    f1 = max((modules['bapo'].token_f1(answer, value) for value in golds), default=0.) if answer is not None else 0.
    legal = answer is not None and modules['bapo'].legal_action_records([{'text':s} for s in spans])
    return dict(answer=answer, normalized_answer=modules['em'].normalize_answer(answer) if answer is not None else None,
                em=em, f1=f1, completed_terminal_answer=answer is not None, legal_complete_trajectory=legal)


def question_summary(question, students, teacher):
    require(len(students) == 5 and [s['response_index'] for s in students] == list(range(5)), 'Five-response group differs')
    complete = [s['normalized_answer'] for s in students if s['completed_terminal_answer']]
    pairs = [(students[i],students[j]) for i in range(5) for j in range(i+1,5)]
    both_complete = [(a,b) for a,b in pairs if a['completed_terminal_answer'] and b['completed_terminal_answer']]
    same = sum(a['normalized_answer'] == b['normalized_answer'] for a,b in both_complete)
    agreed = [s for s in students if s['completed_terminal_answer'] and teacher['completed_terminal_answer']
              and s['normalized_answer'] == teacher['normalized_answer']]
    return dict(question_uid=question['question_uid'], question_position=question['question_position'],
                data_source=question['data_source'], dataset_row_position=question['dataset_row_position'],
                student_correct_count=sum(s['em'] for s in students), student_any_correct=any(s['em'] for s in students),
                student_majority_correct=sum(s['em'] for s in students) >= 3,
                student_mean_em=sum(s['em'] for s in students)/5, student_max_f1=max(s['f1'] for s in students),
                student_complete_count=len(complete), distinct_completed_answers=len(set(complete)),
                consensus_max_count=max(Counter(complete).values(), default=0),
                all_five_complete_and_same=len(complete) == 5 and len(set(complete)) == 1,
                same_completed_pair_count=same, complete_pair_count=len(both_complete), total_pair_count=10,
                teacher=teacher, student_teacher_agreement_count=len(agreed), student_any_teacher_agreement=bool(agreed))


def mean(rows, key):
    return sum(row[key] for row in rows)/len(rows) if rows else None


def official_merges_equivalence(saved, official_pin):
    """HF save adds exactly a BPE format header to the official merge body."""
    if sha(saved)==official_pin:
        return 'BYTE_EXACT_OFFICIAL'
    header=b'#version: 0.2\n'
    require(saved.startswith(header) and sha(saved[len(header):])==official_pin,
            'Official merge body differs beyond the known HF serialization header')
    return 'EXACT_OFFICIAL_BODY_AFTER_SINGLE_HF_BPE_VERSION_HEADER'


def aggregate(questions, students):
    complete_pairs = sum(q['complete_pair_count'] for q in questions)
    same_pairs = sum(q['same_completed_pair_count'] for q in questions)
    both_valid = sum(q['student_complete_count'] for q in questions if q['teacher']['completed_terminal_answer'])
    agreements = sum(q['student_teacher_agreement_count'] for q in questions)
    return dict(questions=len(questions), trajectories=len(students), student=dict(
        em=mean(students,'em'), f1=mean(students,'f1'), completion_rate=mean(students,'completed_terminal_answer'),
        legal_complete_trajectory_rate=mean(students,'legal_complete_trajectory'),
        recorded_executor_done_rate=mean(students,'executor_done'),
        legal_action_fraction=sum(s['strict_legal_actions'] for s in students)/sum(s['tool_turn_count'] for s in students),
        recorded_executor_legal_action_fraction=sum(s['recorded_legal_actions'] for s in students)/sum(s['tool_turn_count'] for s in students),
        search_calls_mean=mean(students,'recorded_valid_search_calls'), generated_model_tokens_mean=mean(students,'generated_model_tokens'),
        any_correct_of_five_rate=mean(questions,'student_any_correct'), majority_correct_of_five_rate=mean(questions,'student_majority_correct'),
        all_five_complete_and_same_rate=mean(questions,'all_five_complete_and_same'),
        same_completed_pair_count=same_pairs, completed_pair_count=complete_pairs,
        completed_pair_agreement_rate=same_pairs/complete_pairs if complete_pairs else None,
        all_pair_agreement_rate=same_pairs/(10*len(questions)),
        distinct_completed_answers_mean=mean(questions,'distinct_completed_answers'),
        consensus_of_five_mean=sum(q['consensus_max_count']/5 for q in questions)/len(questions)),
        privileged_teacher_diagnostic=dict(em=mean([q['teacher'] for q in questions],'em'),
            f1=mean([q['teacher'] for q in questions],'f1'), completion_rate=mean([q['teacher'] for q in questions],'completed_terminal_answer')),
        student_teacher=dict(agreement_count=agreements, both_complete_count=both_valid,
            agreement_rate_when_both_complete=agreements/both_valid if both_valid else None,
            agreement_rate_all_trajectories=agreements/len(students), any_of_five_agreement_rate=mean(questions,'student_any_teacher_agreement')))


def publish(path, raw):
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444),'wb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    return dict(physical_path=str(Path(path).absolute()),sha256=sha(raw),bytes=len(raw))


def analyze(raw_root, repo, outer, output):
    require(type(outer) is int and 1 <= outer <= 5, 'Only accepted first-five raw may be analyzed')
    started, cpu_started = time.monotonic(), time.process_time()
    output = Path(output).absolute(); output.mkdir(exist_ok=False)
    for directory in (output.parent,output):
        fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    raw_root=Path(raw_root).absolute(); records={}
    manifest,records['accepted5_manifest']=json_member(raw_root/'full_state/global_step_5/manifest.json',ACCEPTED5)
    require(manifest['source_commit']==SOURCE and manifest['job_id']=='3304' and manifest['config_sha256']==CONFIG
            and type(manifest['driver_pid']) is int and manifest['driver_pid']==2548319 and manifest['run_root']==RUN,
            'Accepted training identity differs')
    files=manifest['files']; directory=raw_root/'trajectories'/f'outer_{outer:06d}'
    def member(name):
        key=f'trajectory_outer_{outer:06d}/'+name
        origin=files[key]
        require(origin['path']==RUN+f'/trajectories/outer_{outer:06d}/'+name, 'Archive origin differs')
        value, record=json_member(directory/name,origin['sha256'])
        record['origin_path']=origin['path']; records[key]=record
        return value
    paired=member('paired-manifest.json'); identity=paired['identity']
    require(identity==dict(common_manifest_sha256=COMMON,config_sha256=CONFIG,directory=RUN+f'/trajectories/outer_{outer:06d}',
            driver_pid=2548319,job_id='3304',outer_step=outer,profile='multitraj200',source_commit=SOURCE,
            teacher_model_path='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/assets/qwen2.5-3b-instruct-aa8e725',
            teacher_revision=REVISION,version=1), 'Paired identity differs')
    require(paired['student_count']==2560 and paired['teacher_count']==512
            and paired['teacher_diagnostic_answers_used_as_targets'] is False, 'Paired scope/count differs')
    request=member('questions.json'); questions=request['questions']; student_manifest=member('student-manifest.json')
    require(request['identity']==identity and student_manifest['identity']==identity and len(questions)==512
            and student_manifest['recorded_before_balancing'] is True
            and student_manifest['response_count_per_question']==5, 'Question/student identity differs')
    def linked(reference, name):
        record=records[f'trajectory_outer_{outer:06d}/'+name]
        require(reference==dict(path=record['origin_path'],sha256=record['sha256'],bytes=record['bytes']),
                'Paired member reference differs')
    linked(paired['request'],'questions.json');linked(paired['student_manifest'],'student-manifest.json')
    linked(student_manifest['request'],'questions.json')
    require(len({q['question_uid'] for q in questions})==512
            and all(type(q['question_position']) is int and q['question_position']==i for i,q in enumerate(questions)), 'Question grouping differs')
    for q in questions:
        require(q['accepted_answers'] and all(type(v) is str for v in q['accepted_answers']), 'Gold answers differ')
    modules, source_records=source_modules(repo)
    records.update({'source_'+k:v for k,v in source_records.items()})
    hf=raw_root/'checkpoints/actor/global_step_5'
    for name in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json'):
        _,record=read(hf/name,files['actor/'+name]['sha256']); record['origin_path']=files['actor/'+name]['path'];records['tokenizer_'+name]=record
    model,records['official_model_manifest']=json_member(raw_root/'bindings/model_manifest.json',files['model_manifest']['sha256'])
    require(model['repo']=='Qwen/Qwen2.5-3B-Instruct' and model['revision']==REVISION, 'Official model revision differs')
    official={r['file']:r for r in model['files']}
    require(records['tokenizer_vocab.json']['sha256']==official['vocab.json']['sha256'], 'Official vocabulary differs')
    merges_raw,_=read(hf/'merges.txt',records['tokenizer_merges.txt']['sha256'])
    merge_equivalence=official_merges_equivalence(merges_raw,official['merges.txt']['sha256'])
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(hf,local_files_only=True,trust_remote_code=False)
    teachers={}; teacher_cost=[]
    for rank in range(8):
        meta=member(f'teacher-rank-{rank}.manifest.json')
        require(meta['identity']==identity and type(meta['rank']) is int and meta['rank']==rank and meta['world_size']==8
                and meta['uid']==0 and meta['hostname']=='h100-10' and meta['execution_device']=='cuda'
                and meta['question_count']==64 and meta['rng_restored'] is True and meta['model_state_restored'] is True
                and meta['diagnostic_answers_used_as_targets'] is False, 'Teacher execution/scope differs')
        name=f'teacher-rank-{rank}.jsonl'; key=f'trajectory_outer_{outer:06d}/'+name
        raw,record=read(directory/name,files[key]['sha256']);record['origin_path']=files[key]['path'];records[key]=record
        linked(meta['teacher'],name)
        pair=paired['teacher_rank_members'][rank]
        require(type(pair['rank']) is int and pair['rank']==rank, 'Paired teacher rank order differs')
        linked(pair['manifest'],f'teacher-rank-{rank}.manifest.json');linked(pair['teacher'],name)
        require(raw.endswith(b'\n') and len(raw.splitlines())==64, 'Teacher JSONL incomplete')
        for line in raw.splitlines():
            row=parse(line); position=row['question_position']
            require(type(position) is int and 0 <= position < 512 and position not in teachers, 'Teacher cohort repeats/differs')
            q=questions[position]
            require(row['question_uid']==q['question_uid'] and row['question']==q['question']
                    and row['accepted_answers']==q['accepted_answers'] and row['data_source']==q['data_source']
                    and row['privileged_training_answers'] is True and row['is_full_teacher_search_trajectory'] is False
                    and row['not_used_as_student_or_distillation_or_evaluation_input'] is True
                    and row['generated_by']=='fixed_official_teacher.model.generate' and row['tool_turns']==[]
                    and row['decoding']==dict(do_sample=False,max_new_tokens=128,num_beams=1,use_cache=False,
                        temperature_applies=False,independent_of_student_history=True), 'Teacher question/privileged scope differs')
            ids=row['output_token_ids'];int_list(ids,'Teacher output tokens')
            require(row['output_attention_mask']==[1]*len(ids) and row['diagnostic_loss_mask']==[0]*len(ids)
                    and row['output_text']==tokenizer.decode(ids,skip_special_tokens=False), 'Teacher output authorship differs')
            teachers[position]=dict(**answer_metrics([row['output_text']],q['accepted_answers'],modules),
                output_tokens=len(ids),rank=rank,historical_worker_pid=meta['pid'],historical_remote_uid=meta['uid'])
        teacher_cost.append({k:meta[k] for k in ('rank','pid','uid','hostname','input_tokens','generated_tokens','host_wall_seconds',
            'diagnostic_cpu_seconds','diagnostic_cuda_stream_interval_seconds','teacher_measure_host_interval_seconds','cost_scope')})
    require(len(teachers)==512, 'Teacher 512 cohort incomplete')
    students=[];path,before,handle=open_physical(directory/'student.jsonl');hashed=hashlib.sha256();size=0
    with handle:
        for index,line in enumerate(handle):
            require(line.endswith(b'\n') and index<2560, 'Student JSONL incomplete/oversized')
            hashed.update(line);size+=len(line);row=parse(line);pos,ri=divmod(index,5);q=questions[pos]
            require(type(row['question_position']) is int and row['question_position']==pos
                    and type(row['response_index']) is int and row['response_index']==ri
                    and row['question_uid']==q['question_uid']
                    and row['training_targets_are_teacher_q_on_student_history'] is True, 'Student five-response identity differs')
            require(row['tensor_sha256']==sha(canonical(row['tensors'])), 'Student tensor digest differs')
            mask=authored_mask(row['tensors']);validate_turns(row,q,tokenizer)
            spans=modules['protocol'].decode_model_spans(tokenizer,row['tensors']['responses'],mask)
            metrics=answer_metrics(spans,q['accepted_answers'],modules);turns=row['tool_turns']
            students.append(dict(**metrics,question_uid=q['question_uid'],question_position=pos,response_index=ri,
                trajectory_uid=row['trajectory_uid'],data_source=q['data_source'],tool_turn_count=len(turns),
                strict_legal_actions=sum(modules['protocol'].parse_model_action(t['text'],strict_prefix=True) is not None for t in turns),
                recorded_legal_actions=sum(t['valid_action'] for t in turns),recorded_valid_search_calls=sum(t['valid_search'] for t in turns),
                executor_done=turns[-1]['done'],generated_model_tokens=sum(mask),
                raw_sampled_token_count=sum(sum(t['raw_response_attention_mask']) for t in turns)))
        require(len(students)==2560 and len({s['trajectory_uid'] for s in students})==2560, 'Student trajectories incomplete/repeated')
        key=f'trajectory_outer_{outer:06d}/student.jsonl'
        record=finish_read(path,before,handle,hashed.hexdigest(),size,files[key]['sha256']);record['origin_path']=files[key]['path'];records[key]=record
    linked(student_manifest['student'],'student.jsonl')
    summaries=[question_summary(q,students[i*5:(i+1)*5],teachers[i]) for i,q in enumerate(questions)]
    cohorts={'all':aggregate(summaries,students)}
    for source in sorted({q['data_source'] for q in questions}):
        cohorts[source]=aggregate([q for q in summaries if q['data_source']==source],[s for s in students if s['data_source']==source])
    outputs={}
    outputs['students']=publish(output/'students-summary.jsonl',b''.join(canonical(s)+b'\n' for s in students))
    outputs['questions']=publish(output/'questions-summary.jsonl',b''.join(canonical(q)+b'\n' for q in summaries))
    import torch
    require(not torch.cuda.is_initialized(), 'Offline parser initialized CUDA')
    for record in records.values():
        recheck(record)
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=repo), 'Frozen source changed during analysis')
    report=dict(version=1,kind='ACTUAL3304_ONE_OUTER_CPU_PAIRED_BEHAVIOR_DIAGNOSTICS',scope='OBSERVATIONAL_TRAINING_COHORT_NOT_EFFECT_ACCEPTANCE_OR_TEACHER_Q_MEASUREMENT',
        identity=identity,accepted5_manifest_sha256=ACCEPTED5,cohorts=cohorts,inputs=records,outputs=outputs,
        execution=dict(pid=os.getpid(),uid=os.getuid(),hostname=socket.gethostname(),python=sys.version,
            torch_version=torch.__version__,transformers_version=__import__('transformers').__version__,cuda_initialized=False,
            wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu_started,
            max_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,max_rss_units='bytes on macOS, KiB on Linux'),
        teacher_recorded_cost=dict(rank_records=teacher_cost,already_included_in_teacher_cumulative=True,
            host_wall_sum_rank_work_seconds=sum(r['host_wall_seconds'] for r in teacher_cost),
            host_wall_max_rank_seconds=max(r['host_wall_seconds'] for r in teacher_cost),
            concurrent_elapsed_critical_path_not_measured=True),
        tokenizer=dict(source='ACCEPTED_HF5_SERIALIZED_TOKENIZER',model_revision=REVISION,
            vocab_byte_exact_official=True,merges_equivalence=merge_equivalence,
            tokenizer_json_and_config_are_signed_saved_HF_bytes_not_claimed_byte_identical_to_official_files=True),
        limitations=['Teacher diagnostic sees training gold and is greedy; it is not a fair baseline or a loss target.',
            'No actual training teacher q/logits/clip rate is stored in these short-answer records.',
            'Only one training question cohort; different outer batches cannot establish a training effect or cause.',
            'Outer1 uses warmup LR zero; this is a sampled pre-update baseline, not a fixed validation set.',
            'Five-response any-correct is an oracle upper bound, not a one-response accuracy or independent-binomial estimate.',
            'Incomplete outputs count as incorrect, not as agreement; conditional agreement denominators are explicit.',
            'Local original UID differs from historical GPU writer UID0; no current remote/live execution is inferred.'])
    publish(output/'report.json',canonical(report)+b'\n')
    print(json.dumps(dict(output=str(output),cohort=cohorts['all'],wall_seconds=report['execution']['wall_seconds']),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--raw-root',required=True);parser.add_argument('--source-repo',required=True)
    parser.add_argument('--outer',type=int,required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();analyze(args.raw_root,args.source_repo,args.outer,args.output)
