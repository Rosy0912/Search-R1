"""Read-only R/PR rollout aggregates; optional isolated remote case copies.

No model, GPU, HTTP, process, lock, signal or production-file modification.
Raw question/trajectory contents are never included in stdout summaries.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import types


QA_SHA = '9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd'
CORE_SHA = '0fbfbe63d0aafb41f79db702e08c9a0efc5f9bbf29a2964310594ec2ec0a83fc'
READS = []


def read(path):
    path = Path(path)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, f) != getattr(after, f) for f in fields):
        raise ValueError('Input changed during read: ' + str(path))
    READS.append(dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    return raw


def obj(path):
    return json.loads(read(path))


def rows(path):
    raw = read(path)
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Partial JSONL: ' + str(path))
    return [json.loads(line) for line in raw.splitlines()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load_science(runtime):
    runtime = Path(runtime)
    qa_path = runtime / 'upstream/Search-R1/search_r1/qa_protocol.py'
    core_path = runtime / 'search_e1/core.py'
    qa_raw, core_raw = read(qa_path), read(core_path)
    if hashlib.sha256(qa_raw).hexdigest() != QA_SHA or hashlib.sha256(core_raw).hexdigest() != CORE_SHA:
        raise ValueError('Frozen QA/core source identity differs')
    package = types.ModuleType('search_r1')
    package.__path__ = [str(qa_path.parent)]
    sys.modules['search_r1'] = package
    qa = types.ModuleType('search_r1.qa_protocol')
    sys.modules[qa.__name__] = qa
    exec(compile(qa_raw, str(qa_path), 'exec'), qa.__dict__)
    core = types.ModuleType('frozen_core')
    exec(compile(core_raw, str(core_path), 'exec'), core.__dict__)
    return qa, core


def classified(row, qa, core):
    spans = [row['response'][a:b] for a, b in row['policy_spans']]
    action = qa.parse_model_action(spans[-1]) if spans else None
    answer = qa.extract_final_answer(spans)
    reward = core.reward(row['response'], row['answers'], row['policy_spans'])
    if reward != row['reward']:
        raise ValueError('Raw native reward differs from frozen strict rescore')
    return dict(correct=reward, final_answer_valid=answer is not None,
        last_action_valid=action is not None, last_action_kind=action[0] if action else None,
        failure=None if reward else 'final_invalid' if answer is None else 'valid_gold_mismatch',
        no_search=row['search_calls'] == 0, search=row['search_calls'],
        total_recorded_tokens=row['token_count'], policy_characters=sum(b-a for a,b in row['policy_spans']),
        policy_segments=len(row['policy_spans']), termination=row['termination'])


def numeric(values):
    ordered = sorted(values)
    return dict(mean=statistics.mean(values), median=statistics.median(values),
        p90=ordered[int(.9*(len(ordered)-1))], minimum=min(values), maximum=max(values))


def rollout_stats(values):
    return dict(slots=len(values), strict_correct=sum(x['correct'] for x in values),
        final_invalid=sum(x['failure']=='final_invalid' for x in values),
        valid_gold_mismatch=sum(x['failure']=='valid_gold_mismatch' for x in values),
        parser_invalid=sum(not x['last_action_valid'] for x in values),
        valid_last_nonanswer=sum(x['last_action_valid'] and x['last_action_kind']!='answer' for x in values),
        no_search=sum(x['no_search'] for x in values),
        search_histogram=dict(sorted(Counter(x['search'] for x in values).items())),
        termination=dict(Counter(x['termination'] for x in values)),
        total_recorded_tokens=numeric([x['total_recorded_tokens'] for x in values]),
        policy_characters=numeric([x['policy_characters'] for x in values]),
        policy_segments=numeric([x['policy_segments'] for x in values]))


def compare(rrows, prows, rpairs, ppairs, qa, core, outer):
    r = {(x['question_id'], x['sibling']):x for x in rrows}
    p = {(x['question_id'], x['sibling']):x for x in prows}
    if len(r) != 2560 or len(p) != 2560 or set(r) != set(p):
        raise ValueError('R/PR must share all original 512x5 slots')
    rc = {k:classified(v,qa,core) for k,v in r.items()}
    pc = {k:classified(v,qa,core) for k,v in p.items()}
    transitions = Counter(('correct' if rc[k]['correct'] else rc[k]['failure'],
        'correct' if pc[k]['correct'] else pc[k]['failure']) for k in r)
    searches = Counter('equal' if rc[k]['search']==pc[k]['search'] else
        'PR_more' if rc[k]['search']<pc[k]['search'] else 'PR_less' for k in r)
    lost = [k for k in r if rc[k]['correct'] and not pc[k]['correct']]
    gained = [k for k in r if not rc[k]['correct'] and pc[k]['correct']]
    qs = list(dict.fromkeys(x['question_id'] for x in rrows))
    rh = {q:sum(rc[(q,s)]['correct'] for s in range(5)) for q in qs}
    ph = {q:sum(pc[(q,s)]['correct'] for s in range(5)) for q in qs}
    pair_maps = {'R':{x['question_id']:x for x in rpairs}, 'PR':{x['question_id']:x for x in ppairs}}
    quality = {}
    for label,pairs in pair_maps.items():
        quality[label] = dict(pairs=len(pairs), reference_strict_correct=sum(x['reference_reward'] for x in pairs.values()),
            student_strict_correct=sum(x['student_reward'] for x in pairs.values()),
            self_pairs=sum(x['mining']['reference_sibling']==x['mining']['student_sibling'] for x in pairs.values()))
    common = set(pair_maps['R']) & set(pair_maps['PR'])
    dtrans = Counter((pair_maps['R'][q]['student_reward'],pair_maps['PR'][q]['student_reward']) for q in common)
    summary = dict(outer_index=outer, R=rollout_stats(list(rc.values())), PR=rollout_stats(list(pc.values())),
        paired=dict(lost_R_correct_PR_wrong=len(lost), gained_R_wrong_PR_correct=len(gained),
            preserved_correct=sum(rc[k]['correct'] and pc[k]['correct'] for k in r),
            both_wrong=sum(not rc[k]['correct'] and not pc[k]['correct'] for k in r),
            lost_PR_final_invalid=sum(pc[k]['failure']=='final_invalid' for k in lost),
            lost_PR_valid_gold_mismatch=sum(pc[k]['failure']=='valid_gold_mismatch' for k in lost),
            gains_from_R_final_invalid=sum(rc[k]['failure']=='final_invalid' for k in gained),
            gains_from_R_valid_gold_mismatch=sum(rc[k]['failure']=='valid_gold_mismatch' for k in gained),
            transitions={a+'->'+b:n for (a,b),n in sorted(transitions.items())}),
        search_direction=dict(searches),
        five_success_histograms={'R':dict(sorted(Counter(rh.values()).items())), 'PR':dict(sorted(Counter(ph.values()).items()))},
        five_success_transitions={str(a)+'->'+str(b):n for (a,b),n in sorted(Counter((rh[q],ph[q]) for q in qs).items())},
        eligibility=dict(R=len(pair_maps['R']),PR=len(pair_maps['PR']),common=len(common),
            R_only=len(set(pair_maps['R'])-set(pair_maps['PR'])),PR_only=len(set(pair_maps['PR'])-set(pair_maps['R']))),
        B_D_quality=quality, common_eligible_D_reward_transitions={str(a)+'->'+str(b):n for (a,b),n in sorted(dtrans.items())},
        common_eligible_D_sibling_equal=sum(pair_maps['R'][q]['mining']['student_sibling']==pair_maps['PR'][q]['mining']['student_sibling'] for q in common),
        question_sibling_order_sha256=dict(R=digest(list(r)), PR=digest(list(p))),
        length_scope='token_count includes complete transcript and observations; policy_characters are characters, not token counts')
    example_groups = {
        'R_correct_PR_valid_wrong':[k for k in lost if pc[k]['failure']=='valid_gold_mismatch'],
        'R_correct_PR_final_invalid':[k for k in lost if pc[k]['failure']=='final_invalid'],
        'R_wrong_PR_correct':gained,
    }
    cases = []
    for label,keys in example_groups.items():
        chosen = sorted(keys, key=lambda k:digest(['precision-trace-example-v1',42,outer,k]))[:3]
        for k in chosen:
            cases.append(dict(category=label,outer_index=outer,slot_key=k,R=r[k],PR=p[k],R_class=rc[k],PR_class=pc[k],
                R_pair=pair_maps['R'].get(k[0]),PR_pair=pair_maps['PR'].get(k[0])))
    return summary, cases


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--r-config',required=True)
    ap.add_argument('--pr-config',required=True)
    ap.add_argument('--outers',type=int,default=10)
    ap.add_argument('--remote-cases-dir')
    args=ap.parse_args()
    if not 1<=args.outers<=10:
        raise ValueError('Only closed first1..10 outers supported')
    configs={'R':obj(args.r_config),'PR':obj(args.pr_config)}
    r,p=configs['R'],configs['PR']
    if (r['source_model_manifest']['sha256']!=p['source_model_manifest']['sha256'] or
        r['common_initial_adapter_tensor_sha256']!=p['common_initial_adapter_tensor_sha256'] or
        r['runtime_manifest']['sha256']!=p['runtime_manifest']['sha256']):
        raise ValueError('Original common source/template/runtime differ')
    qa,core=load_science(p['paths']['runtime'])
    aggregates=[];first_cases=[]
    for outer in range(args.outers):
        actual={}
        for label,cfg in configs.items():
            directory=Path(cfg['paths']['output_root'])/'outers'/('outer-%05d'%outer)
            manifest=obj(directory/'outer-manifest.json')
            completion=obj(directory/'training/outer-complete.json')
            if not completion['accepted'] or manifest['outer_index']!=outer:
                raise ValueError('Requires accepted complete original outer')
            actual[label]=dict(manifest=manifest,completion=completion,
                rollouts=rows(manifest['rollouts']['path']),pairs=rows(manifest['pairs']['path']))
        summary,cases=compare(actual['R']['rollouts'],actual['PR']['rollouts'],actual['R']['pairs'],actual['PR']['pairs'],qa,core,outer)
        summary['K_and_Adam']={k:{f:v['completion'][f] for f in ('P','K','optimizer_updates_delta','optimizer_updates')} for k,v in actual.items()}
        summary['untrained_common_initial']=outer==0 and all(v['manifest']['cumulative_optimizer_updates_before']==0 for v in actual.values())
        aggregates.append(summary)
        if outer==0:first_cases=cases
    output=dict(schema='opsd-R-vs-PR-rollout-precision-comparison-v1',remote_unix=time.time(),
        scope='First outer shares untrained common initial policy; later outers have different learned policies and are descriptive, not isolated precision effects',
        config_paths={'R':args.r_config,'PR':args.pr_config},precision={k:v['precision'] for k,v in configs.items()},
        sampling_dtype={k:v['sampling']['dtype'] for k,v in configs.items()},outers=aggregates,
        summed_slots=args.outers*2560,summed_R_correct=sum(x['R']['strict_correct'] for x in aggregates),
        summed_PR_correct=sum(x['PR']['strict_correct'] for x in aggregates),
        first_outer_case_categories=dict(Counter(x['category'] for x in first_cases)),input_files=READS,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if __file__!='<stdin>' else None,
        no_cuda_or_torch_imported='torch' not in sys.modules)
    if args.remote_cases_dir:
        directory=Path(args.remote_cases_dir)
        if not directory.is_absolute() or not str(directory).startswith('/data/shaoshitong/') or directory.exists():
            raise ValueError('New explicit independent HKUST cases directory required')
        directory.mkdir(parents=True)
        cases_path=directory/'first-outer-example-cases.jsonl'
        cases_path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in first_cases))
        output['remote_case_file']=dict(path=str(cases_path),bytes=cases_path.stat().st_size,sha256=hashlib.sha256(cases_path.read_bytes()).hexdigest(),count=len(first_cases))
        (directory/'summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(output,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
