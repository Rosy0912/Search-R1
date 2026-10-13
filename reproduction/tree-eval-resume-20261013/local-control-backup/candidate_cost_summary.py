#!/usr/bin/env python3
"""Read-only H10018 tree-audit aggregation; only compact scalars return to Mac."""
import importlib.util
import json
import math
import shlex
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:0befbaf99fc44942105831b3fbec94e4bf1b8ccb'
UNIT = 'sst-tree-nested-branch-fresh200-0befbaf9.service'
REMOTE = r'''
import collections,json,math,time
from pathlib import Path
run=Path(RUN_PATH)
fields=['new_generated_tokens','new_retained_model_tokens','new_prefill_tokens','retrieval_calls','scoring_calls','inherited_response_tokens','inherited_observation_tokens','inherited_raw_ledger_actions','generation_wall_seconds_amortized','scoring_wall_seconds_amortized']
ledger_fields=['generated_tokens','prefill_tokens','retrieval_calls','trajectories','padded_rows','raw_slots','raw_emitted_tokens','raw_retained_tokens','raw_discarded_tail_tokens','raw_inherited_leaf_tokens','semantic_prefix_scan_cpu_seconds','semantic_prefix_scan_wall_seconds','scoring_calls','scoring_wall_seconds']
def finite(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def summarize(rows):
 return {'candidate_count':len(rows),'field_sums':{k:sum(x[k] for x in rows if finite(x.get(k))) for k in fields},'field_record_counts':{k:sum(finite(x.get(k)) for x in rows) for k in fields}}
def grouping(rows):
 return {label:summarize([x for x in rows if check(x)]) for label,check in {
 'all':lambda x:True,'accepted':lambda x:x.get('accepted') is True,'discarded':lambda x:x.get('accepted') is False,
 'fallback':lambda x:x.get('fallback') is True,'accepted_without_fallback':lambda x:x.get('accepted') is True and x.get('fallback') is not True,
 'accepted_fallback':lambda x:x.get('accepted') is True and x.get('fallback') is True,
 'unknown_outcome':lambda x:x.get('accepted') not in (True,False)}.items()}
def sum_ledger(rounds):
 return {k:sum(x.get('cost_delta',{}).get(k,0) for x in rounds if finite(x.get('cost_delta',{}).get(k))) for k in ledger_fields}
started=time.time();rollouts=[];skipped=[];seen_schema=[]
for p in sorted((run/'output/tree-audit').glob('*.json')):
 if p.name.endswith('_policy_batch.json'): continue
 before=p.stat()
 try: d=json.loads(p.read_text())
 except (ValueError,OSError) as e: skipped.append({'name':p.name,'reason':type(e).__name__});continue
 after=p.stat()
 if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns or d.get('status')!='ROLLOUT_DIAGNOSTIC_COMPLETE':
  skipped.append({'name':p.name,'status':d.get('status'),'reason':'not stable completed rollout'});continue
 candidates=d.get('candidate_attempts',[]);rounds=d.get('candidate_rounds',[])
 seen_schema.append({'audit_schema_version':d.get('schema_version'),'candidate_fields':sorted(set().union(*(set(x) for x in candidates))),'round_fields':sorted(set().union(*(set(x) for x in rounds))),'round_cost_delta_fields':sorted(set().union(*(set(x.get('cost_delta',{})) for x in rounds)))})
 unique_rounds=[];ids=set();duplicate_rounds=0
 for x in rounds:
  key=x.get('round_id')
  if key in ids: duplicate_rounds+=1;continue
  ids.add(key);unique_rounds.append(x)
 groups=d.get('question_groups',[]);leaves=d.get('leaves',[])
 z={'audit_file':p.name,'audit_file_bytes':after.st_size,'audit_file_mtime':after.st_mtime,'schema_version':d.get('schema_version'),'global_step':d.get('global_step'),'rollout_sequence':d.get('rollout_sequence'),'attempt_id':d.get('attempt_id'),'status':d.get('status'),'audit_algorithm_source_commit':d.get('algorithm_source_commit'),'audit_base_commit':d.get('base_commit'),
 'question_count':len(groups),'training_leaf_count':len(leaves),'accepted_branch_count':len(d.get('branches',[])),'original_root_leaf_count':sum(x.get('branch_sampling',{}).get('continuation')==0 for x in leaves),'root_leaf_quotas':d.get('root_leaf_quotas'),'candidate_outcomes':grouping(candidates),'stages':{},'rounds':[],'duplicate_round_ids_excluded':duplicate_rounds,'candidate_timing_scopes':dict(collections.Counter(x.get('timing_scope') for x in candidates))}
 for stage in sorted(set(x.get('stage','UNKNOWN') for x in candidates)|set(x.get('stage','UNKNOWN') for x in unique_rounds)):
  rows=[x for x in candidates if x.get('stage')==stage];rr=[x for x in unique_rounds if x.get('stage')==stage]
  z['stages'][stage]={'candidate_outcomes':grouping(rows),'round_count':len(rr),'generation_round_elapsed_seconds':sum(x.get('generation_wall_seconds',0) for x in rr),'scoring_round_elapsed_seconds':sum(x.get('scoring_wall_seconds',0) for x in rr),'round_controller_ledger':sum_ledger(rr),'round_elapsed_by_outcome_seconds':None}
 for i,x in enumerate(unique_rounds,1):
  rows=[c for c in candidates if c.get('round_id')==x.get('round_id')]
  z['rounds'].append({'ordinal':i,'stage':x.get('stage'),'candidate_count':x.get('candidate_count'),'accepted_count':sum(c.get('accepted') is True for c in rows),'discarded_count':sum(c.get('accepted') is False for c in rows),'fallback_count':sum(c.get('fallback') is True for c in rows),'generation_elapsed_seconds':x.get('generation_wall_seconds'),'scoring_elapsed_seconds':x.get('scoring_wall_seconds'),'controller_ledger':{k:v for k,v in x.get('cost_delta',{}).items() if k in ledger_fields}})
 z['candidate_round_controller_ledger']=sum_ledger(unique_rounds)
 z['generation_candidate_round_elapsed_seconds']=sum(x.get('generation_wall_seconds',0) for x in unique_rounds)
 z['scoring_candidate_round_elapsed_seconds']=sum(x.get('scoring_wall_seconds',0) for x in unique_rounds)
 z['recorded_round_minus_candidate_residual']={rk:z['candidate_round_controller_ledger'].get(rk,0)-z['candidate_outcomes']['all']['field_sums'][ck] for rk,ck in [('generated_tokens','new_generated_tokens'),('prefill_tokens','new_prefill_tokens'),('retrieval_calls','retrieval_calls'),('scoring_calls','scoring_calls')]}
 z['padding_cost_by_candidate_outcome']=None
 cost=run/'output/training-rollout-cost'/('attempt-%06d.json'%d.get('attempt_id',0))
 if cost.exists():
  cc=json.loads(cost.read_text());z['rollout_controller_cost']={k:cc.get(k) for k in ['status','attempt_id','candidate_outer','completed_outer_before','question_rows','costs','error','start_epoch','end_epoch','wall_seconds','trainer_process_cpu_seconds','code_commit','scope']}
 policy=p.with_name(p.stem+'_policy_batch.json')
 if policy.exists():
  pd=json.loads(policy.read_text());z['policy_batch_small_metadata']={k:pd.get(k) for k in ['status','leaf_count','empty_leaves','diagnostic_check_cost','shared_prefix_credit','rollout_file_sha256']}
 rollouts.append(z)
rollouts.sort(key=lambda x:(x.get('global_step',0),x.get('rollout_sequence',0)))
all_fields={k:sorted(set().union(*(set(x[k]) for x in seen_schema))) for k in ['candidate_fields','round_fields','round_cost_delta_fields']}
result={'remote_observed_start_epoch':started,'remote_observed_end_epoch':time.time(),'run_directory':str(run),'completed_rollout_count':len(rollouts),'completed_rollout_steps':[x['global_step'] for x in rollouts],'skipped_audits':skipped,'schema_observed':all_fields,'completed_rollout_rollup':{'questions':sum(x['question_count'] for x in rollouts),'training_leaves':sum(x['training_leaf_count'] for x in rollouts),'candidate_attempts':sum(x['candidate_outcomes']['all']['candidate_count'] for x in rollouts),'accepted_candidates':sum(x['candidate_outcomes']['accepted']['candidate_count'] for x in rollouts),'discarded_candidates':sum(x['candidate_outcomes']['discarded']['candidate_count'] for x in rollouts),'fallback_candidates':sum(x['candidate_outcomes']['fallback']['candidate_count'] for x in rollouts)},'rollouts':rollouts[-3:],'earlier_rollout_detail_omitted_count':max(0,len(rollouts)-3)}
encoded=json.dumps(result,ensure_ascii=False,separators=(',',':'))
if len(encoded.encode())>80000: raise RuntimeError('compact remote result exceeds 80KB')
print(encoded)
'''


def main():
    poll_path = OUT / 'LIVE_POLL_15.json'
    poll = json.loads(poll_path.read_text())
    source = json.loads((ROOT / 'outputs/STATUS.json').read_text())
    helper_path = ROOT / 'work/tree-speed-repair-20261008/ssh_retained18.py'
    spec = importlib.util.spec_from_file_location('candidate_cost_ssh', helper_path)
    route = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(route)
    command = 'python3 -c ' + shlex.quote(REMOTE.replace('RUN_PATH', repr(RUN)))
    response = route.run(command, target='h100-10', timeout=80)
    if response.returncode:
        raise RuntimeError('read-only aggregate failed: ' + response.stderr[-2000:])
    result = json.loads(response.stdout)
    result.update({
        'summary_schema': 'tree_candidate_cost_compact_v1',
        'summary_created_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit_from_mainline_status': source['source_commit'],
        'unit': UNIT,
        'known_preceding_observations': [
            {'epoch': 1791907020.0105784, 'completed_rollout_count': 1, 'step_1_audit_bytes': 21640018, 'kind': 'read-only remote schema observation'},
            {'epoch': 1791907071.8494556, 'completed_rollout_count': 1, 'candidate_attempts': 2529, 'kind': 'read-only remote scalar observation'},
        ],
        'mainline_stage_metrics_source': str(poll_path.relative_to(ROOT)),
        'mainline_stage_metrics_reused_without_recalculation': [
            {key: value for key, value in row.items()
             if key == 'step' or key.startswith(('tree_variance/', 'timing_s/'))}
            for row in poll.get('iteration_metrics', [])
            if int(row.get('step', -1)) in result['completed_rollout_steps']
        ],
        'interpretation': {
            'accepted_and_discarded': 'Disjoint outcomes. fallback is an overlapping recorded flag; in this snapshot all fallback candidates are accepted training leaves.',
            'disjoint_candidate_partition': ['accepted_without_fallback', 'accepted_fallback', 'discarded', 'unknown_outcome'],
            'candidate_costs': 'Sums of recorded per-candidate fields; inherited prefix sums are repeated inherited token events, not unique emissions or newly generated tokens.',
            'amortized_candidate_times': 'Recorded equal candidate shares of parallel round elapsed; outcome shares are amortized accounting, not separately measured candidate or outcome wall time.',
            'round_wall_times': 'Each distinct round_id is counted once. Stage sums are sequential candidate-round elapsed only; full rollout wall is reported separately.',
            'controller_ledger': 'Round ledger includes native DP padding. Residual between round ledger and candidate rows remains unallocated; no padding share is inferred by candidate/outcome.',
            'root_rows': '512 questions x 2 original roots = 1024 root leaves; 1536 accepted continuation candidates yield 2560 training leaves. Discarded candidates incur generation/retrieval/scoring but are not added to training leaves.',
            'completed_rollout_not_optimizer_credit': 'ROLLOUT_DIAGNOSTIC_COMPLETE is rollout completion; outer/optimizer credit belongs to mainline progress records.',
        },
        'missing_or_not_separately_measured': [
            'Padding token/prefill cost attribution by individual candidate or accepted/discarded/fallback outcome.',
            'Exclusive wall time for each candidate or each outcome inside a shared parallel round.',
            'Unique KV/FLOPs, actual cached-prefill savings, worker CPU or GPU time, and money cost.',
            'Original root per-trajectory generation, retrieval, scoring and prefix detail in candidate_attempts.',
            'New mathematical audit or recomputation of rewards/advantages/variance (existing mainline stage metrics copied unchanged).',
        ],
    })
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if len(encoded.encode()) > 100000:
        raise RuntimeError('summary exceeds 100KB')
    destination = OUT / 'CANDIDATE_COST_COMPACT.json'
    destination.write_text(encoded)
    lines = [
        '# H10018 已完成 rollout 候选成本摘要', '',
        f"实测：{result['summary_created_utc']}；源码 `{result['source_commit_from_mainline_status']}`。",
        f"已完成 rollout：{result['completed_rollout_count']}；详细轮次：{', '.join(map(str, result['completed_rollout_steps']))}。", '',
        '每轮 512 题、1024 原始根轨迹与 1536 accepted continuation，共 2560 训练叶。fallback 是 accepted 的子集；discarded 额外产生费用，不进入训练叶。', '',
        '| rollout / stage | 尝试 | accepted | discarded | fallback（accepted内） | 新生成token（候选记录） | 检索 | 评分 | 继承response / observation token |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for rollout in result['rollouts']:
        for stage, data in rollout['stages'].items():
            outcomes = data['candidate_outcomes']
            costs = outcomes['all']['field_sums']
            lines.append(f"| {rollout['global_step']} / {stage} | {outcomes['all']['candidate_count']} | {outcomes['accepted']['candidate_count']} | {outcomes['discarded']['candidate_count']} | {outcomes['fallback']['candidate_count']} | {costs['new_generated_tokens']} | {costs['retrieval_calls']} | {costs['scoring_calls']} | {costs['inherited_response_tokens']} / {costs['inherited_observation_tokens']} |")
        outcomes = rollout['candidate_outcomes']
        lines.extend(['', f"rollout {rollout['global_step']}：候选 {outcomes['all']['candidate_count']} = accepted {outcomes['accepted']['candidate_count']} + discarded {outcomes['discarded']['candidate_count']}；accepted 中 fallback {outcomes['fallback']['candidate_count']}。", '',
            '| rollout / outcome | 候选数 | 新生成token | 新prefill token | 检索 | 评分 | 继承response / observation token | 生成 / 评分分摊秒 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for label in ('accepted', 'discarded', 'fallback'):
            item = outcomes[label]; fields = item['field_sums']
            lines.append(f"| {rollout['global_step']} / {label} | {item['candidate_count']} | {fields['new_generated_tokens']} | {fields['new_prefill_tokens']} | {fields['retrieval_calls']} | {fields['scoring_calls']} | {fields['inherited_response_tokens']} / {fields['inherited_observation_tokens']} | {fields['generation_wall_seconds_amortized']:.3f} / {fields['scoring_wall_seconds_amortized']:.3f} |")
        lines.extend(['', '| stage | 候选轮数 | 共享生成elapsed秒 | 共享评分elapsed秒 | ledger生成token（含padding） | ledger prefill token（含padding） |', '|---|---:|---:|---:|---:|---:|'])
        for stage, data in rollout['stages'].items():
            ledger = data['round_controller_ledger']
            lines.append(f"| {stage} | {data['round_count']} | {data['generation_round_elapsed_seconds']:.3f} | {data['scoring_round_elapsed_seconds']:.3f} | {ledger['generated_tokens']} | {ledger['prefill_tokens']} |")
        ledger = rollout['candidate_round_controller_ledger']
        total = rollout.get('rollout_controller_cost', {})
        all_costs = outcomes['all']['field_sums']
        lines.extend(['', f"候选记录新生成 {all_costs['new_generated_tokens']} token；候选轮ledger {ledger['generated_tokens']} token、padding rows {ledger['padded_rows']}；差额 {rollout['recorded_round_minus_candidate_residual']['generated_tokens']} token未向accepted/discarded/fallback分摊。", f"候选并行轮生成elapsed共 {rollout['generation_candidate_round_elapsed_seconds']:.3f} 秒、评分elapsed {rollout['scoring_candidate_round_elapsed_seconds']:.3f} 秒，每轮只计一次。完整rollout wall {total.get('wall_seconds')} 秒（controller ledger原值）。"])
    lines.extend(['', '完整的 A1/B1/B2 × accepted/discarded/fallback 交叉成本、9轮明细、字段覆盖数和原有 tree_variance/timing_s 指标见同名 JSON。未新增数学审核。', '', '字段限制：逐候选有生成、retained、prefill、检索、评分、继承前缀及分摊耗时；没有候选级实际独占wall、padding分摊、KV/FLOPs/缓存节省或货币费用。继承前缀是重复继承事件，不能计为新生成token。', '', '只在原 H10018 远端读取本任务JSON并返回小型标量摘要；未搬运raw rollout/tokens/PT/model/evalarrays，未更改训练源码、服务或主线STATUS/REPORT。', ''])
    markdown = OUT / 'CANDIDATE_COST_COMPACT.md'
    markdown.write_text('\n'.join(lines))
    print(json.dumps({'json': str(destination), 'json_bytes': destination.stat().st_size, 'markdown': str(markdown), 'markdown_bytes': markdown.stat().st_size, 'completed_rollouts': result['completed_rollout_count'], 'remote_observed_epoch': result['remote_observed_end_epoch'], 'rollup': result['completed_rollout_rollup']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
