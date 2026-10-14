"""Small native-log/receipt aggregates and same-outer EM differences, on CPU only."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time

MILESTONES = (5, 50, 100, 150)
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def load(path):
    with Path(path).open() as stream:
        return json.load(stream)


def number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Invalid finite scalar: ' + label)
    return value


def native_log(paths):
    """Only completed native step records; no inference from live partial lines."""
    rows = {}
    for path in paths:
        with Path(path).open(errors='replace') as stream:
            for line in stream:
                clean = ANSI.sub('', line)
                match = re.search(r'\bstep:(\d+) - ', clean)
                if match is None:
                    continue
                metrics = {}
                for item in clean[match.end():].split(' - '):
                    key, sep, value = item.rpartition(':')
                    if sep:
                        try:
                            metrics[key] = float(value)
                        except ValueError:
                            continue
                if 'timing_s/step' not in metrics:
                    continue
                outer = int(match.group(1))
                if outer in rows and rows[outer] != metrics:
                    raise ValueError('Conflicting retained native records for outer ' + str(outer))
                for key, value in metrics.items():
                    number(value, key)
                rows[outer] = metrics
    return rows


def evaluation(metrics, expected_count=512):
    if 'val/qa_count/all' not in metrics:
        return None
    n = int(number(metrics['val/qa_count/all'], 'dev count'))
    if n != metrics['val/qa_count/all'] or n != expected_count:
        raise ValueError('dev count differs from complete fixed cohort')
    groups = {}
    for source in ('all', 'nq', 'hotpotqa'):
        count = number(metrics['val/qa_count/' + source], 'QA count')
        em = number(metrics['val/em/' + source], 'native EM')
        hits = round(count * em)
        if count != int(count) or count <= 0 or not 0 <= em <= 1 or not math.isclose(hits, count*em, abs_tol=1e-7):
            raise ValueError('Native EM/count cannot reconstruct integer hits')
        groups[source] = dict(count=int(count), hits=hits, em=hits/int(count),
                              f1=metrics.get('val/f1/' + source))
    if groups['nq']['count'] != n//2 or groups['hotpotqa']['count'] != n//2:
        raise ValueError('Native NQ/Hotpot cohort counts differ from balanced dev')
    if groups['nq']['hits'] + groups['hotpotqa']['hits'] != groups['all']['hits']:
        raise ValueError('Per-source hits differ from native overall hits')
    return groups


def actual_adam(output_roots, condition):
    outers = {}
    for root in output_roots:
        for path in sorted((Path(root)/'attempt-results').glob('attempt-*.json')):
            value = load(path)
            result = value['all_rank_result']
            identity = result['identity']
            if any(identity.get(key) != condition[key] for key in ('run_id', 'config_sha256')):
                raise ValueError('Attempt receipt run/config differs: ' + str(path))
            if result['decision'] != 'COMPLETED_OUTER':
                raise ValueError('Attempt is not a completed outer')
            ranks = result['ranks']
            world = condition['world_size']
            if len(ranks) != world or sorted(r['rank'] for r in ranks) != list(range(world)):
                raise ValueError('Completed attempt omits an actual rank')
            for rank in ranks:
                ledger = rank['opportunity_ledger']
                if len(ledger) != 20 or [x['index'] for x in ledger] != list(range(1, 21)):
                    raise ValueError('Attempt omits actual update opportunities')
                if any(x['actual_writes'] not in (0, 1) for x in ledger):
                    raise ValueError('Invalid committed-write count')
                if sum(x['actual_writes'] for x in ledger) != rank['optimizer_calls']:
                    raise ValueError('Write ledger disagrees with actual Adam')
                if rank['opportunities_skipped'] != 20-rank['optimizer_calls']:
                    raise ValueError('Actual skip accounting differs')
            if len({r['optimizer_calls'] for r in ranks}) != 1:
                raise ValueError('Actual rank write counts differ')
            outer = value['accepted_outer']
            small = dict(outer=outer, adam_each_rank=ranks[0]['optimizer_calls'],
                         skips_each_rank=ranks[0]['opportunities_skipped'],
                         cumulative_adam_each_rank=value['actual_completed_optimizer_calls_per_rank'])
            if outer in outers and outers[outer] != small:
                raise ValueError('Conflicting completed receipts for same outer')
            outers[outer] = small
    return outers


def tree_events(output_roots, condition):
    """Aggregate only closed, explicitly recorded trees at sparse milestones."""
    outers = {}
    for root in output_roots:
        for path in sorted((Path(root)/'recursive-events').glob('*/*/events.jsonl')):
            records = []
            with path.open() as stream:
                for line in stream:
                    records.append(json.loads(line))
            if not records:
                continue
            start = records[0]
            if start['run_id'] != condition['run_id'] or start['config_sha256'] != condition['config_sha256']:
                raise ValueError('Tree event identity differs')
            end = [r for r in records if r['event'] == 'question_end']
            if len(end) != 1:
                continue  # Partial evidence is not a closed tree.
            outer = start['outer']
            o = outers.setdefault(outer, dict(questions=0, selected=0, shortfalls=0, attempts=0,
                candidates=0, accepted=0, all_correct=0, all_wrong=0, mixed=0,
                reward_0_to_1=0, reward_1_to_0=0, candidate_generations=collections.Counter(),
                selected_generations=collections.Counter(), branch_positions=collections.Counter(),
                selected_reward_count_unavailable=0, identities=set()))
            key = start['question_id']
            if key in o['identities']:
                raise ValueError('Duplicate closed tree evidence')
            o['identities'].add(key)
            reward = {}
            generations = {}
            root_ids = set()
            for row in records:
                if row['event'] == 'search_start':
                    root_ids.update(q['id'] for q in row['eligible'] if q['generation'] == 0)
                elif row['event'] == 'branch_start':
                    o['branch_positions'][row['action_index']] += 1
                elif row['event'] == 'branch_end' and row['status'] == 'complete':
                    reward[row['id']] = row['reward']
                    generations[row['id']] = row['generation']
                    o['candidates'] += 1
                    o['candidate_generations'][row['generation']] += 1
                    o['reward_0_to_1'] += int(row['parent_reward'] == 0 and row['reward'] == 1)
                    o['reward_1_to_0'] += int(row['parent_reward'] == 1 and row['reward'] == 0)
            for key in root_ids:
                reward[key] = start['initial_reward']
                generations[key] = 0
            e = end[0]
            unknown = set(e['final_ids'])-set(reward)
            if len(unknown) == 1 and not root_ids:
                # A root with no legal expansion need not appear in search_start.
                key = unknown.pop()
                reward[key] = start['initial_reward']
                generations[key] = 0
            o['questions'] += 1
            o['selected'] += len(e['final_ids'])
            o['shortfalls'] += int(not e['complete_group'])
            o['attempts'] += e['attempts']
            o['accepted'] += e['accepted_branches']
            if all(key in reward for key in e['final_ids']):
                correct = sum(reward[key] for key in e['final_ids'])
                o['all_correct'] += int(correct == len(e['final_ids']))
                o['all_wrong'] += int(correct == 0)
                o['mixed'] += int(0 < correct < len(e['final_ids']))
                for key in e['final_ids']:
                    o['selected_generations'][generations[key]] += 1
            else:
                o['selected_reward_count_unavailable'] += 1
    for o in outers.values():
        o.pop('identities')
        o['scope'] = 'Recorded closed trees only; no statistics at unrecorded outer'
    return outers


def summarize_condition(condition, milestones):
    metrics = native_log(condition.get('logs', []))
    roots = condition.get('output_roots', [])
    adam = actual_adam(roots, condition)
    trees = tree_events(roots, condition)
    # Some reference publications contain only aggregate dev counts, never question rows.
    aggregate_only = {}
    for outer, path in condition.get('aggregate_files', {}).items():
        value = load(path)
        d = value['dev512']
        aggregate_only[int(outer)] = evaluation({
            'val/qa_count/all': d['count'], 'val/em/all': d['hits']/d['count'],
            'val/f1/all': d.get('f1'),
            'val/qa_count/nq': d['nq']['count'], 'val/em/nq': d['nq']['hits']/d['nq']['count'],
            'val/qa_count/hotpotqa': d['hotpot']['count'],
            'val/em/hotpotqa': d['hotpot']['hits']/d['hotpot']['count']})
    records = {}
    for outer in milestones:
        m = metrics.get(outer, {})
        dev = evaluation(m)
        if dev is None:
            dev = aggregate_only.get(outer)
        prefix = {key: value for key, value in metrics.items() if 1 <= key <= outer}
        scalar_coverage = sorted(prefix) == list(range(1, outer+1))
        native_adam = m.get('training/actual_adam_calls_per_rank')
        prefix_adam = {key: value for key, value in adam.items() if 1 <= key <= outer}
        receipt_coverage = sorted(prefix_adam) == list(range(1, outer+1))
        stages = collections.Counter()
        for row in prefix.values():
            for key, value in row.items():
                if key.startswith('timing_s/'):
                    stages[key] += value
        total_generated = sum(row.get('rollout/raw_emitted_tokens', row.get('rollout/generated_tokens', 0))
                              for row in prefix.values())
        token_coverage = scalar_coverage and all('rollout/raw_emitted_tokens' in row or
                                               'rollout/generated_tokens' in row for row in prefix.values())
        records[str(outer)] = dict(outer=outer, dev512=dev,
            dev_status=('COMPLETE_NATIVE_AGGREGATE' if dev else 'MISSING'),
            native_actual_adam_each_rank=native_adam,
            receipts_complete_through_outer=receipt_coverage,
            receipt_adam_each_rank=sum(row['adam_each_rank'] for row in prefix_adam.values()) if receipt_coverage else None,
            receipt_skips_each_rank=sum(row['skips_each_rank'] for row in prefix_adam.values()) if receipt_coverage else None,
            scalar_coverage_count=len(prefix), complete_scalar_prefix=scalar_coverage,
            generated_tokens_recorded_sum=total_generated if prefix else None,
            complete_generated_token_prefix=token_coverage,
            allocated_gpu_hours_from_recorded_outer_wall=stages['timing_s/step']*condition['world_size']/3600 if prefix else None,
            recorded_stage_seconds_sum=dict(stages),
            tree=trees.get(outer), boundary_policy_loss=m.get('actor/pg_loss'),
            boundary_train_selected_reward=m.get('critic/score/mean'),
            cost_scope='Recorded emitted tokens include padding and discarded tails per native convention. '
                       'Stage sums overlap; do not add components to total step. GPU hours = configured '
                       'world size × recorded outer wall, not active GPU utilization or allocation lease.')
    return dict(condition_id=condition['id'], run_id=condition['run_id'],
                config_sha256=condition['config_sha256'], world_size=condition['world_size'],
                milestones=records, sampled_memory=load(condition['memory_snapshot']) if condition.get('memory_snapshot') else None)


def build(study):
    milestones = tuple(study.get('milestones', MILESTONES))
    result = dict(protocol='recursive_bran_ablation150_report_v1', time=time.time(),
                  reference_condition=study.get('reference_condition'), conditions={}, comparisons={},
                  limitations=['Native scalars/receipts are not independent Adam tensor reloads.',
                               'TRAIN cohorts and selected reward are not fixed dev EM.',
                               'Latest human scope is aggregate EM; no paired/bootstrap/replay.'])
    for condition in study['conditions']:
        cid = condition['id']
        if cid in result['conditions']:
            raise ValueError('Duplicate condition identity')
        result['conditions'][cid] = summarize_condition(condition, milestones)
    reference = study.get('reference_condition')
    if reference is not None:
        if reference not in result['conditions']:
            raise ValueError('Reference condition is missing')
        for cid in result['conditions']:
            if cid == reference:
                continue
            comparisons = result['comparisons'][cid] = {}
            for outer in milestones:
                a = result['conditions'][reference]['milestones'][str(outer)]['dev512']
                b = result['conditions'][cid]['milestones'][str(outer)]['dev512']
                comparisons[str(outer)] = dict(status='COMPLETE_AGGREGATE' if a and b else 'MISSING',
                    aggregate_difference_pp=100*(b['all']['em']-a['all']['em']) if a and b else None,
                    scope='Same-outer aggregate EM difference; no paired or causal claim')
    result['analysis_complete'] = all(row['dev512'] is not None
        for condition in result['conditions'].values() for row in condition['milestones'].values())
    return result


def write_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'aggregate.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    fields = ('condition', 'outer', 'hits', 'em', 'nq_hits', 'hotpot_hits', 'adam_each_rank',
              'skips_each_rank', 'generated_tokens_recorded', 'allocated_gpu_hours_recorded')
    lines = ['# 递归 Tree-GRPO 消融结果', '', '只使用完整原生评测与已有日志，按最新授权仅汇总EM，不追加逐题捕获或评测。', '',
             '| 条件 | outer | dev512 | NQ | Hotpot | 真实Adam/rank | 跳过/rank |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    with (directory/'curve.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cid, condition in report['conditions'].items():
            for row in condition['milestones'].values():
                dev = row['dev512']
                writes = row['receipt_adam_each_rank']
                if writes is None:
                    writes = row['native_actual_adam_each_rank']
                skips = row['receipt_skips_each_rank']
                writer.writerow(dict(condition=cid, outer=row['outer'], hits=dev['all']['hits'] if dev else None,
                    em=dev['all']['em'] if dev else None, nq_hits=dev['nq']['hits'] if dev else None,
                    hotpot_hits=dev['hotpotqa']['hits'] if dev else None, adam_each_rank=writes, skips_each_rank=skips,
                    generated_tokens_recorded=row['generated_tokens_recorded_sum'],
                    allocated_gpu_hours_recorded=row['allocated_gpu_hours_from_recorded_outer_wall']))
                score = f"{dev['all']['hits']}/512（{100*dev['all']['em']:.4f}%）" if dev else '缺失'
                lines.append(f"| {cid} | {row['outer']} | {score} | {dev['nq']['hits'] if dev else '未知'} | "
                             f"{dev['hotpotqa']['hits'] if dev else '未知'} | {writes if writes is not None else '未知'} | "
                             f"{skips if skips is not None else '未知'} |")
    if report['comparisons']:
        lines += ['', '与A同轮次的总体EM差值：']
        for cid, rows in report['comparisons'].items():
            for outer, row in rows.items():
                difference = row['aggregate_difference_pp']
                lines.append(f"- {cid}，{outer}轮：{difference:.4f}pp。" if difference is not None else
                             f"- {cid}，{outer}轮：原生完整EM缺失。")
    lines += ['', '耗时分量可能重叠，不能将它们相加当总耗时。GPU小时只按记录的outer wall×卡数计算，'
              '不是GPU实际忙碌时间；生成token遵循原生记录，包括padding与丢弃尾部。',
              '实际记录范围、缺失覆盖、树统计与采样峰值见aggregate.json。单seed不证明跨seed稳定因果。', '']
    (directory/'findings-zh.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    study = load(args.study)
    report = build(study)
    report['study_sha256'] = hashlib.sha256(args.study.read_bytes()).hexdigest()
    write_report(report, args.output)
    print(json.dumps(dict(output=str(args.output), analysis_complete=report['analysis_complete'],
                          conditions=list(report['conditions'])), allow_nan=False))


if __name__ == '__main__':
    main()
