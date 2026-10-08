"""Read-only stdlib facts from seven ablations' committed outer checkpoints."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

sys.dont_write_bytecode = True
VARIANTS = ('C0', 'C1', 'M', 'R', 'W', 'P', 'J')
SHA = re.compile(r'[0-9a-f]{64}\Z')
METRICS = ('raw_loss', 'post_loss', 'weighted_post_loss', 'student_to_teacher_raw_kl',
           'student_to_outer_old_raw_kl', 'student_to_mixture_raw_kl')


class Rejected(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise Rejected(message)


def stamp(s):
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, mode=s.st_mode)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
                                    separators=(',', ':'),allow_nan=False).encode()).hexdigest()


class Reader:
    def __init__(self):
        self.proofs = {}

    def read(self, ref):
        require(isinstance(ref,dict) and isinstance(ref.get('path'),str), 'Explicit path reference required')
        p = Path(ref['path'])
        require(p.is_absolute() and '..' not in p.parts, 'Absolute canonical input path required')
        current = Path(p.anchor)
        for part in p.parts[1:]:
            current /= part
            require(not current.is_symlink(), 'Symlink input refused')
        before = stamp(p.stat())
        require(stat.S_ISREG(before['mode']), 'Nonregular input refused')
        with p.open('rb') as h:
            require(stamp(os.fstat(h.fileno())) == before, 'Input changed before fixed-handle read')
            raw = h.read()
            require(stamp(os.fstat(h.fileno())) == before, 'Input changed during fixed-handle read')
        require(stamp(p.stat()) == before, 'Input changed during read')
        proof = dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),**before)
        if 'sha256' in ref:
            require(SHA.fullmatch(ref['sha256']) and proof['sha256'] == ref['sha256'], 'Input SHA differs')
        for key in before:
            if key in ref:
                require(ref[key] == proof[key], 'Declared input metadata differs: ' + key)
        old = self.proofs.get(str(p))
        require(old is None or old == proof, 'Same input path changed within collection')
        self.proofs[str(p)] = proof
        return json.loads(raw), proof

    def finish(self):
        for proof in self.proofs.values():
            require(stamp(Path(proof['path']).stat()) == {k:proof[k] for k in ('device','inode','bytes','mtime_ns','ctime_ns','mode')},
                    'Input changed after collection')
        return sorted(self.proofs.values(),key=lambda x:x['path'])


def integer(value, low=0):
    require(type(value) is int and value >= low, 'Invalid integer count')
    return value


def number(value):
    require(type(value) in (float,int) and math.isfinite(value), 'Nonfinite/nonnumeric metric')
    return float(value)


def mean_field(pairs, key):
    observed = [number(p[key]) for p in pairs if p.get(key) is not None]
    # Partial scalar coverage is not silently used as a complete trajectory mean.
    return dict(value=sum(observed)/len(pairs) if pairs and len(observed)==len(pairs) else None,
                observed_trajectories=len(observed),expected_trajectories=len(pairs))


def summarize_pairs(pairs):
    weighted = [number(p['pair_weight']) for p in pairs if p.get('pair_weight') is not None]
    require(all(w > 0 for w in weighted), 'Nonpositive trajectory weight')
    for p in pairs:
        if all(p.get(k) is not None for k in ('post_loss','pair_weight','weighted_post_loss')):
            require(math.isclose(number(p['weighted_post_loss']),number(p['post_loss'])*number(p['pair_weight']),
                                 rel_tol=1e-6,abs_tol=1e-9), 'Weighted post-loss does not match native weight')
    clip = [integer(p['clipped_vocabulary_count']) for p in pairs if p.get('clipped_vocabulary_count') is not None]
    return dict(trajectory_denominator=len(pairs),raw_pair_exposure=len(pairs),
        weighted_pair_exposure=sum(weighted) if len(weighted)==len(pairs) else None,
        weight_coverage=len(weighted),equal_trajectory_means={k:mean_field(pairs,k) for k in METRICS},
        clipped_vocabulary_count=sum(clip) if len(clip)==len(pairs) else None,
        clipped_count_coverage=len(clip),clipped_probability_mass=None,clipped_vocabulary_fraction=None)


def updates(refs, reader, run, index, before, K, order, contract, global_seen, cumulative_before=0):
    unique = []; repeated = 0; local = set()
    for ref in refs:
        key = (ref['path'],ref['sha256'])
        if key in local:
            repeated += 1
            continue
        local.add(key); unique.append(ref)
    require(len(unique) == math.ceil(K/32), 'Committed update count differs from usable K')
    exposure = tokens = 0; all_pairs = []; facts = []
    for n,ref in enumerate(unique,1):
        value,proof = reader.read(ref)
        actual = min(32,K-exposure); step = before+n
        require(value.get('schema') == 'opsd-online-real-update-v1' and value.get('accepted') is True
                and value['run_id'] == run and value['outer_index'] == index
                and value['optimizer_updates'] == step and value['global_actual_pairs'] == actual
                and value['contract'] == contract, 'Committed update identity/count differs')
        key = (run,step)
        require(key not in global_seen, 'Same actual update exposed in two completed outers')
        global_seen.add(key)
        ranks = value['all_rank_actual_token_metrics']
        require(len(ranks)==8 and {r['rank'] for r in ranks}==set(range(8)), 'Eight distinct rank metrics required')
        expected = order[exposure:exposure+actual]
        pairs = []
        for rank in ranks:
            require([p['pair_index'] for p in rank['pairs']] == expected[rank['rank']::8], 'Rank partition differs')
            pairs.extend(rank['pairs'])
        require(len(pairs)==actual and len({p['pair_index'] for p in pairs})==actual, 'Pair exposure missing/duplicate')
        count = sum(integer(p['valid_tokens'],1) for p in pairs)
        require(count==value['global_valid_tokens'], 'Native valid-token total differs')
        norms = [number(r['grad_norm']) for r in ranks if r.get('grad_norm') is not None]
        require(not norms or len(norms)==8 and max(norms)==min(norms), 'Rank grad-norm declarations differ')
        scaler = value.get('scaler')
        if scaler is not None:
            require(scaler.get('accepted') is True and scaler.get('adam_advanced') is True
                    and scaler.get('overflow') is False, 'Committed update scaler is not accepted Adam')
            if norms and scaler.get('grad_norm') is not None:
                require(number(scaler['grad_norm'])==norms[0], 'Scaler/rank grad-norm declarations differ')
        exposure += actual; tokens += count; all_pairs.extend(pairs)
        require(value['cumulative_pair_exposure'] == cumulative_before+exposure, 'Invalid cumulative pair exposure')
        facts.append(dict(update=step,global_actual_pairs=actual,global_valid_tokens=count,
            preclip_grad_norm=norms[0] if len(norms)==8 else None,scaler=scaler,
            recorded_retries=integer(value['retries']) if value.get('retries') is not None else None,
            committed_overflow=False if scaler else None,metrics=summarize_pairs(pairs),source=proof))
    require(exposure==K, 'Final committed pair exposure differs')
    return facts,all_pairs,tokens,repeated


def sampling_facts(stats):
    require(stats['questions']==512 and stats['generated_trajectories']==2560, 'Complete TRAIN512x5 sampling required')
    P = integer(stats['P']); classes = stats['classes']; pair_classes = stats['pair_classes']
    require(sum(integer(classes[k]) for k in ('all_wrong','all_correct','mixed'))==512
            and P==512-classes['all_wrong'] and sum(integer(pair_classes[k]) for k in ('CC','CW'))==P,
            'Sampling class/P accounting differs')
    diagnostics = stats.get('diagnostics',{})
    reward = diagnostics.get('strict_reward')
    hits = None
    if reward is not None:
        # Native dict(Counter(...)) omits categories observed zero times.
        require(bool(reward) and set(reward)<={'0','1'}
                and sum(integer(reward.get(k,0)) for k in ('0','1'))==2560,
                'Native strict-reward denominator differs')
        hits = reward.get('1',0)
    return dict(questions=512,trajectories=2560,strict_reward_hits=hits,
        strict_reward_mean=hits/2560 if hits is not None else None,P=P,classes=classes,pair_classes=pair_classes,
        self_pair=integer(stats['self_pair']) if stats.get('self_pair') is not None else None,
        response_text_equal=integer(stats['response_text_equal']) if stats.get('response_text_equal') is not None else None,
        eligible_pair_weight_sum=number(stats['pair_weight_sum']) if stats.get('pair_weight_sum') is not None else None,
        diagnostics_scope='Native recorded strict TRAIN sampling; no rescoring')


def outer(reader, group, config, config_sha, index, global_seen):
    output = Path(group['output_root']); directory = output/'outers'/('outer-%05d'%index)
    completion_path = directory/'training/outer-complete.json'
    if not completion_path.exists():
        return dict(variant=group['variant'],outer=index,status='NOT_COMPLETE',counts=None)
    complete,completion_proof = reader.read(dict(path=str(completion_path)))
    run = group['run_id']; variant = group['variant']
    require(complete.get('schema')=='opsd-online-outer-complete-v1' and complete.get('accepted') is True
            and complete['run_id']==run and complete['outer_index']==index, 'Native outer completion identity differs')
    cp,cp_proof = reader.read(complete['checkpoint']); update = integer(complete['optimizer_updates'])
    expected = output/'checkpoints'/('update-%08d-outer-%05d'%(update,index))/'manifest.json'
    require(cp_proof['path']==str(expected) and cp.get('schema')=='opsd-online-checkpoint-v1'
            and cp.get('accepted') is True and cp['run_id']==run and cp['config_sha256']==config_sha
            and cp['optimizer_updates']==update and cp['cursor']['outer_index']==index
            and 'outer_end' in cp['protection_reason'], 'Pinned completed checkpoint identity differs')
    contract = dict(direction='student_to_target_reverse',target_version='opsd-ablation-reverse-v1',
                    variant=variant,precision=config['precision']['mode'],mixture_alpha=.1 if variant in ('M','J') else 0.)
    require(cp['contract']==complete['contract']==contract and cp['runtime_identity']==config['runtime_manifest']['sha256']
            and cp['source_model_identity']==config['source_model_manifest']['sha256'], 'Scientific contract differs')
    require(cp['training_input']==complete['training_input'], 'Completion/checkpoint inputs differ')
    inputs = cp['training_input']; manifest,mp = reader.read(inputs['outer_manifest'])
    encoding,ep = reader.read(inputs['encoding']); order,op = reader.read(inputs['order'])
    require(manifest['run_id']==run and manifest['config_sha256']==config_sha and manifest['outer_index']==index
            and manifest['contract']==contract and manifest['pairs']==inputs['pairs'], 'Outer training input identity differs')
    require(mp['path']==str(directory/'outer-manifest.json'), 'Outer manifest path differs')
    require(encoding['P']==complete['P']==manifest['P'] and encoding['K']==complete['K']
            and encoding['skipped']==complete['skipped'] and len(encoding['rows'])==encoding['K']
            and canonical_sha(encoding['rows'])==encoding['sha256'], 'Encoded P/K/skips/hash accounting differs')
    K = integer(complete['K']); P = integer(complete['P'])
    require(K<=P<=512 and sum(integer(n) for n in complete['skipped'].values())==P-K, 'Usable/skipped accounting differs')
    require(order['outer_index']==index and sorted(order['pair_indices'])==list(range(K))
            and order['encoded_sha256']==encoding['sha256']==cp['cursor']['encoded_sha256']
            and op['sha256']==cp['cursor']['order_sha256']
            and cp['cursor']['usable_pair_count']==cp['cursor']['next_pair_cursor']==K, 'Committed order/cursor differs')
    before = integer(manifest['cumulative_optimizer_updates_before'])
    require(update==before+math.ceil(K/32) and complete['optimizer_updates_delta']==math.ceil(K/32), 'Outer Adam delta differs')
    facts,pairs,tokens,repeated = updates(cp['outer_update_receipts'],reader,run,index,before,K,
                                        order['pair_indices'],contract,global_seen,
                                        integer(cp['cumulative_pair_exposure'])-K)
    require(tokens==complete['global_valid_tokens'] and cp['cumulative_pair_exposure']==complete['cumulative_pair_exposure'],
            'Completed token/cumulative exposure differs')
    stats = manifest['sampling_stats']
    require(stats['P']==P and stats['variant']==variant and stats['outer_index']==index, 'Sampling stats identity differs')
    return dict(variant=variant,outer=index,status='COMPLETE_FACTS',run_id=run,precision=config['precision']['mode'],
        contract=contract,counts=dict(P=P,K=K,skipped=complete['skipped'],optimizer_updates=update,
            optimizer_updates_before=before,optimizer_updates_delta=len(facts),valid_tokens=tokens,
            cumulative_pair_exposure=complete['cumulative_pair_exposure'],empty_outer_no_adam=K==0),
        sampling=sampling_facts(stats),metrics=summarize_pairs(pairs),updates=facts,
        duplicate_committed_refs_ignored=repeated,scaler_end=cp.get('all_rank_scaler_state'),
        sources=dict(completion=completion_proof,checkpoint=cp_proof,outer_manifest=mp,encoding=ep,order=op),
        precision_scope='Config/contract/scaler declarations; no Tensor or runtime dtype re-audit')


def collect(goal_ref):
    reader = Reader(); goal,gp = reader.read(goal_ref)
    require(goal.get('schema')=='opsd-ablation-outer-backup-goal-v1' and goal.get('accepted') is True
            and goal.get('goal_scope')=='seven-group-outer-end10' and goal.get('cluster')=='HKUST', 'Exact authorized seven-group goal required')
    groups = goal['groups']
    require(len(groups)==7 and tuple(g['variant'] for g in groups)==VARIANTS
            and len({g['run_id'] for g in groups})==7 and len({g['config']['path'] for g in groups})==7, 'Exact seven distinct group bindings required')
    rows = []; seen = set();configs = []
    for group in groups:
        config,proof = reader.read(group['config']);configs.append(proof)
        require(config.get('schema')=='opsd-ablation-config-v1' and config['run_id']==group['run_id']
                and config['variant']==group['variant'] and config['study_id']==goal['study_id']
                and config['scientific']['outer_iterations']==10 and config['paths']['output_root']==group['output_root'],
                'Immutable original group config differs')
        for index in range(10):
            rows.append(outer(reader,group,config,proof['sha256'],index,seen))
    finished = [r for r in rows if r['status']=='COMPLETE_FACTS']
    for variant in VARIANTS:
        chain = {r['outer']:r for r in finished if r['variant']==variant}
        for index,row in chain.items():
            count = row['counts']
            if index==0:
                require(count['optimizer_updates_before']==0 and count['cumulative_pair_exposure']==count['K'],
                        'Fresh outer0 cumulative counts differ')
            elif index-1 in chain:
                previous = chain[index-1]['counts']
                require(count['optimizer_updates_before']==previous['optimizer_updates']
                        and count['cumulative_pair_exposure']==previous['cumulative_pair_exposure']+count['K'],
                        'Adjacent completed outer counter chain differs')
    return dict(schema='opsd-ablation-outer-facts-v1',accepted=True,observed_unix=time.time(),goal=gp,configs=configs,
        coverage=dict(completed_outers=len(finished),expected_outers=70,all70_complete=len(finished)==70,
            by_variant={v:sum(r['variant']==v for r in finished) for v in VARIANTS}),outers=rows,
        totals=dict(committed_unique_updates=len(seen),raw_pair_exposure=sum(r['counts']['K'] for r in finished),
            valid_tokens=sum(r['counts']['valid_tokens'] for r in finished)),inputs=reader.finish(),
        report_scope='Committed record facts only; no optimizer tensor, model, effect, or execution-exit acceptance',
        chinese_report_inputs=dict(
            title='七组 OPSD 逐外轮已提交记录事实表',
            train_curve='每外轮取新的512个TRAIN问题并采样2560条轨迹；TRAIN reward曲线不是固定heldout准确率。',
            loss='逐轨迹token平均后等轨迹平均；加权目标仍除实际轨迹数，不除权重和。loss数值不是梯度贡献或因果证据。',
            denominator='每次更新显式记录实际1–32条轨迹；K0没有Adam更新，loss为NA。',
            missing='null表示记录缺失/覆盖不足的NA，不能补零。未引用的失败token/retry日志不计曝光。',
            retry='recorded_retries仅该committed receipt声明的重试数，不能代表全部历史溢出。',
            coverage='部分已提交记录可汇总；仅all70_complete=true表示70个outer记录齐全，不等于训练终态或效果验收。'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goal',required=True); parser.add_argument('--goal-sha256',required=True)
    args = parser.parse_args()
    print(json.dumps(collect(dict(path=args.goal,sha256=args.goal_sha256)),ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
