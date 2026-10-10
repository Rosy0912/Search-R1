"""Build a small Pearson audit from accepted compact summaries only."""
import datetime
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'g1-probe-evolution-20261010'
if not PREVIOUS.is_dir():
    PREVIOUS = HERE.parent / 'g1_reward_probe_analysis_20261010'


def load(path):
    return json.loads(path.read_text())


def pearson(x, y):
    assert len(x) == len(y) and all(math.isfinite(v) for v in x + y)
    mx, my = sum(x) / len(x), sum(y) / len(y)
    vx, vy = sum((v - mx) ** 2 for v in x), sum((v - my) ** 2 for v in y)
    if vx == 0 or vy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


remote_path = HERE / 'remote-correlations.stdout.json'
execution = load(HERE / 'remote-correlations.execution.json')
assert execution['rc'] == 0
assert hashlib.sha256(remote_path.read_bytes()).hexdigest() == execution['stdout_sha256']
remote = load(remote_path)
old = load(PREVIOUS / 'group-calibration.json')
assert remote['run_id'] == old['run_id'] == 'db474b6abf654ad9892b02ae67ab6f86'
assert not remote['torch_imported'] and remote['readonly'] and remote['cpu_only']
assert {r['round']: r['dump_sha256'] for r in remote['source']} == {
    r['round']: r['dump_sha256'] for r in old['rounds']}
# Independently reconstruct point-biserial correlation from prior accepted
# moments. FP32 wrong floor is 0.05000000074505806. Tiny recomputation rounding
# differences are tolerated explicitly; this is not a bitwise identity claim.
cross_checks = []
for t, complete_n in [(0, 62), (50, 281), (100, 300)]:
    s = old['common_complete_groups_by_round'][str(t)]
    current = remote['common61'][str(t)]
    assert current['questions'] == 61 and current['trajectories'] == 305
    assert remote['all_complete_groups'][str(t)]['questions'] == complete_n
    p = s['raw_reward_correct_count'] / s['nominal_rollouts']
    moments = s['raw_pre_normalization_reward']
    independent = ((moments['mean'] - 0.05000000074505806)
                   * math.sqrt((1 - p) / p) / moments['std_population'])
    difference = abs(independent - current['trajectory_pearson_R_EM'])
    assert difference < 2e-8
    cross_checks.append(dict(round=t, prior_moment_pearson=independent,
                             direct_remote_pearson=current['trajectory_pearson_R_EM'],
                             absolute_difference=difference, accepted=True))

raw_path = PREVIOUS / 'raw-reward-distributions.json'
raw = load(raw_path)
rows = raw['actual_training_g1']
assert [r['iteration'] for r in rows] == list(range(1, 142))
periods = []
for lo, hi in [(1, 50), (51, 100), (101, 141), (1, 141)]:
    selected = [r for r in rows if lo <= r['iteration'] <= hi]
    periods.append(dict(first_iteration=lo, last_iteration=hi,
                        n_iterations=len(selected),
                        pearson_iteration_mean_R_vs_EM=pearson(
                            [r['raw_reward_mean_native'] for r in selected],
                            [r['correct_fraction_console'] for r in selected]),
                        em_precision='console rounded to 0.001; coefficient approximate'))

result = dict(built_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              requested_full_individual_TRAIN_pearson_status='NOT_COMPUTED',
              previous_answer_withdrawn=True,
              withdrawal_reason='141 iteration-mean pairs and 305 calibration pairs are not complete individual TRAIN trajectory pairs requested by the user.',
              raw_definition='C = original strict EM 0/1; R = designed pre-GRPO G1 reward',
              training_iteration_mean_correlations=periods,
              calibration_common61=remote['common61'],
              calibration_all_complete=remote['all_complete_groups'],
              independent_prior_moment_cross_checks=cross_checks,
              provenance_sha256={str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [remote_path, raw_path, PREVIOUS / 'group-calibration.json']},
              historical_individual_TRAIN_correlations_available=False,
              round150_cohort_available=False)
(HERE / 'correlations.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')

lines = ['# 更正：尚未算出用户指定的全量逐轨迹 TRAIN Pearson', '',
         '**撤回将0.998651及0.97–0.98作为用户问题答案的报告。** 前者输入是141个迭代均值配对，后者是305条共同校准子集；均不等于每轮全部TRAIN轨迹的一对一原始/探针奖励配对。下面旧数值保留作错误审计，不作为全量逐轨迹结论。', '',
         '正确目标：每条实际TRAIN轨迹同时保留原始奖励X与G1设计奖励Y，以(run_id, iteration, trajectory_id)逐条join，包含全部轨迹；先计算每轮r_t，再按用户指定范围展平全部配对计算r_all。不能先把每轮或每组压成一个均值。', '',
         'r_all = [N·Σ(XY)−ΣX·ΣY] / sqrt([N·ΣX²−(ΣX)²]·[N·ΣY²−(ΣY)²])。这五个矩须从每一对真实轨迹累计；ΣXY不能用每轮均值乘积替代。', '',
         '当前run源代码规定512题×5轨迹=2560条/轮；100完整轮次应有256000对。先按实际run/保存行数核查，不静默丢弃无效输出或缺失reward的条目。原始reward采用同一条G1轨迹的原始严格EM，不能拿另一个archive run的轨迹来对齐。', '',
         '历史原始R未直接入已知trace schema，但trace保留完整input token与mask，正在进一步核查是否可用当时冻结encoder/head及题目gold完整恢复。尚未实施模型重算，也没有全量TRAIN系数；不因字段名缺失就断言永远不可恢复。', '',
         '这里原始 reward 明确定义为严格答题正确性 C∈{0,1}；R 为 Search-G1 设计奖励，尚未做 GRPO 组内归一化。critic/rewards 本身保存的就是 R，不能把它当另一套独立原始奖励。', '',
         '下面相关仅在其各自限定数据上成立；不能据此判断用户指定的全量历史TRAIN配对相关，也不能据此反驳50→100奖励与正确性变化脱钩。', '',
         '实际 TRAIN 的逐轮均值相关（既有快照完整覆盖1–141轮；EM取三位小数日志）：', '',
         '| 范围 | Pearson：每轮平均R vs 每轮EM |', '| --- | --- |']
for r in periods:
    lines.append(f"| {r['first_iteration']}–{r['last_iteration']} | {r['pearson_iteration_mean_R_vs_EM']:.6f} |")
lines += ['', '以下是共同61个完整五轨迹校准题，每点305条轨迹；各点使用其匹配探针和新轨迹。这不是历史TRAIN每轮2560条的相关，也不是独立测试集。', '',
          '| 校准轮 | 逐轨迹R vs C | 每题平均R vs 平均C | 混合组内Pearson等题均值 | 混合组数 | 全对/全错组数（Pearson未定义） |',
          '| --- | --- | --- | --- | --- | --- |']
for t in [0, 50, 100]:
    r = remote['common61'][str(t)]
    lines.append(f"| {t} | {r['trajectory_pearson_R_EM']:.6f} | {r['group_mean_pearson_R_EM']:.6f} | {r['mixed_group_pearson_summary']['mean']:.6f} | {r['mixed_group_pearson_summary']['count']} | {r['group_pearson_undefined_all_correct']}/{r['group_pearson_undefined_all_wrong']} |")
lines += ['', '原奖励门控使正确条R≥0.98，而错误条R≤0.05，因此正确/错误区间分离，逐条高相关很大程度来自构造本身。对于这里无非法输出的校准组，R=0.05+C·(R_correct−0.05)，额外探针补分改变正确条内部的排序。', '',
          '全对组C恒为1，原GRPO优势全为0；G1可以因b/d及搜索成本差异产生非零优势。共同61题中，将所有组归一化后的优势拼接，Corr(A_G1,A_EM)为：', '']
for t in [0, 50, 100]:
    lines.append(f"- {t}轮：{remote['common61'][str(t)]['all_groups_pooled_GRPO_advantage_pearson']:.6f}。")
lines += ['', '这与原始reward相关是不同统计量。全对组增加非零G1信号会降低上述pooled优势相关；不能把它解释为混合组纠错排序变差或已证实hacking。混合组单独的优势相关分别为0.987706/0.995823/0.999908。', '',
          '第150轮校准样本和历史逐条TRAIN reward不在本次已验收证据中，不填充、不由均值推造。全对/全错组Pearson未定义，不当成0或1。归一化前R与归一化后优势亦不可混名。', '',
          '远端只读CPU实际退出0；直接Pearson与独立历史矩重构一致（误差<2e-8），源run/dump/heads身份随紧凑JSON保留。没有新模型采样、训练、GPU操作或原始数据下载。']
(HERE / 'findings.md').write_text('\n'.join(lines) + '\n')
print(json.dumps({'old_statistics_arithmetic_verified': True,
                  'requested_full_individual_TRAIN_pearson_computed': False,
                  'previous_answer_withdrawn': True, 'periods': periods,
                  'cross_checks': cross_checks}, ensure_ascii=False))
