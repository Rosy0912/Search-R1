"""Derive private C0 curves from existing recorded scalar facts; no remote I/O."""
import csv
import hashlib
import json
import math
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent
INPUT = OUT.parent.parent / 'goal-continuation-20261009-7'
FACTS = INPUT / 'ten-outer-committed-facts.stdout.json'
AGGREGATE = INPUT / 'C0-ten-scalar-aggregate.json'


def file_ref(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


facts = json.loads(FACTS.read_text())
aggregate = json.loads(AGGREGATE.read_text())
assert facts['accepted'] is True and aggregate['accepted'] is True
assert aggregate['source']['sha256'] == file_ref(FACTS)['sha256']
outers = sorted((o for o in facts['outers'] if o['variant'] == 'C0'),
                key=lambda o: o['outer'])
assert [o['outer'] for o in outers] == list(range(10))
rows = []
update_ids = []
cumulative_tokens = 0
for outer in outers:
    assert outer['status'] == 'COMPLETE_FACTS'
    assert outer['contract'] == {
        'direction': 'student_to_target_reverse',
        'target_version': 'opsd-ablation-reverse-v1',
        'variant': 'C0', 'precision': 'bf16', 'mixture_alpha': 0.0}
    count = outer['counts']
    updates = outer['updates']
    k = count['K']
    assert k == count['P'] == outer['metrics']['trajectory_denominator']
    assert len(updates) == count['optimizer_updates_delta'] == math.ceil(k / 32)
    assert sum(u['global_actual_pairs'] for u in updates) == k
    assert sum(u['global_valid_tokens'] for u in updates) == count['valid_tokens']
    assert count['optimizer_updates'] == updates[-1]['update']
    assert outer['sampling']['questions'] == 512
    assert outer['sampling']['trajectories'] == 2560
    assert outer['sampling']['strict_reward_mean'] == outer['sampling']['strict_reward_hits'] / 2560
    norm = [u['preclip_grad_norm'] for u in updates]
    assert all(math.isfinite(v) for v in norm)
    assert all(u['scaler']['accepted'] is True and
               u['scaler']['adam_advanced'] is True and
               u['committed_overflow'] is False for u in updates)
    means = outer['metrics']['equal_trajectory_means']
    for item in means.values():
        assert item['observed_trajectories'] == item['expected_trajectories'] == k
        assert math.isfinite(item['value'])
    assert means['raw_loss']['value'] == means['student_to_teacher_raw_kl']['value']
    assert means['raw_loss']['value'] == means['student_to_mixture_raw_kl']['value']
    assert means['post_loss']['value'] == means['weighted_post_loss']['value']
    assert outer['metrics']['weighted_pair_exposure'] == float(k)
    update_ids.extend(u['update'] for u in updates)
    cumulative_tokens += count['valid_tokens']
    rows.append(dict(
        outer_index=outer['outer'], outer_number=outer['outer'] + 1,
        train_questions=512, sampled_trajectories=2560,
        strict_train_reward_hits=outer['sampling']['strict_reward_hits'],
        strict_train_reward=outer['sampling']['strict_reward_mean'],
        P=count['P'], K=k,
        actual_adam_updates_before=count['optimizer_updates_before'],
        actual_adam_updates_this_outer=count['optimizer_updates_delta'],
        actual_adam_updates_cumulative=count['optimizer_updates'],
        cumulative_pair_exposures=count['cumulative_pair_exposure'],
        valid_policy_tokens=count['valid_tokens'],
        cumulative_valid_policy_tokens=cumulative_tokens,
        raw_reverse_kl_trajectory_mean=means['raw_loss']['value'],
        clipped_component_loss_trajectory_mean=means['post_loss']['value'],
        weighted_clipped_loss_trajectory_mean=means['weighted_post_loss']['value'],
        student_to_outer_old_raw_kl_diagnostic=means['student_to_outer_old_raw_kl']['value'],
        preclip_grad_norm_min=min(norm), preclip_grad_norm_max=max(norm),
        gradnorm_recorded_update_count=len(norm),
        clipped_vocab_contribution_count=outer['metrics']['clipped_vocabulary_count'],
        committed_overflow_count=sum(u['committed_overflow'] for u in updates),
        recorded_retries=sum(u['recorded_retries'] for u in updates),
        cohort_scope='different TRAIN cohort; not fixed-heldout512',
        loss_reduction='token mean within trajectory, then equal trajectory mean',
    ))

assert update_ids == list(range(1, 101))
assert sum(r['K'] for r in rows) == aggregate['pair_exposures'] == 3012
assert sum(r['valid_policy_tokens'] for r in rows) == aggregate['valid_tokens'] == 583074
assert rows[-1]['actual_adam_updates_cumulative'] == aggregate['actual_update_count'] == 100
for field, source_field in [
        ('raw_reverse_kl_trajectory_mean', 'raw_loss'),
        ('clipped_component_loss_trajectory_mean', 'post_loss'),
        ('weighted_clipped_loss_trajectory_mean', 'weighted_post_loss')]:
    observed = sum(r[field] * r['K'] for r in rows) / 3012
    assert math.isclose(observed, aggregate['recorded_equal_pair_means'][source_field],
                        rel_tol=0, abs_tol=1e-12)
assert sum(r['clipped_vocab_contribution_count'] for r in rows) == aggregate['clip_vocab_count'] == 1021

csv_path = OUT / 'C0-ten-outer-curve.csv'
with csv_path.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

plot_files = []
plot_status = {}
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font_path = Path('/System/Library/Fonts/STHeiti Medium.ttc')
    if not font_path.exists():
        font_path = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    assert font_path.exists(), 'No existing Chinese font; do not install/download fonts'
    font = FontProperties(fname=str(font_path))
    matplotlib.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False,
                               'svg.fonttype': 'path', 'font.size': 10})
    x = [r['outer_number'] for r in rows]
    blue, teal, orange = '#205b89', '#197e6e', '#ba6229'
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    fig.patch.set_facecolor('white')
    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xlabel('外轮（outer index 0–9 对应第1–10轮）', fontproperties=font)
        ax.grid(axis='y', color='#dfe5e9', linewidth=.7)
        ax.spines[['top', 'right']].set_visible(False)

    ax = axes[0, 0]
    reward = [r['strict_train_reward'] * 100 for r in rows]
    ax.plot(x, reward, color=blue, marker='o', linewidth=2)
    ax.set_ylim(0, 60)
    ax.set_ylabel('严格 TRAIN reward（%）', fontproperties=font)
    ax.set_title('不同 TRAIN cohort 的严格 reward', fontproperties=font, loc='left')
    for i in (0, 3, 9):
        ax.annotate(f'{reward[i]:.2f}%', (x[i], reward[i]), xytext=(0, 9),
                    textcoords='offset points', ha='center', color=blue, fontsize=9)

    ax = axes[0, 1]
    ax.bar(x, [r['K'] for r in rows], color='#b3cdda', width=.58, label='K：实际训练对数')
    ax.set_ylim(0, 370)
    ax.set_ylabel('K（C0 中 P=K）', fontproperties=font, color=blue)
    ax.set_title('每外轮对数与实际 Adam 累计', fontproperties=font, loc='left')
    twin = ax.twinx()
    twin.plot(x, [r['actual_adam_updates_cumulative'] for r in rows], color=orange,
              marker='o', linewidth=2, label='实际 Adam 累计')
    twin.set_ylim(0, 115)
    twin.set_ylabel('实际 Adam 更新累计', fontproperties=font, color=orange)
    twin.spines['top'].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, prop=font, loc='upper left', frameon=False)

    ax = axes[1, 0]
    ax.plot(x, [r['raw_reverse_kl_trajectory_mean'] for r in rows], color=blue,
            marker='o', linewidth=2, label='原始 reverse KL：student → teacher')
    ax.plot(x, [r['clipped_component_loss_trajectory_mean'] for r in rows], color=teal,
            marker='s', linewidth=2, label='逐词表贡献 clip 后的记录损失')
    ax.set_ylim(.30, .425)
    ax.set_ylabel('等轨迹平均的记录标量', fontproperties=font)
    ax.set_title('原始 reverse KL 与 clip 后损失', fontproperties=font, loc='left')
    ax.legend(prop=font, loc='upper right', frameon=False, fontsize=9)

    ax = axes[1, 1]
    lows = [r['preclip_grad_norm_min'] for r in rows]
    highs = [r['preclip_grad_norm_max'] for r in rows]
    ax.fill_between(x, lows, highs, color='#d7e6ed', alpha=.85,
                    label='本外轮真实更新的 min–max')
    ax.plot(x, lows, color=blue, marker='o', linewidth=1.5)
    ax.plot(x, highs, color=blue, marker='o', linewidth=1.5)
    ax.axhline(1.0, color=orange, linestyle='--', linewidth=1.5, label='grad norm clip 阈值 1')
    ax.set_ylim(0, 1.8)
    ax.set_ylabel('clip 前全局 grad norm', fontproperties=font)
    ax.set_title('逐外轮更新的梯度范数范围', fontproperties=font, loc='left')
    ax.legend(prop=font, loc='upper right', frameon=False)

    fig.suptitle('C0：10轮在线 OPSD 已提交训练记录', fontproperties=font, fontsize=18, x=.05, ha='left')
    fig.text(.05, .923, '每外轮是不同的512题 TRAIN cohort；此图不是固定512题评估的收益曲线。',
             fontproperties=font, fontsize=11, color='#963b30')
    fig.text(.05, .025, '100次实际 Adam · 3012对训练曝光 · 583074个有效 policy token\n'
             'loss：每条轨迹内 token 平均，再等轨迹平均；clip 后数值是目标损失，不称为原始 KL。',
             fontproperties=font, fontsize=10, color='#45515a')
    fig.subplots_adjust(left=.08, right=.92, top=.86, bottom=.16, wspace=.30, hspace=.42)
    for suffix in ('png', 'svg'):
        path = OUT / ('C0-ten-outer-curve.' + suffix)
        fig.savefig(path, dpi=180, facecolor='white')
        plot_files.append(path)
    plt.close(fig)
    plot_status = {'accepted': True, 'matplotlib_version': matplotlib.__version__,
                   'font_path': str(font_path), 'gpu_used': False}
except (ImportError, FileNotFoundError, AssertionError) as error:
    plot_status = {'accepted': False, 'reason': str(error), 'no_network_install': True,
                   'csv_preserved': True}

note = OUT / 'C0-ten-outer-curve.notes.md'
note.write_text('''# C0 逐外轮派生曲线\n\n'
10个完整外轮、100个连续实际 Adam 更新、3012对曝光、583074个有效 policy token，均从已有已提交记录派生。CSV保留原浮点精度；图用于阅读。没有读取节点、重新打分或重算 logits。\n\n'
每外轮使用不同的512题 TRAIN cohort，采样2560条轨迹。TRAIN reward 的分母为2560条采样轨迹；它不是固定512题 heldout 正确率，不能从该曲线推断评测收益或退化时间。\n\n'
C0 的原始目标方向为 student→teacher reverse KL，mixture_alpha=0。记录先对每条轨迹的有效 token 平均，再等轨迹平均；外轮汇总按实际轨迹数合并各更新，包含真实尾批，不是对更新数平均。C0每对权重为1，weighted_post_loss与post_loss相同。逐词表贡献clip后的post_loss不是原始KL，也不是梯度贡献；clip次数没有完整词表/概率质量分母，不能补出比例。\n\n'
梯度面板范围是本外轮各真实更新记录的全局 clip 前范数 min–max，不是置信区间；虚线1为原 grad norm clip 阈值。图不证明梯度或权重的新增数值验收，原验收证据另存。\n\n'
全10轮按轨迹曝光合并：raw reverse KL=0.37264755582515885，post loss=0.36394943558509507。记录中被clip的词表贡献计数1021，已提交overflow=0；该零值不代替所有历史失败尝试的完整审计。\n'''.replace("\n'\n", "\n\n"), encoding='utf-8')
manifest = {'schema': 'C0-ten-outer-derived-curve-v1', 'created_unix': time.time(),
            'inputs': [file_ref(FACTS), file_ref(AGGREGATE)],
            'coverage': {'outer_rows': 10, 'actual_adam_updates': 100,
                         'pair_exposures': 3012, 'valid_policy_tokens': 583074,
                         'all_recorded_losses_and_norms_finite': True,
                         'continuous_update_ids': [1, 100]},
            'loss_contract': {'direction': 'student_to_target_reverse',
                              'target': 'frozen source100 privileged teacher',
                              'variant': 'C0', 'precision': 'bf16', 'mixture_alpha': 0.0,
                              'reduction': 'token mean within each trajectory, then equal trajectory mean',
                              'post_label': 'recorded clipped per-vocabulary-contribution objective, not raw KL',
                              'outer_aggregation': 'weighted by actual trajectory count, including tail updates'},
            'plot': plot_status,
            'scope': 'Local derived TRAIN scalars across ten different cohorts; not fixed heldout512 or scientific/causal acceptance',
            'artifacts': [file_ref(x) for x in [csv_path, note, Path(__file__).resolve()] + plot_files]}
manifest_path = OUT / 'derived-manifest.json'
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'manifest': file_ref(manifest_path), 'plot': plot_status}, ensure_ascii=False))
