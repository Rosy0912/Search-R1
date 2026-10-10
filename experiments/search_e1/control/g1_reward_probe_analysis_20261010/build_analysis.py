"""Build compact, reproducible charts from remote CPU aggregates; no raw samples."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(p):
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    p = args.base
    g = read(p / 'observation-2.json')
    calibration = read(p / 'calibration-distributions.json')
    archive = read(p / 'archive/observation-second.json')
    rounds = sorted(calibration['rounds'], key=lambda r: r['round'])
    assert calibration['same300_question_ids_across_all_rounds']
    assert len({r['question_ids_set_sha256'] for r in rounds}) == 1
    for r in rounds:
        for key in ['b_question_equal_question', 'd_valid_searched_rollout', 'ground_valid_searched_rollout']:
            v = r['distributions'][key]
            assert sum(v['histogram_counts']) == v['count']
            assert 0 <= v['min'] <= v['mean'] <= v['max'] <= 1
        assert sum(r['conditional_distributions'][k]['ground']['count'] for k in ['em_correct', 'em_wrong']) == r['cohort']['valid_searched_trajectory_count']
    train = g['training_log']['scalar_rows']
    exact = {r['step']: r for r in g['actual_updates']['complete_rows']}
    assert [r['step'] for r in train] == list(range(1, len(train) + 1))
    assert [r['iteration'] for r in archive['curve']] == list(range(1, 200))
    def points(rows, stepkey, metric, exact_metric=False):
        return [[r[stepkey], exact[r[stepkey]]['metrics'][metric] if exact_metric else r['metrics'][metric]] for r in rows if r[stepkey] <= 150]
    series = {
        'g1_em': points(train, 'step', 'train/perf/em'),
        'g1_reward': points(train, 'step', 'critic/rewards/mean', True),
        'g1_entropy': points(train, 'step', 'actor/entropy_loss', True),
        'g1_search': points(train, 'step', 'env/actual_retrieval_calls/mean'),
        'archive_em': points(archive['curve'], 'iteration', 'audit/strict_em'),
        'archive_reward': points(archive['curve'], 'iteration', 'critic/rewards/mean'),
        'archive_entropy': points(archive['curve'], 'iteration', 'actor/entropy_loss'),
        'archive_search': points(archive['curve'], 'iteration', 'env/number_of_valid_search'),
    }
    windows = []
    for lo, hi in [(1, 10), (41, 50), (91, 100), (131, 140)]:
        row = {'iterations': [lo, hi], 'g1': {}, 'archive': {}}
        for side in ['g1', 'archive']:
            for m in ['em', 'reward', 'entropy', 'search']:
                row[side][m] = statistics.mean(v for x, v in series[f'{side}_{m}'] if lo <= x <= hi)
        windows.append(row)
    slimrounds = [{
        'round': r['round'], 'version': r['version'], 'cohort': r['cohort'],
        'b': r['distributions']['b_question_equal_question'],
        'd': r['distributions']['d_valid_searched_rollout'],
        'ground': r['distributions']['ground_valid_searched_rollout'],
        'correct_ground_mean': r['conditional_distributions']['em_correct']['ground']['mean'],
        'wrong_ground_mean': r['conditional_distributions']['em_wrong']['ground']['mean'],
        'counterfactual_invalid_count': r['cohort']['valid_searched_trajectory_count'] - r['counterfactual_valid_count'],
        'cv_auc_b': r['report']['boundary_cv_auc'], 'cv_auc_d': r['report']['commit_cv_auc'],
        'b_head': r['heads']['boundary'], 'd_head': r['heads']['commit'],
        'manifest_sha256': r['manifest_sha256'], 'dump_sha256': r['dump_sha256'],
    } for r in rounds]
    direct_validation = read(p/'observation-4.json')['fixed_development_validation']
    archive_summary = read(p/'archive/summary.json')
    fixed_g1 = [[r['step'], r['weighted_em']] for r in direct_validation['rows']]
    assert next(y for x,y in fixed_g1 if x==50)==202/512
    assert next(y for x,y in fixed_g1 if x==100)==205/512
    fixed_archive = [[0, archive_summary['initial_fixed512']['EM']]] + [[r['iteration'],r['exact_em_from_unique_count']] for r in archive_summary['native_fixed512']]
    payload = {'stamp': '2026-10-10 10:54–10:58 CST', 'rounds': slimrounds, 'series': series, 'windows': windows,
               'fixed512': {'g1': fixed_g1, 'archive': fixed_archive},
               'limits': ['calibration fitted-sample predictions, not full historical TRAIN scores',
                          'same300 questions, fresh trajectories and replacement encoder/readout',
                          'reward scales and dev protocols are not controlled between runs',
                          'display scope0–150; archive has no b/d and is already human-stopped']}
    (p/'chart-data.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white'})
    colors = ['#2563eb', '#ea580c', '#0f766e']
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.4))
    for ax, key, title in zip(axs, ['b', 'd', 'ground'], ['b: closed-book readout', 'd: evidence-deletion sensitivity', 'g = d(1-b)']):
        for j, r in enumerate(slimrounds):
            v = r[key]
            edges = v['histogram_edges']
            # Uniform ten-bin probabilities, no interpolation to fake individual scores.
            y = [n/v['count'] for n in v['histogram_counts']]
            ax.stairs(y, edges, color=colors[j], linewidth=2.2, linestyle=['-', '--', ':'][j], label=f"round {r['round']} (n={v['count']})")
        ax.set(xlim=(0, 1), ylim=(0, .8), xlabel='Probe score', title=title)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', alpha=.15)
        ax.legend(fontsize=8, loc='upper center')
    axs[0].set_ylabel('Fraction in each 0.1-wide bin')
    fig.suptitle('Search-G1: score mass shifts toward low b and high d / g', fontsize=14, fontweight='bold', y=1.01)
    fig.text(.5, -.03, 'Matching fitted heads on fixed 300 calibration questions; fresh sampled trajectories. Not historical live TRAIN score distributions.', ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(p/'probe-distributions.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8.2))
    def curve(ax, key, label, color, style='-'):
        pts = series[key]
        ax.plot([x for x,y in pts], [y for x,y in pts], color=color, alpha=.18, lw=.8)
        sm = [[pts[i][0], statistics.mean(y for x,y in pts[i-4:i+1])] for i in range(4, len(pts))]
        ax.plot([x for x,y in sm], [y for x,y in sm], color=color, lw=2, ls=style, label=label+' (5-round mean)')
    curve(axs[0,0], 'g1_reward', 'G1 shaped reward', colors[0])
    curve(axs[0,0], 'g1_em', 'G1 TRAIN strict EM', colors[1], '--')
    curve(axs[0,0], 'archive_reward', 'Archive reward / strict EM', colors[2])
    axs[0,0].set(title='Reward and accuracy: different scales', ylim=(0, .75), ylabel='Reward / EM')
    for name,c in [('g1',colors[0]),('archive',colors[2])]:
        curve(axs[0,1], name+'_search', name.title()+' executed searches', c)
        curve(axs[1,0], name+'_entropy', name.title()+' actor entropy', c)
    axs[0,1].set(title='G1 converges near one search per trajectory', ylabel='Search count (run-specific scorer)')
    axs[1,0].set(title='Both policies become more concentrated', ylabel='Logged actor entropy', ylim=(0, 1))
    for name,c in [('g1',colors[0]),('archive',colors[2])]:
        pts=payload['fixed512'][name]
        axs[1,1].plot([x for x,y in pts], [y for x,y in pts], color=c, marker='o', ls='--', label=name.title())
        for x,y in pts: axs[1,1].annotate(f'{int(round(y*512))}/512', (x,y), xytext=(0, 8 if name=='archive' else -16), textcoords='offset points', ha='center', fontsize=8)
    axs[1,1].set(title='Each run\'s native fixed-dev512; protocol differences', ylabel='Strict EM', ylim=(0, .55))
    axs[1,1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axs.flat:
        ax.set_xlim(0, 150); ax.set_xlabel('Completed outer iteration'); ax.grid(alpha=.15); ax.legend(fontsize=8)
        ax.axvline(51, color='#64748b', lw=.8, ls=':'); ax.axvline(101, color='#64748b', lw=.8, ls=':')
    fig.suptitle('Search-G1 vs 4-card48GiB Search-E1 archive: iterations 0–150 (GRPO)', fontsize=13, fontweight='bold')
    fig.text(.5, .002, 'Grey: probe versions activate at 51 / 101. No extrapolation beyond observed G1 iteration140. Native dev is descriptive, not a controlled algorithm comparison.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.025,1,.97)); fig.savefig(p/'reward-and-behavior.png',dpi=180,bbox_inches='tight');plt.close(fig)
    inputs=['observation-2.json','calibration-distributions.json','archive/observation-second.json','observation-4.json','archive/summary.json']
    (p/'build-input-manifest.json').write_text(json.dumps({'inputs':[{ 'path':n, 'sha256':hashlib.sha256((p/n).read_bytes()).hexdigest()} for n in inputs], 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'assertions':'cohort identity; bin totals; finite bounded means; conditional counts; contiguous iteration history'},indent=2)+'\n')


if __name__=='__main__':
    main()
