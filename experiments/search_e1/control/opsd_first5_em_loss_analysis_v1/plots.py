"""Four SHA-bound CPU scientific figures from accepted first-five statistics.

No models, scoring, questions, trajectories, network or GPU. Aggregate input
values are preserved in figure-data.json; private paths only in inputs.private.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import stat
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GROUPS = ('C0', 'C1', 'M', 'pooled')
COLORS = dict(C0='#cc5b38', C1='#2879b9', M='#29946e', pooled='#606972')
MAIN = ('teacher_kl', 'target_post_loss', 'outer_old_kl', 'answer_student_logp',
        'answer_student_teacher_gap', 'student_entropy')
MAIN_LABELS = ('Teacher raw KL', 'Target clipped loss', 'Outer-old raw KL',
               'Answer student logp', 'Answer student-teacher gap', 'Student entropy')
REGIONS = ('think_text', 'search_query', 'final_answer')
REGION_LABELS = ('Think', 'Search query', 'Final answer')
CORE = ('ordinary_controls', 'ordinary_student_distributions', 'student_plus_reference_metadata',
        'student_plus_reference_and_teacher')
CORE_LABELS = ('Ordinary\ncontrols', '+ Student\ndistributions', '+ Reference\nmetadata', '+ Teacher\ndistributions')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()


def stamp(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode)


def fixed(ref):
    p = Path(ref['path'])
    require(p.is_absolute() and '..' not in p.parts, 'Absolute SHA-bound input required')
    current = Path(p.anchor)
    for part in p.parts[1:]:
        current /= part
        require(not current.is_symlink(), 'Symlink input refused')
    before = stamp(p.stat())
    require(stat.S_ISREG(before[-1]), 'Regular aggregate input required')
    with p.open('rb') as f:
        require(stamp(os.fstat(f.fileno())) == before, 'Input changed before read')
        raw = f.read()
        require(stamp(os.fstat(f.fileno())) == before, 'Input changed during read')
    require(stamp(p.stat()) == before, 'Input replaced after read')
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == ref['sha256'] and ('bytes' not in ref or len(raw) == ref['bytes']), 'Input SHA/bytes differs')
    return raw, dict(path=str(p), sha256=digest, bytes=len(raw), stat=list(before))


def correlation_item(summary, group, feature, outcome, regional=False):
    rows = summary['regional_token_mean_associations'] if regional else summary['univariate']
    selected = [r for r in rows if (r['group'], r['feature'], r['outcome']) == (group, feature, outcome)]
    require(len(selected) == 1, 'Unique aggregate correlation row required')
    row = selected[0]
    value = row['metrics']['pearson']
    interval = row['marginal_question_bootstrap_ci']['pearson']
    require(value is None or isinstance(value, (int, float)) and math.isfinite(value), 'Correlation must be finite or explicit NA')
    return dict(group=group, feature=feature, outcome=outcome, pearson=value,
                lower=interval['lower'], upper=interval['upper'], confidence=interval['confidence'],
                finite_replicates=interval['finite_replicates'], observed_pairs=row['observed_pairs'],
                structural_absence_pairs=row['structural_absence_pairs'])


def dots(ax, rows, labels, groups=GROUPS):
    offset = {g: (i - (len(groups)-1)/2) * .15 for i, g in enumerate(groups)}
    for i, label in enumerate(labels):
        for g in groups:
            row = rows[i, g]
            y = i + offset[g]
            if row['pearson'] is None:
                ax.text(-.97, y, 'NA', color=COLORS[g], fontsize=8)
                continue
            if row['lower'] is not None and row['upper'] is not None:
                ax.hlines(y, row['lower'], row['upper'], color=COLORS[g], linewidth=1.5, alpha=.9)
            ax.scatter([row['pearson']], [y], color=COLORS[g], s=26, zorder=3)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels)-.45, -.5)
    ax.set_xlim(-1.02, 1.02)
    ax.axvline(0., color='#777777', linewidth=.8, linestyle='--')
    ax.grid(axis='x', color='#cccccc', alpha=.5)
    ax.set_xlabel('Pearson r (95% marginal question-bootstrap interval)')


def legend(fig, groups=GROUPS, y=.92):
    fig.legend(handles=[Line2D([], [], marker='o', linestyle='', color=COLORS[g], label=g) for g in groups],
               loc='upper center', bbox_to_anchor=(.5, y), ncol=len(groups), frameon=False)


def save(fig, out, name, footer):
    fig.text(.5, .018, footer, ha='center', va='bottom', fontsize=8, color='#555555')
    fig.savefig(out / (name + '.png'), dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(out / (name + '.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_main(summary, out):
    rows = {(i, g): correlation_item(summary, g, name, 'sampling_siblings_success_fraction')
            for i, name in enumerate(MAIN) for g in GROUPS}
    fig, ax = plt.subplots(figsize=(10.6, 6.6))
    dots(ax, rows, MAIN_LABELS)
    fig.suptitle('First five outers: feature association with empirical five-sibling success fraction', fontsize=12)
    legend(fig)
    fig.subplots_adjust(top=.82, left=.25, right=.96, bottom=.16)
    save(fig, out, '01-main-sampling-fraction-associations',
         'Eligible native D only; s/5 is observed outer-start sampling, not future success probability. Intervals condition on fixed runs.')
    return list(rows.values())


def plot_regions(summary, out):
    metrics = ('teacher_kl', 'student_observed_logp', 'student_teacher_observed_logp_gap')
    labels = ('Teacher KL', 'Student observed logp', 'Student - teacher observed logp')
    outcomes = ('selected_D_strict_em', 'sampling_siblings_success_fraction')
    fig, axes = plt.subplots(3, 2, figsize=(13., 11.4))
    data = []
    for j, (metric, label) in enumerate(zip(metrics, labels)):
        for k, outcome in enumerate(outcomes):
            rows = {(i, g): correlation_item(summary, g, region+'::'+metric, outcome, True)
                    for i, region in enumerate(REGIONS) for g in GROUPS[:3]}
            dots(axes[j, k], rows, REGION_LABELS, groups=GROUPS[:3])
            axes[j, k].set_title(label + ' vs ' + ('selected-D strict EM' if k == 0 else 'empirical s/5'), fontsize=10)
            axes[j, k].set_xlabel('Pearson r (95% marginal interval)', fontsize=9)
            data.extend(rows.values())
    fig.suptitle('Where the recorded distributions differ: regional token means', fontsize=13)
    legend(fig, groups=GROUPS[:3], y=.965)
    fig.subplots_adjust(top=.91, left=.12, right=.98, bottom=.08, hspace=.55, wspace=.48)
    save(fig, out, '02-regional-distribution-associations',
         'Absent-region token means are NA. Selected D correctness is constrained by native mining. Correlation is not a regional causal effect.')
    return data


def plot_oof(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6))
    data = []
    for gi, group in enumerate(GROUPS):
        auc = [summary['oof']['metrics'][group][family]['auc'] for family in CORE]
        rmse = [summary['oof']['sampling_fraction_oof'][group]['metrics'][family]['rmse'] for family in CORE]
        require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in auc + rmse), 'Finite complete four-family OOF metrics required')
        for ax, values in zip(axes, (auc, rmse)):
            ax.plot(range(4), values, color=COLORS[group], marker='o', linewidth=1.6, label=group)
            ax.set_xticks(range(4), CORE_LABELS)
            ax.grid(axis='y', color='#cccccc', alpha=.5)
        for family, a, r in zip(CORE, auc, rmse):
            data.append(dict(group=group, family=family, binary_selected_D_auc=a, empirical_s_over5_rmse=r))
    axes[0].set_ylabel('OOF AUC: recorded selected-D strict EM')
    axes[0].set_ylim(0., 1.)
    axes[0].axhline(.5, color='#777777', linestyle='--', linewidth=.8)
    axes[0].set_title('Completed-trajectory correctness discrimination')
    axes[1].set_ylabel('OOF RMSE: empirical five-sibling s/5')
    axes[1].set_ylim(bottom=0.)
    axes[1].set_title('Conditional empirical-fraction regression (lower is better)')
    fig.suptitle('Question-block OOF: ordinary evidence, reference metadata and teacher distributions', fontsize=12)
    legend(fig, y=.93)
    fig.subplots_adjust(top=.78, bottom=.24, left=.07, right=.97, wspace=.30)
    save(fig, out, '03-oof-information-increments',
         'Points are actual OOF estimates, not fitting-set scores. Increment CIs are in figure-data.json. Retrospective only; no sampling calibration claim.')
    return dict(point_metrics=data, increment_intervals=summary['oof']['increments'],
                empirical_fraction_increment_intervals={g: summary['oof']['sampling_fraction_oof'][g]['increments'] for g in GROUPS})


def plot_bins(bins, out):
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.))
    data = []
    for col, feature in enumerate(('teacher_kl', 'answer_student_logp')):
        for group in GROUPS[:3]:
            rows = [r for r in bins if r['group'] == group and r['feature'] == feature and r['bin'] != 'structural_absence']
            rows.sort(key=lambda r: int(r['bin']))
            require(rows, 'Completed descriptive bins required')
            parsed = []
            for r in rows:
                lo, hi = float(r['observed_min']), float(r['observed_max'])
                x = .5 * (lo + hi)
                if feature == 'answer_student_logp':
                    lo, hi, x = math.exp(lo), math.exp(hi), math.exp(x)
                parsed.append(dict(group=group, feature=feature, bin=int(r['bin']), pairs=int(r['pairs']),
                    x=x, observed_min=lo, observed_max=hi, strict_correct_fraction=float(r['strict_correct_fraction']),
                    sampling_success_fraction_mean=float(r['sampling_success_fraction_mean']),
                    representative_scope='Bin midpoint in original feature; answer probability is exp(midpoint logp), not arithmetic probability mean.'))
            data.extend(parsed)
            for row, key in enumerate(('strict_correct_fraction', 'sampling_success_fraction_mean')):
                axes[row, col].plot([r['x'] for r in parsed], [r[key] for r in parsed], marker='o', color=COLORS[group], linewidth=1.4)
                axes[row, col].set_ylim(0., 1.)
                axes[row, col].grid(color='#cccccc', alpha=.4)
                axes[row, col].set_xlabel('Teacher raw KL bin midpoint' if col == 0 else 'Answer-token geometric probability (student)')
        axes[0, col].set_ylabel('Selected-D strict-correct fraction')
        axes[1, col].set_ylabel('Mean empirical five-sibling s/5')
    axes[0, 0].set_title('Teacher divergence bins')
    axes[0, 1].set_title('Recorded answer confidence bins')
    fig.suptitle('Descriptive confidence/divergence relationship on eligible trajectories', fontsize=12)
    legend(fig, groups=GROUPS[:3], y=.95)
    fig.subplots_adjust(top=.87, bottom=.14, left=.08, right=.98, wspace=.25, hspace=.45)
    save(fig, out, '04-answer-probability-and-teacher-kl-bins',
         'Per-group descriptive quantile bins; counts and absence rows retained in JSON. Geometric token probability is not trajectory EM probability.')
    return dict(observed_bins=data, structural_absence_bins=[r for r in bins if r['feature'] == 'answer_student_logp' and r['bin'] == 'structural_absence'])


def run(manifest_ref, output):
    raw, mp = fixed(manifest_ref)
    manifest = json.loads(raw)
    require(manifest['schema'] == 'opsd-first5-em-loss-plot-inputs-v1', 'Plot manifest schema differs')
    sr, sp = fixed(manifest['summary']); br, bp = fixed(manifest['bins'])
    summary = json.loads(sr)
    require(summary['schema'] == 'opsd-first5-em-loss-stats-v1' and summary['accepted'] is True
            and summary['pair_rows'] == 4529 and summary['cohort_rows'] == 7680, 'Accepted exact first-five statistics required')
    bins = list(csv.DictReader(io.StringIO(br.decode())))
    require(bins, 'Descriptive bin CSV empty')
    out = Path(output)
    require(out.is_absolute() and not out.exists(), 'Absolute new plot output required')
    out.mkdir(parents=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    figures = dict(main_sampling_associations=plot_main(summary, out), regional_associations=plot_regions(summary, out),
                   oof_information=plot_oof(summary, out), confidence_bins=plot_bins(bins, out))
    data = dict(schema='opsd-first5-em-loss-scientific-figures-v1', accepted=True,
                input_sha256=dict(summary=sp['sha256'], bins=bp['sha256']),
                figures=figures, interpretation='Eligible retrospective training-trajectory analysis; not new inference or heldout causal effects.')
    (out / 'figure-data.json').write_bytes(canonical(data) + b'\n')
    (out / 'inputs.private.json').write_bytes(canonical(dict(manifest=mp, summary=sp, bins=bp)) + b'\n')
    artifacts = {p.name: dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size) for p in out.iterdir()}
    (out / 'derived-manifest.json').write_bytes(canonical(dict(schema='opsd-first5-em-loss-figure-artifacts-v1',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), artifacts=artifacts)) + b'\n')
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True); p.add_argument('--manifest-sha256', required=True); p.add_argument('--output', required=True)
    a = p.parse_args()
    result = run(dict(path=a.manifest, sha256=a.manifest_sha256), a.output)
    print(json.dumps(dict(accepted=result['accepted'], figures=4, png=4, svg=4), sort_keys=True))


if __name__ == '__main__':
    main()
