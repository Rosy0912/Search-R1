"""CPU-only retrospective first-five-outer correctness/loss statistics.

No models, tokenizers, scoring, GPU, network or training imports/actions. Explicit
SHA-bound extractor files are the only inputs. Predictions concern recorded,
already generated D trajectories, not pre-generation calibrated success odds.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import stat

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

GROUPS = ('C0', 'C1', 'M')
MAIN = ('teacher_kl', 'target_post_loss', 'outer_old_kl',
        'answer_student_logp', 'answer_student_teacher_gap', 'student_entropy')
STUDENT = ('outer_old_kl', 'answer_student_logp', 'student_entropy')
PRIVILEGED = ('teacher_kl', 'target_post_loss', 'answer_student_teacher_gap')
PRIMARY_REGIONS = ('think_text', 'search_query', 'final_answer')
REGIONS = ('all_policy', 'think_text', 'search_query', 'structural_tags',
           'final_answer', 'other', 'boundary')
CONTRACT = 'opsd-first5-em-loss-stats-v1'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode()


def finite(x):
    return x is not None and not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


def number(x):
    if x is None:
        return np.nan
    require(finite(x), 'Nonfinite/non-numeric feature')
    return float(x)


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    return x


def stamp(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode)


def fixed_bytes(ref):
    p = Path(ref['path'])
    require(p.is_absolute() and '..' not in p.parts, 'Explicit canonical absolute input path required')
    current = Path(p.anchor)
    for part in p.parts[1:]:
        current /= part
        require(not current.is_symlink(), 'Symlink input refused')
    before = stamp(p.stat())
    require(stat.S_ISREG(before[-1]), 'Regular file input required')
    with p.open('rb') as f:
        require(stamp(os.fstat(f.fileno())) == before, 'Input replaced before read')
        raw = f.read()
        require(stamp(os.fstat(f.fileno())) == before, 'Input changed during read')
    require(stamp(p.stat()) == before, 'Input replaced after read')
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == ref['sha256'] and ('bytes' not in ref or len(raw) == ref['bytes']), 'Input SHA/size differs')
    return raw, dict(path=str(p), sha256=digest, bytes=len(raw), stat=list(before))


def rows(ref):
    raw, proof = fixed_bytes(ref)
    require(raw.endswith(b'\n'), 'Partial JSONL input refused')
    result = [json.loads(line) for line in raw.splitlines()]
    require(all(isinstance(r, dict) for r in result), 'JSONL object rows required')
    return result, proof


def validate(pairs, cohorts):
    ci = {}
    qident = {}
    totals = Counter()
    for c in cohorts:
        require(c['schema'] == 'opsd-first5-cohort-feature-v1', 'Cohort schema differs')
        g, o, q = c['group'], c['outer_index'], c['question_id']
        require(g in GROUPS and type(o) is int and 0 <= o < 5 and isinstance(q, str) and q, 'Cohort group/outer/qid differs')
        key = (g, q)
        require(key not in ci, 'Duplicate cohort group/question')
        rewards = c['raw_rewards']
        require(len(rewards) == 5 and all(type(v) is int and v in (0, 1) for v in rewards), 'Five strict binary native rewards required')
        s = sum(rewards)
        require(c['siblings_success_count'] == s and math.isclose(c['sampling_success_fraction'], s / 5), 'Cohort success fields differ')
        require(type(c['loss_available']) is bool and (not c['loss_available'] or s > 0), 'All-wrong cohort has fabricated loss')
        require((o, c['data_source']) == qident.setdefault(q, (o, c['data_source'])), 'Common cohort question identity changed')
        ci[key] = c
        totals[g, o] += 1
    require(set(totals) == {(g, o) for g in GROUPS for o in range(5)} and set(totals.values()) == {512}, 'Exactly three groups x five complete512 cohorts required')
    require(all({q for g2, q in ci if g2 == g} == {q for g2, q in ci if g2 == 'C0'} for g in GROUPS), 'Different question cohorts')
    seen = set()
    for r in pairs:
        require(r['schema'] == 'opsd-first5-pair-feature-v1', 'Pair schema differs')
        g, q = r['group'], r['question_id']
        key = (g, q)
        require(key in ci and key not in seen, 'Foreign/duplicate pair')
        seen.add(key)
        c = ci[key]
        require(c['loss_available'] and r['outer_index'] == c['outer_index'] and r['data_source'] == c['data_source'], 'Pair/cohort identity differs')
        require(type(r['selected_student_strict_em']) is int and r['selected_student_strict_em'] in (0, 1) and r['reference_strict_em'] == 1, 'Native strict labels/reference differ')
        require(r['siblings_success_count'] == c['siblings_success_count'] and math.isclose(r['sampling_success_fraction'], c['sampling_success_fraction']), 'Pair/cohort native outcome differs')
        require(c['selected_student_strict_em'] == r['selected_student_strict_em'], 'Selected native strict label differs')
        require(type(r['valid_tokens']) is int and r['valid_tokens'] > 0 and set(r['regions']) == set(REGIONS), 'Pair regions/token denominator differs')
        require(finite(r['normalized_update_progress']) and 0 <= r['normalized_update_progress'] <= 1, 'Explicit pre-update progress required')
        for name, region in r['regions'].items():
            n = region['valid_tokens']
            require(type(n) is int and n >= 0, 'Region count differs')
            if name == 'all_policy':
                require(n == r['valid_tokens'], 'Whole-policy denominator differs')
            for metric in ('teacher_kl', 'target_post_loss', 'outer_old_kl', 'student_entropy',
                           'student_observed_logp', 'student_teacher_observed_logp_gap',
                           'clipped_vocabulary_count', 'clipped_vocabulary_fraction',
                           'clipped_student_probability_mass', 'clipped_teacher_probability_mass',
                           'clipped_target_probability_mass', 'clip_loss_removed'):
                mean = region['means'][metric]
                require((n > 0 and finite(mean)) or (n == 0 and mean is None), 'Region absence/nonfinite scalar misrepresented')
                contribution = region['trajectory_contributions'][metric]
                require(finite(contribution) and (n > 0 or contribution == 0), 'Legal region absence must contribute zero')
    require(seen == {k for k, c in ci.items() if c['loss_available']}, 'Missing pair feature for available native loss')
    return ci, qident


def feature(r, name):
    if '::' in name:
        region, metric = name.split('::')
        return number(r['regions'][region]['means'][metric])
    region = r['regions']['final_answer' if name.startswith('answer_') else 'all_policy']
    key = {'answer_student_logp': 'student_observed_logp',
           'answer_student_teacher_gap': 'student_teacher_observed_logp_gap'}.get(name, name)
    return number(region['means'][key])


def weighted_corr(x, y, w):
    z = w.sum()
    if z <= 1:
        return np.nan
    mx, my = np.dot(w, x) / z, np.dot(w, y) / z
    a, b = x - mx, y - my
    den = math.sqrt(max(0., np.dot(w, a * a) * np.dot(w, b * b)))
    return np.dot(w, a * b) / den if den > 0 else np.nan


class Ranking:
    """Weighted midranks/AUC from exact recorded-value ties, no random jitter."""
    def __init__(self, x):
        _, self.groups = np.unique(x, return_inverse=True)
        self.k = int(self.groups.max()) + 1

    def ranks(self, w):
        masses = np.bincount(self.groups, weights=w, minlength=self.k)
        return (np.cumsum(masses) - .5 * masses)[self.groups]

    def auc(self, y, w):
        p = np.bincount(self.groups, weights=w * y, minlength=self.k)
        n = np.bincount(self.groups, weights=w * (1 - y), minlength=self.k)
        denom = p.sum() * n.sum()
        return np.dot(p, np.cumsum(n) - .5 * n) / denom if denom > 0 else np.nan


def stats_values(x, y, w, rank, binary):
    values = dict(pearson=weighted_corr(x, y, w),
                  spearman=weighted_corr(rank.ranks(w), Ranking(y).ranks(w), w))
    if binary:
        n1, n0 = np.dot(w, y), np.dot(w, 1 - y)
        m1 = np.dot(w * y, x) / n1 if n1 else np.nan
        m0 = np.dot(w * (1 - y), x) / n0 if n0 else np.nan
        v1 = np.dot(w * y, (x - m1) ** 2) / max(n1 - 1, 1) if n1 else np.nan
        v0 = np.dot(w * (1 - y), (x - m0) ** 2) / max(n0 - 1, 1) if n0 else np.nan
        pooled = ((n1 - 1) * v1 + (n0 - 1) * v0) / max(n1 + n0 - 2, 1) if min(n1, n0) else np.nan
        values.update(auc_higher_feature_means_correct=rank.auc(y, w), mean_correct=m1,
                      mean_incorrect=m0, mean_correct_minus_incorrect=m1 - m0,
                      cohen_d=(m1 - m0) / math.sqrt(pooled) if pooled > 0 else np.nan)
    return values


def ci(values):
    a = np.asarray(values, dtype=float)
    valid = a[np.isfinite(a)]
    return dict(lower=float(np.quantile(valid, .025)) if len(valid) else None,
                upper=float(np.quantile(valid, .975)) if len(valid) else None,
                finite_replicates=len(valid), total_replicates=len(a), confidence=.95)


def boot_weights(qident, reps, seed):
    qids = sorted(qident)
    qi = {q: i for i, q in enumerate(qids)}
    strata = defaultdict(list)
    for q in qids:
        strata[qident[q]].append(qi[q])
    rng = np.random.default_rng(seed)
    result = np.zeros((reps, len(qids)), dtype=np.uint16)
    for indices in strata.values():
        idx = np.asarray(indices)
        for b in range(reps):
            sampled = rng.choice(idx, size=len(idx), replace=True)
            result[b] += np.bincount(sampled, minlength=len(qids)).astype(np.uint16)
    return qids, qi, result


def fold_questions(qident, folds, seed):
    strata = defaultdict(list)
    for q in sorted(qident):
        strata[qident[q]].append(q)
    rng = np.random.default_rng(seed)
    assigned = {}
    for questions in strata.values():
        questions = list(questions)
        rng.shuffle(questions)
        for i, q in enumerate(questions):
            assigned[q] = i % folds
    require(set(assigned.values()) == set(range(folds)), 'Insufficient question blocks for CV')
    return assigned


def control_values(r, reference_metadata=False):
    out = [math.log1p(r['valid_tokens']), number(r['student_search_calls']),
           number(r['normalized_update_progress'])]
    if reference_metadata:
        out += [number(r['reference_search_calls']), math.log1p(r['reference_response_characters']),
                number(r['reference_token_count'])]
    return out


def design(train, test, names, reference_metadata=False):
    """Fit every category, imputation and standardization on the train fold."""
    tr, te = np.asarray([control_values(r, reference_metadata) for r in train]), np.asarray([control_values(r, reference_metadata) for r in test])
    median = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0. for c in tr.T])
    missing_tr, missing_te = ~np.isfinite(tr), ~np.isfinite(te)
    tr, te = np.where(missing_tr, median, tr), np.where(missing_te, median, te)
    mean, sd = tr.mean(0), tr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.)
    xtr = [np.ones((len(train), 1)), (tr - mean) / sd, missing_tr.astype(float)]
    xte = [np.ones((len(test), 1)), (te - mean) / sd, missing_te.astype(float)]
    proof = dict(reference_metadata_controls=reference_metadata, control_medians=median.tolist(), control_means=mean.tolist(), control_sd=sd.tolist(), category_levels={})
    for key in ('group', 'data_source', 'outer_index'):
        levels = sorted({str(r[key]) for r in train})
        proof['category_levels'][key] = levels
        for level in levels[1:]:
            xtr.append(np.array([[float(str(r[key]) == level)] for r in train]))
            xte.append(np.array([[float(str(r[key]) == level)] for r in test]))
        xtr.append(np.zeros((len(train), 1)))
        xte.append(np.array([[float(str(r[key]) not in levels)] for r in test]))
    proof['feature_transforms'] = {}
    for name in names:
        a, b = np.array([feature(r, name) for r in train]), np.array([feature(r, name) for r in test])
        am, bm = ~np.isfinite(a), ~np.isfinite(b)
        median_f = float(np.nanmedian(a)) if np.isfinite(a).any() else 0.
        a, b = np.where(am, median_f, a), np.where(bm, median_f, b)
        mu, scale = float(a.mean()), float(a.std())
        scale = scale if scale > 1e-12 else 1.
        a, b = (a - mu) / scale, (b - mu) / scale
        proof['feature_transforms'][name] = dict(median=median_f, mean=mu, sd=scale,
                                                 train_observed=int((~am).sum()), test_observed=int((~bm).sum()))
        xtr.extend([a[:, None], am[:, None].astype(float)])
        xte.extend([b[:, None], bm[:, None].astype(float)])
        # Group-specific slopes; these use the same train-only numeric transform.
        for group in ('C1', 'M'):
            gt, ge = np.array([r['group'] == group for r in train]), np.array([r['group'] == group for r in test])
            xtr.extend([(a * gt)[:, None], (am * gt)[:, None].astype(float)])
            xte.extend([(b * ge)[:, None], (bm * ge)[:, None].astype(float)])
    return np.hstack(xtr), np.hstack(xte), proof


def fit_logistic(x, y, regularization=1.):
    require(np.isfinite(y).all() and y.min() >= 0 and y.max() <= 1 and np.ptp(y) > 0,
            'Nonconstant bounded binary/fractional outcomes required in training fold')
    penalty = np.ones(x.shape[1]); penalty[0] = 0.
    def f(beta):
        z = x @ beta
        loss = np.logaddexp(0., z).sum() - np.dot(y, z) + .5 * regularization * np.dot(penalty * beta, beta)
        grad = x.T @ (expit(z) - y) + regularization * penalty * beta
        return loss, grad
    initial = np.zeros(x.shape[1]); initial[0] = math.log(y.mean() / (1 - y.mean()))
    result = minimize(f, initial, jac=True, method='L-BFGS-B',
                      options=dict(maxiter=400, ftol=1e-10, gtol=1e-6, maxls=40))
    require(result.success and np.isfinite(result.x).all(), 'CPU logistic convergence failed: ' + str(result.message))
    return result.x, dict(success=True, iterations=int(result.nit), objective=float(result.fun),
                          slope_l2_lambda=regularization, objective_scale='sum binary logloss + lambda/2 sum nonintercept_beta^2')


def prediction_metrics(y, p, w, rank=None):
    eps = np.finfo(float).eps
    clipped = np.clip(p, eps, 1 - eps)
    return dict(auc=(rank or Ranking(p)).auc(y, w), brier=np.dot(w, (p - y) ** 2) / w.sum(),
                logloss=np.dot(w, -y * np.log(clipped) - (1 - y) * np.log1p(-clipped)) / w.sum())


def fraction_metrics(y, p, w):
    mse = np.dot(w, (p - y) ** 2) / w.sum()
    return dict(brier_fraction=mse, rmse=math.sqrt(mse), pearson_prediction_to_s_over5=weighted_corr(p, y, w))


def oof(pairs, qident, qi, weights, folds, seed):
    assigned = fold_questions(qident, folds, seed)
    family = {'ordinary_controls': ((), False), 'ordinary_student_distributions': (STUDENT, False),
              'student_plus_reference_metadata': (STUDENT, True), 'student_plus_reference_and_teacher': (MAIN, True)}
    for region in PRIMARY_REGIONS:
        family['teacher_region_' + region] = (STUDENT + tuple(region + '::' + k for k in
            ('teacher_kl', 'student_teacher_observed_logp_gap', 'target_post_loss')), True)
    preds = {name: np.full(len(pairs), np.nan) for name in family}
    core = tuple(list(family)[:4])
    fraction_preds = {name: np.full(len(pairs), np.nan) for name in core}
    fits = []
    for fold in range(folds):
        tr = [i for i, r in enumerate(pairs) if assigned[r['question_id']] != fold]
        te = [i for i, r in enumerate(pairs) if assigned[r['question_id']] == fold]
        train, test = [pairs[i] for i in tr], [pairs[i] for i in te]
        require(not ({r['question_id'] for r in train} & {r['question_id'] for r in test}), 'Question CV leakage')
        y = np.array([r['selected_student_strict_em'] for r in train], dtype=float)
        yf = np.array([r['sampling_success_fraction'] for r in train], dtype=float)
        for name, (features, reference_metadata) in family.items():
            a, b, transform = design(train, test, features, reference_metadata)
            beta, fit = fit_logistic(a, y)
            preds[name][te] = expit(b @ beta)
            fits.append(dict(fold=fold, family=name, train_rows=len(tr), test_rows=len(te),
                             train_questions=len({r['question_id'] for r in train}),
                             test_questions=len({r['question_id'] for r in test}), outcome='selected_D_strict_em', preprocessing=transform, fit=fit))
            if name in core:
                fb, ff = fit_logistic(a, yf)
                fraction_preds[name][te] = expit(b @ fb)
                fits.append(dict(fold=fold, family=name, outcome='empirical_s_over5', train_rows=len(tr), test_rows=len(te),
                                 preprocessing=transform, fit=ff,
                                 scope='Fractional cross-entropy as retrospective regression; no independent-five-rollout likelihood assumption.'))
    require(all(np.isfinite(p).all() for p in preds.values()), 'Incomplete OOF predictions')
    require(all(np.isfinite(p).all() for p in fraction_preds.values()), 'Incomplete fractional OOF predictions')
    y = np.array([r['selected_student_strict_em'] for r in pairs], dtype=float)
    idx = np.array([qi[r['question_id']] for r in pairs])
    metrics = {}
    increments = {}
    fractional = {}
    fraction_y = np.array([r['sampling_success_fraction'] for r in pairs], dtype=float)
    for group in ('pooled',) + GROUPS:
        mask = np.ones(len(pairs), dtype=bool) if group == 'pooled' else np.array([r['group'] == group for r in pairs])
        yy, ids = y[mask], idx[mask]
        ranks = {name: Ranking(p[mask]) for name, p in preds.items()}
        baseline = {name: prediction_metrics(yy, p[mask], np.ones(mask.sum()), ranks[name]) for name, p in preds.items()}
        metrics[group] = baseline
        comparisons = [('ordinary_controls', 'ordinary_student_distributions'),
                       ('ordinary_student_distributions', 'student_plus_reference_metadata'),
                       ('student_plus_reference_metadata', 'student_plus_reference_and_teacher'),
                       ('ordinary_controls', 'student_plus_reference_and_teacher')]
        comparisons += [('student_plus_reference_metadata', 'teacher_region_' + r) for r in PRIMARY_REGIONS]
        for left, right in comparisons:
            name = left + '->' + right
            values = {k: [] for k in ('auc', 'brier', 'logloss')}
            for w in weights[:, ids]:
                a, b = prediction_metrics(yy, preds[left][mask], w, ranks[left]), prediction_metrics(yy, preds[right][mask], w, ranks[right])
                for k in values:
                    values[k].append(b[k] - a[k])
            increments.setdefault(group, {})[name] = {k: dict(delta=baseline[right][k] - baseline[left][k], ci=ci(v),
                benefit_direction='positive' if k == 'auc' else 'negative') for k, v in values.items()}
        fy = fraction_y[mask]
        fm = {name: fraction_metrics(fy, p[mask], np.ones(mask.sum())) for name, p in fraction_preds.items()}
        fi = {}
        for left, right in comparisons[:4]:
            values = {k: [] for k in ('brier_fraction', 'rmse', 'pearson_prediction_to_s_over5')}
            for w in weights[:, ids]:
                a, b = fraction_metrics(fy, fraction_preds[left][mask], w), fraction_metrics(fy, fraction_preds[right][mask], w)
                for k in values:
                    values[k].append(b[k] - a[k])
            fi[left + '->' + right] = {k: dict(delta=fm[right][k] - fm[left][k], ci=ci(v),
                benefit_direction='positive' if k.startswith('pearson') else 'negative') for k, v in values.items()}
        fractional[group] = dict(metrics=fm, increments=fi)
    records = [dict(group=r['group'], question_id=r['question_id'], outer_index=r['outer_index'],
                    selected_student_strict_em=r['selected_student_strict_em'], sampling_success_fraction=r['sampling_success_fraction'],
                    fold=assigned[r['question_id']], predictions={name: float(p[i]) for name, p in preds.items()},
                    empirical_fraction_predictions={name: float(p[i]) for name, p in fraction_preds.items()})
               for i, r in enumerate(pairs)]
    return dict(metrics=metrics, increments=increments, fits=fits, question_block_folds=folds,
                feature_families={k: dict(distribution_features=list(v[0]), reference_metadata_controls=v[1]) for k, v in family.items()},
                sampling_fraction_oof=fractional,
                scope='Retrospective discrimination of completed selected D, not ex ante sampling success/calibration; privileged teacher features use successful B.',
                bootstrap_scope='Conditional fixed OOF predictions, not repeated refitting or independent training-run variance.'), records


def selection(pairs, cohorts):
    result = {}
    for group in GROUPS:
        cs, ps = [r for r in cohorts if r['group'] == group], [r for r in pairs if r['group'] == group]
        bad = [r for r in ps if r['selected_student_strict_em'] != int(r['siblings_success_count'] == 5)]
        result[group] = dict(cohort_questions=len(cs), sampled_trajectories=5 * len(cs), eligible_loss_questions=len(ps),
            native_correct_siblings=sum(r['siblings_success_count'] for r in cs),
            sampling_pass1_empirical=sum(r['siblings_success_count'] for r in cs) / (5 * len(cs)),
            sampling_pass_at5_empirical=sum(r['siblings_success_count'] > 0 for r in cs) / len(cs),
            sibling_success_histogram=dict(Counter(r['siblings_success_count'] for r in cs)),
            missing_loss_reasons=dict(Counter(r['loss_missing_reason'] for r in cs if not r['loss_available'])),
            selected_D_correct=sum(r['selected_student_strict_em'] for r in ps),
            D_correct_iff_all5_correct_checked_pairs=len(ps), D_correct_iff_all5_correct_violations=len(bad),
            association_selection_boundary='D correctness is constrained by native failed-priority selection, not an unbiased sample of generated-trajectory correctness.',
            loss_missingness='All-five-wrong questions have no invented loss; conditional eligible metrics do not estimate the full cohort.')
    return result


def univariate(pairs, qi, weights, features=MAIN, include_bins=True):
    output, bins = [], []
    for group in ('pooled',) + GROUPS:
        rs = pairs if group == 'pooled' else [r for r in pairs if r['group'] == group]
        for name in features:
            observed = [r for r in rs if math.isfinite(feature(r, name))]
            x = np.array([feature(r, name) for r in observed])
            ids = np.array([qi[r['question_id']] for r in observed])
            require(len(x) > 1, 'Insufficient observed main feature')
            rank = Ranking(x)
            for outcome, binary in (('selected_D_strict_em', True), ('sampling_siblings_success_fraction', False)):
                y = np.array([r['selected_student_strict_em'] if binary else r['sampling_success_fraction'] for r in observed], dtype=float)
                point = stats_values(x, y, np.ones(len(x)), rank, binary)
                bootstrap = {k: [] for k in point}
                for w in weights[:, ids]:
                    for key, value in stats_values(x, y, w, rank, binary).items():
                        bootstrap[key].append(value)
                output.append(dict(group=group, feature=name, outcome=outcome, observed_pairs=len(observed),
                                   structural_absence_pairs=len(rs) - len(observed), metrics=point,
                                   marginal_question_bootstrap_ci={k: ci(v) for k, v in bootstrap.items()}))
            if not include_bins:
                continue
            # Full-sample deciles are descriptive, not prediction preprocessing.
            cuts = np.unique(np.quantile(x, np.linspace(0., 1., 11)))
            bucket = np.searchsorted(cuts[1:-1], x, side='right')
            for b in sorted(set(bucket)):
                chosen = [r for i, r in enumerate(observed) if bucket[i] == b]
                xx = x[bucket == b]
                bins.append(dict(group=group, feature=name, bin=int(b), observed_min=float(xx.min()), observed_max=float(xx.max()),
                                 pairs=len(chosen), strict_correct=sum(r['selected_student_strict_em'] for r in chosen),
                                 strict_correct_fraction=np.mean([r['selected_student_strict_em'] for r in chosen]),
                                 sampling_success_fraction_mean=np.mean([r['sampling_success_fraction'] for r in chosen]),
                                 scope='Full-observed-sample descriptive quantile bin; not OOF or calibrated ex ante probability.'))
            if len(observed) < len(rs):
                absent = [r for r in rs if not math.isfinite(feature(r, name))]
                bins.append(dict(group=group, feature=name, bin='structural_absence', pairs=len(absent),
                                 strict_correct=sum(r['selected_student_strict_em'] for r in absent),
                                 strict_correct_fraction=np.mean([r['selected_student_strict_em'] for r in absent]),
                                 sampling_success_fraction_mean=np.mean([r['sampling_success_fraction'] for r in absent]),
                                 scope='No textual final_answer-region token; observed logp is NA, not zero.'))
    return output, bins


def outer_trends(pairs):
    """15 observed group/outer cells, descriptive only, no lag or heldout claim."""
    result = []
    for group in GROUPS:
        for outer in range(5):
            rs = [r for r in pairs if r['group'] == group and r['outer_index'] == outer]
            for name in MAIN:
                observed = [r for r in rs if math.isfinite(feature(r, name))]
                x = np.array([feature(r, name) for r in observed])
                for outcome, binary in (('selected_D_strict_em', True), ('sampling_siblings_success_fraction', False)):
                    y = np.array([r['selected_student_strict_em'] if binary else r['sampling_success_fraction'] for r in observed], dtype=float)
                    metrics = stats_values(x, y, np.ones(len(x)), Ranking(x), binary) if len(x) > 1 else {}
                    result.append(dict(group=group, outer_index=outer, feature=name, outcome=outcome,
                                       observed_pairs=len(observed), structural_absence_pairs=len(rs)-len(observed), metrics=metrics,
                                       scope='Observed outer cohort, no CI/significance or cross-outer lag/learning claim.'))
    return result


def regions(pairs, qi, weights):
    exploratory = []
    primary = []
    by_group = {g: {r['question_id']: r for r in pairs if r['group'] == g} for g in GROUPS}
    for g in GROUPS:
        rs = list(by_group[g].values())
        y = np.array([r['selected_student_strict_em'] for r in rs], dtype=float)
        for region in PRIMARY_REGIONS + ('structural_tags',):
            for metric in ('teacher_kl', 'target_post_loss', 'outer_old_kl'):
                x = np.array([r['regions'][region]['trajectory_contributions'][metric] for r in rs], dtype=float)
                exploratory.append(dict(group=g, region=region, metric=metric, questions=len(rs),
                                        pearson=weighted_corr(x, y, np.ones(len(rs))),
                                        spearman=weighted_corr(Ranking(x).ranks(np.ones(len(rs))), Ranking(y).ranks(np.ones(len(rs))), np.ones(len(rs))),
                                        scope='Descriptive eligible-D association; no p-value/familywise or causal claim.'))
    for left, right in (('C0', 'C1'), ('C1', 'M')):
        shared = sorted(set(by_group[left]) & set(by_group[right]))
        ids = np.array([qi[q] for q in shared])
        for region in PRIMARY_REGIONS:
            d = np.array([by_group[right][q]['regions'][region]['trajectory_contributions']['outer_old_kl'] -
                          by_group[left][q]['regions'][region]['trajectory_contributions']['outer_old_kl'] for q in shared])
            values = [np.dot(w, d) / w.sum() for w in weights[:, ids]]
            primary.append(dict(comparison=left + '->' + right, region=region,
                                metric='equal_trajectory_region_outer_old_raw_kl_contribution', shared_eligible_questions=len(shared),
                                right_minus_left=float(d.mean()), ci=ci(values),
                                scope='Same question, potentially different D/reference/model preupdate state; post-selection intersection, not identical-token or causal contrast.'))
    return dict(primary_six_contrasts=primary, exploratory=exploratory,
                multiplicity='No p-values reported. CIs are marginal descriptive intervals, not simultaneous familywise intervals.')


def shared_first_outer(pairs, qi, weights):
    by_group = {g: {r['question_id']: r for r in pairs if r['group'] == g and r['outer_index'] == 0} for g in GROUPS}
    shared = sorted(set.intersection(*(set(v) for v in by_group.values())))
    accepted, mismatched = [], []
    keys = ('student_response_sha256', 'reference_response_sha256', 'token_mask_alignment_sha256')
    for q in shared:
        triplet = [by_group[g][q] for g in GROUPS]
        require(all(all(isinstance(r.get(k), str) and len(r[k]) == 64 for k in keys) for r in triplet), 'Common-history explicit SHA fields required')
        equal = all(len({r[k] for r in triplet}) == 1 for k in keys)
        if equal:
            require(len({r['selected_student_strict_em'] for r in triplet}) == 1, 'Identical native response has changed strict label')
            accepted.append(q)
        else:
            mismatched.append(q)
    contrasts = []
    for left, right in (('C0', 'C1'), ('C1', 'M')):
        for label, em in (('all', None), ('selected_D_correct', 1), ('selected_D_incorrect', 0)):
            selected = [q for q in accepted if em is None or by_group[left][q]['selected_student_strict_em'] == em]
            for name in MAIN:
                usable = [q for q in selected if math.isfinite(feature(by_group[left][q], name)) and math.isfinite(feature(by_group[right][q], name))]
                if not usable:
                    contrasts.append(dict(comparison=left + '->' + right, stratum=label, feature=name,
                                          matched_questions=0, structural_absence_questions=len(selected), mean_delta=None, ci=None))
                    continue
                d = np.array([feature(by_group[right][q], name) - feature(by_group[left][q], name) for q in usable])
                ids = np.array([qi[q] for q in usable])
                boot = [np.dot(w, d) / w.sum() if w.sum() else np.nan for w in weights[:, ids]]
                contrasts.append(dict(comparison=left + '->' + right, stratum=label, feature=name,
                                      matched_questions=len(usable), structural_absence_questions=len(selected) - len(usable),
                                      mean_delta=float(d.mean()), ci=ci(boot)))
    return dict(outer_index=0, common_question_candidates=len(shared), exact_three_SHA_matched=len(accepted),
                mismatched_questions=len(mismatched), contrasts=contrasts,
                identity_fields=list(keys), scope='Same D/reference/mask, sampled before optimization; current preupdate checkpoints follow different optimization paths. Not same model state or gained/lost performance.')


def clip_associations(pairs):
    names = ('clipped_vocabulary_count', 'clipped_vocabulary_fraction', 'clipped_student_probability_mass',
             'clipped_teacher_probability_mass', 'clipped_target_probability_mass', 'clip_loss_removed')
    result = []
    for group in ('pooled',) + GROUPS:
        rs = pairs if group == 'pooled' else [r for r in pairs if r['group'] == group]
        for key in names:
            for region in ('all_policy',) + PRIMARY_REGIONS:
                observed = [r for r in rs if r['regions'][region]['means'].get(key) is not None]
                x = np.array([number(r['regions'][region]['means'][key]) for r in observed])
                for outcome in ('selected_D_strict_em', 'sampling_siblings_success_fraction'):
                    y = np.array([r['selected_student_strict_em'] if outcome == 'selected_D_strict_em' else r['sampling_success_fraction'] for r in observed], dtype=float)
                    metrics = stats_values(x, y, np.ones(len(x)), Ranking(x), outcome == 'selected_D_strict_em') if len(x) > 1 else {}
                    result.append(dict(group=group, region=region, feature=key, outcome=outcome, observed_pairs=len(x),
                                       structural_absence_pairs=len(rs) - len(x), metrics=metrics,
                                       scope='Exploratory recorded-clip association, no bootstrap/p-value/causal claim. M hit masses follow mixed-target clip set.'))
    return result


def write_csv(path, records):
    # Nested metrics remain JSON in CSV cells instead of silently flattened.
    keys = sorted({k for r in records for k in r})
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
        for r in records:
            writer.writerow({k: json.dumps(clean(v), ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else clean(v) for k, v in r.items()})


def run(manifest_ref, output):
    raw, manifest_proof = fixed_bytes(manifest_ref)
    manifest = json.loads(raw)
    require(manifest['schema'] == 'opsd-first5-em-loss-stats-inputs-v1', 'Statistics manifest schema differs')
    pairs, pair_proof = rows(manifest['pair_features'])
    cohorts, cohort_proof = rows(manifest['cohort_features'])
    audit_raw, audit_proof = fixed_bytes(manifest['extraction_summary'])
    audit = json.loads(audit_raw)
    require(audit.get('accepted') is True and audit['counts'] == dict(pairs=4529, valid_tokens=940295,
            cohort_rows=7680, outers=15, optimizer_updates=151), 'Accepted native-extractor first-five counts required')
    execution_raw, execution_proof = fixed_bytes(manifest['root_execution'])
    # Preserve and bind the root's actual execution proof. The statistics reader
    # does not invent an extractor return code or repeat its native token audit.
    execution = json.loads(execution_raw)
    require(isinstance(execution, dict) and execution['returncode'] == 0, 'Root actual extractor returncode0 proof required')
    for key in ('extractor_source_sha256', 'request_sha256', 'stdout_sha256', 'stderr_sha256'):
        require(isinstance(execution[key], str) and len(execution[key]) == 64 and
                all(c in '0123456789abcdef' for c in execution[key]), 'Explicit extractor source/request/original-stdout SHA required')
    require(audit['request_sha256'] == execution['request_sha256'], 'Accepted extractor summary/request execution identity differs')
    _, qident = validate(pairs, cohorts)
    require(len(pairs) == audit['counts']['pairs'] and len(cohorts) == audit['counts']['cohort_rows']
            and sum(r['valid_tokens'] for r in pairs) == audit['counts']['valid_tokens'], 'Statistics rows do not reproduce accepted extractor counts')
    reps, folds, seed = manifest['bootstrap_replicates'], manifest['cv_folds'], manifest['seed']
    require(type(reps) is int and 500 <= reps <= 2000 and folds == 5 and seed == 42, 'Explicit bounded1000-style bootstrap, fivefold seed42 required')
    require(len(pairs) <= 7680, 'First-five bounded pair count exceeded')
    qids, qi, weights = boot_weights(qident, reps, seed)
    unary, binned = univariate(pairs, qi, weights)
    regional_names = tuple(region + '::' + metric for region in PRIMARY_REGIONS for metric in
                           ('teacher_kl', 'student_observed_logp', 'student_teacher_observed_logp_gap'))
    regional_means, _ = univariate(pairs, qi, weights, features=regional_names, include_bins=False)
    predicted, oof_rows = oof(pairs, qident, qi, weights, folds, seed)
    region_result = regions(pairs, qi, weights)
    summary = clean(dict(schema=CONTRACT, accepted=True, groups=GROUPS, outer_indices=list(range(5)),
        selection=selection(pairs, cohorts), pair_rows=len(pairs), cohort_rows=len(cohorts), common_unique_questions=len(qids),
        univariate=unary, region_associations=region_result, oof=predicted,
        regional_token_mean_associations=regional_means, per_outer_descriptive_associations=outer_trends(pairs),
        shared_first_outer=shared_first_outer(pairs, qi, weights), recorded_clip_associations=clip_associations(pairs),
        statistics=dict(seed=seed, bootstrap_replicates=reps, bootstrap_unit='Common question across group, stratified outer/data_source',
                        pointbiserial='Pearson between feature and recorded binary strict D outcome',
                        auc='Higher recorded feature treated as positive score; not orientation-selected AUC',
                        probability_scope='s/5 is empirical five-sibling outer-start sampling success fraction; pass_at5 is at least one strict success, not independent-event extrapolation.'),
        limitations=['Selected-D label may be exactly native all-five-correct fallback, not unbiased generated EM.',
                     'All-five-wrong cohort loss is structurally unavailable, not zero.',
                     'Sampling labels precede this outer training; forward features use complete D at current preupdate state.',
                     'Teacher features contain privileged successful reference; student-only and teacher-augmented OOF families separated.',
                     'OOF is retrospective trajectory discrimination, not ex ante success prediction or causal performance explanation.',
                     'Ordinary outer-old target resets each outer; old KL is not cumulative source100 ordinary policy drift.',
                     'No TRAIN-to-heldout question join, no correlation of three group-endpoint EM scalars.',
                     'Question-bootstrap intervals condition on fixed runs/OOF predictions, not independent training seed variance.']))
    out = Path(output)
    require(out.is_absolute() and not out.exists(), 'Absolute new output required')
    out.mkdir(parents=True)
    (out / 'summary.json').write_bytes(canonical(summary) + b'\n')
    (out / 'inputs.private.json').write_bytes(canonical(dict(manifest=manifest_proof, pairs=pair_proof, cohorts=cohort_proof,
        extraction_summary=audit_proof, root_execution=execution_proof,
        native_audit_scope='Root supplied accepted native extractor counts/source execution; this reader validates aggregate feature/schema/outcome consistency, not logits or token files.')) + b'\n')
    write_csv(out / 'univariate.csv', unary)
    write_csv(out / 'descriptive-bins.csv', binned)
    with (out / 'oof-predictions.private.jsonl').open('xb') as f:
        for r in oof_rows:
            f.write(canonical(clean(r)) + b'\n')
    artifacts = {p.name: dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size) for p in out.iterdir()}
    (out / 'derived-manifest.json').write_bytes(canonical(dict(schema=CONTRACT + '-artifacts', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), artifacts=artifacts)) + b'\n')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True); p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = run(dict(path=a.manifest, sha256=a.manifest_sha256), a.output)
    print(json.dumps(dict(schema=CONTRACT, accepted=result['accepted'], pair_rows=result['pair_rows'], cohort_rows=result['cohort_rows']), sort_keys=True))


if __name__ == '__main__':
    main()
