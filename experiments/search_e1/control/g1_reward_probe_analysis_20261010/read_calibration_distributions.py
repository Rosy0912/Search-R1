"""CPU-only read-only G1 calibration scoring; never emits features or rows."""
from pathlib import Path
import datetime, hashlib, json, os, re, socket, sys
import numpy as np
import joblib

BASE = Path('/data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512')
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def summary(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if not a.size:
        return dict(count=0)
    assert np.isfinite(a).all()
    edges = np.linspace(0., 1., 11)
    counts, _ = np.histogram(a, bins=edges)
    return dict(count=int(a.size), mean=float(a.mean()), std_population=float(a.std()),
                min=float(a.min()), max=float(a.max()),
                quantile_probabilities=[0., .05, .25, .5, .75, .95, 1.],
                quantiles=np.quantile(a, [0., .05, .25, .5, .75, .95, 1.]).tolist(),
                histogram_edges=edges.tolist(), histogram_counts=counts.tolist(),
                count_below_point1=int((a < .1).sum()),
                count_above_point9=int((a > .9).sum()))

run_path = BASE / 'logs/run-manifest.json'
run = json.loads(run_path.read_text())
out = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), node=socket.gethostname(),
           base=str(BASE), run_id=run['run_id'], source_commit=run['source_commit'],
           readonly=True, cpu_only=True, CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
           interpretation='Matching round fitted heads scored on that round calibration features; in-sample distributions, not live training scores or independent held-out fidelity.',
           rounds=[])
question_sets = []
for directory in sorted((BASE / 'artifacts').glob('round_*')):
    manifest_path = directory / 'version-manifest.json'
    if not manifest_path.is_file():
        continue
    mb = manifest_path.read_bytes()
    manifest = json.loads(mb)
    assert manifest['run_id'] == run['run_id'] and manifest['round'] in (0, 50, 100, 150)
    head_paths = {k: Path(manifest[k]) for k in ('commit', 'boundary')}
    for kind, path in head_paths.items():
        assert path.is_file() and path.resolve().is_relative_to(BASE.resolve())
        assert sha(path) == manifest['head_sha256'][kind + '_probe.pkl']
    heads = {kind: joblib.load(path) for kind, path in head_paths.items()}
    dump_paths = sorted((directory / 'dumps').glob('*.npz'))
    assert len(dump_paths) == 1
    dump_path = dump_paths[0]
    assert sha(dump_path) == manifest['dump_sha256'][dump_path.name]
    report_path = directory / 'heads/calibration-report.json'
    assert sha(report_path) == manifest['calibration_report_sha256']
    report = json.loads(report_path.read_text())
    proof_path = directory / 'dumps/calibration-label-proof.json'
    proof = json.loads(proof_path.read_text())
    expected_ids = set(run['calibration_question_ids'])
    actual_ids = set(proof['question_ids'])
    assert len(actual_ids) == len(expected_ids) == 300 and actual_ids == expected_ids
    question_sets.append(actual_ids)
    row = dict(round=manifest['round'], version=manifest['version'],
               manifest_path=str(manifest_path), manifest_sha256=hashlib.sha256(mb).hexdigest(),
               report_path=str(report_path), report_sha256=sha(report_path), report=report,
               dump_path=str(dump_path), dump_sha256=manifest['dump_sha256'][dump_path.name],
               labels_proof_path=str(proof_path), labels_proof_sha256=sha(proof_path),
               question_ids_set_sha256=hashlib.sha256(json.dumps(sorted(actual_ids)).encode()).hexdigest(),
               heads={kind: dict(path=str(head_paths[kind]), sha256=manifest['head_sha256'][kind + '_probe.pkl'],
                                model=heads[kind].get('model'), layer=heads[kind]['layer'],
                                label=heads[kind].get('label'), position=heads[kind].get('position'),
                                classifier_type=type(heads[kind]['clf']).__name__) for kind in heads})
    with np.load(dump_path, allow_pickle=False) as z:
        row['npz_keys'] = z.files
        commit = heads['commit']; boundary = heads['boundary']
        d = commit['clf'].predict_proba(z[commit['position'] + '_' + str(commit['layer'])])[:, 1]
        b_question = boundary['clf'].predict_proba(z['Xp_question_' + str(boundary['layer'])])[:, 1]
        b_rollout = boundary['clf'].predict_proba(z['Xp_' + str(boundary['layer'])])[:, 1]
        assert np.isfinite(d).all() and np.isfinite(b_question).all() and np.isfinite(b_rollout).all()
        assert all(((v >= 0) & (v <= 1)).all() for v in (d, b_question, b_rollout))
        assert len(d) == len(b_rollout)
        ground = d * (1. - b_rollout)
        em = z['y_em'].astype(int); flip = z['y_flip'].astype(int)
        cb_q = z['boundary_q_cb'].astype(int)
        cb_rollout = z['q_cb'].astype(int)
        assert len(cb_q) == 300 and len(em) == len(d)
        row['cohort'] = dict(unique_questions=300, valid_searched_trajectory_count=len(d),
                             nominal_generated_trajectories=300 * proof['group_size'],
                             evidence_sensitive_labels=int(flip.sum()),
                             closedbook_correct_question_labels=int(cb_q.sum()),
                             correct_valid_searched_trajectory_labels=int(em.sum()),
                             same_reserved_question_set_as_other_rounds=True,
                             same_sampled_trajectories_as_other_rounds=False)
        row['distributions'] = dict(b_question_equal_question=summary(b_question),
                                     b_valid_searched_rollout_weighted=summary(b_rollout),
                                     d_valid_searched_rollout=summary(d),
                                     ground_valid_searched_rollout=summary(ground))
        row['conditional_distributions'] = {}
        for label, mask in [('em_correct', em == 1), ('em_wrong', em == 0),
                            ('flip_true', flip == 1), ('flip_false', flip == 0),
                            ('closedbook_correct', cb_rollout == 1), ('closedbook_wrong', cb_rollout == 0)]:
            row['conditional_distributions'][label] = dict(d=summary(d[mask]), b=summary(b_rollout[mask]), ground=summary(ground[mask]))
        if 'counterfactual_valid' in z.files:
            row['counterfactual_valid_count'] = int(z['counterfactual_valid'].sum())
        row['calibration_reward_terms_no_cost'] = dict(
            correct_searched_additive_bonus=summary(.6 * ground[em == 1]),
            correct_searched_reward_before_search_cost=summary(1. + .6 * ground[em == 1]))
        row['reward_limitations'] = 'No parsed n_search in saved NPZ: cannot reconstruct exact search cost or full calibration reward; summaries are explicit no-cost terms only.'
    out['rounds'].append(row)
out['same300_question_ids_across_all_rounds'] = bool(question_sets) and all(s == question_sets[0] for s in question_sets)

# Read a bounded prefix to test whether existing native sampling traces record probe scores.
out['behavior_trace_schema_probe'] = []
paths = sorted((BASE / 'artifacts/behavior-traces').glob('attempt-*.json'), key=lambda p: p.stat().st_mtime_ns)
for path in (paths[:1] + paths[-1:]):
    with path.open('rb') as stream:
        prefix = stream.read(32768)
    names = re.findall(rb'"([A-Za-z_][A-Za-z_0-9]*)"\s*:', prefix)
    out['behavior_trace_schema_probe'].append(dict(path=str(path), bytes=path.stat().st_size,
                                                   sampled_prefix_bytes=len(prefix),
                                                   unique_key_names=sorted(set(n.decode() for n in names)),
                                                   probe_terms_present_in_prefix=[t.decode() for t in (b'boundary', b'flip', b'ground', b'probe') if t in prefix],
                                                   bounded_prefix_not_full_file_scan=True))
out['torch_imported'] = 'torch' in sys.modules
assert not out['torch_imported']
out['observed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(out, ensure_ascii=False))
