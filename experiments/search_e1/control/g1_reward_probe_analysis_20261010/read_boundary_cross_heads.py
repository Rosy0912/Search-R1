"""Read-only CPU cross-head scores of existing fixed-question prompt features."""
from pathlib import Path
import datetime, hashlib, json, os, socket, sys
import numpy as np
import joblib

BASE = Path('/data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512')
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

run = json.loads((BASE / 'logs/run-manifest.json').read_text())
out = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), node=socket.gethostname(),
           run_id=run['run_id'], readonly=True, cpu_only=True,
           interpretation='Off-diagonal cells are diagnostic CPU scores only, never actual served training rewards. Both feature encoder and learned head may change; this matrix separates their numerical contributions on the same 300 prompts, not a causal attribution of learning.',
           input_files=[], score_matrix=[])
rounds = {}
for directory in (BASE / 'artifacts').glob('round_*'):
    path = directory / 'version-manifest.json'
    if not path.is_file():
        continue
    manifest = json.loads(path.read_text())
    step = manifest['round']
    assert manifest['run_id'] == run['run_id']
    if step not in (0, 50, 100):
        continue
    hp = Path(manifest['boundary'])
    assert sha(hp) == manifest['head_sha256']['boundary_probe.pkl']
    dumps = list((directory / 'dumps').glob('*.npz'))
    assert len(dumps) == 1 and sha(dumps[0]) == manifest['dump_sha256'][dumps[0].name]
    head = joblib.load(hp)
    with np.load(dumps[0], allow_pickle=False) as z:
        ids = z['boundary_question_id'].astype(str).tolist()
        labels = z['boundary_q_cb'].astype(int)
        features = {layer: z['Xp_question_' + str(layer)].copy() for layer in (21, 24, 27)}
    assert len(ids) == len(set(ids)) == 300
    assert set(ids) == set(run['calibration_question_ids'])
    rounds[step] = dict(head=head, ids=ids, labels=labels, features=features)
    out['input_files'].append(dict(round=step, manifest_path=str(path), manifest_sha256=sha(path),
                                   head_path=str(hp), head_sha256=manifest['head_sha256']['boundary_probe.pkl'],
                                   dump_path=str(dumps[0]), dump_sha256=manifest['dump_sha256'][dumps[0].name],
                                   actual_question_order_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
assert set(rounds) == {0, 50, 100}
out['same_question_order'] = all(row['ids'] == rounds[0]['ids'] for row in rounds.values())
for encoder_step, encoded in sorted(rounds.items()):
    for head_step, fitted in sorted(rounds.items()):
        head = fitted['head']
        scores = head['clf'].predict_proba(encoded['features'][head['layer']])[:, 1]
        assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
        out['score_matrix'].append(dict(feature_encoder_round=encoder_step, boundary_head_round=head_step,
                                        count=len(scores), actual_served_pair=encoder_step == head_step,
                                        head_layer=head['layer'], head_model=head.get('model'),
                                        mean=float(scores.mean()), median=float(np.median(scores)),
                                        quantiles=np.quantile(scores, [.05, .25, .5, .75, .95]).tolist(),
                                        closedbook_correct_current_labels=int(encoded['labels'].sum())))
assert 'torch' not in sys.modules
out['torch_imported'] = False
out['observed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(out))
