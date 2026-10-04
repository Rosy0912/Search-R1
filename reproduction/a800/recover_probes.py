"""Refit round200 heads from preserved genuine interventions, without relabeling.

This publishes an initial evaluator for a NEW continuation segment. It does not
modify the original live manifest or claim the original trainer adopted round200.
"""
import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
import joblib
import numpy as np
from reproduction.a800.prepare_run import digest
from reproduction.a800.owned_processes import publish
from reproduction.a800.probe_admission import calibrated_version
from reproduction.a800.recovery_binding import verify_interventions


def recover(base, dump):
    run=json.loads((base/'logs/run-manifest.json').read_text())
    assert run['completed_updates']==200 and run['initial_probe_version'] is None
    assert not (base/'artifacts/live/manifest.json').exists()
    parent=Path(run['parent']['parent_base'])
    assert dump.resolve().is_relative_to((parent/'artifacts').resolve())
    assert digest(parent/'logs/run-manifest.json')==run['parent']['parent_manifest_sha256']
    assert digest(Path(json.loads((parent/'logs/run-manifest.json').read_text())['train_log']))==run['parent']['parent_train_log_sha256']
    marker=Path(run['actor_checkpoint'])/'complete.json'
    assert digest(marker)==run['parent']['parent_checkpoint_marker_sha256']
    for name,value in run['actor_checkpoint_sha256'].items():assert digest(Path(run['actor_checkpoint'])/name)==value
    proof=json.loads((dump/'calibration-label-proof.json').read_text())
    assert Path(proof['model']).resolve()==Path(run['actor_checkpoint']).resolve()
    assert set(proof['question_ids'])==set(json.loads((base/'data/audit.json').read_text())['calibration_question_ids'])
    files=sorted(dump.glob('*.npz'))+[dump/'calibration-label-proof.json']
    assert len(files)>1, 'No genuine hidden/label arrays'
    hashes=verify_interventions(dump,run['recovery_dump_sha256'])
    for p in files:
        if p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as z:
                for key in z.files:
                    if z[key].dtype.kind in 'fc':assert np.isfinite(z[key]).all(), 'Nonfinite calibration '+key
    round_dir=base/'artifacts'/('round_200-recovered-'+uuid.uuid4().hex)
    round_dir.mkdir(parents=True,exist_ok=False);heads=round_dir/'heads'
    with (round_dir/'fit.log').open('w') as log:
        subprocess.run([str(base/'env/bin/python'),str(base/'repo/ts_probe_train.py'),
            '--dumps',str(dump),'--layers','21,24,27','--commit-position','Xa','--commit-label','flip','--save',str(heads)],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    verify_interventions(dump,run['recovery_dump_sha256'])
    report=json.loads((heads/'calibration-report.json').read_text())
    for name in ('commit_probe.pkl','boundary_probe.pkl'):
        obj=joblib.load(heads/name);assert list(obj['clf'].classes_)==[0,1] and np.isfinite(obj['auc'])
    payload=dict(run_id=run['run_id'],round=200,version=run['run_id']+':200:'+uuid.uuid4().hex,
        encoder=run['actor_checkpoint'],encoder_sha256=run['actor_checkpoint_sha256'],
        commit=str(heads/'commit_probe.pkl'),boundary=str(heads/'boundary_probe.pkl'),
        head_sha256={name:digest(heads/name) for name in ('commit_probe.pkl','boundary_probe.pkl')},
        calibration_report=report,label_proof=proof,published_unix=time.time(),
        recovery=dict(parent_run_id=run['parent']['parent_run_id'],dump_sha256=hashes,
            semantics='preserved actual checkpoint200 closed-book/evidence-deletion interventions; heads refitted with reviewed small-sample guard; new segment identity'))
    payload['archived_manifest']=str(round_dir/'manifest.json')
    calibrated_version(run,payload)
    publish(round_dir/'manifest.json',payload)
    live=base/'artifacts/live';live.mkdir(exist_ok=False);publish(live/'manifest.json',payload)
    print('SEARCH_G1_RECOVERED_PROBES_FITTED='+payload['version'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dumps',type=Path,required=True);args=ap.parse_args()
    recover(Path(os.environ['SEARCH_G1_BASE']),args.dumps)
