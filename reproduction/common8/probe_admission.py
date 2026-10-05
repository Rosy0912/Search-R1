"""Require common8 preparation, actual resource admission and genuine fitted probes."""
import argparse
import json
import os
from pathlib import Path
from reproduction.common8.verify_data import digest, verify
from reproduction.a800 import probe_admission as generic
from reproduction.a800.recovery_binding import finite_fitted_values


def check_common(base):
    import joblib
    run=json.loads((base/'logs/run-manifest.json').read_text())
    assert run['profile']=='common8' and run['completed_updates']==0
    assert run['acceptance_update']==5 and run['target_updates']==300
    assert run['initial_model']==run['reference_model']
    assert run['model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
    verify(base/'data')
    budget=json.loads((base/'repo/reproduction/common8/budget.json').read_text())
    assert run['budget']==budget
    # These must be produced by real admission, never by prepare_run.
    for name in ('eight-rank-admission.json','retrieval-admission.json','probe-coexistence-admission.json'):
        proof=json.loads((base/'logs'/name).read_text())
        assert proof['accepted'] is True and proof['run_id']==run['run_id']
    assert run['probe_visible_device'] is not None
    live=json.loads((base/'artifacts/live/manifest.json').read_text())
    report=live['calibration_report'];protocol=live['label_proof']['calibration_protocol']
    assert report['require_all_cv_folds'] is True and report['configured_cv_folds']==2
    assert protocol['rollout_max_new']==500 and protocol['correction_feedback'] is True
    assert live['label_proof']['evidence_deletion']=='typed_retrieval_ranges_only_preserve_model_and_correction'
    for name,key in (('commit_probe.pkl','commit'),('boundary_probe.pkl','boundary')):
        path=Path(live[key]);assert digest(path)==live['head_sha256'][name]
        fitted=joblib.load(path);finite_fitted_values(vars(fitted['clf']))
    return run


def verify_training_gate(base):
    check_common(base)
    return generic.verify_training_gate(base)


def admit(base):
    check_common(base)
    generic.admit(base)
    verify_training_gate(base)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    base=Path(os.environ['SEARCH_G1_BASE'])
    if a.verify:verify_training_gate(base)
    else:admit(base)
