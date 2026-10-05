"""Require exact immutable recovery intervention file set and content hashes."""
from pathlib import Path
from reproduction.a800.prepare_run import digest


def intervention_fingerprint(dump):
    dump=Path(dump).resolve()
    files=sorted(dump.glob('*.npz'))+[dump/'calibration-label-proof.json']
    assert len(files)>1, 'No genuine intervention arrays'
    return {str(p):digest(p) for p in files}


def verify_interventions(dump, expected):
    actual=intervention_fingerprint(dump)
    assert actual==expected, 'Preserved intervention file set/content changed after binding'
    return actual


def finite_fitted_values(obj, seen=None):
    """Inspect fitted Pipeline estimators, including MLP/scaler arrays."""
    import numpy as np
    if seen is None:seen=set()
    if id(obj) in seen:return
    seen.add(id(obj))
    if isinstance(obj,np.ndarray):
        if obj.dtype.kind in 'fc':assert np.isfinite(obj).all(), 'Nonfinite fitted array'
    elif isinstance(obj,(float,np.floating)):
        assert np.isfinite(obj), 'Nonfinite fitted scalar'
    elif isinstance(obj,(list,tuple)):
        for value in obj:finite_fitted_values(value,seen)
    elif isinstance(obj,dict):
        for value in obj.values():finite_fitted_values(value,seen)
    elif hasattr(obj,'__dict__') and type(obj).__module__.startswith('sklearn.'):
        for key,value in vars(obj).items():
            if key.endswith('_') or key=='steps':finite_fitted_values(value,seen)


def verify_repair_calibration(dump, acceptance):
    import json,math,joblib
    dump=Path(dump).resolve();acceptance=Path(acceptance).resolve()
    accepted=json.loads(acceptance.read_text())
    assert accepted['accepted'] is True and acceptance.parent==dump.parent
    assert {p.name:digest(p) for p in dump.iterdir() if p.is_file()}==accepted['dump_sha256']
    proof=json.loads((dump/'calibration-label-proof.json').read_text())
    assert proof['question_count']==300 and proof['group_size']==5
    assert len(set(proof['question_ids']))==300
    assert proof['calibration_protocol']['rollout_max_new']==500
    assert proof['calibration_protocol']['correction_feedback'] is True
    assert proof['evidence_deletion']=='typed_retrieval_ranges_only_preserve_model_and_correction'
    model=Path(proof['model']);assert str(model)==accepted.get('model',str(model))
    assert {str(p.relative_to(model)):digest(p) for p in model.rglob('*') if p.is_file()}==accepted['model_sha256']
    report=accepted['report'];assert report['require_all_cv_folds'] is True and report['configured_cv_folds']==2
    assert report==json.loads((acceptance.parent/'heads/calibration-report.json').read_text())
    heads=acceptance.parent/'heads'
    assert {p.name:digest(p) for p in heads.iterdir() if p.is_file()}==accepted['head_sha256']
    for name in ('commit_probe.pkl','boundary_probe.pkl'):
        obj=joblib.load(heads/name);assert list(obj['clf'].classes_)==[0,1]
        assert math.isfinite(obj['auc']) and 0<=obj['auc']<=1
        finite_fitted_values(vars(obj['clf']))
    return accepted
