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
