"""Actual fitted-Head endpoint admission; never frozen teacher reward inference."""
import json
import math
import os
import subprocess
from pathlib import Path

from reproduction.common8.actor_probe_head_service import ROLE, verify_manifest
from reproduction.common8.verify_data import digest
from search_g1.actor_probe_contract import head_round
from search_g1.actor_probe_namespace import namespace_origin
from search_g1.actor_probe_recovery import identity

ARCHIVE = 'logs/actor-probe-head-service-manifest.json'


def verify_endpoint_documents(run, manifest, health, smoke, manifest_sha):
    origin = namespace_origin(run); logical = origin['completed_outer'] + 1
    verify_manifest(run,manifest,logical_update=logical)
    assert manifest['round'] == head_round(logical)
    if origin['completed_outer'] == 0:
        assert manifest['version'].startswith(run['run_id']+':0:'), 'Fresh probe Head must belong to this official-root run'
    for document in (health,smoke):
        assert document['role'] == ROLE and document['identity'] == identity(run)
        assert document['version'] == manifest['version'] and document['round'] == manifest['round']
        assert document['head_sha256'] == manifest['head_sha256']
        assert document['frozen_encoder_loaded'] is False
    assert health['ok'] is True and health['heads_loaded'] is True and health['suspended'] is False
    assert health['actor_step'] == manifest['actor_step'] and health['encoder'] == manifest['encoder']
    assert health['live_manifest_sha256'] == manifest_sha and health['cuda_model_allocated_bytes'] == 0
    assert smoke['synthetic_cpu_inputs'] is True and smoke['task_accuracy_claimed'] is False
    assert smoke['actor_features_exercised'] is False and smoke['producer'] == health['producer']
    assert set(smoke['readouts']) == {'commit','boundary'}
    for values in smoke['readouts'].values():
        assert len(values) == 2 and all(type(v) in (int,float) and math.isfinite(v) and 0 <= v <= 1 for v in values)
    return origin


def _owned(base,relative):
    path=base/relative
    assert path == path.resolve() and path.is_relative_to(base) and path.is_file()
    assert path.stat().st_uid == os.getuid() and path.stat().st_size <= 32*1024**2
    return path


def verify_owned_endpoint(base,run,producer):
    from reproduction.a800.owned_processes import identity as process_identity
    path=_owned(base,'logs/process-ledger.json'); ledger=json.loads(path.read_bytes())
    assert ledger['run_id']==run['run_id']
    expected=ledger['jobs']['probe-server']['leader']
    assert set(producer)=={'pid','start_ticks','sid','uid'}
    assert all(type(producer[k]) is int and producer[k]==expected[k] for k in producer)
    actual=process_identity(producer['pid'])
    assert all(actual[k]==producer[k] for k in producer) and actual['state'] not in ('Z','X')
    assert actual['uid']==os.getuid()
    return producer


def verify_inputs_and_source(base,run):
    from reproduction.common8.source_files import runtime_files
    for name,sha in run['input_sha256'].items():
        assert Path(name).name==name and digest(_owned(base,'data/'+name))==sha
    repo=base/'repo'
    actual={str(path.relative_to(repo)):digest(_owned(base,'repo/'+str(path.relative_to(repo))))
            for path in runtime_files(repo)}
    assert actual==run['source_sha256'], 'Actor Head gate source map changed'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==run['source_commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()


def _endpoint(base,run,manifest,sha):
    import requests
    session=requests.Session(); session.trust_env=False
    response=session.get('http://127.0.0.1:18012/health',timeout=60); response.raise_for_status();health=response.json()
    response=session.post('http://127.0.0.1:18012/readout_smoke',json={'version':manifest['version']},timeout=60)
    response.raise_for_status();smoke=response.json()
    origin=verify_endpoint_documents(run,manifest,health,smoke,sha)
    verify_owned_endpoint(base,run,health['producer'])
    assert digest(_owned(base,'artifacts/live/manifest.json'))==sha
    return origin,health,smoke


def verify_training_gate(base):
    base=Path(base).resolve(); run=json.loads(_owned(base,'logs/run-manifest.json').read_bytes())
    verify_inputs_and_source(base,run)
    accepted=json.loads(_owned(base,'logs/initial-probes-accepted.json').read_bytes())
    live_path=_owned(base,'artifacts/live/manifest.json'); live=json.loads(live_path.read_bytes());sha=digest(live_path)
    assert accepted['role']=='actual_actor_binary_head_endpoint_admission' and accepted['accepted'] is True
    assert accepted['identity']==identity(run) and accepted['origin']==namespace_origin(run)
    assert accepted['version']==run['initial_probe_version']==live['version']
    assert accepted['live_manifest_sha256']==accepted['archived_manifest_sha256']==sha
    assert accepted['actor_features_exercised'] is False and accepted['task_accuracy_claimed'] is False
    archive=_owned(base,ARCHIVE)
    assert digest(archive)==sha and json.loads(archive.read_bytes())==live
    _endpoint(base,run,live,sha)
    return run


def admit(base):
    from reproduction.common8.calibration_runtime import write_once
    from reproduction.a800.probe_admission import atomic
    base=Path(base).resolve(); run_path=_owned(base,'logs/run-manifest.json');run=json.loads(run_path.read_bytes())
    verify_inputs_and_source(base,run)
    target=base/'logs/initial-probes-accepted.json'
    if target.exists() or target.is_symlink():return verify_training_gate(base)
    live_path=_owned(base,'artifacts/live/manifest.json');live=json.loads(live_path.read_bytes());sha=digest(live_path)
    assert run['initial_probe_version'] in (None,live['version'])
    origin,health,smoke=_endpoint(base,run,live,sha)
    archive=base/ARCHIVE
    raw=live_path.read_bytes();assert digest(live_path)==sha
    import hashlib
    assert hashlib.sha256(raw).hexdigest()==sha
    with archive.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    fd=os.open(archive.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    assert digest(archive)==sha
    # An unchanged recovery version does not mutate its prepared candidate.
    if run['initial_probe_version'] is None:
        run['initial_probe_version']=live['version'];atomic(run_path,run)
    proof=dict(role='actual_actor_binary_head_endpoint_admission',accepted=True,
        run_id=run['run_id'],identity=identity(run),origin=origin,version=live['version'],
        live_manifest_sha256=sha,archived_manifest=ARCHIVE,archived_manifest_sha256=digest(archive),
        actual_http_health=health,actual_http_readouts=smoke,
        actor_features_exercised=False,task_accuracy_claimed=False,
        semantics='actual owned CPU binary Head plumbing; reward features require current actor detached RPC; no encoder inference')
    write_once(target,proof)
    verify_training_gate(base)
    return proof
