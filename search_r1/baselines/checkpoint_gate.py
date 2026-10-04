"""Fail-closed fifth-outer pause. This module never produces PASS receipts."""
import hashlib
import json
import os
import secrets
import subprocess
import time
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def validate_gate_config(config):
    """Bind the release counters to the actual resolved scheduler config."""
    try:
        trainer = config['trainer']
        optim = config['actor_rollout_ref']['actor']['optim']
    except (KeyError, TypeError):
        raise ValueError('Missing actual 300/114 gate configuration')
    if (type(trainer.get('total_training_steps')) is not int
            or trainer['total_training_steps'] != 301
            or type(optim.get('num_warmup_steps')) is not int
            or optim['num_warmup_steps'] != 114
            or trainer.get('e1_gate_enabled') is not True
            or type(trainer.get('acceptance_step')) is not int
            or trainer['acceptance_step'] != 5):
        raise ValueError('Actual resolved config must use steps301/warmup114/step5 gate')
    from .full_state import resolve_warmup_steps
    return trainer['total_training_steps'] - 1, resolve_warmup_steps(optim)


def write_manifest(paths, directory, config, source, run, step, reports):
    horizon, warmup_steps = validate_gate_config(config)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    run=Path(run).resolve()
    if run not in directory.resolve().parents:raise ValueError('Checkpoint directory outside exact run')
    config_bytes=json.dumps(config,sort_keys=True,allow_nan=False).encode()
    config_path=directory/'resolved-config.json';config_path.write_bytes(config_bytes)
    files=[]
    for p in sorted(set(Path(p).resolve()for p in paths)|{config_path.resolve()}):
        if run not in p.parents or not p.is_file() or not p.stat().st_size:raise ValueError('Missing/outside/empty checkpoint member '+str(p))
        files.append(dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size))
    value=dict(version=1,run_root=str(Path(run).resolve()),source_commit=source,global_step=step,
               actual_outer_updates=step,horizon=horizon,warmup_steps=warmup_steps,config_sha256=digest(config_path),
               pause_nonce=secrets.token_hex(32),rank_roundtrips=reports,files=files)
    target=directory/'manifest.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,target)
    return target


def validate_release(receipt, manifest_path, repo, expected_driver_pid=None):
    manifest_path=Path(manifest_path);m=json.loads(manifest_path.read_text());r=json.loads(Path(receipt).read_text())
    if r.get('status')!='PASS_INDEPENDENT_E1_STEP5_REVIEW':raise ValueError('Independent E1 step5 release missing')
    for key in ('run_root','source_commit','global_step','actual_outer_updates','horizon','warmup_steps','config_sha256','pause_nonce'):
        if r.get(key)!=m.get(key):raise ValueError('Release identity mismatch '+key)
    if any(type(m.get(k)) is not int or type(r.get(k)) is not int for k in ('global_step','actual_outer_updates','horizon','warmup_steps')):
        raise ValueError('Gate counters must be integers')
    if m['global_step']!=5 or m['actual_outer_updates']!=5 or m['horizon']!=300 or m['warmup_steps']!=114:
        raise ValueError('Not a fifth-outer 300/114 gate')
    if r.get('checkpoint_manifest_sha256')!=digest(manifest_path):raise ValueError('Checkpoint manifest changed')
    source=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    if source!=m['source_commit'] or dirty:raise ValueError('Current source changed')
    run=Path(m['run_root']).resolve();root=run.parent.parent
    if run not in manifest_path.resolve().parents:raise ValueError('Manifest outside exact run')
    reports=m['rank_roundtrips']
    if (len(reports)!=8 or {x.get('rank')for x in reports}!=set(range(8))
            or any(x.get('status')!='EXACT_FULL_STATE_RELOAD_OK' or x.get('step')!=5
                   or x.get('world_size')!=8 or x.get('scheduler_last_epoch')!=5
                   or x.get('model_state_dict_type')!='SHARDED_STATE_DICT'
                   or type(x.get('model_dtensor_entries')) is not int or x['model_dtensor_entries']<=0
                   or x.get('optimizer_state_entries',0)<=0 for x in reports)):
        raise ValueError('Eight full-state reload reports incomplete')
    members={x['path']:x['sha256']for x in m['files']}
    if len(members)!=len(m['files']):raise ValueError('Duplicate checkpoint members')
    configs=[x for x in m['files']if Path(x['path']).name=='resolved-config.json']
    if (len(configs)!=1 or configs[0]['sha256']!=m['config_sha256']
            or digest(configs[0]['path'])!=m['config_sha256']):
        raise ValueError('Config identity differs from actual resolved config')
    resolved_config = json.loads(Path(configs[0]['path']).read_text())
    horizon, warmup_steps = validate_gate_config(resolved_config)
    if m['horizon'] != horizon or m['warmup_steps'] != warmup_steps:
        raise ValueError('Manifest counters differ from actual scheduler config')
    native_paths=set()
    for report in reports:
        state_path=Path(report.get('path',''))
        report_path=state_path.with_suffix('.json')
        if (state_path.name!=f"rank_{report['rank']}.pt" or str(state_path) in native_paths
                or members.get(str(state_path))!=report.get('sha256')
                or str(report_path) not in members or digest(report_path)!=members[str(report_path)]
                or json.loads(report_path.read_text())!=report
                or type(report.get('pid')) is not int or report['pid']<=0
                or not isinstance(report.get('state_digest'),str) or len(report['state_digest'])!=64):
            raise ValueError('Native rank state/report identity not bound')
        native_paths.add(str(state_path))
    drivers=[x for x in m['files']if Path(x['path']).name=='driver-reload.json']
    driver_states=[x for x in m['files']if Path(x['path']).name=='driver.pt']
    if len(drivers)!=1 or len(driver_states)!=1:
        raise ValueError('Unique driver state/reload proof required')
    driver=json.loads(Path(drivers[0]['path']).read_text())
    import torch
    from search_r1.baselines.full_state import state_digest
    driver_state=torch.load(driver_states[0]['path'],map_location='cpu',weights_only=False)
    if (driver.get('status')!='EXACT_DRIVER_RELOAD_OK' or driver.get('step')!=5
            or driver.get('path')!=driver_states[0]['path']
            or driver.get('sha256')!=driver_states[0]['sha256']
            or digest(driver_states[0]['path'])!=driver_states[0]['sha256']
            or driver.get('state_digest')!=state_digest(driver_state)
            or type(driver.get('pid')) is not int or driver['pid']<=0
            or driver.get('pid')!=driver_state.get('driver_pid')
            or (expected_driver_pid is not None and
                (type(expected_driver_pid) is not int or expected_driver_pid<=0
                 or driver.get('pid')!=expected_driver_pid))
            or driver_state.get('global_step')!=5 or driver_state.get('next_global_step')!=6
            or driver_state.get('actual_outer_updates')!=5):
        raise ValueError('Actual driver sampler/RNG/BAPO reload not proven')
    evidence=r.get('evidence',[]);bindings={x['path']:x['sha256']for x in evidence}
    if len(bindings)!=len(evidence):raise ValueError('Duplicate release evidence')
    if str(run/'source-commit.txt') not in bindings:raise ValueError('Run source not directly bound')
    if str(run/'train.log') in bindings:raise ValueError('Use immutable step5 prefix, not mutable whole train.log')
    prefix=r.get('train_log_prefix',{})
    snapshot=Path(prefix.get('snapshot_path',''))
    length=prefix.get('bytes')
    if (prefix.get('live_path')!=str(run/'train.log') or type(length) is not int or length<=0
            or not snapshot.is_absolute() or root not in snapshot.resolve().parents
            or not snapshot.is_file() or snapshot.stat().st_size!=length
            or digest(snapshot)!=prefix.get('sha256') or bindings.get(str(snapshot))!=prefix.get('sha256')):
        raise ValueError('Immutable step5 log snapshot missing/unbound')
    with (run/'train.log').open('rb') as handle:
        h=hashlib.sha256();remaining=length
        while remaining:
            block=handle.read(min(1048576,remaining))
            if not block:raise ValueError('Actual live log shorter than signed prefix')
            h.update(block);remaining-=len(block)
    if h.hexdigest()!=prefix['sha256']:raise ValueError('Actual live prefix differs from snapshot')
    events=[]
    for line in snapshot.read_text().split('\n'):
        if 'BAPO_TRAIN_METRICS_JSON=' in line:
            events.append(json.loads(line.split('BAPO_TRAIN_METRICS_JSON=',1)[1]))
    if [x.get('step')for x in events]!=[1,2,3,4,5]:raise ValueError('Signed prefix must contain exact five outer summaries')
    if any(x.get('metrics',{}).get('bapo/actual_outer_updates')!=x['step']for x in events):
        raise ValueError('Signed prefix actualouter mismatch')
    if (run/'source-commit.txt').read_text().strip()!=m['source_commit']:
        raise ValueError('Run-start source marker changed')
    for item in m['files']:
        p=Path(item['path'])
        if run not in p.resolve().parents or digest(p)!=item['sha256'] or bindings.get(str(p))!=item['sha256']:
            raise ValueError('Checkpoint member changed/unbound '+str(p))
    if bindings.get(str(manifest_path.resolve()))!=digest(manifest_path):raise ValueError('Manifest not directly bound')
    review=r.get('independent_review',{})
    for item in evidence+[review]:
        p=Path(item.get('path',''))
        if not p.is_absolute() or root not in p.resolve().parents or not p.is_file() or digest(p)!=item.get('sha256'):
            raise ValueError('Missing/changed owned evidence')
    return r


def wait_for_release(receipt, manifest_path, repo, interval=5):
    print('E1_STEP5_PAUSED_JSON='+json.dumps(dict(manifest_path=str(manifest_path),manifest_sha256=digest(manifest_path),pid=os.getpid())),flush=True)
    while not Path(receipt).exists():time.sleep(interval)
    # An invalid present receipt is a hard failure, never a silent release.
    result=validate_release(receipt,manifest_path,repo,expected_driver_pid=os.getpid())
    print('E1_STEP5_RELEASED_JSON='+json.dumps(dict(receipt=str(receipt),receipt_sha256=digest(receipt),pid=os.getpid())),flush=True)
    return result
