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
    from .fresh200_contract import validate_fresh_config
    if config['trainer'].get('fresh200_diagnostic_only',False):
        raise ValueError('Diagnostic-only runs cannot release formal fresh200 training')
    return validate_fresh_config(config)


def validate_first_five_attempts(attempts, progress, driver_pid):
    """Replay retained writes and accepted outers; five is not five attempts."""
    from .fresh200_driver import Fresh200Progress
    if type(driver_pid) is not int or driver_pid<=0:raise ValueError('Exact original driver PID required')
    state=Fresh200Progress()
    def equal_accounting(observed):
        # Python equality considers True == 1 and 20.0 == 20; raw receipt
        # metadata must preserve the exact typed counters produced by replay.
        return json.dumps(observed,sort_keys=True,allow_nan=False)==json.dumps(state.state_dict(),sort_keys=True,allow_nan=False)
    accepted=[]
    for attempt in attempts:
        outer=state.completed_outers+1
        attempt_id=state.begin_attempt(outer)
        if (any(type(attempt.get(key)) is not int for key in ('attempt_id','candidate_outer','driver_pid'))
                or attempt.get('attempt_id')!=attempt_id or attempt.get('candidate_outer')!=outer
                or attempt.get('driver_pid')!=driver_pid):
            raise ValueError('First-five attempt/PID/candidate identity differs')
        complete=state.record(attempt['rank_write_ledger'],outer)
        if (attempt.get('accepted_complete_outer') is not complete
                or attempt.get('status')!=attempt['rank_write_ledger'][0]['status']
                or not equal_accounting(attempt.get('progress'))):
            raise ValueError('First-five raw attempt accounting differs')
        if complete:accepted.append(attempt)
        if state.completed_outers>5:raise ValueError('Fifth gate contains later accepted updates')
    if state.completed_outers!=5 or not equal_accounting(progress):
        raise ValueError('First-five gate needs five accepted complete outers, actual writes bound')
    return accepted


def write_manifest(paths, directory, config, source, run, step, reports, progress):
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
    if progress.get('completed_outers')!=5:raise ValueError('Gate requires fifth accepted outer')
    value=dict(version=2,run_root=str(Path(run).resolve()),source_commit=source,global_step=step,
               actual_outer_updates=step,horizon=horizon,warmup_steps=warmup_steps,config_sha256=digest(config_path),
               pause_nonce=secrets.token_hex(32),rank_roundtrips=reports,files=files,
               protocol='BAPO_FRESH200_V1', mini_batch_switch_outer=50,
               update_semantics='SKIP_CONTINUE_ACCEPTED_OUTERS_V1',attempt_accounting=progress,
               optimizer_opportunities_before_switch=20,optimizer_opportunities_after_switch=10)
    target=directory/'manifest.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,target)
    return target


def validate_release(receipt, manifest_path, repo, expected_driver_pid=None):
    manifest_path=Path(manifest_path);m=json.loads(manifest_path.read_text());r=json.loads(Path(receipt).read_text())
    if r.get('status')!='PASS_INDEPENDENT_FRESH200_STEP5_REVIEW':raise ValueError('Independent FRESH200 step5 release missing')
    for key in ('run_root','source_commit','global_step','actual_outer_updates','horizon','warmup_steps','config_sha256','pause_nonce','protocol','mini_batch_switch_outer','update_semantics','attempt_accounting','optimizer_opportunities_before_switch','optimizer_opportunities_after_switch'):
        if r.get(key)!=m.get(key):raise ValueError('Release identity mismatch '+key)
    if any(type(m.get(k)) is not int or type(r.get(k)) is not int for k in ('global_step','actual_outer_updates','horizon','warmup_steps')):
        raise ValueError('Gate counters must be integers')
    if m['global_step']!=5 or m['actual_outer_updates']!=5 or m['horizon']!=200 or m['warmup_steps']!=57:
        raise ValueError('Not a fifth-outer 200/57 gate')
    if (m.get('version')!=2 or m.get('update_semantics')!='SKIP_CONTINUE_ACCEPTED_OUTERS_V1'
            or m.get('protocol')!='BAPO_FRESH200_V1' or m.get('mini_batch_switch_outer')!=50
            or m.get('optimizer_opportunities_before_switch')!=20 or m.get('optimizer_opportunities_after_switch')!=10):
        raise ValueError('Fresh200 variable optimizer counts unbound')
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
            or driver_state.get('actual_outer_updates')!=5
            or driver_state.get('progress')!=m.get('attempt_accounting')):
        raise ValueError('Actual driver sampler/RNG/BAPO reload not proven')
    sampling_proof=validate_first_sampling_evidence(members,resolved_config,source,run,driver['pid'],reports)
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
    events=[];attempts=[]
    for line in snapshot.read_text().split('\n'):
        if 'BAPO_TRAIN_METRICS_JSON=' in line:
            events.append(json.loads(line.split('BAPO_TRAIN_METRICS_JSON=',1)[1]))
        if 'FRESH200_ATTEMPT_JSON=' in line:
            attempts.append(json.loads(line.split('FRESH200_ATTEMPT_JSON=',1)[1]))
    accepted=validate_first_five_attempts(attempts,m['attempt_accounting'],driver['pid'])
    for attempt in attempts:
        path=Path(attempt.get('ledger_path',''))
        if (attempt.get('source_commit')!=m['source_commit']
                or not path.is_absolute() or run not in path.resolve().parents or str(path) not in members
                or digest(path)!=members[str(path)] or json.loads(path.read_text())!=attempt):
            raise ValueError('Raw per-attempt all-rank ledger missing or unbound')
    complete_events=[event for event in events if event.get('accepted_complete_outer') is True]
    if [x.get('step')for x in complete_events]!=[1,2,3,4,5]:raise ValueError('Prefix must contain exact five accepted outer summaries')
    if len(events)!=len(attempts):raise ValueError('Each attempt must have one metrics summary')
    for event,attempt in zip(events,attempts):
        if (event.get('step')!=attempt['candidate_outer'] or event.get('attempt_id')!=attempt['attempt_id']
                or event.get('accepted_complete_outer') is not attempt['accepted_complete_outer']
                or event.get('metrics',{}).get('actor/optimizer_steps')!=attempt['rank_write_ledger'][0]['optimizer_steps']):
            raise ValueError('Actual metrics differ from rank write ledger')
    if not any(a['attempt_id']==sampling_proof['attempt_id'] and a['candidate_outer']==sampling_proof['actual_outer']
               and a['accepted_complete_outer'] is True for a in attempts):
        raise ValueError('Verified raw sampling proof must belong to an accepted actual attempt')
    if any(x.get('metrics',{}).get('bapo/actual_outer_updates')!=x['step']for x in complete_events):
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
    print('FRESH200_STEP5_PAUSED_JSON='+json.dumps(dict(manifest_path=str(manifest_path),manifest_sha256=digest(manifest_path),pid=os.getpid())),flush=True)
    while not Path(receipt).exists():time.sleep(interval)
    # An invalid present receipt is a hard failure, never a silent release.
    result=validate_release(receipt,manifest_path,repo,expected_driver_pid=os.getpid())
    print('FRESH200_STEP5_RELEASED_JSON='+json.dumps(dict(receipt=str(receipt),receipt_sha256=digest(receipt),pid=os.getpid())),flush=True)
    return result


def validate_first_sampling_evidence(members, config, source, run, driver_pid, rank_reports=None):
    """Replay actual transported probabilities, bind raw native initial evidence."""
    import torch
    from verl import DataProto
    from .fresh200_contract import verify_sampling_policy_binding, critical_source_hashes
    def member(name):
        found=[Path(p) for p in members if Path(p).name==name]
        if len(found)!=1 or digest(found[0])!=members[str(found[0])]:
            raise ValueError('Unique immutable first sampling/initial evidence missing: '+name)
        return found[0]
    path=member('formal-first-sampling-batch.pt');report=json.loads(member('formal-first-sampling-batch.json').read_text())
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if (report.get('path')!=str(path) or report.get('sha256')!=members[str(path)]
            or report.get('sampling_verifier_status')!='PASS_ACTUAL_BATCH_BINDING_NOT_RELEASE'
            or payload.get('source_commit')!=source or payload.get('source_sha256')!=critical_source_hashes()
            or payload.get('driver_pid')!=driver_pid or payload.get('run_root')!=str(run)
            or type(payload.get('attempt_id')) is not int or payload['attempt_id']<1
            or type(payload.get('actual_outer')) is not int or not 1<=payload['actual_outer']<=5):
        raise ValueError('First formal raw sampling identity differs')
    for key in ('version','status','source_commit','source_sha256','run_root','driver_pid','attempt_id','actual_outer','sampling_binding_sha256','max_selected_logprob_abs_diff','selected_tokens','sampling_logprob_atol'):
        if json.dumps(report.get(key),sort_keys=True,allow_nan=False)!=json.dumps(payload.get(key),sort_keys=True,allow_nan=False):
            raise ValueError('First raw proof/report typed identity differs: '+key)
    data=DataProto.from_dict(tensors=payload['tensors'],non_tensors=payload['non_tensor_batch'],meta_info=payload['meta_info'])
    if (type(data.meta_info.get('attempt_id')) is not int or data.meta_info['attempt_id']!=payload['attempt_id']
            or type(data.meta_info.get('actual_outer')) is not int or data.meta_info['actual_outer']!=payload['actual_outer']
            or data.meta_info.get('fresh200_shared_input_bindings')!=payload.get('shared_input_bindings')):
        raise ValueError('First batch actual attempt/shared inputs differ')
    binding=verify_sampling_policy_binding(data,config['actor_rollout_ref']['actor']['policy_safety'])
    mask=data.batch['sampling_mask'].bool()
    gap=float((data.batch['old_log_probs'][mask].double()-data.batch['sampling_log_probs'][mask].double()).abs().max())
    if (binding!=payload.get('sampling_binding_sha256') or binding!=report.get('sampling_binding_sha256')
            or gap!=payload.get('max_selected_logprob_abs_diff') or gap!=report.get('max_selected_logprob_abs_diff')
            or int(mask.sum())!=report.get('selected_tokens')):
        raise ValueError('First raw sampler numerical replay differs')
    alignment=json.loads(member('native-initial-official-alignment.json').read_text())
    official=member('native-initial-official-hf.pt')
    if alignment.get('path')!=str(official) or alignment.get('sha256')!=members[str(official)]:
        raise ValueError('Actual official initial alignment is not directly bound')
    checkpoint_pids={r['rank']:r['pid'] for r in (rank_reports or [])}
    for rank in range(8):
        state=member(f'native-initial-rank_{rank}.pt');initial=json.loads(member(f'native-initial-rank_{rank}.json').read_text())
        if (initial.get('path')!=str(state) or initial.get('sha256')!=members[str(state)]
                or initial.get('official_alignment')!=alignment or initial.get('source_sha256')!=payload['source_sha256']
                or type(initial.get('optimizer_state_entries')) is not int or initial['optimizer_state_entries']!=0
                or type(initial.get('scheduler_last_epoch')) is not int or initial['scheduler_last_epoch']!=0
                or initial.get('lr')!=[1e-6/57] or initial.get('status')!='ACTUAL_INITIAL_STATE_CAPTURED_NOT_RELEASE'
                or type(initial.get('native_process',{}).get('rank')) is not int
                or initial.get('native_process',{}).get('rank')!=rank
                or (rank_reports is not None and (type(initial.get('native_process',{}).get('pid')) is not int
                    or initial['native_process']['pid']!=checkpoint_pids.get(rank)))):
            raise ValueError('Eight rank actual fresh initial evidence is not directly bound')
    return report
