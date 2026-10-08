"""Explicit failed-before-Adam C0 recovery; original training config is immutable."""
import os
from pathlib import Path
import subprocess
try:
    from .utils import *
except ImportError:
    from utils import *


def absent(identity, reader):
    try:
        actual=reader(identity['pid'])
    except (FileNotFoundError, ProcessLookupError):
        return True
    require(actual is None or any(actual.get(k)!=identity[k] for k in ('pid','uid','starttick')),
            'Previous exact owned process is still alive; never signal/restart it')
    return True


def previous_values(value,train,tf):
    r=value['recovery']
    require(r['schema']=='opsd-ablation-c0-before-adam-recovery-v1'
        and r['training_config']==tf and train['variant']=='C0'
        and r['run_id']==train['run_id'] and r['output_root']==train['paths']['output_root'],
        'Recovery is outside exact original C0 config/run/output')
    started,_=checked_json(r['driver_start']);status,_=checked_json(r['training_failure'])
    terminal,_=checked_json(r['actual_training_exit']);released,_=checked_json(r['actual_training_release'])
    require(status['phase']=='FAILED_OWNED_RELEASE_CONFIRMED' and status['signals_sent'] is False
        and status['run_id']==train['run_id'] and status['config_sha256']==tf['sha256']
        and terminal['actual_wait_returncode']!=0 and terminal['identity']==status['owner']
        and released['accepted'] is True, 'Prior failure/actual wait/release is not accepted')
    return r,started,status


def dead_pane(session, expected):
    pane=subprocess.run(['tmux','list-panes','-t',session,'-F',
        '#{session_id} #{pane_id} #{pane_pid} #{pane_dead} #{pane_dead_status}'],
        capture_output=True,text=True,check=True).stdout.strip().splitlines()
    require(pane==[expected] and expected.endswith(' 1 2'),
        'Actual previous study driver terminal is not same pane rc2')
    return pane[0]


def validate_previous(value,train,tf,admit,owner):
    r,started,status=previous_values(value,train,tf)
    pane=dead_pane(r['previous_session'],r['expected_dead_pane'])
    identities=[started['owner'],status['owner']]+status['known_children']
    for identity in identities:
        require(identity['uid']==owner['uid'],'Prior ownership UID differs')
        absent(identity,admit.process)
    outer=Path(train['paths']['output_root'])/'outers'/'outer-00000'
    manifest,mp=checked_json(r['outer_manifest'])
    require(Path(mp['path'])==outer/'outer-manifest.json' and manifest['outer_index']==0
        and manifest['run_id']==train['run_id'] and manifest['config_sha256']==tf['sha256']
        and manifest['cumulative_optimizer_updates_before']==0 and manifest['explicit_fresh'] is True,
        'Only failed-before-first-Adam C0 may resume without a checkpoint')
    require(not list((Path(train['paths']['output_root'])/'checkpoints').glob('*/manifest.json'))
        and not (outer/'training'/'outer-complete.json').exists(),
        'Unexpected committed checkpoint; separate state restoration required')
    roles,_=checked_json(fileproof(train['gates']['scoped_cpu_processes_path']))
    observer_role=validate_old_roles(value,train,tf,roles,admit)
    role_driver=next(role['identity'] for role in roles['roles'] if role['role']=='study_driver')
    if role_driver not in identities:identities.append(role_driver)
    observed=admit.memory_observer_admission(train)
    scanner_manifest,_=checked_json(train['helper_source_manifest'])
    row=scanner_manifest['files']['owned_process_audit.py']
    scanner=fresh_module('dtype_previous_scanner',Path(train['paths']['helper_root'])/'owned_process_audit.py',row['sha256'])
    observer=observer_role['identity']
    audit=scanner.audit_same_uid(train['run_id'],owner['uid'],status['owner']['starttick'],identities,
        [str(Path(train['paths']['helper_root'])/n) for n in ('sampling.py','trainer.py','teacher_diag.py')]
        +[train['paths']['output_root']],
        exclusions=scanner.load_exclusions(train.get('gates',{}).get('process_exclusions',[])),
        ignored_exact=[owner,observer],ancestry_anchors=identities)
    require(audit['accepted_no_owned_or_opaque'],'Previous owned/opaque worker remains; preserve original failure')
    return dict(accepted=True,previous_driver=started['owner'],previous_controller=status['owner'],
        actual_dead_pane=pane,original_failure=r['training_failure'],original_outer=mp,
        no_checkpoint_or_adam_resumed=True,no_sampling_repeated=True,owned_release=audit,
        observer=observed,no_signals=True)


def validate_old_roles(value,train,tf,old,admit):
    r,started,_=previous_values(value,train,tf)
    keys=('role_previous_driver_start','role_previous_session','role_previous_dead_pane')
    require(not any(k in r for k in keys) or all(k in r for k in keys),
        'Additional previous role driver requires complete start/session/dead-pane bindings')
    if all(k in r for k in keys):
        started,_=checked_json(r['role_previous_driver_start'])
        dead_pane(r['role_previous_session'],r['role_previous_dead_pane'])
    require(old['run_id']==train['run_id'] and old['config_sha256']==tf['sha256']
        and sorted(r['role'] for r in old['roles'])==['memory_observer','study_driver'],
        'Original exact two CPU roles differ')
    previous=next(r for r in old['roles'] if r['role']=='study_driver')
    require(previous['identity']==started['owner'] and previous['input_config']==started['input_manifest'],
        'Unrelated previous driver CPU role/start/input')
    previous_input,_=checked_json(previous['input_config'])
    require(previous_input['study_id']==train['study_id'] and previous_input['run_id']==previous['identity']['run_id']
        and any(group['training_config']['path']==tf['path'] and group['training_config']['sha256']==tf['sha256']
            for group in previous_input['groups']), 'Previous role driver is outside this immutable study/config')
    absent(previous['identity'],admit.process)
    observer=next(r for r in old['roles'] if r['role']=='memory_observer')
    live=admit.process(observer['identity']['pid'])
    require(admit.same_process(live,observer['identity']) and live['uid']==previous['identity']['uid']
        and all(live.get(k)==observer['identity'].get(k) for k in ('argv','run_id','pgid'))
        and observer['input_config']==tf,'Existing observer identity/config changed')
    # A surviving observer can legitimately be reparented after its driver exits.
    # Preserve its original binding; only the dynamic role uses current /proc identity.
    return dict(observer,identity=live)


def repair_command(value,train,tf,index):
    repair,proof=checked_json(value['dtype_repair'])
    require(repair['schema']=='opsd-ablation-dtype-repair-manifest-v1'
        and repair['study_id']==train['study_id'],'Wrong dtype repair study')
    candidates=[g for g in repair['original_configs'] if g['path']==tf['path']]
    require(len(candidates)==1 and candidates[0]['sha256']==tf['sha256']
        and ((index==0)==(train['variant']=='C0')),'Overlay group/scientific input differs')
    source=Path(repair['helper_root'])/'controller.py'
    manifest,_=checked_json(repair['source_manifest'])
    fileproof(source,manifest['files']['controller.py']['sha256'])
    command=[str(source),'--repair-manifest',proof['path'],'--repair-manifest-sha256',proof['sha256'],
        '--config',tf['path'],'--config-sha256',tf['sha256'],'--run-id',train['run_id']]
    if index==0:command.append('--resume')
    return command
