"""Reuse completed C0 only after its old study parent actually exits; no training."""
import shlex
import subprocess
from pathlib import Path
try:
    from .utils import *
    from . import completion
except ImportError:
    from utils import *
    import completion


def same_ref(a,b):
    return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('path','sha256'))


def absent(identity,reader):
    try: actual=reader(identity['pid'])
    except (FileNotFoundError,ProcessLookupError): return True
    require(actual is None or any(actual.get(k)!=identity[k] for k in ('pid','uid','starttick')),
        'Previous exact owned process remains; no completed-C0 evaluation yet')
    return True


def preflight(value):
    """Bound file checks only; no process creation, training, HTTP or CUDA."""
    r=value['completed_c0']
    require(r['schema']=='opsd-ablation-completed-c0-eval-recovery-v1','Wrong completed-C0 recovery schema')
    item=value['groups'][0];train,tf=checked_json(item['training_config'])
    require(train['variant']=='C0' and train['study_id']==value['study_id'] and train['scientific']['outer_iterations']==10
        and same_ref(r['training_config'],tf),'Only this immutable C0 may bypass training')
    start,sp=checked_json(r['previous_driver_start']);old,op=checked_json(r['previous_study_manifest'])
    require(same_ref(start['input_manifest'],op) and start['owner']['run_id']==old['run_id']
        and old['study_id']==value['study_id'] and old['run_id']!=value['run_id']
        and [x['variant'] for x in old['groups']]==list(VARIANTS)
        and all(same_ref(a['training_config'],b['training_config']) for a,b in zip(old['groups'],value['groups'])),
        'Previous parent/input is outside the same exact seven-group study')
    terminal,tp=checked_json(r['completed_training_terminal']);status,stp=checked_json(r['training_status'])
    current,cp=checked_json(fileproof(Path(train['paths']['output_root'])/'status.json'))
    require(current==status and cp['sha256']==stp['sha256'],'Completed C0 status changed after terminal binding')
    require(terminal['schema']=='opsd-ablation-group-foreground-terminal-v1' and terminal['accepted'] is True
        and terminal['actual_exit_code']==0 and terminal['signal_termination_not_exit0'] is False
        and terminal['run_id']==train['run_id'] and terminal['config_sha256']==tf['sha256']
        and terminal['dtype_repair']==value['dtype_repair']
        and terminal['study_driver']['owner']==start['owner'] and same_ref(terminal['study_driver']['input_manifest'],op)
        and same_ref(terminal['status'],stp)
        and status['phase']=='TRAIN_COMPLETE' and status['outer_iterations']==10 and status['owner']==terminal['controller']
        and status['run_id']==train['run_id'] and status['config_sha256']==tf['sha256']
        and same_ref(status['latest_checkpoint'],r['raw10_checkpoint']),'Missing exact successful C0 tenth-outer terminal')
    exited,_=checked_json(terminal['actual_child_exit']);released,_=checked_json(terminal['actual_release'])
    require(exited['actual_wait_returncode']==0 and exited['identity']==terminal['controller']
        and exited['run_id']==train['run_id'] and released['accepted'] is True,
        'Training terminal lacks actual wait0/release evidence')
    parent,pp=checked_json(r['previous_driver_terminal'])
    require(parent['schema']=='opsd-ablation-study-actual-terminal-v1' and parent['accepted'] is True
        and parent['actual_exit_code']==2 and parent['signal_termination_not_exit0'] is False
        and parent['owner']==start['owner'] and same_ref(parent['driver_start'],sp)
        and same_ref(parent['input_manifest'],op) and parent['session']==r['previous_session']
        and parent['pane_dead_record']==r['expected_dead_pane'],'Actual old study parent rc2 must be separately preserved')
    # A single foreground command preserves Python's status. No chained shell commands.
    tokens=shlex.split(parent['pane_start_command'])
    if tokens[:1]==['exec']:tokens=tokens[1:]
    require(tokens[:1]==['env'] and tokens[-3] in ('>','>>') and tokens[-2:]==[parent['foreground_log'],'2>&1'],
        'Old pane is not the exact single foreground command with fixed log redirection')
    argv=start['owner']['argv'];prefix=tokens[1:len(tokens)-len(argv)-3]
    require(tokens[len(prefix)+1:-3]==argv and prefix==['CUDA_VISIBLE_DEVICES=',
        'E1_RUN_ID='+old['run_id'],'PYTHONDONTWRITEBYTECODE=1']
        and argv[0]==train['paths']['python'] and argv[1:3]==['-B',str(Path(old['helper_root'])/'study_runner.py')]
        and argv[3:]==['--study-manifest',op['path'],'--study-manifest-sha256',op['sha256']],
        'Old pane grammar/run/source/input differs from actual recorded parent')
    return dict(recovery=r,train=train,training_ref=tf,start=start,old=old,terminal=terminal,status=status,
        previous_driver_terminal=pp,training_terminal=tp,training_status=stp)


def check_dead_parent(value,reader):
    p=preflight(value);r=p['recovery'];parent=p['start']['owner']
    result=subprocess.run(['tmux','list-panes','-t',r['previous_session'],'-F',
        '#{session_id} #{pane_id} #{pane_pid} #{pane_dead} #{pane_dead_status}'],capture_output=True,text=True,check=True)
    require(result.stdout.strip().splitlines()==[r['expected_dead_pane']] and r['expected_dead_pane'].endswith(' 1 2'),
        'Current old pane is not the bound actual dead parent rc2')
    command=subprocess.run(['tmux','display-message','-p','-t',r['previous_session'],'#{pane_start_command}'],
        capture_output=True,text=True,check=True).stdout.rstrip('\n')
    terminal,_=checked_json(r['previous_driver_terminal'])
    require(command==terminal['pane_start_command'],'Old pane foreground command changed')
    absent(parent,reader)
    return p


def validate_old_roles(value,train,tf,old,admit):
    p=check_dead_parent(value,admit.process);previous=p['start'];old_input=p['old']
    require(old['run_id']==train['run_id'] and old['config_sha256']==tf['sha256']
        and sorted(x['role'] for x in old['roles'])==['memory_observer','study_driver'],'Previous exact two CPU roles differ')
    driver=next(x for x in old['roles'] if x['role']=='study_driver')
    require(driver['identity']==previous['owner'] and same_ref(driver['input_config'],previous['input_manifest'])
        and driver['helper_source_manifest']==old_input['helper_source_manifest'] and driver['publication']==old_input['publication'],
        'Cannot replace an unrelated live or differently sourced study driver')
    inventory,mp=checked_json(driver['helper_source_manifest']);pub,_=checked_json(driver['publication'])
    source=Path(old_input['helper_root'])/'study_runner.py';entry=inventory['files']['study_runner.py']
    require(Path(driver['source']['path']).resolve()==source.resolve() and driver['source']['sha256']==entry['sha256']
        and fileproof(source,entry['sha256'])['bytes']==entry['bytes']
        and driver['argv_sha256']==object_sha(driver['identity']['argv'])
        and pub['accepted'] is True and pub['source_commit']==pub['remote_readback_sha']==old_input['source_commit']
        and pub['source_manifest_sha256']==mp['sha256'],'Prior CPU source/publication/argv binding changed')
    observer=next(x for x in old['roles'] if x['role']=='memory_observer');actual=admit.process(observer['identity']['pid'])
    require(admit.same_process(actual,observer['identity']) and actual['uid']==previous['owner']['uid']
        and all(actual.get(k)==observer['identity'].get(k) for k in ('argv','run_id','pgid'))
        and same_ref(observer['input_config'],tf),'Retained observer identity/config changed')
    # PPID legitimately changes after the previous driver exits; immutable binding remains.
    admit.memory_observer_admission(dict(train,_config_sha256=tf['sha256']))
    return dict(observer,identity=actual)


def resume_completed_training(driver,item,evaluation):
    """No spawn/torchrun/Adam/sampling call on the C0 completion path."""
    from controller import durable
    from study_runner import bind_final_evaluation
    p=preflight(driver.value);train=p['train'];tf=p['training_ref'];train['_config_sha256']=tf['sha256']
    modules,_=frozen_helpers(evaluation);modules['prepare'].validate_config(train,launch=True);admit=modules['gates']
    check_dead_parent(driver.value,admit.process)
    r=p['recovery'];final=bind_final_evaluation(evaluation,r['completed_training_terminal'],p['status'])
    readiness=completion.validate(final)  # full10/miner/diagnostics/raw9 before any GPU evaluator.
    history,_=rows(fileproof(Path(train['paths']['output_root'])/'child-history.jsonl'))
    known=[p['start']['owner'],p['terminal']['controller']]+[x['identity'] for x in history]
    for identity in known:
        require(identity['uid']==driver.owner['uid'],'C0 owned UID differs')
        absent(identity,admit.process)
    driver.observer(train,tf,evaluation,0,admit)  # exact new role + unchanged binding/fresh sample.
    release=admit.require_external_study_release(train,train['run_id'],driver.owner,known_children=known)
    after=admit.require_launch(train,'cpu-merge',train['run_id'],driver.owner,known_children=[])
    durable(driver.output/'00-C0'/'completed-training-reused.json',dict(accepted=True,
        previous_driver_terminal=p['previous_driver_terminal'],completed_training_terminal=r['completed_training_terminal'],
        training_status=r['training_status'],raw10_checkpoint=r['raw10_checkpoint'],full10=readiness,
        actual_release=release,after_release=after,no_training_spawn=True,no_sampling_repeated=True,no_adam_repeated=True))
    return r['completed_training_terminal'],p['status']
