"""Scheduling-only correction requested by the human: train first, tests pending.

No model/objective/optimizer/trajectory settings are modified. Saved outer
completions remain the authority for the same-run resume. Deferred is explicitly
not an accepted evaluation, and the study terminal is training-only.
"""
import json,os,shutil,subprocess,time
from pathlib import Path
from recovery_common import checked_json, require


def startup(repair,configs,gates):
    require(repair['human_instruction']['defer_evaluation_and_continue_training'] is True,
            'Original train-first instruction missing')
    status=subprocess.check_output(['tmux','display-message','-p','-t','%47',
        '#{pane_pid}|#{pane_dead}|#{pane_dead_status}'],text=True).strip()
    require(status=='1861244|1|1','Actual previous failed supervisor terminal differs')
    terminal=checked_json(repair['previous_controller_terminal'])
    require(terminal['actual_wait_returncode']==1 and terminal['controller']['pid']==1977766
        and terminal['controller']['starttick']==690962026,'Failed controller terminal differs')
    for pid in repair['previous_processes']:
        require(not Path('/proc',str(pid)).exists(),'Previous owned child still present')
    cp=checked_json(repair['resume_checkpoint']);audit=checked_json(repair['independent_state_audit'])
    require(cp['accepted'] and cp['optimizer_updates']==88 and cp['cursor']['outer_index']==8,
        'Original last committed CP88/outer8 required')
    require(audit['accepted'] and audit['optimizer_updates']==88 and audit['outer_index']==8
        and audit['all_four_exact_full_adam'] and audit['adapter_finite_fp32']
        and audit['checkpoint']['sha256']==repair['resume_checkpoint']['sha256'],
        'Independent full CP88 CPU readback missing')
    cfg=next(c for c,ref in configs.values() if c['variant']=='G_NEG')
    output=Path(cfg['paths']['output_root']);pending=output/'outers/outer-00009'
    require((output/'outers/outer-00008/training/outer-complete.json').exists(),
        'Nine committed G_NEG outers missing')
    require(not (pending/'training/outer-complete.json').exists()
        and not (pending/'training/live-update.json').exists(),
        'Unexpected new outer9 update: rebind actual committed state')
    sampling=checked_json(repair['pending_sampling']);teacher=checked_json(repair['pending_teacher'])
    require(sampling['accepted'] and sampling['train_em']['overall']['denominator']==2560
        and teacher['accepted'],'Existing full sampling/teacher diagnostic not accepted')
    root=Path(repair['helper_root']).parents[2]
    preserve=root/'preserved'/('port-resume-'+repair['supervisor_run_id'])
    preserve.mkdir(parents=True,exist_ok=False);copies=[]
    for source in [root/'study-status.json',output/'status.json',
        Path(cfg['gates']['scoped_cpu_processes_path']),output/'failure-actual-exits.json',
        pending/'trainer-logs/attempt-00000.log']:
        if source.exists():
            target=preserve/(source.parent.name+'-'+source.name)
            shutil.copyfile(source,target);copies.append(dict(original=str(source),preserved=str(target)))
    (preserve/'reason.json').write_text(json.dumps(dict(previous_terminal=status,
        previous_processes_absent=repair['previous_processes'],copies=copies,
        reason='EADDRINUSE before trainer model load; zero new outer9 Adam',
        same_run=True,sampling_and_teacher_reused=True,no_signals_sent=True),indent=2)+'\n')
    return dict(accepted=True,resume_checkpoint=repair['resume_checkpoint'],
        previous_terminal=status,previous_processes_absent=True,preserved=str(preserve),
        pending_sampling=repair['pending_sampling'],checked_unix=time.time())


def deferred_boundary(controller,completed_outer,checkpoint_ref,repair):
    require(completed_outer in (15,25) and checkpoint_ref is not None,'Explicit postraw endpoint required')
    value=dict(schema='opsd-signal-evaluation-deferred-v1',accepted=False,training_checkpoint_retained=True,
        evaluation_pending=True,run_id=controller.run_id,config_sha256=controller.config_ref['sha256'],
        completed_outer=completed_outer,raw_checkpoint=checkpoint_ref,human_instruction=repair['human_instruction'])
    path=controller.output/'evaluations'/('deferred-outer-%02d.json'%completed_outer)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(json.loads(path.read_text())==value,'Deferred endpoint identity changed')
    else:path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    return value


def study_schedule(source):
    old="""        for iteration in (15,25):
            parent, _ = checked_json(fileproof(output / 'evaluations' / ('outer-%02d' % iteration) / 'parent-complete.json'))
            require(parent['accepted'] and parent['completed_outer'] == iteration, 'Formal15/25 parent evidence absent')
"""
    new="""        for iteration in (15,25):
            pending, _ = checked_json(fileproof(output / 'evaluations' / ('deferred-outer-%02d.json' % iteration)))
            require(pending['accepted'] is False and pending['evaluation_pending'] is True
                and pending['completed_outer'] == iteration, 'Deferred raw endpoint identity absent')
"""
    require(source.count(old)==1,'Original study evaluation scheduling source changed')
    source=source.replace(old,new)
    source=source.replace('FOUR_GROUPS_TRAIN_AND_EVAL_COMPLETE','FOUR_GROUPS_TRAIN_COMPLETE_TESTS_PENDING')
    source=source.replace('analysis_still_required=True))','analysis_still_required=True, evaluation_pending=True))')
    return source


def load_scheduled(path,mode,repair):
    source=Path(path).read_text()
    if mode=='study':source=study_schedule(source)
    else:
        require(mode=='controller','Only CPU scheduling code may be overlaid')
        source=source.replace('Twenty-five complete outers, full TRAIN EM and accepted postraw15/25 native512; resources released',
            'Twenty-five complete outers and full TRAIN EM; postraw15/25 tests deferred and pending; resources released')
    namespace=dict(__file__=str(path),__name__='_signal_training_priority_'+mode)
    exec(compile(source,str(path),'exec'),namespace)
    if mode=='controller':
        cls=namespace['Controller'];original=cls.evaluate_boundary
        def boundary(self,completed_outer,checkpoint_ref=None):
            if completed_outer==0:return original(self,completed_outer,checkpoint_ref)
            return deferred_boundary(self,completed_outer,checkpoint_ref,repair)
        cls.evaluate_boundary=boundary
    return namespace
