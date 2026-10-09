"""Consume the actual mandatory70/5632 closure before any L resource/model work."""
from pathlib import Path
try:
    from .utils import *
except ImportError:
    from utils import *

GROUPS=('C0','C1','M','R','W','P','J')
LABELS={'C0','C1','M','R','W','P','J','bf16_zero','fp16_zero','P_fp16','J_fp16'}


def totals(values,run):
    import math
    require(len(values)==10,'NOT_READY: ten committed outers per group required')
    updates=exposure=0
    for index,value in enumerate(values):
        require(value['accepted'] is True and value['run_id']==run and value['outer_index']==index,'Wrong completed outer sequence')
        P,K=value['P'],value['K'];require(type(P) is int and type(K) is int and 0<=K<=P<=512,'Invalid P/K')
        delta=math.ceil(K/32);updates+=delta;exposure+=K
        require(value['optimizer_updates_delta']==delta and value['optimizer_updates']==updates
            and value['cumulative_pair_exposure']==exposure and value['scheduler_advanced_updates']==0
            and value['empty_outer_no_adam'] is (K==0),'Broken actual Adam/exposure tail chain')
    return dict(outer_iterations=10,optimizer_updates=updates,cumulative_pair_exposure=exposure)


def validate(config):
    reference=config['mandatory_completion'];value,vp=checked_json(reference)
    require(value['schema']=='opsd-legacy-mandatory-closure-inputs-v1' and value['state']=='READY'
        and value['study_id']==config['study_id'] and set(value['groups'])==set(GROUPS),
        'NOT_READY: explicit actual mandatory study closure required')
    complete,cp=checked_json(value['driver_complete']);start,sp=checked_json(value['driver_start'])
    manifest,ip=checked_json(value['study_manifest']);terminal,tp=checked_json(value['foreground_terminal'])
    require(complete['accepted'] is True and complete['primary_slots']==5632 and complete['model_count']==11
        and complete['actual_group_train_and_eval_exit0'] is True and complete['input_manifest']==value['study_manifest']
        and start['input_manifest']==value['study_manifest'] and manifest['study_id']==config['study_id'],
        'NOT_READY: actual eleven-model driver completion missing')
    require(terminal['accepted'] is True and terminal['actual_exit_code']==0
        and terminal['signal_termination_not_exit0'] is False and terminal['owner']==start['owner']
        and terminal['run_id']==manifest['run_id'] and terminal['input_manifest']==value['study_manifest'],
        'NOT_READY: actual successful main foreground terminal missing')
    waited,wp=checked_json(terminal['actual_wait_evidence'])
    require(waited['identity']==terminal['owner'] and waited['actual_wait_returncode']==0
        and waited['exit_status_source'] in ('Popen.wait','retained_same_pid_foreground_pane'),
        'No actual main wait0; absence cannot prove success')
    summary,smp=checked_json(complete['summary'])
    require(summary['schema']=='opsd-ablation-eleven-fixed-model-summary-v1' and summary['accepted'] is True
        and summary['study_id']==config['study_id'] and summary['primary_slots']==5632
        and set(summary['models'])==LABELS and len(summary['inputs'])==11,'Mandatory5632 summary incomplete')
    train,_=original_config(config)
    require(summary['identities']['source_model_identity']==SOURCE_MANIFEST_SHA
        and summary['identities']['runtime_identity']==RUNTIME_SHA
        and summary['identities']['validation_sha256']==VALIDATION_SHA
        and summary['identities']['common_initial_adapter_tensor_sha256']==COMMON_TENSOR_SHA,
        'Main and L common model/data/template differ')
    require([x['variant'] for x in manifest['groups']]==list(GROUPS),'Mandatory original seven-group order differs')
    proofs=[];known=[start['owner']];chains={};scopes=[]
    for variant in GROUPS:
        item=value['groups'][variant]
        group,gp=checked_json(item['config']);status,stp=checked_json(item['status']);end,ep=checked_json(item['terminal'])
        original=next(x for x in manifest['groups'] if x['variant']==variant)['training_config']
        require(all(gp[k]==original[k] for k in ('path','sha256')),'Group is not the main immutable study configuration')
        require(group['schema']=='opsd-ablation-config-v1' and group['variant']==variant
            and group['study_id']==config['study_id'] and group['scientific']['outer_iterations']==10,
            'Wrong original mandatory group')
        root=Path(group['paths']['output_root']).resolve()
        require(stp['path']==str(root/'status.json') and status['phase']=='TRAIN_COMPLETE'
            and status['outer_iterations']==10 and status['run_id']==group['run_id']
            and status['config_sha256']==gp['sha256'],'Actual group TRAIN_COMPLETE10 missing')
        require(end['accepted'] is True and end['actual_exit_code']==0 and end['run_id']==group['run_id']
            and end['config_sha256']==gp['sha256'] and status['owner']==end['controller'],
            'Actual group successful terminal differs')
        completions=[];previous=None
        for index,ref in enumerate(item['outer_completions']):
            outer,op=checked_json(ref)
            require(op['path']==str(root/'outers'/f'outer-{index:05d}'/'training/outer-complete.json'),
                'Completion outside actual group outer path')
            checkpoint,fp=checked_json(outer['checkpoint'])
            require(checkpoint['run_id']==group['run_id'] and checkpoint['config_sha256']==gp['sha256']
                and checkpoint['optimizer_updates']==outer['optimizer_updates'] and checkpoint['cursor']['outer_index']==index
                and checkpoint['cursor']['next_pair_cursor']==checkpoint['cursor']['usable_pair_count']==outer['K']
                and checkpoint['source_model_identity']==SOURCE_MANIFEST_SHA and checkpoint['runtime_identity']==RUNTIME_SHA,
                'Committed mandatory checkpoint/cursor differs')
            previous=outer['checkpoint'];completions.append(outer);proofs.extend([op,fp])
        chain=totals(completions,group['run_id'])
        require(status['latest_checkpoint']==previous and status['actual_optimizer_updates']==chain['optimizer_updates'],
            'Group final actual update chain differs')
        history,hp=rows(item['child_history']);require(hp['path']==str(root/'child-history.jsonl'),'Wrong actual child history')
        group_known=[end['controller']] + [r['identity'] for r in history];known+=group_known
        scopes.append(dict(run_id=group['run_id'],starttick=end['controller']['starttick'],known=group_known,
            argv_roots=[str(Path(group['paths']['helper_root'])/n) for n in ('sampling.py','trainer.py')]+[str(root)]))
        # All group inference results must already be accepted by the original parent.
        result,rp=checked_json(item['result_accepted']);require(result['accepted'] is True,'Group evaluation not accepted')
        proofs.extend([gp,stp,ep,hp,rp]);chains[variant]=chain
    eval_runs=set()
    for entry in summary['inputs']:
        native,np=checked_json(entry['summary'])
        require(native['accepted'] is True and native['study_id']==config['study_id']
            and native['identities']==summary['identities'],'Closed native summary identity differs')
        if native['run_id'] in eval_runs:continue
        eval_runs.add(native['run_id'])
        parent_ref=fileproof(Path(np['path']).parent/'controller-start.json')
        parent,pr=checked_json(parent_ref);phase_known=[parent['owner']];phases=0
        for ref in native['inputs']:
            if not ref['path'].endswith('/phase-complete.json'):continue
            phase,pp=checked_json(ref)
            require(phase['accepted'] is True and phase['expected_worker_count']==phase['actual_worker_count']==8
                and len(phase['child_exit_evidence'])==8
                and all(x['actual_wait_returncode']==0 for x in phase['child_exit_evidence']),
                'Native model actual eight exits not rc0')
            phase_known += [r['identity'] for r in phase['child_exit_evidence']];phases+=1;proofs.append(pp)
        require(phases==len(native['models']) and parent['run_id']==native['run_id'],'Native phase/parent count differs')
        scopes.append(dict(run_id=native['run_id'],starttick=parent['owner']['starttick'],known=phase_known,
            argv_roots=[str(Path(np['path']).parent)]))
        proofs.extend([np,pr]);known+=phase_known
    scopes.append(dict(run_id=manifest['run_id'],starttick=start['owner']['starttick'],known=known,
        argv_roots=[str(Path(manifest['helper_root'])/'study_runner.py'),manifest['output_root']]))
    # Original private primary refs are checked by the main producer; L must bind
    # the same exact summaries for its two prespecified paired comparisons.
    for label in ('bf16_zero','C0'):
        native=next(r for r in summary['inputs'] if r['label']==label)
        require(config['comparison_inputs'][label]['summary']==native['summary'],'Comparison differs from closed eleven-model study')
    return dict(accepted=True,outer_iterations=70,primary_slots=5632,model_count=11,groups=chains,
        known_identities=known,release_scopes=scopes,study_manifest=ip,driver_complete=cp,driver_start=sp,foreground_terminal=tp,
        actual_wait_evidence=wp,summary=smp,mandatory_input=vp,group_file_proofs=proofs)
