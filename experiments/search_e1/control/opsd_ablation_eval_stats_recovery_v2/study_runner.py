"""CPU sequential seven-group driver: real train exit -> immediate own raw10 eval.

No model import, signals, silent resume or resource-lock ownership by this parent.
The published native child controllers retain their own shared/study locks.
"""
import argparse
import os
from pathlib import Path
import subprocess
import time
try:
    from .utils import *
    from . import gates,prepare,completion
    from .controller import capture_child_identity,hold_evidence,durable
except ImportError:
    from utils import *
    import gates,prepare,completion
    from controller import capture_child_identity,hold_evidence,durable


def validate_manifest(value):
    require(value['schema']=='opsd-ablation-seven-sequential-study-v1' and len(value['run_id'])==32
        and all(c in '0123456789abcdef' for c in value['run_id']),'Wrong independent study driver')
    require(value.get('state')=='READY','NOT_READY study manifest cannot start a training or evaluation child')
    require([x['variant'] for x in value['groups']]==list(VARIANTS),'Require explicit C0/C1/M/R/W/P/J order')
    common=None;run_ids=[];outputs=[]
    require(value.get('completed_c0') and value.get('dtype_repair') and not value.get('recovery'),
        'Explicit completed C0 eval recovery required; before-Adam recovery cannot be reused')
    for index,item in enumerate(value['groups']):
        train,tf=checked_json(item['training_config']);evaluation,ef=checked_json(item['evaluation_template'])
        require(train['schema']=='opsd-ablation-config-v1' and train['variant']==item['variant']
            and train['study_id']==value['study_id'] and train['scientific']['outer_iterations']==10,'Wrong study group config')
        fingerprint={k:train[k] for k in ('source_model_manifest','train_identity','validation_identity','common_cohort_manifest',
            'common_initial_adapter_manifest','common_initial_adapter_tensor_sha256')}
        fingerprint['runtime_sha256']=train['runtime_manifest']['sha256'];fingerprint['helper_sha256']=train['helper_source_manifest']['sha256']
        if common is None:common=fingerprint
        require(fingerprint==common,'Seven groups do not share exact source/data/cohorts/initial adapter/published training source')
        prepare.validate_config(evaluation)
        require(evaluation.get('source_commit') and evaluation.get('helper_source_manifest') and all(evaluation['gates'].get(k)
            for k in ('publication','storage','memory_observer','memory_observer_config','memory_observer_source')),
            'NOT_READY: explicit source/storage/new-group observer refs required before starting this study')
        require(evaluation['evaluation_role']=='group' and evaluation['training']['config']==tf
            and evaluation['training']['terminal'] is None and evaluation['training']['raw10_checkpoint'] is None,
            'Evaluation template must be unfilled and bound to the immutable training group')
        require(evaluation['paths']['helper_root']==value['helper_root'],'Driver/evaluator deployed source differs')
        run_ids.extend([train['run_id'],evaluation['run_id']]);outputs.extend([train['paths']['output_root'],evaluation['paths']['output_root']])
        if index==0:
            zero,zf=checked_json(value['common_zero_template']);prepare.validate_config(zero)
            require(zero.get('source_commit') and zero.get('helper_source_manifest') and all(zero['gates'].get(k)
                for k in ('publication','storage','memory_observer','memory_observer_config','memory_observer_source')),
                'NOT_READY: common-zero source/storage/observer refs missing')
            require(zero['evaluation_role']=='common_zero' and zero['training']['config']==tf
                and zero['training']['terminal'] is None and zero['training']['raw10_checkpoint'] is None
                and zero['paths']['helper_root']==value['helper_root'],'One common-zero run must be bound to C0')
            run_ids.append(zero['run_id']);outputs.append(zero['paths']['output_root'])
    require(len(set(run_ids+[value['run_id']]))==len(run_ids)+1,'Study/train/eval run identities are not distinct')
    paths=[Path(p).resolve() for p in outputs+[value['output_root']]]
    require(len(set(paths))==len(paths) and not any(a!=b and (a.is_relative_to(b) or b.is_relative_to(a)) for a in paths for b in paths),
        'Independent study/train/eval output roots alias or overlap')
    return True


def bind_final_evaluation(template,terminal_ref,status):
    """Fill terminal/checkpoint and planned observer binding SHA; science is immutable."""
    import copy
    final=copy.deepcopy(template)
    require(final['training']['terminal'] is None and final['training']['raw10_checkpoint'] is None,'No replacement of already bound eval inputs')
    require(status['phase']=='TRAIN_COMPLETE' and status['outer_iterations']==10
        and status['run_id']==final['training']['run_id'] and status['config_sha256']==final['training']['config']['sha256'],
        'NOT_READY: actual group ten-outer terminal missing')
    final['training']['terminal']=terminal_ref;final['training']['raw10_checkpoint']=status['latest_checkpoint']
    if final.get('gates',{}).get('memory_observer',{}).get('sha256') is None and final.get('gates',{}).get('memory_observer'):
        # The planned binding path is immutable; only the actual observer's
        # newly created immutable binding SHA is materialized at this boundary.
        final['gates']['memory_observer']=fileproof(final['gates']['memory_observer']['path'])
    final['state']='BOUND: actual rc0 plus raw tenth checkpoint; fresh admission still mandatory'
    return final


class Driver:
    def __init__(self,value,ref):
        self.value=value;self.ref=ref;self.output=Path(value['output_root']);self.owner=None;self.observers=[]

    def role(self,identity,source,manifest,publication,input_config,name):
        return dict(role=name,identity=identity,argv_sha256=object_sha(identity['argv']),source=source,helper_source_manifest=manifest,
            publication=publication,input_config=input_config,no_gpu_or_training=True)

    def observer(self,train,tf,evaluation,index,admit):
        """Bootstrap only published CPU observation; no model or signal here."""
        receipt_path=Path(train['gates']['scoped_cpu_processes_path'])
        expected=Path(train['paths']['receipts']).resolve()/'scoped-cpu-processes.json'
        require(receipt_path.resolve()==expected,'Dynamic exact CPU roles outside planned group receipts')
        driver_role=self.role(self.owner,fileproof(Path(self.value['helper_root'])/'study_runner.py'),
            self.value['helper_source_manifest'],self.value['publication'],self.ref,'study_driver')
        role_value=dict(schema='opsd-ablation-scoped-cpu-processes-v1',accepted=True,run_id=train['run_id'],config_sha256=tf['sha256'],roles=[driver_role],updated_unix=time.time())
        if receipt_path.exists():
            old,old_ref=checked_json(fileproof(receipt_path))
            require(index==0 and old['run_id']==train['run_id'] and old['config_sha256']==tf['sha256'],
                'Only explicitly recovered C0 may migrate an existing exact CPU role receipt')
            from completed_c0 import validate_old_roles
            live_observer_role=validate_old_roles(self.value,train,tf,old,admit)
            durable(self.output/'previous-scoped-cpu-roles.json',dict(reference=old_ref,value=old))
            # Same live memory observer is retained; only dead study-driver role is replaced.
            role_value.update(roles=[driver_role,live_observer_role])
            atomic_json(receipt_path,role_value)
        else:atomic_json(receipt_path,role_value,exclusive=True)
        # This dedicated stage has all normal source/storage/transition/oracle/
        # empty-GPU gates, and only omits the not-yet-created observer sample.
        admit.require_launch_without_memory(train,train['run_id'],self.owner,known_children=[])
        binding_path=Path(train['gates']['memory_binding_path']);scope=self.output/(f'{index:02d}-'+train['variant'])/'observer'
        if binding_path.exists():
            binding,bp=checked_json(fileproof(binding_path));identity=admit.process(binding['observer_identity']['pid'])
            require(admit.same_process(identity,binding['observer_identity']) and identity['run_id']!=train['run_id'],
                'Existing observer binding is not live; no overwrite/restart')
            job=None
        else:
            command=[str(Path(train['paths']['helper_root'])/'memory_observer.py'),'--config',tf['path'],
                '--output',str(Path(train['paths']['receipts'])/'memory-live.json'),'--binding-output',str(binding_path)]
            job=self.spawn(command,'observer-'+train['run_id'],train['paths']['python'],scope,admit.process);self.observers.append(job)
            identity=job['identity'];require(type(identity.get('starttick')) is int and not job.get('identity_error'),'Observer actual identity unavailable')
        while True:
            if job and job['process'].poll() is not None:
                rc=job['process'].wait();job['handle'].close();durable(scope/'actual-observer-exit.json',dict(identity=identity,actual_wait_returncode=rc,no_signals=True))
                raise Hold('New observer exited before readiness; no training child started')
            try:
                observed=admit.memory_observer_admission(train)
                require(admit.same_process(observed['observer_identity'],identity),'Actual observer changed before fresh sample')
                break
            except Exception as error:
                hold_evidence(lambda:durable((scope if job else self.output/(f'{index:02d}-'+train['variant']))/
                    ('observer-uncertain-'+str(time.time_ns())+'.json'),dict(error_type=type(error).__name__,reason=str(error),no_training_started=True)))
                time.sleep(2)
        live_identity=admit.process(identity['pid'])
        require(admit.same_process(live_identity,identity)
            and all(live_identity.get(k)==identity.get(k) for k in ('argv','run_id','pgid')),
            'Observer identity changed while collecting its fresh sample')
        observer_role=self.role(live_identity,fileproof(Path(train['paths']['helper_root'])/'memory_observer.py'),
            train['helper_source_manifest'],train['gates']['publication'],tf,'memory_observer')
        role_value.update(roles=[driver_role,observer_role],updated_unix=time.time());atomic_json(receipt_path,role_value)
        validated=admit.scoped_cpu_processes(train)
        require(len(validated['identities'])==2,'Exact driver/observer roles not admitted')
        destination=self.output/(f'{index:02d}-'+train['variant'])/'observer-start-accepted.json'
        durable(destination,dict(accepted=True,role_receipt=fileproof(receipt_path),actual_observer=observed,
            reused_existing_binding=job is None,new_group_config=tf,no_signals=True))
        return observed

    def wait(self,job,scope,observe,release):
        """Persist actual wait return; observation uncertainty never invents failure."""
        child=job['process']
        while child.poll() is None:
            try:observe()
            except Exception as error:
                hold_evidence(lambda:durable(scope/('live-uncertain-'+str(time.time_ns())+'.json'),
                    dict(error_type=type(error).__name__,reason=str(error),not_a_child_returncode=True)))
            time.sleep(5)
        rc=child.wait();job['handle'].close()
        actual=dict(identity=job['identity'],command=job['command'],actual_wait_returncode=rc,log=fileproof(job['handle'].name),
            signal_termination_not_exit0=rc<0,run_id=job['run_id'],no_signals=True)
        hold_evidence(lambda:durable(scope/'actual-child-exit.json',actual))
        while True:
            try:
                result=release();require(result.get('accepted') is True,'Current child resources not released')
                hold_evidence(lambda:durable(scope/'actual-child-release.json',result));break
            except Exception as error:
                hold_evidence(lambda:durable(scope/('release-uncertain-'+str(time.time_ns())+'.json'),dict(error_type=type(error).__name__,reason=str(error))))
                time.sleep(5)
        require(rc==0 and type(job['identity'].get('starttick')) is int and not job['identity'].get('capture_status'),
            'Actual child nonzero or unknown identity preserved; no next group/evaluation')
        return actual,result

    def spawn(self,command,run_id,python,scope,identity_reader):
        scope.mkdir(parents=True,exist_ok=False)
        env=dict(os.environ,E1_RUN_ID=run_id,CUDA_VISIBLE_DEVICES='',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1')
        log=(scope/'foreground.log').open('x');child=subprocess.Popen([python,'-B']+command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        job=dict(process=child,handle=log,run_id=run_id,command=[python,'-B']+command,
            identity=dict(pid=child.pid,uid=self.owner['uid'],starttick=None))
        # Popen is registered before identity reads; quick-exit is UNKNOWN and
        # still waited/released, never spun forever or inferred rc0.
        try:
            actual=capture_child_identity(job,identity_reader);job['identity']=actual
            require(actual['uid']==self.owner['uid'] and actual['ppid']==self.owner['pid'] and actual['run_id']==run_id,
                'Actual child identity/run/parent differs')
        except Exception as error:job['identity_error']=str(error)
        hold_evidence(lambda:durable(scope/'child-start.json',dict(identity=job['identity'],command=job['command'],identity_error=job.get('identity_error'))))
        return job

    def training(self,item,index,evaluation):
        if index==0:
            from completed_c0 import resume_completed_training
            return resume_completed_training(self,item,evaluation)
        train,tf=checked_json(item['training_config']);train['_config_sha256']=tf['sha256']
        modules,_=frozen_helpers(evaluation);modules['prepare'].validate_config(train,launch=True);admit=modules['gates']
        # The new group observer must be live and bound to this group config.
        # Nothing is inferred from the stopped old200 observer.
        from recovery import repair_command
        observer=self.observer(train,tf,evaluation,index,admit)
        require(observer['observer_identity']['run_id']!=self.value['run_id'],'Study driver cannot masquerade as group memory observer')
        admit.require_launch(train,'cpu-merge',train['run_id'],self.owner,known_children=[])
        require(Path(train['paths']['output_root']).exists() if index==0 else not Path(train['paths']['output_root']).exists(), 'Only exact failed C0 may reuse output; other six are fresh')
        scope=self.output/(f'{index:02d}-'+item['variant'])/'training'
        command=repair_command(self.value,train,tf,index)
        job=self.spawn(command,train['run_id'],train['paths']['python'],scope,admit.process)
        known=[job['identity']]
        def release():
            result=admit.require_external_study_release(train,train['run_id'],self.owner,known_children=known)
            after=admit.require_launch(train,'cpu-merge',train['run_id'],self.owner,known_children=[])
            return dict(accepted=True,owned=result,after=after)
        actual,released=self.wait(job,scope,lambda:admit.require_launch(train,'sampling-running',train['run_id'],self.owner,known_children=known),release)
        status,sp=checked_json(fileproof(Path(train['paths']['output_root'])/'status.json'))
        require(status['phase']=='TRAIN_COMPLETE' and status['owner']==job['identity'] and status['outer_iterations']==10
            and status['run_id']==train['run_id'] and status['config_sha256']==tf['sha256'],'Actual rc0 not this group TRAIN_COMPLETE10')
        terminal=dict(schema='opsd-ablation-group-foreground-terminal-v1',accepted=True,run_id=train['run_id'],config_sha256=tf['sha256'],
            actual_exit_code=actual['actual_wait_returncode'],dtype_repair=self.value['dtype_repair'],controller=job['identity'],actual_child_exit=fileproof(scope/'actual-child-exit.json'),
            actual_release=fileproof(scope/'actual-child-release.json'),status=sp,signal_termination_not_exit0=False,
            study_driver=dict(owner=self.owner,input_manifest=self.ref),no_signals=True)
        # Freeze this group's end sample. The persistent observer may see later
        # groups, but those later global minima are not this group's training min.
        binding,_=checked_json(fileproof(train['gates']['memory_binding_path']))
        end_report,ep=checked_json(fileproof(binding['live_report_path']))
        durable(scope/'group-end-memory-snapshot.json',dict(accepted=True,group_run_id=train['run_id'],config=tf,
            report_ref=ep,report=end_report,scope='Samples until this group foreground exit; later persistent observer samples excluded'))
        return durable(scope/'group-terminal.json',terminal),status

    def evaluation(self,template,terminal,status,scope):
        final=bind_final_evaluation(template,terminal,status);prepare.validate_config(final)
        # Full ten-outer chain, native exact mining and final checkpoint file SHA
        # are checked before spawning the child, not by a self-reported phase.
        proof=completion.validate(final)
        finalref=durable(scope.parent/'evaluation-config.json',final)
        native=gates.native_admission(final)
        command=[str(Path(final['paths']['helper_root'])/'controller.py'),'--config',finalref['path'],'--config-sha256',finalref['sha256']]
        job=self.spawn(command,final['run_id'],final['paths']['python'],scope,native.process);known=[job['identity']]
        def release():
            owned=gates.owned_release(final,self.owner,known)
            current=gates.require_launch(final,self.owner,[],'released')
            return dict(accepted=True,owned=owned,current=current)
        self.wait(job,scope,lambda:gates.require_launch(final,self.owner,known,'running'),release)
        complete,cp=checked_json(fileproof(Path(final['paths']['output_root'])/'controller-complete.json'))
        summary,smp=checked_json(complete['summary'])
        require(complete['accepted'] is True and complete['run_id']==final['run_id'] and complete['config']==finalref
            and summary['accepted'] is True and summary['primary_slots']==final['scientific']['total_primary_slots'],
            'Actual evaluator rc0 lacks exact accepted512 model slots')
        durable(scope/'result-accepted.json',dict(accepted=True,readiness=proof,config=finalref,controller_completion=cp,summary=smp))
        return summary,smp

    def run(self):
        validate_manifest(self.value)
        from completed_c0 import preflight
        preflight(self.value)  # Actual ten-outer refs before output creation or Popen.
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules
            and os.environ.get('E1_RUN_ID')==self.value['run_id'],'Explicit CPU-only independent study parent required')
        view=dict(run_id=self.value['run_id'],helper_source_manifest=self.value['helper_source_manifest'],source_commit=self.value['source_commit'],
            paths=dict(helper_root=self.value['helper_root']),gates=dict(publication=self.value['publication']))
        gates.publication(view)
        first,_=checked_json(self.value['groups'][0]['evaluation_template']);self.owner=gates.process(first,os.getpid())
        require(self.owner['run_id']==self.value['run_id'] and self.owner['uid']==first['owner']['uid'],'Study parent actual identity differs')
        require(not self.output.exists(),'Existing partial study is preserved; no implicit rerun')
        self.output.mkdir(parents=True,exist_ok=False);durable(self.output/'driver-start.json',dict(owner=self.owner,input_manifest=self.ref,
            no_resource_locks_held_by_parent=True,no_training_in_parent=True))
        models={}
        for index,item in enumerate(self.value['groups']):
            template,_=checked_json(item['evaluation_template']);terminal,status=self.training(item,index,template)
            result,ref=self.evaluation(template,terminal,status,self.output/(f'{index:02d}-'+item['variant'])/'evaluation')
            models[item['variant']]=dict(summary=ref,version='raw10_bf16')
            if item['variant'] in ('P','J'):models[item['variant']+'_fp16']=dict(summary=ref,version='raw10_fp16')
            if index==0:
                zero,_=checked_json(self.value['common_zero_template'])
                result,ref=self.evaluation(zero,terminal,status,self.output/'common-zero'/'evaluation')
                models.update({v:dict(summary=ref,version=v) for v in ('bf16_zero','fp16_zero')})
        require(set(models)=={'C0','C1','M','R','W','P','J','bf16_zero','fp16_zero','P_fp16','J_fp16'},'Eleven fixed512 models incomplete')
        compare_ref=durable(self.output/'eleven-model-inputs.json',dict(schema='opsd-ablation-eleven-fixed-model-inputs-v1',models=models))
        try:from .summary import compare
        except ImportError:from summary import compare
        result=compare(compare_ref);durable(self.output/'eleven-model-summary.json',result)
        return durable(self.output/'driver-complete.json',dict(accepted=True,primary_slots=5632,model_count=11,summary=fileproof(self.output/'eleven-model-summary.json'),
            input_manifest=self.ref,actual_group_train_and_eval_exit0=True,no_signals=True,no_resource_locks_held_by_parent=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study-manifest',required=True);parser.add_argument('--study-manifest-sha256',required=True)
    a=parser.parse_args();value,ref=checked_json(dict(path=a.study_manifest,sha256=a.study_manifest_sha256))
    try:result=Driver(value,ref).run()
    except Exception as error:
        print(json.dumps(dict(accepted=False,decision='HOLD_OR_REJECTED_STUDY',reason=str(error),error_type=type(error).__name__,no_signals=True)),flush=True);return 2
    print(json.dumps(dict(accepted=True,complete=result)));return 0


if __name__=='__main__':raise SystemExit(main())
