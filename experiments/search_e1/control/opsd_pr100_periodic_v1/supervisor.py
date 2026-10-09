"""One CPU pipeline: train10 -> native512 -> same-state resume, through100.

Only the dedicated pipeline lock is held by this parent. Each sequential child
holds the exact shared GPU leases itself; no GPU phase can overlap or be orphaned.
"""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import subprocess
import time
try:
    from .utils import *
    from . import gates,prepare,completion,summary
    from .controller import durable,hold_evidence,capture_child_identity
except ImportError:
    from utils import *
    import gates,prepare,completion,summary
    from controller import durable,hold_evidence,capture_child_identity

def validate_manifest(value):
    require(value['schema']=='opsd-pr100-periodic-pipeline-v1' and value['state']=='READY','Pipeline is not READY')
    require(len(value['run_id'])==32 and all(c in '0123456789abcdef' for c in value['run_id']),'Wrong pipeline run identity')
    train,tf=checked_json(value['training_config'])
    require(train['schema']=='opsd-pr100-config-v1' and train['variant']=='PR' and train['study_id']==value['study_id']
        and train['scientific']['outer_iterations']==100,'Wrong immutable PR100 training')
    require([x['completed_outers'] for x in value['evaluations']]==prepare.BOUNDARIES,'Require every10 through100 in exact order')
    require(value['resources']==dict(exclusive_gpu_sequence=True,parent_holds_gpu_locks=False),'Parent cannot hold child GPU leases')
    runs=[value['run_id'],train['run_id']];outputs=[value['output_root'],train['paths']['output_root']]
    for item in value['evaluations']:
        evaluation,ef=checked_json(item['template']);prepare.validate_config(evaluation)
        require(evaluation['training']['config']==tf and evaluation['completed_outers']==item['completed_outers']
            and all(evaluation['training'][k] is None for k in ('terminal','boundary_checkpoint','boundary')),'Template already bound or wrong boundary')
        require(evaluation['paths']['helper_root']==value['helper_root'] and evaluation.get('source_commit')
            and evaluation.get('helper_source_manifest') and all(evaluation['gates'].get(k) for k in ('publication','storage','memory_observer')),'Unpublished eval template')
        runs.append(evaluation['run_id']);outputs.append(evaluation['paths']['output_root'])
    require(len(set(runs))==len(runs),'Run IDs alias')
    paths=[Path(p).resolve() for p in outputs]
    require(len(set(paths))==len(paths) and not any(a!=b and (a.is_relative_to(b) or b.is_relative_to(a)) for a in paths for b in paths),'Independent outputs overlap')
    previous,_=checked_json(value['startup_previous_driver_terminal'])
    successful=previous.get('accepted') is True and previous.get('actual_exit_code')==0
    observed_failure=(previous.get('schema')=='pr100-prior-driver-observed-terminal-v1'
        and previous.get('original_driver_pid')==1894801 and previous.get('original_driver_absent') is True
        and previous.get('original_driver_actual_foreground_exit_code')==2
        and previous.get('pane',{}).get('raw')=='$74\t%74\t1894801\t1\t2\tbash\n'
        and previous.get('gpu_compute',{}).get('actual_command_returncode')==0
        and not Path('/proc/1894801').exists())
    require(successful or observed_failure,'Previous eight-card driver remains live or terminal evidence is missing')
    baseline,bp=checked_json(value['baseline_summary'])
    require(baseline['accepted'] is True and value['baseline_version']=='bf16_zero'
        and baseline['models']['bf16_zero']['overall']['questions']==512
        and baseline['identities']['source_model_identity']==train['source_model_manifest']['sha256']
        and baseline['identities']['runtime_identity']==train['runtime_manifest']['sha256']
        and baseline['identities']['validation_sha256']==train['validation_identity']['sha256'],'Existing native baseline identity differs')
    return train,tf

def bind_evaluation(template,terminal,status,boundary):
    final=copy.deepcopy(template);n=final['completed_outers']
    require(status['completed_outers']==n and status['phase']==('TRAIN_COMPLETE' if n==100 else 'TRAIN_BOUNDARY'),'Actual segment boundary differs')
    final['training'].update(terminal=terminal,boundary_checkpoint=status['latest_checkpoint'],boundary=boundary)
    final['gates']['memory_observer']=fileproof(final['gates']['memory_observer']['path'])
    final['state']='BOUND: actual training foreground0/complete raw boundary; resource admission still required'
    return final

class Supervisor:
    def __init__(self,value,ref):
        self.value=value;self.ref=ref;self.train,self.tf=validate_manifest(value)
        self.output=Path(value['output_root']);self.owner=None;self.last_eval=[];self.mirror_jobs=[]
        self.eval0,_=checked_json(value['evaluations'][0]['template']);self.admit=gates.native_admission(self.eval0)
        self.train['_config_sha256']=self.tf['sha256']
    def status(self,phase,**extra):
        return atomic_json(self.output/'status.json',dict(schema='opsd-pr100-periodic-status-v1',run_id=self.value['run_id'],owner=self.owner,
            input_config=self.ref,training_run_id=self.train['run_id'],phase=phase,updated_unix=time.time(),**extra))
    def role(self,identity,source,manifest,publication,input_config,name):
        return dict(role=name,identity=identity,argv_sha256=object_sha(identity['argv']),source=source,helper_source_manifest=manifest,
            publication=publication,input_config=input_config,no_gpu_or_training=True)
    def observer(self):
        train=self.train;path=Path(train['gates']['scoped_cpu_processes_path'])
        driver=self.role(self.owner,fileproof(Path(self.value['helper_root'])/'supervisor.py'),self.value['helper_source_manifest'],
            self.value['publication'],self.ref,'study_driver')
        value=dict(schema='opsd-ablation-scoped-cpu-processes-v1',accepted=True,run_id=train['run_id'],config_sha256=self.tf['sha256'],roles=[driver],updated_unix=time.time())
        require(not path.exists(),'Fresh PR pipeline cannot replace an old CPU registry')
        atomic_json(path,value,exclusive=True)
        # The original driver is already terminal and the normal resource/oracle
        # transition still rejects any other live GPU process here.
        self.admit.require_launch_without_memory(train,train['run_id'],self.owner,known_children=[])
        scope=self.output/'observer'; binding=Path(train['gates']['memory_binding_path'])
        require(not binding.exists(),'Fresh PR observer binding must not alias an old experiment')
        command=[str(Path(train['paths']['helper_root'])/'memory_observer.py'),'--config',self.tf['path'],
            '--output',str(Path(train['paths']['receipts'])/'memory-live.json'),'--binding-output',str(binding)]
        job=self.spawn(command,'pr100-observer-'+self.value['run_id'],scope,self.admit.process)
        while True:
            if job['process'].poll() is not None:
                rc=job['process'].wait();durable(scope/'actual-exit.json',dict(identity=job['identity'],actual_wait_returncode=rc));raise Hold('Observer exited before readiness')
            try:
                observed=self.admit.memory_observer_admission(train);require(gates.same(observed['observer_identity'],job['identity']),'Observer identity differs');break
            except Exception:time.sleep(2)
        live=self.admit.process(job['identity']['pid'])
        observer=self.role(live,fileproof(Path(train['paths']['helper_root'])/'memory_observer.py'),train['helper_source_manifest'],train['gates']['publication'],self.tf,'memory_observer')
        value.update(roles=[driver,observer],updated_unix=time.time());atomic_json(path,value)
        self.admit.scoped_cpu_processes(train);durable(scope/'accepted.json',dict(accepted=True,binding=fileproof(binding),observer=observed))
        # Independent CPU observer deliberately continues; it has no GPU context.
        return job
    def spawn(self,command,run,scope,reader):
        scope.mkdir(parents=True,exist_ok=False);log=(scope/'foreground.log').open('x')
        argv=[self.train['paths']['python'],'-B']+command
        env=dict(os.environ,E1_RUN_ID=run,CUDA_VISIBLE_DEVICES='',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1')
        child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        job=dict(process=child,handle=log,identity=dict(pid=child.pid,uid=self.owner['uid'],starttick=None),command=argv,run_id=run)
        try:
            job['identity']=capture_child_identity(job,reader)
            require(job['identity']['uid']==self.owner['uid'] and job['identity']['ppid']==self.owner['pid'] and job['identity']['run_id']==run,'Child ownership differs')
        except Exception as error:job['identity_error']=str(error)
        hold_evidence(lambda:durable(scope/'child-start.json',dict(identity=job['identity'],command=argv,identity_error=job.get('identity_error'))))
        return job
    def wait(self,job,scope,observe,release):
        child=job['process']
        while child.poll() is None:
            try:observe()
            except Exception as error:durable(scope/('live-uncertain-'+str(time.time_ns())+'.json'),dict(reason=str(error),not_a_returncode=True))
            time.sleep(5)
        rc=child.wait();job['handle'].close()
        actual=dict(identity=job['identity'],command=job['command'],actual_wait_returncode=rc,run_id=job['run_id'],
            log=fileproof(job['handle'].name),signal_termination_not_exit0=rc<0,no_signals=True)
        er=hold_evidence(lambda:durable(scope/'actual-child-exit.json',actual))
        while True:
            try:
                released=release();require(released['accepted'] is True,'Child resource release missing');rr=durable(scope/'actual-child-release.json',released);break
            except Exception as error:durable(scope/('release-uncertain-'+str(time.time_ns())+'.json'),dict(reason=str(error)));time.sleep(5)
        require(rc==0 and type(job['identity'].get('starttick')) is int and not job.get('identity_error'),'Actual child failed/unknown; no next stage')
        return actual,er,rr
    def training(self,n):
        scope=self.output/f'outer-{n:03d}'/'training';command=[str(Path(self.train['paths']['helper_root'])/'controller.py'),
            '--config',self.tf['path'],'--config-sha256',self.tf['sha256'],'--run-id',self.train['run_id'],'--until-outer',str(n)]
        if n>10:command.append('--resume')
        self.admit.require_launch(self.train,'cpu-merge',self.train['run_id'],self.owner,known_children=[])
        self.status('TRAINING',until_outer=n)
        job=self.spawn(command,self.train['run_id'],scope,self.admit.process);known=[job['identity']]
        def release():
            owned=self.admit.require_external_study_release(self.train,self.train['run_id'],self.owner,known_children=known)
            again=self.admit.require_launch(self.train,'cpu-merge',self.train['run_id'],self.owner,known_children=[])
            return dict(accepted=True,owned=owned,after=again)
        actual,er,rr=self.wait(job,scope,lambda:self.admit.require_launch(self.train,'sampling-running',self.train['run_id'],self.owner,known_children=known),release)
        status,sp=checked_json(fileproof(Path(self.train['paths']['output_root'])/'status.json'))
        require(status['owner']==job['identity'] and status['completed_outers']==n and status['phase']==('TRAIN_COMPLETE' if n==100 else 'TRAIN_BOUNDARY'),'Actual training status differs')
        # Copy the mutable current status as a new immutable boundary snapshot.
        frozen=durable(scope/'training-status.json',status)
        terminal=durable(scope/'training-terminal.json',dict(schema='opsd-pr100-training-foreground-terminal-v1',accepted=True,
            run_id=self.train['run_id'],config_sha256=self.tf['sha256'],completed_outers=n,actual_exit_code=actual['actual_wait_returncode'],
            controller=job['identity'],actual_child_exit=er,actual_release=rr,status=frozen,source_status=sp,no_signals=True))
        boundary=fileproof(Path(self.train['paths']['output_root'])/'boundaries'/f'outer-{n:05d}.json')
        return terminal,status,boundary
    def mirror(self,n,checkpoint):
        # CPU copy proceeds concurrently and never blocks the next GPU phase.
        scope=self.output/f'outer-{n:03d}'/'checkpoint-mirror'
        request=durable(scope.parent/'mirror-request.json',dict(schema='opsd-pr100-boundary-mirror-request-v1',
            completed_outers=n,training_run_id=self.train['run_id'],training_config_sha256=self.tf['sha256'],checkpoint=checkpoint,
            destination=str(Path(self.value['mirror_root'])/f'outer-{n:03d}')))
        command=[str(Path(self.value['helper_root'])/'mirror.py'),'--request',request['path'],'--request-sha256',request['sha256']]
        job=self.spawn(command,'pr100-mirror-'+str(n)+'-'+self.value['run_id'],scope,self.admit.process)
        job.update(scope=scope,request=request,completed_outers=n);self.mirror_jobs.append(job)
    def finish_mirrors(self):
        results=[]
        for job in self.mirror_jobs:
            rc=job['process'].wait();job['handle'].close()
            actual=durable(job['scope']/'actual-exit.json',dict(identity=job['identity'],command=job['command'],actual_wait_returncode=rc,
                source=fileproof(Path(self.value['helper_root'])/'mirror.py'),no_gpu=True))
            require(rc==0 and not job.get('identity_error'),'Boundary mirror failed; preserve remote source and report')
            proof=fileproof(Path(self.value['mirror_root'])/f"outer-{job['completed_outers']:03d}-complete.json")
            value,_=checked_json(proof);require(value['accepted'] is True and value['request']==job['request'],'Mirror receipt differs')
            results.append(dict(complete=proof,actual_exit=actual))
        return results
    def evaluation(self,item,terminal,status,boundary):
        template,_=checked_json(item['template']);n=item['completed_outers'];final=bind_evaluation(template,terminal,status,boundary)
        scope=self.output/f'outer-{n:03d}'/'evaluation';finalref=durable(scope.parent/'evaluation-config.json',final)
        completion.validate(final)
        command=[str(Path(final['paths']['helper_root'])/'controller.py'),'--config',finalref['path'],'--config-sha256',finalref['sha256']]
        self.status('EVALUATING',completed_outers=n,checkpoint=status['latest_checkpoint'])
        job=self.spawn(command,final['run_id'],scope,self.admit.process);known=[job['identity']]
        def release():
            owned=gates.owned_release(final,self.owner,known);current=gates.require_launch(final,self.owner,[],'released')
            return dict(accepted=True,owned=owned,current=current)
        actual,er,rr=self.wait(job,scope,lambda:gates.require_launch(final,self.owner,known,'running'),release)
        complete,cp=checked_json(fileproof(Path(final['paths']['output_root'])/'controller-complete.json'));result,smp=checked_json(complete['summary'])
        require(complete['accepted'] is True and complete['config']==finalref and result['primary_slots']==512 and result['accepted'] is True,'Native512 completion differs')
        baseline,_=checked_json(self.value['baseline_summary']);br,_=rows(baseline['private_primary_records']);current,_=rows(result['private_primary_records'])
        base=sorted([x for x in br if x['version']==self.value['baseline_version']],key=lambda x:x['validation_index'])
        now=sorted(current,key=lambda x:x['validation_index'])
        require(result['identities']==baseline['identities'],'Native512 baseline/current protocol identity differs')
        paired=summary.paired(base,now)
        entry=dict(completed_outers=n,training_run_id=self.train['run_id'],checkpoint=status['latest_checkpoint'],
            actual_optimizer_updates=status['actual_optimizer_updates'],evaluation_run_id=final['run_id'],summary=smp,
            models=result['models'],baseline_summary=self.value['baseline_summary'],paired_against_source100=paired,
            actual_exit=er,actual_release=rr,accepted=True)
        ref=durable(scope.parent/'boundary-result.json',entry);self.last_eval.append(dict(reference=ref,**entry))
        durable(self.output/f'curve-through-{n:03d}.json',dict(accepted=True,run_id=self.value['run_id'],rows=self.last_eval,
            primary_slots=n//10*512,scope='Each10 outer raw checkpoint; fixed native512. No per-Adam EM interpolation.'))
        if n==10:durable(self.output/'first10-training-and512-accepted.json',dict(accepted=True,training_terminal=terminal,boundary=boundary,
            evaluation_summary=smp,evaluation_actual_exit=er,evaluation_release=rr,run_id=self.value['run_id'],continues_to100=True))
        return entry
    def run(self):
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules and os.environ.get('E1_RUN_ID')==self.value['run_id'],'Supervisor must be CPU-only exact run')
        self.owner=self.admit.process(os.getpid());require(self.owner['uid']==self.train['owner']['uid'],'Wrong supervisor UID')
        # Pipeline publication has the same file scope, but its own run binding.
        pubcfg=dict(self.eval0,run_id=self.value['run_id'],source_commit=self.value['source_commit'],helper_source_manifest=self.value['helper_source_manifest'])
        pubcfg['gates']=dict(self.eval0['gates'],publication=self.value['publication']);gates.publication(pubcfg)
        lock=Path(self.value['pipeline_lock']);lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a+') as handle:
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            require(not self.output.exists(),'Existing pipeline requires explicit audited continuation, never fresh duplicate')
            self.output.mkdir(parents=True);durable(self.output/'supervisor-start.json',dict(owner=self.owner,input_config=self.ref,no_gpu=True))
            self.status('OBSERVER_STARTUP');self.observer()
            for item in self.value['evaluations']:
                terminal,status,boundary=self.training(item['completed_outers']);self.mirror(item['completed_outers'],status['latest_checkpoint']);self.evaluation(item,terminal,status,boundary)
            mirrors=self.finish_mirrors()
            self.status('COMPLETE',completed_outers=100,primary_slots=5120)
            return durable(self.output/'pipeline-complete.json',dict(accepted=True,run_id=self.value['run_id'],input_config=self.ref,
                completed_outers=100,evaluation_count=10,primary_slots=5120,results=self.last_eval,boundary_mirrors=mirrors,no_training_restarts_from_zero=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--pipeline-config',required=True);p.add_argument('--pipeline-config-sha256',required=True);a=p.parse_args()
    value,ref=checked_json(dict(path=a.pipeline_config,sha256=a.pipeline_config_sha256))
    try:result=Supervisor(value,ref).run()
    except Exception as error:
        print(json.dumps(dict(accepted=False,decision='HOLD_OR_FAILURE_PRESERVED',reason=str(error),error_type=type(error).__name__,no_signals=True)),flush=True);return 2
    print(json.dumps(dict(accepted=True,pipeline_complete=result)),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
