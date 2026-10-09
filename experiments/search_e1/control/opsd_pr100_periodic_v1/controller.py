"""Independent result run, eight locks, no partial reuse or process signalling."""
import argparse
import os
from pathlib import Path
import subprocess
import time
try:
    from .utils import *
    from . import completion,prepare,gates,summary
except ImportError:
    from utils import *
    import completion,prepare,gates,summary

def hold_evidence(operation,sleeper=time.sleep):
    # Called only while owning leases. Evidence failures must never free leases
    # while a GPU child is alive. No next stage until evidence is durable.
    while True:
        try:return operation()
        except Exception:sleeper(5)

def durable(path,value):
    if Path(path).exists():
        existing,fp=checked_json(fileproof(path));require(existing==value,'Immutable result evidence differs');return fp
    return atomic_json(path,value,exclusive=True)

def wait_children(jobs,check_running,release,write,sleeper=time.sleep,expected_count=None):
    """Transient admission is uncertainty, actual nonzero is hard failure."""
    finished={};uncertain=[]
    while len(finished)<len(jobs):
        for job in jobs:
            child=job['process'];key=child.pid
            if key in finished:continue
            rc=child.poll()
            if rc is not None:
                row=dict(identity=job['identity'],actual_wait_returncode=rc)
                hold_evidence(lambda row=row:write('child-'+str(key),row),sleeper);finished[key]=row
        try:check_running()
        except Exception as error:
            entry=dict(error_type=type(error).__name__,reason=str(error),checked_unix=time.time())
            hold_evidence(lambda:write('admission-'+str(time.time_ns()),entry),sleeper);uncertain.append(entry)
        if len(finished)<len(jobs):sleeper(5)
    terminal=dict(child_exit_evidence=[finished[j['process'].pid] for j in jobs],all_actual_wait_zero=all(r['actual_wait_returncode']==0 for r in finished.values()))
    hold_evidence(lambda:write('actual-terminal',terminal),sleeper)
    while True:
        try:
            released=release();require(released.get('accepted') is True,'Terminal current release not accepted')
            hold_evidence(lambda:write('release',released),sleeper);break
        except Exception as error:
            hold_evidence(lambda:write('release-uncertain-'+str(time.time_ns()),dict(error_type=type(error).__name__,reason=str(error))),sleeper);sleeper(5)
    identity_complete=all(type(r['identity'].get('uid')) is int and type(r['identity'].get('starttick')) is int
                         and not r['identity'].get('capture_status') for r in finished.values())
    result=dict(accepted=terminal['all_actual_wait_zero'] and identity_complete and (expected_count is None or len(jobs)==expected_count),
        child_exit_evidence=terminal['child_exit_evidence'],release=released,expected_worker_count=expected_count,actual_worker_count=len(jobs),
        admission_uncertain_count=len(uncertain),actual_exit_codes_never_inferred=True,no_signals=True)
    hold_evidence(lambda:write('phase-complete',result),sleeper)
    require(result['accepted'],'Actual evaluation child nonzero; artifacts preserved after current release')
    return result

def capture_child_identity(job,reader,sleeper=time.sleep):
    """Never wait for an exited PID to reappear; preserve UNKNOWN with actual rc."""
    while True:
        try:return reader(job['process'].pid)
        except Exception as error:
            rc=job['process'].poll()
            if rc is not None:
                job['identity']['capture_status']='UNKNOWN_EXITED_BEFORE_FULL_IDENTITY'
                job['identity']['capture_error_type']=type(error).__name__
                job['identity']['actual_wait_returncode']=rc
                raise Hold('Spawned child exited before identity capture; preserve actual rc and reject this result attempt')
            sleeper(.05)

class Controller:
    def __init__(self,config,ref):
        self.config=config;self.ref=ref;self.owner=gates.process(config,os.getpid());self.known=[]
        config['_config_path']=ref['path']
        self.output=Path(config['paths']['output_root']);self.base=Path(config['paths']['base'])
    def gate(self,phase):return gates.require_launch(self.config,self.owner,self.known,phase)
    def released(self):
        gpu=self.gate('released');owned=gates.owned_release(self.config,self.owner,self.known)
        again=self.gate('released');return dict(accepted=True,gpu=gpu,all_owned_release=owned,after=again)
    def phase(self,version):
        directory=self.output/version;directory.mkdir(exist_ok=False);self.gate('start');jobs=[]
        model_ref=None
        ticket=dict(schema='opsd-pr100-native512-launch-ticket-v1',accepted=True,run_id=self.config['run_id'],config_sha256=self.ref['sha256'],
            controller=self.owner,version=version,output=str(directory),worker_count=8,logical_rank_map=list(range(8)),model=model_ref)
        ticket_ref=durable(directory/'launch-ticket.json',ticket)
        def write(name,value):return durable(directory/(name+'.json'),value)
        failure=None
        try:
            for rank in range(8):
                self.gate('running')
                command=[self.config['paths']['python'],'-B',str(Path(self.config['paths']['helper_root'])/'worker.py'),
                    '--config',self.ref['path'],'--config-sha256',self.ref['sha256'],'--ticket',ticket_ref['path'],
                    '--ticket-sha256',ticket_ref['sha256'],'--version',version,'--rank',str(rank),'--output',str(directory)]
                env=dict(os.environ,E1_RUN_ID=self.config['run_id'],CUDA_VISIBLE_DEVICES=str(rank),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1')
                train,_=original_config(self.config);env['PYTHONPATH']=str(Path(train['paths']['runtime'])/'upstream/Search-R1')+os.pathsep+train['paths']['runtime']
                handle=(directory/f'worker-{rank}.log').open('x');child=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
                # Register the actual Popen immediately, before any identity/hash can fail.
                job=dict(process=child,handle=handle,identity=dict(pid=child.pid,uid=self.owner['uid'],starttick=None),command=command);jobs.append(job)
                identity=capture_child_identity(job,lambda pid:gates.process(self.config,pid))
                job['identity']=identity;self.known.append(identity)
                hold_evidence(lambda:write('child-start-'+str(rank),dict(identity=identity,command=command,rank=rank,version=version)))
                require(identity['run_id']==self.config['run_id'] and identity['uid']==self.owner['uid'] and identity['ppid']==self.owner['pid'],'Spawn ownership differs')
        except Exception as error:failure=error
        # Even a later spawn/identity/gate failure cannot orphan earlier children.
        result=wait_children(jobs,lambda:self.gate('running'),self.released,
            lambda name,value:write(name,dict(value,version=version,run_id=self.config['run_id'])),expected_count=8)
        for job in jobs:job['handle'].close()
        if failure:raise failure
        require(len(jobs)==8,'Not all eight workers launched')
        return result

    def run(self):
        prepare.validate_config(self.config);require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules,'Controller remains CPU-only')
        require(os.environ.get('E1_RUN_ID')==self.config['run_id'],'Explicit independent evaluator run required')
        gates.publication(self.config)
        # No model process can spawn before this group has the requested completed boundary
        # of outers and a successful foreground terminal; no other group is awaited.
        current=completion.validate(self.config)
        with gates.Leases(self.config):
            fresh=completion.validate(self.config)
            require(current['totals']==fresh['totals'] and current['final_checkpoint']==fresh['final_checkpoint'],'Group final state changed after acquiring leases')
            gates.training_release(self.config,self.owner);self.gate('start')
            require(not self.output.exists(),'Existing partial/completed result requires separate run/output; no silent slot reuse')
            source_files(self.config);runtime_files(self.config);cohort(self.config)
            for version in self.config['scientific']['versions']:prepare.adapter(self.config,version)
            self.output.mkdir(parents=True,exist_ok=False);durable(self.output/'fresh-boundary-completion.json',fresh)
            durable(self.output/'controller-start.json',dict(owner=self.owner,config=self.ref,run_id=self.config['run_id'],leases=gates.locks(self.config)))
            for version in self.config['scientific']['versions']:self.phase(version)
            result=summary.collect(self.config);result['training_completion']=fileproof(self.output/'fresh-boundary-completion.json');result['final_release']=self.released()
            durable(self.output/'summary.json',result);durable(self.output/'controller-complete.json',dict(accepted=True,run_id=self.config['run_id'],config=self.ref,
                primary_slots=result['primary_slots'],summary=fileproof(self.output/'summary.json'),release=result['final_release']))
            return result

def main():
    p=argparse.ArgumentParser()
    for n in ('config','config-sha256'):p.add_argument('--'+n,required=True)
    a=p.parse_args();cfg,fp=checked_json(dict(path=a.config,sha256=a.config_sha256))
    try:result=Controller(cfg,fp).run()
    except Exception as error:
        print(json.dumps(dict(accepted=False,decision='HOLD_OR_REJECTED_RESULT',reason=str(error),error_type=type(error).__name__,training_not_stopped=True,no_signals=True)),flush=True);return 2
    print(json.dumps(dict(accepted=True,primary_slots=result['primary_slots'],models=result['models'],paired=result['paired'])));return 0

if __name__=='__main__':raise SystemExit(main())
