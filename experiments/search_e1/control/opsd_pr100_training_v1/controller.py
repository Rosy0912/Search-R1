"""Exclusive online OPSD orchestration. No signals, GRPO or optimizer reset."""
import argparse
import fcntl
import importlib.util
import uuid
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

try:
    from .checkpoint import (require,checked_json,checked_file,fileproof,object_sha,exclusive_json,atomic_json,
                             check_checkpoint,check_outer_complete,science_contract)
    from .prepare import validate_config,verify_runtime,verify_common
    from .scoring import verify_scoring
    from .sampling import cohort,write_rows,read_rows,verify_rollouts,bind_model_files,verify_model_metadata,verify_http_trace
except ImportError:
    from checkpoint import (require,checked_json,checked_file,fileproof,object_sha,exclusive_json,atomic_json,
                            check_checkpoint,check_outer_complete,science_contract)
    from prepare import validate_config,verify_runtime,verify_common
    from scoring import verify_scoring
    from sampling import cohort,write_rows,read_rows,verify_rollouts,bind_model_files,verify_model_metadata,verify_http_trace


def process_identity(pid):
    try:
        from .gates import process
    except ImportError:from gates import process
    try:return process(pid)
    except FileNotFoundError:return None


def load_admission(config):
    try:from . import gates
    except ImportError:import gates
    manifest,_=checked_json(config['helper_source_manifest'])
    expected=manifest['files']['gates.py'];expected=expected['sha256'] if isinstance(expected,dict) else expected
    checked_file(dict(path=str(Path(config['paths']['helper_root'])/'gates.py'),sha256=expected))
    require(Path(gates.__file__).resolve()==Path(config['paths']['helper_root']).resolve()/'gates.py','Admission module shadowed')
    gates.require_launch.require_owned_release=gates.require_owned_release
    return gates.require_launch


def atomic_evidence(path,value):
    """Unique temporary paths make an evidence I/O retry safe after a partial write."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending-'+uuid.uuid4().hex)
    with temporary.open('x') as handle:
        json.dump(value,handle,sort_keys=True,ensure_ascii=False,indent=2);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)
    fd=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return fileproof(path)


def hold_evidence(callback,pause=time.sleep):
    """A failed evidence write never releases the held lease around a live child."""
    while True:
        try:return callback()
        except Exception:pause(5)


def append_history(path,value):
    payload=(json.dumps(value,sort_keys=True,ensure_ascii=False)+'\n').encode()
    path=Path(path)
    with path.open('a+b') as handle:
        handle.seek(0,2);start=handle.tell()
        try:handle.write(payload);handle.flush();os.fsync(handle.fileno())
        except Exception:
            handle.seek(start);handle.truncate();handle.flush();os.fsync(handle.fileno());raise


def latest_resume_checkpoint(output_root,config,run_id,outer_index):
    """Only accepted complete actual state; incomplete pending checkpoints never count."""
    candidates=[]
    for path in (Path(output_root)/'checkpoints').glob('*/manifest.json'):
        # A crash after manifest fsync but before the atomic directory rename
        # leaves a pending manifest with final-path references. It is not committed.
        if re.fullmatch(r'update-\d{8}-outer-\d{5}',path.parent.name) is None:
            continue
        if (path.parent/'payload-prune-intent.json').exists() or (path.parent/'payload-pruned.json').exists():continue
        value=json.loads(path.read_text())
        if value.get('accepted') is True and value.get('run_id')==run_id and value.get('cursor',{}).get('outer_index')==outer_index:
            ref=fileproof(path);checked,_=check_checkpoint(ref,config,run_id,expected_outer=outer_index)
            candidates.append((checked['optimizer_updates'],checked['cursor']['next_pair_cursor'],ref))
    if not candidates:return None
    candidates.sort(key=lambda x:(x[0],x[1]));return candidates[-1][2]


class Controller:
    def __init__(self,config_ref,run_id,resume=False,admit=None,until_outer=None):
        self.config,self.config_ref=checked_json(config_ref);validate_config(self.config,launch=True)
        require(self.config['run_id']==run_id,'Config/CLI run identity mismatch')
        self.config['_config_sha256']=self.config_ref['sha256'];self.config['_run_id']=run_id;self.run_id=run_id;self.resume=resume
        self.owner=process_identity(os.getpid())
        require(self.owner and self.owner['uid']==self.config['owner']['uid'] and self.owner['run_id']==run_id,'Controller needs actual bound UID/run environment')
        self.admit=admit or load_admission(self.config);self.known=[];self.active_jobs=[];self.phase='CREATED'
        self.output=Path(self.config['paths']['output_root']);self.lock=None;self.shared_locks=[];self.owns_output=False
        self.until_outer=validate_until_outer(until_outer,self.config['scientific']['outer_iterations'])

    def gate(self,phase,children=None):
        result=self.admit(self.config,phase,self.run_id,self.owner,known_children=self.known if children is None else children)
        require(result.get('accepted') is True,'HOLD: strict launch gate rejected')
        return result

    def stage(self,phase,**fields):
        self.phase=phase
        value=dict(schema='opsd-ablation-controller-status-v1',run_id=self.run_id,variant=self.config['variant'],
            contract=science_contract(self.config),owner=self.owner,phase=phase,updated_at_unix=time.time(),
            config_sha256=self.config_ref['sha256'],known_children=self.known,**fields)
        return hold_evidence(lambda:atomic_evidence(self.output/'status.json',value))

    def await_gate(self,phase,children=None):
        while True:
            try:return self.gate(phase,children)
            except Exception as exc:
                self.stage('HOLD_ADMISSION_LEASE_HELD',child_phase=phase,error=str(exc),no_signal=True)
                time.sleep(5)

    def spawn(self,command,log,phase,extra_env=None):
        self.await_gate(phase)
        runtime=Path(self.config['paths']['runtime'])
        env=dict(os.environ,PYTHONPATH=str(runtime/'upstream'/'Search-R1')+os.pathsep+str(runtime),PYTHONUNBUFFERED='1',
                 PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',E1_RUN_ID=self.run_id,E1_ONLINE_PHASE=phase,
                 CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_ATTENTION_BACKEND='XFORMERS',
                 E1_INFERENCE_UTIL=str(self.config['sampling']['gpu_memory_utilization']))
        env.update(extra_env or {})
        handle=Path(log).open('x')
        process=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        identity=dict(pid=process.pid,uid=self.owner['uid'],starttick=None,run_id=self.run_id,identity_uncertain=True)
        job=(process,handle,identity);self.active_jobs.append(job);self.known.append(identity)
        # Registered Popen first. A fast exited child must not wait for /proc to reappear.
        while True:
            try:
                actual=process_identity(process.pid)
                require(actual and actual['uid']==self.owner['uid'] and actual['run_id']==self.run_id,'Spawn actual UID/run mismatch')
                identity.clear();identity.update(actual);break
            except Exception as exc:
                rc=process.poll()
                if rc is not None:
                    identity['identity_error']=str(exc);identity['actual_polled_returncode']=rc;break
                self.stage('HOLD_CHILD_IDENTITY_LEASE_HELD',child_pid=process.pid,error=str(exc));time.sleep(1)
        hold_evidence(lambda:append_history(self.output/'child-history.jsonl',dict(time=time.time(),phase=phase,
            identity=identity,command=command,log=str(log))))
        return job

    def wait(self,jobs,phase):
        while True:
            codes=[p.poll() for p,_,_ in jobs]
            if all(c is not None for c in codes):break
            try:self.gate(phase)
            except Exception as exc:
                # An admission observation is not a child return code or a sticky worker failure.
                self.stage('HOLD_RUNNING_ADMISSION_LEASE_HELD',child_phase=phase,error=str(exc),observed_returncodes=codes)
            time.sleep(5)
        evidence=[]
        for process,handle,identity in jobs:
            rc=process.wait()
            if not handle.closed:hold_evidence(handle.close)
            evidence.append(dict(identity=dict(identity),actual_wait_returncode=rc,log=hold_evidence(lambda h=handle:fileproof(h.name))))
        value=dict(schema='opsd-ablation-actual-phase-terminal-v1',run_id=self.run_id,variant=self.config['variant'],
            config_sha256=self.config_ref['sha256'],phase=phase,child_exit_evidence=evidence,checked_unix=time.time())
        path=self.output/'phase-exits'/(uuid.uuid4().hex+'.json')
        terminal_ref=hold_evidence(lambda:atomic_evidence(path,value))  # Persist actual codes BEFORE any gate/release failure.
        release=self.wait_release([i for _,_,i in jobs])
        self.await_gate('cpu-merge',[])
        accepted=all(e['actual_wait_returncode']==0 and e['identity'].get('starttick') is not None
                     and not e['identity'].get('identity_uncertain') for e in evidence)
        require(accepted,'Actual nonzero/unknown child exit preserved; no implicit successful reuse')
        return dict(accepted=True,child_exit_evidence=evidence,gpu_release=release,actual_terminal=terminal_ref)

    def wait_release(self,children):
        audit=getattr(self.admit,'require_owned_release',None)
        require(callable(audit),'Full ownership scanner missing')
        complete=[x for x in self.known if x.get('starttick') is not None]
        while True:
            try:
                for expected in children:
                    if expected.get('starttick') is not None:
                        actual=process_identity(expected['pid'])
                        require(actual is None or actual['starttick']!=expected['starttick'],'Owned child still alive after wait')
                current=self.gate('cpu-merge',[])
                proof=audit(self.config,self.run_id,self.owner,known_children=complete)
                require(proof.get('accepted') is True,'Owned/opaque descendant remains')
                return dict(resource_admission=current,all_owned_release=proof)
            except Exception as exc:
                self.stage('HOLD_RELEASE_LEASE_HELD',error=str(exc),signals_sent=False)
                time.sleep(5)

    def contain_failure(self,exc):
        self.stage('FAILED_LEASE_HELD',error_type=type(exc).__name__,error=str(exc),signals_sent=False)
        if self.active_jobs:
            while any(p.poll() is None for p,_,_ in self.active_jobs):time.sleep(5)
            evidence=[]
            for process,handle,identity in self.active_jobs:
                rc=process.wait()
                if not handle.closed:hold_evidence(handle.close)
                evidence.append(dict(identity=identity,actual_wait_returncode=rc))
            hold_evidence(lambda:atomic_evidence(self.output/'failure-actual-exits.json',dict(run_id=self.run_id,children=evidence)))
            self.wait_release([i for _,_,i in self.active_jobs])
        self.stage('FAILED_OWNED_RELEASE_CONFIRMED',error_type=type(exc).__name__,error=str(exc),signals_sent=False)

    def cpu_merge(self,outer,index,previous,updates):
        require('torch' not in sys.modules,'CPU-only controller imported Torch before merge boundary')
        canonical=outer/'merged-model.json';request_path=outer/'cpu-merge-request.json';receipt_path=outer/'cpu-merge-child.json'
        parent_path=outer/'cpu-merge-parent.json'
        if canonical.exists():
            require(parent_path.exists(),'Merged artifact lacks actual CPU-child/release evidence; preserved HOLD')
            parent,_=checked_json(fileproof(parent_path));child,_=checked_json(parent['child']);identity,_=checked_json(fileproof(canonical))
            require(parent.get('accepted') is True and child['result']==identity and parent['actual_terminal']==parent['exit_proof']['actual_terminal'],
                    'CPU merge provenance differs')
            require(parent['run_id']==self.run_id and parent['config_sha256']==self.config_ref['sha256']
                    and parent['parent_cpu']['torch_imported'] is False,'CPU merge parent identity/device boundary differs')
            self.verify_exit(parent['exit_proof'],'cpu-merge',1)
            request,_=checked_json(parent['request'])
            require(request['config']==self.config_ref and request['checkpoint']==previous
                    and request['optimizer_updates']==updates and request['outer_index']==index
                    and Path(request['output']).resolve()==(outer/'merged-model-cache').resolve(),
                    'Existing CPU merge request differs from this outer-start')
            require(all(x['actual_wait_returncode']==0 for x in parent['exit_proof']['child_exit_evidence']),'CPU merge actual exit not zero')
            bind_model_files(self.config,identity)
            self.verify_merge_identity(identity,index,previous,updates);return identity
        require(not request_path.exists() and not receipt_path.exists() and not (outer/'merged-model-cache').exists(),
                'Partial CPU merge preserved; independent recovery must review before reuse')
        self.await_gate('cpu-merge',[])
        require(shutil.disk_usage(self.output).free>=self.config['resources']['minimum_disk_bytes'],'CPU merge disk gate failed')
        request=dict(schema='opsd-ablation-cpu-merge-request-v1',run_id=self.run_id,config=self.config_ref,
            checkpoint=previous,outer_index=index,optimizer_updates=updates,output=str(outer/'merged-model-cache'))
        request_ref=exclusive_json(request_path,request)
        command=[self.config['paths']['python'],str(Path(self.config['paths']['helper_root'])/'sampling.py'),
            '--merge-request',request_ref['path'],'--merge-request-sha256',request_ref['sha256'],'--merge-receipt',str(receipt_path)]
        job=self.spawn(command,outer/'cpu-merge.log','cpu-merge',dict(CUDA_VISIBLE_DEVICES=''))
        exit_proof=self.wait([job],'cpu-merge');child,child_ref=checked_json(fileproof(receipt_path));identity=child['result']
        require(child.get('accepted') is True and child['request']==request_ref and child['run_id']==self.run_id,
                'CPU merge result is not bound to this request')
        self.verify_merge_identity(identity,index,previous,updates)
        canonical_ref=exclusive_json(canonical,identity)
        exclusive_json(parent_path,dict(schema='opsd-ablation-cpu-merge-parent-v1',accepted=True,run_id=self.run_id,
            config_sha256=self.config_ref['sha256'],request=request_ref,child=child_ref,canonical=canonical_ref,
            actual_terminal=exit_proof['actual_terminal'],exit_proof=exit_proof,parent_cpu=dict(torch_imported='torch' in sys.modules)))
        require('torch' not in sys.modules,'CPU controller imported Torch during isolated merge')
        return identity

    def verify_merge_identity(self,identity,index,previous,updates):
        require(identity['checkpoint']==previous and identity['optimizer_updates']==updates
                and identity['source_model_identity']==self.config['source_model_manifest']['sha256']
                and identity['common_initial_adapter']==self.config['common_initial_adapter_manifest']
                and identity['variant']==self.config['variant'] and identity['precision']==self.config['precision'],
                'Merged sampling identity differs from current outer-start raw adapter')
        expected=(checked_json(previous)[0]['all_rank_adapter_sha256'][0] if previous else self.config['common_initial_adapter_tensor_sha256'])
        require(identity['adapter_tensor_sha256']==expected,'Merged raw adapter digest mismatch')
        proof=identity['device_proof']
        require(all(proof[k]==dict(cuda_initialized=False,visible_cuda_devices=0,cuda_visible_devices='') for k in ('before','after')),
                'CPU merge CUDA-free proof missing')
        require(proof['peft_parameters']['all_cpu'] is True and proof['merged_parameters']['all_cpu'] is True
                and proof['merge_calls']==1 and proof['cast_after_merge_count']==(1 if self.config['variant'] == 'PR' else 0),
                'CPU merge actual parameter device/method differs')
        bind_model_files(self.config,identity)

    def sample(self,outer,questions,identity):
        attempts=list(outer.glob('sampling-attempt-*'));attempt=outer/f'sampling-attempt-{len(attempts):05d}';attempt.mkdir()
        identity=bind_model_files(self.config,identity)
        model_ref=exclusive_json(attempt/'sampling-model.json',identity)
        jobs=[]
        self.gate('sampling-start',[])
        for rank in range(8):
            shard=attempt/f'questions-{rank}.jsonl';write_rows(shard,questions[rank::8])
            output=attempt/f'rollouts-{rank}.jsonl';trace=attempt/f'generation-trace-{rank}.jsonl'
            http_trace=attempt/f'http-requests-{rank}.jsonl'
            command=[self.config['paths']['python'],str(Path(self.config['paths']['helper_root'])/'sampling.py'),
                '--config',self.config_ref['path'],'--config-sha256',self.config_ref['sha256'],'--questions',str(shard),
                '--model-identity',model_ref['path'],'--model-identity-sha256',model_ref['sha256'],
                '--rank',str(rank),'--output',str(output),'--trace',str(trace),'--http-trace',str(http_trace)]
            jobs.append(self.spawn(command,attempt/f'worker-{rank}.log','sampling-running',
                dict(CUDA_VISIBLE_DEVICES=str(self.config['owner']['physical_gpus'][rank]))))
        exit_proof=self.wait(jobs,'sampling-running')
        rows=[];worker_proofs=[];http_proofs=[]
        for rank in range(8):
            ref=fileproof(attempt/f'rollouts-{rank}.jsonl.worker-complete.json');value,proof=checked_json(ref)
            require(value['accepted'] is True and value['rank']==rank and value['rows']==320 and value['run_id']==self.run_id,'Wrong/missing worker completion')
            require(value['config_sha256']==self.config_ref['sha256'] and value['variant']==self.config['variant']
                    and value['precision']==self.config['precision'] and value['pid']==jobs[rank][0].pid
                    and value['uid']==self.owner['uid'],'Worker completion group/process differs')
            require(value['actual_llm_load']['actual_dtype']=='torch.'+self.config['sampling']['dtype'],
                    'Worker did not load the authorized actual inference dtype')
            require(value['sampling_model_identity']==identity['sha256'],'Worker sampled different weight identity')
            checked_file(value['rollouts']);checked_file(value['generation_trace'])
            http_proof=verify_http_trace(value['http_trace'],rank,identity['sha256'],value['http_requests'])
            require(Path(http_proof['path']).resolve()==(attempt/f'http-requests-{rank}.jsonl').resolve(),'Worker HTTP trace path mismatch')
            rows.extend(read_rows(value['rollouts']['path']));worker_proofs.append(proof);http_proofs.append(http_proof)
        sys.path.insert(0,self.config['paths']['runtime'])
        scoring_proof=verify_scoring(self.config)
        from search_e1 import core
        pairs,stats=verify_rollouts(rows,questions,core,self.config['variant'],int(outer.name.split('-')[-1]))
        rollout_ref=write_rows(attempt/'rollouts.jsonl',rows);pair_ref=write_rows(attempt/'pairs.jsonl',pairs)
        receipt=dict(schema='opsd-online-sampling-complete-v1',accepted=True,run_id=self.run_id,
            sampling_model_identity=identity,workers=worker_proofs,exit_proof=exit_proof,
            private_http_traces=http_proofs,scoring_identity=scoring_proof,
            rollouts=rollout_ref,pairs=pair_ref,stats=stats)
        return receipt,exclusive_json(outer/'sampling-complete.json',receipt)

    def verify_exit(self,value,phase,count):
        require(value.get('accepted') is True and len(value['child_exit_evidence'])==count,'Missing accepted actual phase exits')
        actual,_=checked_json(value['actual_terminal'])
        require(actual['run_id']==self.run_id and actual['config_sha256']==self.config_ref['sha256']
                and actual['phase']==phase and actual['child_exit_evidence']==value['child_exit_evidence'],
                'Actual exit receipt is outside this phase/config/run')
        require(all(x['actual_wait_returncode']==0 and x['identity'].get('starttick') is not None
                    and x['identity']['uid']==self.owner['uid'] and x['identity']['run_id']==self.run_id
                    for x in value['child_exit_evidence']),'Unknown/nonzero exit cannot be borrowed as success')
        require(value['gpu_release']['all_owned_release']['accepted'] is True,'Phase full release proof missing')

    def teacher_diagnostic(self,outer,index,manifest):
        """Independent source100/BF16 greedy child; never contributes to training inputs."""
        if self.config['diagnostics']['teacher_autoregressive'] is False:
            root=outer/'teacher-diagnostics';root.mkdir(exist_ok=True);path=root/'complete.json'
            value=dict(schema='opsd-pr100-teacher-diagnostic-disabled-v1',accepted=True,run_id=self.run_id,
                config_sha256=self.config_ref['sha256'],variant='PR',outer_index=index,disabled=True,count=0,
                pairs=manifest['pairs'],scope='Latest human reduces debug output; frozen training teacher q is unchanged',
                training_teacher_enabled=True,diagnostic_child_started=False)
            if path.exists():
                saved,ref=checked_json(fileproof(path));require(saved==value,'Disabled diagnostic receipt changed')
            else:ref=exclusive_json(path,value)
            return value
        try:from .teacher_diag import select_cases,VERSION
        except ImportError:from teacher_diag import select_cases,VERSION
        require('torch' not in sys.modules,'Teacher diagnostic supervisor must remain CPU-only')
        pairs_ref=checked_file(manifest['pairs']);selection=select_cases(read_rows(pairs_ref['path']),index)
        root=outer/'teacher-diagnostics';root.mkdir(exist_ok=True)
        selected_path=root/'selection.json';request_path=root/'request.json';child_path=root/'child-complete.json'
        completed_path=root/'complete.json'
        if selected_path.exists():
            saved,selection_ref=checked_json(fileproof(selected_path));require(saved==selection,'Diagnostic resume selection changed')
        else:selection_ref=exclusive_json(selected_path,selection)
        if completed_path.exists():
            value,_=checked_json(fileproof(completed_path))
            require(value.get('accepted') is True and value['schema']=='opsd-ablation-teacher-diagnostic-parent-v1'
                and value['run_id']==self.run_id and value['config_sha256']==self.config_ref['sha256']
                and value['pairs']==pairs_ref and value['selection']==selection_ref and value['count']==selection['count']
                and value['diagnostic_teacher_autoregressive'] is True and value['original_training_objective'] is False
                and value['diagnostic_precision_common_bf16'] is True,'Diagnostic completion group/input differs')
            if selection['not_applicable']:
                require(value['not_applicable'] is True and value['child'] is None and value['exit_proof'] is None,
                        'No-success NA must not invent a teacher child')
            else:
                self.verify_exit(value['exit_proof'],'sampling-running',1)
                request,request_ref=checked_json(value['request'])
                child,_=checked_json(value['child'])
                self.verify_teacher_child(child,request,request_ref,selection,index,value['exit_proof'])
            return value
        if selection['not_applicable']:
            value=dict(schema='opsd-ablation-teacher-diagnostic-parent-v1',accepted=True,run_id=self.run_id,
                config_sha256=self.config_ref['sha256'],variant=self.config['variant'],outer_index=index,
                pairs=pairs_ref,selection=selection_ref,count=0,not_applicable=True,
                not_applicable_reason=selection['not_applicable_reason'],request=None,child=None,exit_proof=None,
                diagnostic_teacher_autoregressive=True,original_training_objective=False,
                diagnostic_precision_common_bf16=True,training_or_validation_denominator_contribution=0)
            exclusive_json(completed_path,value);return value
        require(not request_path.exists() and not child_path.exists() and not (root/'cases').exists(),
                'Partial teacher diagnostic preserved; actual exit/provenance recovery review required')
        self.gate('cpu-merge',[])
        source=bind_model_files(self.config,dict(kind='raw-source100',path=self.config['paths']['model'],
            sha256=self.config['source_model_manifest']['sha256'],source_model_identity=self.config['source_model_manifest']['sha256']))
        request=dict(schema=VERSION,config=self.config_ref,run_id=self.run_id,variant=self.config['variant'],
            outer_index=index,pairs=pairs_ref,selection=selection_ref,source_identity=source,output_root=str(root/'cases'))
        request_ref=exclusive_json(request_path,request)
        command=[self.config['paths']['python'],str(Path(self.config['paths']['helper_root'])/'teacher_diag.py'),
            '--request',request_ref['path'],'--request-sha256',request_ref['sha256'],'--receipt',str(child_path)]
        self.stage('TEACHER_DIAGNOSTIC',outer_index=index,count=selection['count'],diagnostic_precision_common_bf16=True)
        self.gate('sampling-start',[])
        job=self.spawn(command,root/'worker.log','sampling-running',
            dict(CUDA_VISIBLE_DEVICES=str(self.config['owner']['physical_gpus'][0])))
        exit_proof=self.wait([job],'sampling-running')
        child,child_ref=checked_json(fileproof(child_path))
        self.verify_teacher_child(child,request,request_ref,selection,index,exit_proof)
        value=dict(schema='opsd-ablation-teacher-diagnostic-parent-v1',accepted=True,run_id=self.run_id,
            config_sha256=self.config_ref['sha256'],variant=self.config['variant'],outer_index=index,
            pairs=pairs_ref,selection=selection_ref,count=selection['count'],not_applicable=False,
            request=request_ref,child=child_ref,exit_proof=exit_proof,
            diagnostic_teacher_autoregressive=True,original_training_objective=False,
            diagnostic_precision_common_bf16=True,training_or_validation_denominator_contribution=0)
        exclusive_json(completed_path,value);return value

    def verify_teacher_child(self,child,request,request_ref,selection,index,exits):
        require(child['schema']=='opsd-ablation-teacher-diagnostic-child-v1' and child.get('accepted') is True
            and child['request']==request_ref and request['config']==self.config_ref and request['run_id']==self.run_id
            and request['variant']==self.config['variant'] and request['outer_index']==index
            and child['selection']==request['selection'] and child['pairs']==request['pairs']
            and child['run_id']==self.run_id and child['variant']==self.config['variant']
            and child['config_sha256']==self.config_ref['sha256'] and child['count']==selection['count']
            and len(child['cases'])==selection['count'],'Teacher child provenance/count differs')
        self.verify_exit(exits,'sampling-running',1)
        actual=exits['child_exit_evidence'][0]['identity']
        require(child['pid']==actual['pid'] and child['uid']==actual['uid'],'Teacher child actual process differs')
        require(child['diagnostic_teacher_autoregressive'] is True and child['original_training_objective'] is False
            and child['diagnostic_precision_common_bf16'] is True
            and child['training_or_validation_denominator_contribution']==0,'Teacher diagnostic scope changed')
        source=request['source_identity']
        require(child['source_identity']==source and source['kind']=='raw-source100'
            and source['source_model_identity']==self.config['source_model_manifest']['sha256']
            and Path(source['path']).resolve()==Path(self.config['paths']['model']).resolve(),'Teacher source changed')
        verify_model_metadata(source)
        pairs=read_rows(checked_file(request['pairs'])['path']);verify_scoring(self.config)
        from search_e1 import core
        for ref,item in zip(child['cases'],selection['selected']):
            case,_=checked_json(ref);require(case['selected']==item and case['run_id']==self.run_id
                and case['variant']==self.config['variant'] and case['outer_index']==index
                and case['source_model_identity']==self.config['source_model_manifest']['sha256']
                and case['original_pairs']==request['pairs'],'Teacher per-case provenance differs')
            pair=pairs[item['pair_index']];result=case['result']
            require(object_sha(pair)==item['pair_sha256'] and case['original_B']==pair['reference']
                and case['original_D']==pair['student'] and result['question_id']==pair['question_id']
                and result['answers']==pair['answers'] and result['reward']==core.reward(result['response'],pair['answers'],result['policy_spans']),
                'Teacher actual B/D/strict result differs')
            trace=case['trace'];require(not trace['logging_errors'] and trace['return_objects_modified'] is False
                and trace['generation_options_modified'] is False,'Teacher transparent trace incomplete')
            require(case['diagnostic_teacher_autoregressive'] is True and case['original_training_objective'] is False
                and case['diagnostic_precision_common_bf16'] is True,'Teacher case scope differs')
            http=checked_file(case['http']);records=read_rows(http['path'])
            verify_http_trace(http,0,'source100-teacher-diagnostic',len(records))

    def verify_outer_inputs(self,manifest,index,updates,previous,cursor):
        require(manifest['run_id']==self.run_id and manifest['outer_index']==index
                and manifest['cumulative_optimizer_updates_before']==updates,'Wrong immutable outer start')
        require(manifest['config_sha256']==self.config_ref['sha256']
                and manifest['runtime_identity']==self.config['runtime_manifest']['sha256']
                and manifest['source_model_identity']==self.config['source_model_manifest']['sha256'],'Immutable outer source/config differs')
        require(manifest['contract']==science_contract(self.config) and manifest['variant']==self.config['variant']
                and manifest['mining_version']==self.config['mining']['version']
                and manifest['common_cohort_manifest']==self.config['common_cohort_manifest']
                and manifest['common_initial_adapter_manifest']==self.config['common_initial_adapter_manifest'],
                'Immutable variant/miner/cohort/common-init contract differs')
        require(manifest['previous_complete_checkpoint']==previous and manifest['explicit_fresh']==(previous is None)
                and manifest['outer_old_adapter']==(previous or self.config['common_initial_adapter_manifest']),
                'Original outer-start p_old checkpoint changed')
        selected=checked_file(manifest['questions']);questions=read_rows(selected['path'])
        expected_rows,selection=cohort(self.config,cursor)
        require(questions==expected_rows and manifest['train_cursor']==selection,'Resume questions differ from common TRAIN cohort')
        checked_file(manifest['rollouts']);checked_file(manifest['pairs']);verify_scoring(self.config)
        from search_e1 import core
        pairs,stats=verify_rollouts(read_rows(manifest['rollouts']['path']),questions,core,self.config['variant'],index)
        require(read_rows(manifest['pairs']['path'])==pairs and manifest['sampling_stats']==stats and manifest['P']==len(pairs),
                'Immutable strict score/miner/weight accounting differs')
        sampling,_=checked_json(fileproof(Path(manifest['outer_dir'])/'sampling-complete.json'))
        require(sampling['rollouts']==manifest['rollouts'] and sampling['pairs']==manifest['pairs']
                and sampling['sampling_model_identity']==manifest['sampling_model_identity'],'Sampling completion differs from outer inputs')
        self.verify_exit(sampling['exit_proof'],'sampling-running',8)
        return questions

    def reclaim_completed_cache(self,outer,index,completed_ref,terminal,checkpoint):
        """Only this new PR100 completed outer's rebuildable sampling merge cache."""
        require(self.config['schema']=='opsd-pr100-config-v1' and self.config['variant']=='PR'
                and self.config['retention']['merged_cache_cleanup_authorized'] is True,
                'New PR100 cache-only housekeeping authorization missing')
        base=Path(self.config['paths']['base'])
        root=Path(self.config['paths']['output_root'])
        require(base.name.startswith('search-e1-opsd-pr100-') and root==base/'runs'
                and not base.is_symlink() and not root.is_symlink(),
                'Cache cleanup is restricted to the new PR100 namespace')
        require(outer==root/'outers'/f'outer-{index:05d}' and not outer.is_symlink(),
                'Cache cleanup outer escapes this run')
        self.verify_exit(terminal,'training-running',1)
        require(terminal['gpu_release']['all_owned_release']['accepted'] is True,
                'Cache cannot be reclaimed before actual trainer0 and full GPU release')
        completed,_=checked_json(completed_ref)
        require(completed.get('accepted') is True and completed['run_id']==self.run_id
                and completed['outer_index']==index and completed['checkpoint']==checkpoint,
                'Cache cleanup completion/checkpoint identity differs')
        canonical=outer/'merged-model.json';cache=outer/'merged-model-cache'
        identity,identity_ref=checked_json(fileproof(canonical))
        require(cache.exists() and cache.is_dir() and not cache.is_symlink()
                and cache.resolve().parent==outer.resolve() and Path(identity['path']).resolve()==cache.resolve(),
                'Completed merge cache path differs or is symlinked')
        parent_ref=fileproof(outer/'cpu-merge-parent.json');parent,_=checked_json(parent_ref)
        child,_=checked_json(parent['child'])
        require(parent['accepted'] is True and parent['run_id']==self.run_id and child['result']==identity,
                'Cache cleanup needs original accepted CPU merge provenance')
        # Current metadata must still match the immutable accepted actual merge;
        # original full payload SHAs remain in identity/child, never rewritten.
        verify_model_metadata(identity)
        expected={Path(f['path']).resolve():f for f in identity['files']}
        actual=[]
        for path in cache.rglob('*'):
            require(not path.is_symlink(),'Cache cleanup refuses symlinked descendants')
            if path.is_file():actual.append(path.resolve())
        require(set(actual)==set(expected) and all(p.is_relative_to(cache.resolve()) for p in actual),
                'Cache cleanup encountered unbound files or escaped descendants')
        reclaimed_bytes=sum(p.stat().st_size for p in actual)
        receipt=outer/'merged-cache-reclaimed.json'
        require(not receipt.exists(),'Immutable cache cleanup receipt already exists')
        preflight=exclusive_json(outer/'merged-cache-reclaim-intent.json',dict(
            schema='opsd-pr100-merged-cache-reclaim-intent-v1',run_id=self.run_id,config=self.config_ref,
            outer_index=index,cache=str(cache),model_manifest=identity_ref,cpu_merge_parent=parent_ref,
            completed=completed_ref,checkpoint=checkpoint,actual_train_exit=terminal['actual_terminal'],
            bytes=reclaimed_bytes,files=len(actual),scope='Only completed own rebuildable merged-model-cache'))
        shutil.rmtree(cache)
        require(not cache.exists(),'Cache cleanup did not remove only the bound cache')
        return exclusive_json(receipt,dict(schema='opsd-pr100-merged-cache-reclaimed-v1',accepted=True,
            run_id=self.run_id,config_sha256=self.config_ref['sha256'],outer_index=index,intent=preflight,
            original_model_manifest=identity_ref,original_model_manifest_sha256=identity_ref['sha256'],
            bytes_reclaimed=reclaimed_bytes,files_reclaimed=len(actual),cache_absent=True,
            scope='Own completed rebuildable merged cache only; original metadata/CPU merge receipts/checkpoints/raw debug retained',
            reconstruction='Same bound source100 + original outer-start504 FP32 adapter;CPU FP32 merge thenFP16 cast',
            no_old_namespace_touched=True))

    def run(self):
        # Approval is checked before creating experimental outputs or merging a model.
        initial=self.gate('cpu-merge',[])
        Path(self.config['paths']['base']).mkdir(parents=True,exist_ok=True)
        self.lock=open(Path(self.config['paths']['base'])/'online-experiment.lock','a+')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in self.config['resources']['shared_gpu_lease_paths']+[self.config['resources']['study_lease_path']]:
            lease_path=Path(name);require(lease_path.parent.is_dir(),'Shared lease parent must be CPU prepared')
            lease=lease_path.open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);self.shared_locks.append(lease)
        self.gate('cpu-merge',[])
        verify_runtime(self.config)
        verify_common(self.config)
        if not self.resume:
            require(not self.output.exists(),'Fresh output root already exists; explicit same-run resume required')
            self.output.mkdir(parents=True)
            exclusive_json(self.output/'run-identity.json',dict(schema='opsd-online-run-identity-v1',run_id=self.run_id,
                variant=self.config['variant'],study_id=self.config['study_id'],contract=science_contract(self.config),
                config=self.config_ref,config_sha256=self.config_ref['sha256'],initial_owner=self.owner,initial_gate=initial,
                retention=self.config['retention'],no_old_adam_restored=True))
            self.owns_output=True
        else:
            identity=json.loads((self.output/'run-identity.json').read_text())
            require(identity['run_id']==self.run_id and identity['config_sha256']==self.config_ref['sha256'],'Cannot resume another run/config')
            self.owns_output=True
        previous=None;updates=0;cursor=dict(cycle=0,offset=0)
        completed_indices=[int(p.parents[1].name.split('-')[-1]) for p in (self.output/'outers').glob('outer-*/training/outer-complete.json')]
        require(not completed_indices or max(completed_indices)<self.until_outer,
                'Requested training boundary precedes already committed outer progress')
        for index in range(self.until_outer):
            outer=self.output/'outers'/f'outer-{index:05d}';outer.mkdir(parents=True,exist_ok=True)
            completed=outer/'training'/'outer-complete.json'
            manifest_path=outer/'outer-manifest.json'
            if completed.exists():
                manifest,_=checked_json(fileproof(manifest_path))
                self.verify_outer_inputs(manifest,index,updates,previous,cursor)
                require((outer/'teacher-diagnostics'/'complete.json').exists(),'Completed outer teacher diagnostic proof missing')
                self.teacher_diagnostic(outer,index,manifest)
                exits=list((outer/'trainer-logs').glob('attempt-*-exit.json'))
                require(exits,'Completed outer has no actual trainer exit proof; independent terminal recovery required')
                verified=False
                for path in exits:
                    value,_=checked_json(fileproof(path))
                    if value.get('accepted') is True:self.verify_exit(value,'training-running',1);verified=True
                require(verified,'Completed outer has no successful committed trainer phase')
                value,_=check_outer_complete(fileproof(completed),self.config,self.run_id,index,updates,manifest['P'])
                updates=value['optimizer_updates'];previous=value['checkpoint'];cursor=manifest['train_cursor']['after'];continue
            if manifest_path.exists():
                manifest,manifest_ref=checked_json(fileproof(manifest_path))
                questions=self.verify_outer_inputs(manifest,index,updates,previous,cursor)
            else:
                questions_path=outer/'questions.jsonl';selection_path=outer/'train-selection.json'
                if questions_path.exists() or selection_path.exists():
                    require(questions_path.exists() and selection_path.exists(),'Incomplete CPU selection preserved; root review required')
                    questions=read_rows(questions_path);selection,_=checked_json(fileproof(selection_path))
                    expected_rows,expected_selection=cohort(self.config,cursor)
                    require(questions==expected_rows and selection==expected_selection,'Resumed selected questions differ from bound TRAIN stream')
                else:
                    questions,selection=cohort(self.config,cursor);write_rows(questions_path,questions);exclusive_json(selection_path,selection)
                if previous is None and self.config['precision']['mode']=='bf16':
                    model_identity=dict(kind='raw-source100',path=self.config['paths']['model'],sha256=self.config['source_model_manifest']['sha256'],
                        source_model_identity=self.config['source_model_manifest']['sha256'],optimizer_updates=0,adapter=None,
                        checkpoint=None,adapter_tensor_sha256=self.config['common_initial_adapter_tensor_sha256'],
                        variant=self.config['variant'],precision=self.config['precision'],common_initial_adapter=self.config['common_initial_adapter_manifest'])
                else:
                    model_identity=self.cpu_merge(outer,index,previous,updates)
                self.stage('SAMPLING',outer_index=index,optimizer_updates=updates,latest_sampling_model=model_identity['sha256'])
                sampling_path=outer/'sampling-complete.json'
                if sampling_path.exists():
                    sampling,_=checked_json(fileproof(sampling_path));require(all(sampling['sampling_model_identity'].get(k)==model_identity.get(k) for k in ('kind','path','sha256','source_model_identity','checkpoint','adapter_tensor_sha256','optimizer_updates')),'Changed sampling model on resume')
                    checked_file(sampling['rollouts']);checked_file(sampling['pairs'])
                    self.verify_exit(sampling['exit_proof'],'sampling-running',8)
                else:sampling,_=self.sample(outer,questions,model_identity)
                model_identity=sampling['sampling_model_identity']
                manifest=dict(schema='opsd-online-outer-manifest-v1',run_id=self.run_id,config_sha256=self.config_ref['sha256'],
                    runtime_identity=self.config['runtime_manifest']['sha256'],source_model_identity=self.config['source_model_manifest']['sha256'],
                    outer_index=index,outer_dir=str(outer),questions=fileproof(questions_path),train_cursor=selection,
                    variant=self.config['variant'],contract=science_contract(self.config),mining_version=self.config['mining']['version'],
                    common_cohort_manifest=self.config['common_cohort_manifest'],common_initial_adapter_manifest=self.config['common_initial_adapter_manifest'],
                    outer_old_adapter=previous or self.config['common_initial_adapter_manifest'],
                    rollouts=sampling['rollouts'],pairs=sampling['pairs'],P=sampling['stats']['P'],sampling_stats=sampling['stats'],
                    sampling_model_identity=model_identity,previous_complete_checkpoint=previous,explicit_fresh=previous is None,
                    cumulative_optimizer_updates_before=updates)
                manifest_ref=exclusive_json(manifest_path,manifest)
            self.teacher_diagnostic(outer,index,manifest)
            resume=latest_resume_checkpoint(self.output,self.config,self.run_id,index)
            command=[self.config['paths']['python'],'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                str(Path(self.config['paths']['helper_root'])/'trainer.py'),'--config',self.config_ref['path'],
                '--config-sha256',self.config_ref['sha256'],'--outer-manifest',manifest_ref['path'],
                '--outer-manifest-sha256',manifest_ref['sha256']]
            if resume:
                command+=['--resume-checkpoint',resume['path'],'--resume-checkpoint-sha256',resume['sha256']]
                try:from .retry_state import select_retry_capsule
                except ImportError:from retry_state import select_retry_capsule
                retry=select_retry_capsule(outer/'training',resume,self.config)
                if retry:command+=['--retry-capsule-manifest',retry['path'],'--retry-capsule-sha256',retry['sha256']]
            self.stage('TRAINING',outer_index=index,optimizer_updates_before=updates,resume_checkpoint=resume)
            logs=outer/'trainer-logs';logs.mkdir(exist_ok=True);attempt=len(list(logs.glob('attempt-*.log')))
            self.gate('training-start',[])
            job=self.spawn(command,logs/f'attempt-{attempt:05d}.log','training-running',dict(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'))
            terminal=self.wait([job],'training-running');exclusive_json(logs/f'attempt-{attempt:05d}-exit.json',terminal)
            value,evidence=check_outer_complete(fileproof(completed),self.config,self.run_id,index,updates,manifest['P'])
            updates=value['optimizer_updates'];previous=value['checkpoint'];cursor=manifest['train_cursor']['after']
            cache_receipt=self.reclaim_completed_cache(outer,index,fileproof(completed),terminal,previous)
            self.stage('OUTER_COMPLETE',outer_index=index,actual_optimizer_updates=updates,P=value['P'],K=value['K'],checkpoint=previous,
                merged_cache_reclaimed=cache_receipt)
            # No checkpoint/debug or unapproved merge-cache deletion. Keep-all is explicit.
        release=self.wait_release([])
        phase='TRAIN_COMPLETE' if self.until_outer==100 else 'TRAIN_BOUNDARY'
        boundary=dict(schema='opsd-pr100-training-boundary-v1',accepted=True,run_id=self.run_id,
            variant='PR',config=self.config_ref,config_sha256=self.config_ref['sha256'],
            immutable_outer_iterations=100,completed_outers=self.until_outer,boundary_outer=self.until_outer,
            actual_optimizer_updates=updates,latest_checkpoint=previous,checkpoint=previous,train_cursor=cursor,
            last_outer_complete=fileproof(self.output/'outers'/f'outer-{self.until_outer-1:05d}'/'training'/'outer-complete.json'),
            resource_release=release,owner=self.owner,phase=phase,
            scope='Same-run online state boundary; parent actual wait0 required separately before native512')
        boundary_path=self.output/'boundaries'/f'outer-{self.until_outer:05d}.json'
        if boundary_path.exists():
            accepted,ref=checked_json(fileproof(boundary_path))
            require(all(accepted.get(k)==boundary[k] for k in ('schema','run_id','variant','config','immutable_outer_iterations',
                'completed_outers','boundary_outer','actual_optimizer_updates','latest_checkpoint','train_cursor','last_outer_complete','phase')),
                'Existing immutable boundary differs from same-run committed state')
        else:ref=exclusive_json(boundary_path,boundary)
        self.stage(phase,outer_iterations=100,completed_outers=self.until_outer,boundary_outer=self.until_outer,
            actual_optimizer_updates=updates,latest_checkpoint=previous,resource_release=release,boundary=ref,
            scope='Native512 follows each boundary;100-outers train completion still has separate effect acceptance')


def validate_until_outer(value,total=100):
    require(total==100 and type(value) is int and value in range(10,101,10),
            'Explicit until-outer must be10,20,...100; immutable science budget remains100')
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--until-outer',required=True,type=int);a=p.parse_args()
    controller=None
    try:
        controller=Controller(dict(path=a.config,sha256=a.config_sha256),a.run_id,a.resume,until_outer=a.until_outer);controller.run()
    except Exception as exc:
        if controller is not None and controller.owns_output:
            controller.contain_failure(exc)
        raise


if __name__=='__main__':main()
