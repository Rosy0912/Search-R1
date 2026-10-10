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
        value=json.loads(path.read_text())
        if value.get('accepted') is True and value.get('run_id')==run_id and value.get('cursor',{}).get('outer_index')==outer_index:
            ref=fileproof(path);checked,_=check_checkpoint(ref,config,run_id,expected_outer=outer_index)
            candidates.append((checked['optimizer_updates'],checked['cursor']['next_pair_cursor'],ref))
    if not candidates:return None
    candidates.sort(key=lambda x:(x[0],x[1]));return candidates[-1][2]


class Controller:
    def __init__(self,config_ref,run_id,resume=False,admit=None):
        self.config,self.config_ref=checked_json(config_ref);validate_config(self.config,launch=True)
        require(self.config['run_id']==run_id,'Config/CLI run identity mismatch')
        self.config['_config_sha256']=self.config_ref['sha256'];self.config['_run_id']=run_id;self.run_id=run_id;self.resume=resume
        self.owner=process_identity(os.getpid())
        require(self.owner and self.owner['uid']==self.config['owner']['uid'] and self.owner['run_id']==run_id,'Controller needs actual bound UID/run environment')
        self.admit=admit or load_admission(self.config);self.known=[];self.active_jobs=[];self.phase='CREATED'
        self.output=Path(self.config['paths']['output_root']);self.lock=None;self.shared_locks=[];self.owns_output=False

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
                and proof['merge_calls']==1 and proof['cast_after_merge_count']==(1 if self.config['precision']['mode']=='fp32_master_fp16' else 0),
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
        root=outer/'teacher-diagnostics';root.mkdir(exist_ok=True)
        done=root/'complete.json'
        if done.exists():return checked_json(fileproof(done))[0]
        request=exclusive_json(root/'paired-request.json',dict(config=self.config_ref,pairs=manifest['pairs'],outer_index=index,student_checkpoint=manifest['previous_complete_checkpoint'],output=str(root/'paired')))
        command=[self.config['paths']['python'],'-B',str(Path(self.config['paths']['helper_root'])/'paired_diag.py'),'--request',request['path'],'--request-sha256',request['sha256']]
        self.gate('sampling-start',[])
        job=self.spawn(command,root/'paired.log','sampling-running',dict(CUDA_VISIBLE_DEVICES='0'))
        terminal=self.wait([job],'sampling-running')
        child,ref=checked_json(fileproof(root/'paired/complete.json'));require(child['accepted'],'Paired autonomous diagnostic incomplete')
        return exclusive_json(done,dict(accepted=True,child=ref,exit_proof=terminal,original_training_objective=False))

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

    def evaluate(self,number,checkpoint):
        destination=Path(self.config['paths']['base'])/'periodic512'/f'eval-{number:03d}'
        done=destination/'summary.json'
        if done.exists():
            value=json.loads(done.read_text());require(value['checkpoint']==checkpoint and value['count']==512,'Changed existing evaluation');return
        destination.mkdir(parents=True,exist_ok=True)
        self.stage('EVALUATING_512',completed_outers=number,checkpoint=checkpoint)
        self.gate('sampling-start',[])
        jobs=[]
        for rank in range(8):
            command=[self.config['paths']['python'],'-B',str(Path(self.config['paths']['helper_root'])/'eval_worker.py'),'--config',self.config_ref['path'],'--checkpoint',checkpoint['path'],'--checkpoint-sha256',checkpoint['sha256'],'--rank',str(rank),'--output',str(destination)]
            jobs.append(self.spawn(command,destination/f'worker-{rank}.log','sampling-running',dict(CUDA_VISIBLE_DEVICES=str(rank))))
        terminal=self.wait(jobs,'sampling-running');exclusive_json(destination/'actual-exit-release.json',terminal)
        rows=[]
        for rank in range(8):
            rows.extend(read_rows(destination/f'results-{rank}.jsonl'))
        require(len(rows)==512 and sorted(x['index'] for x in rows)==list(range(512)),'Incomplete or duplicate512')
        totals={}
        for x in rows:
            v=totals.setdefault(x['dataset'],dict(count=0,hits=0));v['count']+=1;v['hits']+=x['em']
        exclusive_json(done,dict(count=512,hits=sum(x['em'] for x in rows),em=sum(x['em'] for x in rows)/512,by_dataset=totals,checkpoint=checkpoint,completed_outers=number,actual_exit_release=terminal,finished_unix=time.time(),protocol='native HF SDPA BF16base FP32unmergedadapter greedy rep1.05',exclusive_GPU=True))

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
                retention=self.config['retention'],no_old_adam_restored=True,trajectory_generator='fixed_source100_ordinary'))
            self.owns_output=True
        else:
            identity=json.loads((self.output/'run-identity.json').read_text())
            require(identity['run_id']==self.run_id and identity['config_sha256']==self.config_ref['sha256'],'Cannot resume another run/config')
            self.owns_output=True
        previous=None;updates=0;cursor=dict(cycle=0,offset=0)
        for index in range(self.config['scientific']['outer_iterations']):
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
                updates=value['optimizer_updates'];previous=value['checkpoint'];cursor=manifest['train_cursor']['after']
                if (index+1)%10==0:self.evaluate(index+1,previous)
                continue
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
                model_identity=dict(kind='raw-source100',path=self.config['paths']['model'],sha256=self.config['source_model_manifest']['sha256'],source_model_identity=self.config['source_model_manifest']['sha256'],optimizer_updates=0,adapter=None,checkpoint=None,adapter_tensor_sha256=self.config['common_initial_adapter_tensor_sha256'],variant=self.config['variant'],precision=self.config['precision'],common_initial_adapter=self.config['common_initial_adapter_manifest'])
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
            self.stage('OUTER_COMPLETE',outer_index=index,actual_optimizer_updates=updates,P=value['P'],K=value['K'],checkpoint=previous)
            if (index+1)%10==0:self.evaluate(index+1,previous)
            from checkpoint import prune_merged_caches
            prune_merged_caches(self.output,self.run_id,self.config_ref['sha256'])
        release=self.wait_release([])
        self.stage('TRAIN_COMPLETE',outer_iterations=200,actual_optimizer_updates=updates,latest_checkpoint=previous,resource_release=release,
            scope='Full200 online outers including preserved first5; no512 evaluation')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    controller=None
    try:
        controller=Controller(dict(path=a.config,sha256=a.config_sha256),a.run_id,a.resume);controller.run()
    except Exception as exc:
        if controller is not None and controller.owns_output:
            controller.contain_failure(exc)
        raise


if __name__=='__main__':main()
