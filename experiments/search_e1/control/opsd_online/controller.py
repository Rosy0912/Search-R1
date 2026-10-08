"""Exclusive online OPSD orchestration. No signals, GRPO or optimizer reset."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

try:
    from .checkpoint import (require,checked_json,checked_file,fileproof,object_sha,exclusive_json,atomic_json,
                             check_checkpoint,check_outer_complete,prune_merged_caches)
    from .prepare import validate_config,verify_runtime
    from .scoring import verify_scoring
    from .sampling import cohort,write_rows,read_rows,verify_rollouts,merge_latest,bind_model_files,verify_http_trace
except ImportError:
    from checkpoint import (require,checked_json,checked_file,fileproof,object_sha,exclusive_json,atomic_json,
                            check_checkpoint,check_outer_complete,prune_merged_caches)
    from prepare import validate_config,verify_runtime
    from scoring import verify_scoring
    from sampling import cohort,write_rows,read_rows,verify_rollouts,merge_latest,bind_model_files,verify_http_trace


def process_identity(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    stat=(p/'stat').read_text().rsplit(')',1)[1].split()
    status=(p/'status').read_text().splitlines()
    uid=int(next(s for s in status if s.startswith('Uid:')).split()[2])
    args=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    environ=dict(x.split('=',1) for x in (p/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in x)
    return dict(pid=pid,uid=uid,starttick=int(stat[19]),state=stat[0],ppid=int(stat[1]),pgid=int(stat[2]),
                argv=args,run_id=environ.get('E1_RUN_ID'),cuda_visible_devices=environ.get('CUDA_VISIBLE_DEVICES'))


def gpu_snapshot():
    cards=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    return dict(time=time.time(),cards=[dict(index=int(x[0]),uuid=x[1],free_mib=int(x[2])) for x in
                ([v.strip() for v in s.split(',')] for s in cards.splitlines())],
                apps=[dict(uuid=x[0],pid=int(x[1]),used_mib=int(x[2])) for x in
                ([v.strip() for v in s.split(',')] for s in apps.splitlines() if s.strip())])


def require_gpu_release(config,children):
    identities={c['pid']:c for c in children}
    for pid,expected in identities.items():
        actual=process_identity(pid)
        require(actual is None or actual['starttick']!=expected['starttick'],'Owned worker remains after wait (including zombie)')
    snapshot=gpu_snapshot()
    require(not any(a['pid'] in identities for a in snapshot['apps']),'Sampling/trainer still holds GPU after child exit')
    for app in snapshot['apps']:
        actual=process_identity(app['pid'])
        require(actual is not None,'Opaque GPU PID after phase')
        require(actual['run_id']!=config['_run_id'],'Owned descendant still holds GPU after parent exit')
    cards={c['index']:c for c in snapshot['cards']}
    for index,uuid in zip(config['owner']['physical_gpus'],config['owner']['gpu_uuids']):
        require(cards[index]['uuid']==uuid and cards[index]['free_mib']>=config['resources']['startup_minimum_free_mib'],'GPU UUID/free gate before next phase failed')
    return snapshot


def load_admission(config):
    manifest,_=checked_json(config['helper_source_manifest'])
    expected=manifest['files']['acceptance.py']
    checked_file(dict(path=str(Path(config['paths']['helper_root'])/'acceptance.py'),
                      **(expected if isinstance(expected,dict) else dict(sha256=expected))))
    path=Path(config['paths']['helper_root'])/'acceptance.py'
    require(path.is_file(),'HOLD: acceptance helper not installed/published')
    spec=importlib.util.spec_from_file_location('opsd_online_acceptance_bound',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(callable(getattr(module,'require_launch',None)),'Missing strict admission interface')
    require(callable(getattr(module,'require_owned_release',None)),'Missing full owned-release audit interface')
    module.require_launch.require_owned_release=module.require_owned_release
    return module.require_launch


def latest_resume_checkpoint(output_root,config,run_id,outer_index):
    """Only accepted complete actual state; incomplete pending checkpoints never count."""
    candidates=[]
    for path in (Path(output_root)/'checkpoints').glob('*/manifest.json'):
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
        return atomic_json(self.output/'status.json',dict(schema='opsd-online-controller-status-v1',run_id=self.run_id,
            owner=self.owner,phase=phase,updated_at_unix=time.time(),config_sha256=self.config_ref['sha256'],**fields))

    def spawn(self,command,log,phase,extra_env=None):
        self.gate(phase)
        runtime=Path(self.config['paths']['runtime'])
        env=dict(os.environ,PYTHONPATH=str(runtime/'upstream'/'Search-R1')+os.pathsep+str(runtime),PYTHONUNBUFFERED='1',
                 TOKENIZERS_PARALLELISM='false',E1_RUN_ID=self.run_id,E1_ONLINE_PHASE=phase,
                 VLLM_ATTENTION_BACKEND='XFORMERS',E1_INFERENCE_UTIL=str(self.config['sampling']['gpu_memory_utilization']))
        env.update(extra_env or {})
        handle=Path(log).open('x')
        process=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        identity=dict(pid=process.pid,uid=self.owner['uid'],starttick=None,run_id=self.run_id,identity_uncertain=True)
        self.known.append(identity);job=(process,handle,identity);self.active_jobs.append(job)
        actual=process_identity(process.pid)
        require(actual and actual['uid']==self.owner['uid'] and actual['run_id']==self.run_id,'Spawn identity mismatch (child registered; lease retained during recovery)')
        identity.clear();identity.update(actual)
        with (self.output/'child-history.jsonl').open('a') as history:
            history.write(json.dumps(dict(time=time.time(),phase=phase,identity=identity,command=command,log=str(log)))+'\n');history.flush();os.fsync(history.fileno())
        return job

    def wait(self,jobs,phase):
        failed=False
        while True:
            codes=[p.poll() for p,_,_ in jobs]
            if any(c is not None and c!=0 for c in codes):
                failed=True;self.stage('FAILED_CHILD_WAITING_OWNED_TERMINATION',child_phase=phase,exit_codes=codes,
                    scientific_updates_not_inferred_from_exit=True)
            # Do not stop a process. The root supervisor can perform separately authorized narrow recovery.
            try:self.gate(phase)
            except Exception as exc:
                failed=True;self.stage('UNCERTAIN_ADMISSION_WAITING_OWNED_TERMINATION',child_phase=phase,error=str(exc),exit_codes=codes)
            if all(c is not None for c in codes):break
            time.sleep(5)
        evidence=[]
        for p,h,identity in jobs:
            rc=p.wait();
            if not h.closed:h.close()
            evidence.append(dict(identity=identity,actual_wait_returncode=rc))
        release=self.wait_release([i for _,_,i in jobs])
        require(not failed and all(e['actual_wait_returncode']==0 for e in evidence),'Worker failure preserved; no automatic signal/restart')
        return dict(accepted=True,child_exit_evidence=evidence,gpu_release=release)

    def wait_release(self,children):
        while True:
            try:
                release=require_gpu_release(self.config,children)
                self.gate('cpu-merge',[])
                # load_admission attaches an explicit full owned-process-release callback.
                audit=getattr(self.admit,'require_owned_release',None)
                require(callable(audit),'HOLD: full same-UID V2 owned release audit unavailable')
                proof=audit(self.config,self.run_id,self.owner,known_children=self.known)
                require(proof.get('accepted') is True,'Owned descendant/opaque process release audit failed')
                return dict(gpu=release,all_owned_release=proof)
            except Exception as exc:
                self.stage('UNCERTAIN_RELEASE_LEASE_HELD',error=str(exc),known_children=self.known,
                    signals_sent=False,recovery='Wait for actual terminal evidence; no next CUDA phase')
                time.sleep(5)

    def contain_failure(self,exc):
        self.stage('FAILED_LEASE_HELD',error_type=type(exc).__name__,error=str(exc),known_children=self.known,signals_sent=False)
        if self.active_jobs:
            while any(p.poll() is None for p,_,_ in self.active_jobs):
                self.stage('FAILED_WAITING_OWNED_TERMINATION',error=str(exc),known_children=self.known,signals_sent=False)
                time.sleep(5)
            for process,handle,_ in self.active_jobs:
                process.wait()
                if not handle.closed:handle.close()
            self.wait_release([i for _,_,i in self.active_jobs])
        self.stage('FAILED_OWNED_RELEASE_CONFIRMED',error_type=type(exc).__name__,error=str(exc),known_children=self.known,signals_sent=False)

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
            require(value['sampling_model_identity']==identity['sha256'],'Worker sampled different weight identity')
            checked_file(value['rollouts']);checked_file(value['generation_trace'])
            http_proof=verify_http_trace(value['http_trace'],rank,identity['sha256'],value['http_requests'])
            require(Path(http_proof['path']).resolve()==(attempt/f'http-requests-{rank}.jsonl').resolve(),'Worker HTTP trace path mismatch')
            rows.extend(read_rows(value['rollouts']['path']));worker_proofs.append(proof);http_proofs.append(http_proof)
        sys.path.insert(0,self.config['paths']['runtime'])
        scoring_proof=verify_scoring(self.config)
        from search_e1 import core
        pairs,stats=verify_rollouts(rows,questions,core)
        rollout_ref=write_rows(attempt/'rollouts.jsonl',rows);pair_ref=write_rows(attempt/'pairs.jsonl',pairs)
        receipt=dict(schema='opsd-online-sampling-complete-v1',accepted=True,run_id=self.run_id,
            sampling_model_identity=identity,workers=worker_proofs,exit_proof=exit_proof,
            private_http_traces=http_proofs,scoring_identity=scoring_proof,
            rollouts=rollout_ref,pairs=pair_ref,stats=stats)
        return receipt,exclusive_json(outer/'sampling-complete.json',receipt)

    def run(self):
        # Approval is checked before creating experimental outputs or merging a model.
        initial=self.gate('cpu-merge',[])
        Path(self.config['paths']['base']).mkdir(parents=True,exist_ok=True)
        self.lock=open(Path(self.config['paths']['base'])/'online-experiment.lock','a+')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in self.config['resources']['shared_gpu_lease_paths']:
            lease_path=Path(name);require(lease_path.parent.is_dir(),'Shared lease parent must be CPU prepared')
            lease=lease_path.open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);self.shared_locks.append(lease)
        self.gate('cpu-merge',[])
        verify_runtime(self.config)
        if not self.resume:
            require(not self.output.exists(),'Fresh output root already exists; explicit same-run resume required')
            self.output.mkdir(parents=True)
            exclusive_json(self.output/'run-identity.json',dict(schema='opsd-online-run-identity-v1',run_id=self.run_id,
                config=self.config_ref,config_sha256=self.config_ref['sha256'],initial_owner=self.owner,initial_gate=initial,
                retention=self.config['retention'],no_old_adam_restored=True))
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
                value,_=check_outer_complete(fileproof(completed),self.config,self.run_id,index,updates,manifest['P'])
                updates=value['optimizer_updates'];previous=value['checkpoint'];cursor=manifest['train_cursor']['after'];continue
            if manifest_path.exists():
                manifest,manifest_ref=checked_json(fileproof(manifest_path))
                require(manifest['run_id']==self.run_id and manifest['outer_index']==index and manifest['cumulative_optimizer_updates_before']==updates,'Wrong immutable outer resume')
                require(manifest['config_sha256']==self.config_ref['sha256'] and manifest['runtime_identity']==self.config['runtime_manifest']['sha256'] and manifest['source_model_identity']==self.config['source_model_manifest']['sha256'],'Immutable outer source/config mismatch')
                require(manifest['previous_complete_checkpoint']==previous and manifest['explicit_fresh']==(previous is None),'Immutable outer-start checkpoint mismatch')
                selected_ref=checked_file(manifest['questions']);questions=read_rows(selected_ref['path'])
                expected_rows,expected_selection=cohort(self.config,cursor)
                require(questions==expected_rows and manifest['train_cursor']==expected_selection,'Outer resume questions not bound to TRAIN stream')
                checked_file(manifest['rollouts']);checked_file(manifest['pairs'])
                sys.path.insert(0,self.config['paths']['runtime'])
                verify_scoring(self.config)
                from search_e1 import core
                expected_pairs,expected_stats=verify_rollouts(read_rows(manifest['rollouts']['path']),questions,core)
                require(read_rows(manifest['pairs']['path'])==expected_pairs and manifest['sampling_stats']==expected_stats and manifest['P']==len(expected_pairs),'Immutable outer mining/scoring mismatch')
            else:
                questions_path=outer/'questions.jsonl';selection_path=outer/'train-selection.json'
                if questions_path.exists() or selection_path.exists():
                    require(questions_path.exists() and selection_path.exists(),'Incomplete CPU selection preserved; root review required')
                    questions=read_rows(questions_path);selection,_=checked_json(fileproof(selection_path))
                    expected_rows,expected_selection=cohort(self.config,cursor)
                    require(questions==expected_rows and selection==expected_selection,'Resumed selected questions differ from bound TRAIN stream')
                else:
                    questions,selection=cohort(self.config,cursor);write_rows(questions_path,questions);exclusive_json(selection_path,selection)
                if previous is None:
                    model_identity=dict(kind='raw-source100',path=self.config['paths']['model'],sha256=self.config['source_model_manifest']['sha256'],
                        source_model_identity=self.config['source_model_manifest']['sha256'],optimizer_updates=0,adapter=None,
                        checkpoint=None,adapter_tensor_sha256='ZERO_EFFECT_FRESH_LORA')
                else:
                    checkpoint,_=check_checkpoint(previous,self.config,self.run_id,updates)
                    merge_receipt=outer/'merged-model.json'
                    if merge_receipt.exists():model_identity,_=checked_json(fileproof(merge_receipt))
                    else:
                        self.gate('cpu-merge',[])
                        require(shutil.disk_usage(self.output).free>=self.config['resources']['minimum_disk_bytes'],'Disk headroom before CPU merge failed')
                        model_identity=merge_latest(self.config,checkpoint,outer/'merged-model-cache',previous);exclusive_json(merge_receipt,model_identity)
                    require(model_identity['checkpoint_optimizer_updates']==updates,'Sampler merge is not latest checkpoint')
                self.stage('SAMPLING',outer_index=index,optimizer_updates=updates,latest_sampling_model=model_identity['sha256'])
                sampling_path=outer/'sampling-complete.json'
                if sampling_path.exists():
                    sampling,_=checked_json(fileproof(sampling_path));require(all(sampling['sampling_model_identity'].get(k)==model_identity.get(k) for k in ('kind','path','sha256','source_model_identity','checkpoint','adapter_tensor_sha256','optimizer_updates')),'Changed sampling model on resume')
                    checked_file(sampling['rollouts']);checked_file(sampling['pairs'])
                else:sampling,_=self.sample(outer,questions,model_identity)
                model_identity=sampling['sampling_model_identity']
                manifest=dict(schema='opsd-online-outer-manifest-v1',run_id=self.run_id,config_sha256=self.config_ref['sha256'],
                    runtime_identity=self.config['runtime_manifest']['sha256'],source_model_identity=self.config['source_model_manifest']['sha256'],
                    outer_index=index,outer_dir=str(outer),questions=fileproof(questions_path),train_cursor=selection,
                    rollouts=sampling['rollouts'],pairs=sampling['pairs'],P=sampling['stats']['P'],sampling_stats=sampling['stats'],
                    sampling_model_identity=model_identity,previous_complete_checkpoint=previous,explicit_fresh=previous is None,
                    cumulative_optimizer_updates_before=updates)
                manifest_ref=exclusive_json(manifest_path,manifest)
            resume=latest_resume_checkpoint(self.output,self.config,self.run_id,index)
            command=[self.config['paths']['python'],'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                str(Path(self.config['paths']['helper_root'])/'trainer.py'),'--config',self.config_ref['path'],
                '--config-sha256',self.config_ref['sha256'],'--outer-manifest',manifest_ref['path'],
                '--outer-manifest-sha256',manifest_ref['sha256']]
            if resume:command+=['--resume-checkpoint',resume['path'],'--resume-checkpoint-sha256',resume['sha256']]
            self.stage('TRAINING',outer_index=index,optimizer_updates_before=updates,resume_checkpoint=resume)
            logs=outer/'trainer-logs';logs.mkdir(exist_ok=True);attempt=len(list(logs.glob('attempt-*.log')))
            self.gate('training-start',[])
            job=self.spawn(command,logs/f'attempt-{attempt:05d}.log','training-running',dict(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'))
            terminal=self.wait([job],'training-running');exclusive_json(logs/f'attempt-{attempt:05d}-exit.json',terminal)
            value,evidence=check_outer_complete(fileproof(completed),self.config,self.run_id,index,updates,manifest['P'])
            updates=value['optimizer_updates'];previous=value['checkpoint'];cursor=manifest['train_cursor']['after']
            self.stage('OUTER_COMPLETE',outer_index=index,actual_optimizer_updates=updates,P=value['P'],K=value['K'],checkpoint=previous)
            prune_merged_caches(self.output,self.run_id,self.config_ref['sha256'])
        self.stage('COMPLETE',outer_iterations=self.config['scientific']['outer_iterations'],actual_optimizer_updates=updates,latest_checkpoint=previous)


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
