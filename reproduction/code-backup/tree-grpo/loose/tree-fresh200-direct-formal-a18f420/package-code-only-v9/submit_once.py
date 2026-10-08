"""Cluster-bound production candidate dispatcher. Not executed by preparation.

Existence/unknown is irreversible for this intent; no removal, resubmit, restart,
or cancellation path. All tests inject an in-memory fake scheduler.
"""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import time
from production_contract import ROOT,require,read_bound,production_inputs,DIRECT_MODE,descriptor,SOURCE,CHAIN_SCOPE
from slurm_identity import (CLUSTER,NODE,local_environment,cluster_identity,query,
    create_journal,finalize_journal,timeout_output,verify_management_scope)

TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
INTENT_RE=r'SUBMISSION_ONCE-current-fresh200-chain-mask-v9-[0-9a-f]{7}\.json'
HISTORICAL_SOURCES={
    '3319':'a18f4200ba4fd3e76ceeeae0cf409a43e25f7b98',
    '3321':'a18f4200ba4fd3e76ceeeae0cf409a43e25f7b98',
    '3322':'a18f4200ba4fd3e76ceeeae0cf409a43e25f7b98',
    '3326':'2e26b5505ce8ddfad8b627bba74337fbf4e23373',
    '3327':'4a65e9a3ab87e84e1e9dd283b664cdef3a224c8a',
    '3328':'4a65e9a3ab87e84e1e9dd283b664cdef3a224c8a',
    '3330':'9693f0c846fab483a03e1d70ce1109aeddc5e118',
    '3331':'9693f0c846fab483a03e1d70ce1109aeddc5e118',
}
HISTORICAL_JOURNALS={
    '3321':('SUBMISSION_ONCE-fresh200-direct-a18f420.json','33fd8b5351a0a28e0a7a1e5742ce5c49db604d950985f35f234ba36e2a7de504'),
    '3322':('SUBMISSION_ONCE-fresh200-direct-a18f420-v3.json','3f21730c9665dc25072d71c8e79b7e91becfcf5e96ad0aa136df26545cb2fad4'),
    '3326':('SUBMISSION_ONCE-fresh200-direct-2e26b55-v4.json','b9e7de77b1d1e168dd02af2f3be474af7d0dd2a510d13e4ee812fe6d6c4b9108'),
    '3327':('SUBMISSION_ONCE-fresh200-direct-4a65e9a-v5.json','cda27d64b7114417f64cf80f23772b288cdbc7ec9ef2ce9e7c86338e08dbf1c9'),
    '3328':('SUBMISSION_ONCE-fresh200-direct-4a65e9a-v6.json','ef2817200f999c74deca43ea3de501f6bf18276ea838b38b35342ebf3368265f'),
    '3330':('SUBMISSION_ONCE-current-fresh200-chain-mask-v7-9693f0c.json','5f7071c0c6886b3a5b88abd216cd3b18bc0b14a1fd6293418717d742ece4ccff'),
    '3331':('SUBMISSION_ONCE-current-fresh200-chain-mask-v8-9693f0c.json','336a0a60df561fd3fcf50e7fdc03cee344a0b33e9d916193c7b4a4716b1b6a12'),
}

def queue_clear(run,environment):
    raw=query(['squeue','--local','-h','-w',NODE,'-o','%i|%j|%T|%b'],run,environment,management=True)
    for line in raw.splitlines():
        if not line.strip():continue
        parts=line.split('|');require(len(parts)==4,'Ambiguous local queue snapshot')
        if parts[1].startswith('tree-'):
            # Only an explicitly zero GPU request can coexist for this task.
            zero=parts[3] in ('gres/gpu:0','gres/gpu=0','gpu:0')
            require(zero,'Another Tree GPU allocation/pending lineage exists; never duplicate')
    return raw

def prior_journals_clear(path,run,environment,proof_closure):
    require(not path.exists() and not path.is_symlink(),'Intent already exists; no retry even UNKNOWN')
    observations=[]
    prior_paths=set(path.parent.glob('TREE_FRESH200_TRAIN_ONCE-*.json'))|set(path.parent.glob('SUBMISSION_ONCE-fresh200-direct-*.json'))|set(path.parent.glob('SUBMISSION_ONCE-current-fresh200-chain-*.json'))
    for prior in sorted(prior_paths):
        value,sha,_=read_bound(prior)
        require(value.get('status')=='SUBMITTED' and value.get('cluster_name')==CLUSTER
                and type(value.get('job_id')) is int and value['job_id']>0,'Previous unknown/legacy/foreign intent blocks dispatch')
        job=str(value['job_id'])
        output=value.get('parsable_stdout');require(output in (job,job+';'+CLUSTER),'Foreign prior submit suffix')
        match=[proof for proof in proof_closure.values() if proof.get('once_journal',{}).get('path')==str(prior)]
        if match:
            require(len(match)==1,'Multiple retained terminal proofs claim the same once journal')
            proof=match[0]
            require(sha==proof['once_journal']['sha256'] and job==str(proof['job_id'])
                    and value['manifest_sha256']==proof['prior_manifest_sha256'],
                    'Exact retained failed journal/source binding differs')
            observations.append(dict(path=str(prior),sha256=sha,positive_terminal_cleanup_proof=proof,
                accounting_disabled_not_interpreted_as_terminal=True))
            continue
        raw=query(['sacct','--local','-X','-n','-P','-j',job,'-o','JobIDRaw,Cluster,State,ExitCode'],run,environment,management=True)
        rows=[line.split('|') for line in raw.splitlines() if line.strip()]
        require(len(rows)==1 and len(rows[0])==4 and rows[0][0]==job and rows[0][1]==CLUSTER
                and rows[0][2] in TERMINAL and re.fullmatch(r'\d+:\d+',rows[0][3]),'Previous own terminal lineage unknown; accounting failure never means stopped')
        observations.append(dict(path=str(prior),sha256=sha,accounting_stdout=raw))
    return observations

def submission_output(value):
    if type(value) is bytes:return value.decode('utf-8',errors='strict')
    require(type(value) is str,'Unexpected native submit output type')
    return value

def read_regular_bound(path,expected):
    path=Path(path);require(stat.S_ISREG(path.lstat().st_mode),'Pinned package/script must be regular file')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:
        before=os.fstat(stream.fileno());raw=stream.read();after=os.fstat(stream.fileno())
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
            and hashlib.sha256(raw).hexdigest()==expected,'Pinned package/script bytes changed: '+str(path))
    return raw

def parse_submission(result):
    stdout,stderr=submission_output(result.stdout),submission_output(result.stderr)
    require(result.returncode==0 and stderr=='','Submit not exactly confirmed')
    require(re.fullmatch(r'[1-9][0-9]*(;wuhan-h100)?\n?',stdout),'Foreign/ambiguous submit suffix')
    parsable=stdout.rstrip('\n');return int(parsable.split(';')[0]),parsable

def predecessor_ended(manifest,proof_path,proof_sha):
    pins=manifest['terminal_proof_closure']
    require(set(pins)==set(HISTORICAL_SOURCES) and pins['3331']==dict(path=str(proof_path),sha256=proof_sha),
            'Direct V9 must bind latest3331 and all retained historical terminal proofs')
    require(pins['3319']['sha256']=='a4b9656f02427090983405b91afe3757473103a91b727c09aa51384b2f876816'
            and pins['3321']['sha256']=='484b1f5485d63c4f082c6e520e3d17914fa12cb6fefd2457509b0e2b975d51ef'
            and pins['3322']['sha256']=='85a8bfa5e9d9649c1fec6db8fcc2a3e21a8250501931aee264117e09df6c0930'
            and pins['3326']['sha256']=='9d4a4754d5364affd930ede6205fa419c63102fbbcbb7b31f935cbaa0d83696b'
            and pins['3327']['sha256']=='1d0d957f96b82c5a22f2f6e69dfa841193fe222c3320258e07a4974becbc5909'
            and pins['3328']['sha256']=='9e907f68bdac597d5b3b091e3e1199fcd715170694e71cd1f9b42b80a75e482a'
            and pins['3330']['sha256']=='f8b15bdfaa72923b59821a5c37b40b5ab4ce804946f7262bcb6364077dcbc047',
            'Retained historical terminal proof bytes cannot be replaced or relabelled')
    closure={}
    for job,pin in sorted(pins.items()):
        proof,digest,_=descriptor(pin)
        historical_source=HISTORICAL_SOURCES[job]
        calibration=job=='3319'
        run_name=('tree-fresh200-' if calibration else ('tree-current200-continue-' if job in ('3330','3331') else 'tree-fresh200-train-'))+job
        status='OWN_EXTRA_CALIBRATION_ENDED_FOR_DIRECT_TRAIN_WITH_CLEANUP' if calibration else 'OWN_DIRECT_TRAIN_FAILED_WITH_CLEANUP'
        require(pin['path']==str(ROOT/'runs'/run_name/'direct-training-predecessor-proof.json')
                and proof['status']==status and str(proof['job_id'])==job and proof['code_commit']==historical_source
                and proof['scheduler_state']=='FAILED' and proof['scheduler_exit_code']=='1:0'
                and proof['cleanup_residual']=={} and (proof.get('original_eight_pid_start_times_preserved') is None if job=='3331' else proof['original_eight_pid_start_times_preserved'] is True),
                'Own historical positive terminal/source/cleanup/protected-process proof differs: '+job)
        result,result_sha,_=descriptor(proof['run_result']);cost,cost_sha,_=descriptor(proof['cost'])
        require(str(result['job_id'])==job and result['code_commit']==historical_source
                and result['cleanup_residual']=={} and (result.get('original_eight_pid_start_times_preserved') is None if job=='3331' else result['original_eight_pid_start_times_preserved'] is True)
                and str(cost['job_id'])==job,'Historical native run-result/cost binding differs: '+job)
        item=dict(path=pin['path'],sha256=digest,job_id=int(job),code_commit=historical_source,
            run_result_sha256=result_sha,cost_sha256=cost_sha,scheduler_state=proof['scheduler_state'],scheduler_exit_code=proof['scheduler_exit_code'],
            actual_optimizer_committed=result.get('optimizer_committed'),scheduler_advanced=result.get('scheduler_advanced'),
            failed_unrecorded_partial_optimizer_writes_unknown=result.get('failed_unrecorded_partial_optimizer_writes_unknown'))
        if job=='3322':
            captured=proof['positive_scheduler_observation'];fields=captured['fields']
            require(re.fullmatch('[0-9a-f]{64}',captured['observation_sha256'])
                    and captured['returncode']==0 and captured['stderr']==''
                    and captured['raw_stdout_retained'] is False
                    and all(fields.get(key)==value for key,value in dict(JobId='3322',JobState='FAILED',ExitCode='1:0',
                        EndTime='2026-10-07T15:26:24',RunTime='00:47:52').items()),
                    'Retained positive3322 parsed scheduler observation differs; missing raw stdout is never synthesized')
            item['positive_scheduler_observation']=captured
        if job=='3326':
            captured=proof['positive_scheduler_observation'];fields=captured['fields']
            require(re.fullmatch('[0-9a-f]{64}',captured['observation_sha256'])
                    and captured['returncode']==0 and captured['stderr']==''
                    and captured['raw_stdout_retained'] is False
                    and all(fields.get(key)==value for key,value in dict(JobId='3326',JobState='FAILED',ExitCode='1:0',
                        BatchHost=NODE,NodeList=NODE,EndTime='2026-10-07T16:29:14',RunTime='00:50:13').items()),
                    'Exact byte-bound positive3326 scheduler terminal observation missing')
            item['positive_scheduler_observation']=captured
            observed=proof['direct_completed_outer_observation']
            require(observed['credited_outer']==1 and observed['actual_optimizer_calls_per_rank']==[20]*8
                    and observed['actual_scheduler_steps_per_rank']==[1]*8 and observed['skips_per_rank']==[0]*8
                    and observed['logical_optimizer_writes']==20 and observed['physical_rank_optimizer_calls']==160
                    and proof['model_optimizer_scheduler_checkpoint_available'] is False,
                    '3326 native completed credit observation/checkpoint availability differs')
            item.update(direct_completed_outer_observation=observed,
                model_optimizer_scheduler_checkpoint_available=False,
                checkpoint_inheritance=False,
                scope='Native result null fields remain null; separate real attempt credit is historical and never resumes into V6')
        if job=='3327':
            captured=proof['positive_scheduler_observation'];fields=captured['fields']
            require(re.fullmatch('[0-9a-f]{64}',captured['observation_sha256'])
                    and captured['returncode']==0 and captured['stderr']==''
                    and all(fields.get(key)==value for key,value in dict(JobId='3327',JobState='FAILED',ExitCode='1:0',
                        BatchHost=NODE,NodeList=NODE,EndTime='2026-10-07T16:44:13',RunTime='00:02:02').items()),
                    'Exact byte-bound positive3327 scheduler terminal observation missing')
            item['positive_scheduler_observation']=captured
            item.update(checkpoint_inheritance=False,
                scope='Native null counters remain null; actual pre-actor wrapper version fault never creates inherited credit')
        if job=='3328':
            captured=proof['positive_scheduler_observation'];fields=captured['fields']
            require(captured['observation_sha256']=='b82054dde7ee2e8b66aa59eae21ad9dd524fcbcdab9a4cd939da73f0f073c79d'
                and captured['returncode']==0 and captured['stderr']=='' and captured['raw_stdout_retained'] is True
                and all(fields.get(key)==value for key,value in dict(JobId='3328',JobState='FAILED',ExitCode='1:0',
                    BatchHost=NODE,NodeList=NODE,EndTime='2026-10-07T18:37:07',RunTime='01:44:53').items()),
                'Exact positive3328 native terminal observation differs')
            require(result_sha=='7386e708ee492f618085b9dcede97d22704dd58bc6676abf1a5650fa8a41cbf6'
                and cost_sha=='abd0336c14967af8eac5c01327f427b529ba8f021c7ac3143a2d10b3f3dc00b2',
                'Exact3328 native terminal files differ')
            observed=proof['direct_completed_outer_observation']
            require(observed==dict(credited_outer=2,training_attempts=2,actual_optimizer_calls_per_rank=[40]*8,
                actual_scheduler_steps_per_rank=[2]*8,logical_optimizer_writes=40,physical_rank_optimizer_calls=320),
                'Actual3328 completed2/40 Adam/scheduler2 observation differs')
            require(proof['current_chain_parent_manifest']==dict(path=str(ROOT/'runs/tree-fresh200-train-3328/checkpoints/training_state/rolling-completed/global_step_2/manifest.json'),sha256='067d71ee3c7fa0e00da471bebbd9e8913ec7e5de0b1c98bdb7598d7a022565a4')
                and proof['model_optimizer_scheduler_checkpoint_available'] is True,
                'Exact current-chain3328 completed native parent differs')
            item.update(positive_scheduler_observation=captured,direct_completed_outer_observation=observed,
                current_chain_parent_manifest=proof['current_chain_parent_manifest'],checkpoint_inheritance=True,
                scope='Only rolling2 is explicitly migrated into this same current fresh200 chain; failed third generation is separate')
        if job=='3330':
            captured=proof['positive_scheduler_observation'];fields=captured['fields']
            require(captured['observation_sha256']=='1028ad0189ebac9dd08ce3c56a745cf89f81134178b641a7377595996d5fc156'
                and captured['returncode']==0 and captured['stderr']=='' and captured['raw_stdout_retained'] is True
                and all(fields.get(key)==value for key,value in dict(JobId='3330',JobState='FAILED',ExitCode='1:0',
                    BatchHost=NODE,NodeList=NODE,EndTime='2026-10-07T19:01:32',RunTime='00:02:28').items()),
                'Exact positive3330 startup terminal observation differs')
            require(result_sha=='354324b39e0bf5cd9ca7c6fdc758a5e681081b650743b13562a045aeca45e4b2'
                and cost_sha=='a74e23582aa65ded0f153b90c4fc3a0fd9588fe2149eab5acb476365bbe842ae'
                and result.get('optimizer_committed') is None and result.get('scheduler_advanced') is None,
                'Exact3330 native terminal files/null counters differ')
            require(proof['actual_migration_receipt_present'] is False and proof['actual_initialization_receipt_present'] is False
                and proof['child_training_current_counts'] is None and proof['unpublished_rank_optimizer_calls_unknown'] is True,
                'Missing3330 startup/training evidence must remain unknown')
            item.update(positive_scheduler_observation=captured,actual_migration_receipt_present=False,
                child_training_current_counts=None,checkpoint_inheritance=False,
                scope='Native null counters remain null; startup wrapper fault published no current-chain migration/training receipt')
        if job=='3331':
            captured=proof['positive_scheduler_observation'];fields=captured['fields']
            require(captured['observation_sha256']=='e2f336e19af337150eaa34c268202fb8d71ca3e5d3a9d0905ccd2f0b51f9e634'
                and captured['returncode']==0 and captured['stderr']=='' and captured['raw_stdout_retained'] is True
                and all(fields.get(key)==value for key,value in dict(JobId='3331',JobState='FAILED',ExitCode='1:0',
                    BatchHost=NODE,NodeList=NODE,EndTime='2026-10-07T21:27:00',RunTime='02:20:29').items()),
                'Exact native3331 positive terminal differs')
            require(result_sha=='1d53066e4d88eb50013bed5db8608bbc71123aa030bf65eb8c9f7f65cd7e684f'
                and cost_sha=='2491c56d38c0370dddaa67a97015e12f691075e041fb0137ad776d3fb66fccb5'
                and proof['own_cgroup_process_count_at_terminal_observation']==0
                and proof['native_final_gpu_processes']==result['final_gpu']['processes']==[],
                'Exact native3331 own cleanup/native terminal files differ')
            history=proof['historical_original_background_missing_observation']
            reads=history['observation']['native']['native_original_eight_proc_reads']
            require(history['sha256']=='25ef248763a6693de8135cda88ffaef2754bac697a6df25951ef8a737008f3d9'
                and len(reads)==8 and {row['pid'] for row in reads}=={1680782,1680783,1680784,1680785,1696443,1696444,1696445,1696446}
                and all(row['error_type']=='FileNotFoundError' and row['error_errno']==2 and row['path_exists'] is False for row in reads)
                and proof['background_identity_disposition']=='ORIGINAL_BACKGROUND_MISSING_AT_FAILED3331_CAUSE_UNKNOWN',
                'Original8 missing identity must remain explicit native unknown, never preserved=True')
            from current_chain_runtime import PARENT_MANIFEST,PARENT_SHA
            require(proof['current_chain_parent_manifest']==dict(path=str(PARENT_MANIFEST),sha256=PARENT_SHA)
                and proof['direct_completed_outer_observation']['credited_outer']==4
                and proof['direct_completed_outer_observation']['actual_optimizer_calls_per_rank']==[80]*8,
                'Exact current-chain completed rolling4 parent differs')
            item.update(positive_scheduler_observation=captured,original_eight_pid_start_times_preserved=None,
                background_identity_disposition=proof['background_identity_disposition'],own_cgroup_process_count_at_terminal_observation=0,
                direct_completed_outer_observation=proof['direct_completed_outer_observation'],current_chain_parent_manifest=proof['current_chain_parent_manifest'],
                checkpoint_inheritance=True,scope='Explicit same current chain3331 rolling4 migration; fifth writes unknown; original background disappearance is separately archived')
        if job in HISTORICAL_JOURNALS:
            journal_name,prior_manifest_sha=HISTORICAL_JOURNALS[job]
            journal,journal_sha,_=descriptor(proof['once_journal'])
            require(proof['once_journal']['path']==str(ROOT/'instruct300'/journal_name)
                    and journal['status']=='SUBMITTED' and journal['job_id']==int(job) and journal['cluster_name']==CLUSTER
                    and journal['manifest_sha256']==prior_manifest_sha,
                    'Historical proof must bind exact submitted once journal and original source manifest: '+job)
            item.update(once_journal=dict(path=proof['once_journal']['path'],sha256=journal_sha),prior_manifest_sha256=prior_manifest_sha)
        closure[job]=item
    return closure['3331'],closure

def submit(package_path,manifest_sha,review_path,review_sha,script_path,script_sha,predecessor_proof,predecessor_proof_sha,*,run=subprocess.run,hostname=None):
    require((socket.gethostname() if hostname is None else hostname).split('.')[0]==NODE,'Dispatcher only runs on own h100-9')
    from calibration_contract import forbid_recovery_environment
    forbid_recovery_environment(os.environ)
    require(not any(k in os.environ for k in ('TREE_PPO_CALIBRATION_ONLY','TREE_FRESH_INITIALIZATION_ONLY')),'Production dispatch forbids inherited zero-only flags')
    verify_management_scope(os.environ,hostname)
    environment=local_environment(os.environ);identity=cluster_identity(run,environment,management=True)
    package_path=Path(package_path)
    require(package_path.is_absolute() and package_path.is_relative_to(ROOT/'instruct300/candidates'),'Own candidate directory required')
    manifest,_,_=read_bound(package_path/'manifest.json',manifest_sha)
    require(manifest['scope']==CHAIN_SCOPE and manifest['execution_mode']==DIRECT_MODE and manifest['packet_revision']==9
            and manifest['stage']=='fresh_mu_train200' and manifest['bootstrap_thresholds_accepted'] is False
            and manifest['production_thresholds_accepted'] is False,'Only explicit human-authorized direct200 package may dispatch')
    _,pins=production_inputs(manifest,manifest_sha256=manifest_sha,dispatch_review_descriptor=dict(path=str(review_path),sha256=review_sha))
    release,_,_=read_bound(review_path,review_sha)
    path=Path(manifest['dispatch']['once_intent'])
    require(path.parent==ROOT/'instruct300' and re.fullmatch(INTENT_RE,path.name)
            and manifest['dispatch']['cluster_name']==CLUSTER and release['dispatch_intent_path']==str(path),'Fixed reviewed NEW production intent/cluster required')
    for name,expected in manifest['files'].items():
        relative=Path(name);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe production package member')
        read_regular_bound(package_path/relative,expected)
    require(Path(__file__).resolve()==(package_path/'submit_once.py').resolve(),'Dispatcher is not the reviewed package source')
    script=read_regular_bound(script_path,script_sha)
    require(release['dispatch_renderer_sha256']==manifest['files']['render_dispatch.py']
            and release['dispatch_submitter_sha256']==manifest['files']['submit_once.py'],'Exact direct dispatcher/renderer source not bound')
    # Verify rendered actual argv/script exactly, not substring directives.
    from render_dispatch import render
    wall=manifest['dispatch']['train_wall_hours']
    require(script==render(package_path,manifest_sha,review_path,review_sha,wall).encode(),'Submitted script differs from exact reviewed production renderer')
    predecessor,proof_closure=predecessor_ended(manifest,predecessor_proof,predecessor_proof_sha)
    lock_path=ROOT/'instruct300/FRESH200_TRAIN_DISPATCH.lock'
    fd=os.open(lock_path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode),'Dispatch lock must be ordinary file')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=prior_journals_clear(path,run,environment,proof_closure);queue=queue_clear(run,environment)
        # The immutable bytes below, not the filename, are the submitted program.
        submit_argv=['sbatch','--parsable']
        initial=dict(status='SUBMITTING',execution_mode=DIRECT_MODE,scope=CHAIN_SCOPE,
            current_chain_resume=manifest['current_chain_resume'],code_commit=SOURCE,independent_acceptance=False,training_accepted=False,
            production_thresholds_accepted=False,original_sampler_actor_gate_accepted=False,
            cluster_name=CLUSTER,local_cluster_identity=identity,node_list=NODE,
            manifest_sha256=manifest_sha,review_sha256=review_sha,dispatch_intent_path=str(path),train_wall_hours=wall,
            dispatch_script_sha256=script_sha,submit_argv=submit_argv,
            submitted_input_sha256=hashlib.sha256(script).hexdigest(),submitted_input_bytes=len(script),
            submitted_input_base64=base64.b64encode(script).decode('ascii'),submit_mode='verified_bytes_via_stdin',
            selected_script_path=str(script_path),production_input_pins=pins,prior_accounting=prior,
            predecessor_terminal_cleanup=predecessor,terminal_proof_closure=proof_closure,
            queue_stdout=queue,start_epoch=time.time(),stdout=None,stderr=None,job_id=None)
        create_journal(path,initial)
        result=None
        try:
            result=run(submit_argv,input=script,env=environment,capture_output=True,timeout=60)
            job,parsable=parse_submission(result)
            final=dict(initial,status='SUBMITTED',job_id=job,run_id='tree-current200-continue-'+str(job),
                stdout=submission_output(result.stdout),stderr=submission_output(result.stderr),returncode=result.returncode,parsable_stdout=parsable,
                submitted_result_stdout_raw=timeout_output(result.stdout),submitted_result_stderr_raw=timeout_output(result.stderr),end_epoch=time.time())
        except BaseException as error:
            final=dict(initial,status='UNKNOWN',error_type=type(error).__name__,error=str(error),end_epoch=time.time(),
                timeout_stdout=timeout_output(getattr(error,'stdout',None)),timeout_stderr=timeout_output(getattr(error,'stderr',None)),
                submitted_returncode=None if result is None else result.returncode,
                submitted_result_stdout_raw=timeout_output(None if result is None else result.stdout),
                submitted_result_stderr_raw=timeout_output(None if result is None else result.stderr))
            finalize_journal(path,final)
            raise
        finalize_journal(path,final);return final
    finally:os.close(fd)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('package','manifest-sha','release-path','release-sha','script','script-sha','predecessor-proof','predecessor-proof-sha'):p.add_argument('--'+key,required=True)
    args=p.parse_args();print(json.dumps(submit(args.package,args.manifest_sha,args.release_path,args.release_sha,args.script,args.script_sha,args.predecessor_proof,args.predecessor_proof_sha),sort_keys=True))
