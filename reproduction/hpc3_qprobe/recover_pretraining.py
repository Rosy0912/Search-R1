"""Once-only recovery of the actual pretraining allocation failure 717554.

Use a fresh target, preserve the terminal source/logs, and reject any prior
actor/teacher pipeline work. No live job, shared asset or old binding is edited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from reproduction.hpc3_qprobe.control import prepare, link, publish
from reproduction.common8.native_allocation import fields

ROOT=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
FAILED=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-20261014')
TARGET=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery717554-20261014')
FAILED_COMMIT='300734e78705936b9b3885248ac3ffe7f8cbd3aa'
JOB='717554'


def validate_original_submission(prior):
    if JOB=='717554':
        accepted=prior.get('status')=='REPLACEMENT_HELD' and prior.get('source_commit')==FAILED_COMMIT
    else:
        accepted=prior.get('returncode')==0 and prior.get('commit')==FAILED_COMMIT
    if prior.get('job')!=JOB or not accepted:raise ValueError('Missing actual own submission binding')


def select_failed_job(job):
    global JOB,FAILED,TARGET,FAILED_COMMIT
    if job=='717554':return
    if job not in ('717607','717637','717698','717765'):raise ValueError('Only actually observed own startup failures are supported')
    JOB=job
    parent={'717607':'717554','717637':'717607','717698':'717637','717765':'717698'}[job]
    FAILED=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery'+parent+'-20261014')
    TARGET=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery'+job+'-20261014')
    FAILED_COMMIT={'717607':'1552de146c2dd44e4fd40eb94b27839ac8648ef1',
        '717637':'f30895d421e190bdb13334e9ad6f718cb394d787',
        '717698':'7bf7a30fd530f0741c4deadf7d89788ca0070c0f',
        '717765':'1e0ac4e588faa7811f0aeabb1c9b3a9e28f0012e'}[job]


def diagnostic_gate(commit):
    if JOB not in ('717607','717637','717698','717765'):return None
    root=ROOT/'preparation/gpu-inventory717607'
    receipt=json.loads((root/'submission.json').read_text())
    if receipt['returncode']!=0:raise ValueError('Actual GPU diagnostic submission failed')
    proof=json.loads((root/('job-'+receipt['job'])/'accepted.json').read_text())
    detail=fields(proof['actual_job_detail'])
    original=ROOT/'source'/receipt['commit']
    repaired=ROOT/'source'/commit
    if proof['accepted'] is not True or proof['job_id']!=receipt['job'] or proof['uid']!=14090 or proof['loaded_model'] is not False:
        raise ValueError('Actual single GPU XML/CUDA diagnostic not accepted')
    if detail['Command']!=str(original/'reproduction/hpc3_qprobe/gpu_inventory_diagnostic.sh'):
        raise ValueError('Actual GPU diagnostic source entry differs')
    name='reproduction/common8/native_allocation.py'
    if hashlib.sha256((original/name).read_bytes()).digest()!=hashlib.sha256((repaired/name).read_bytes()).digest():
        raise ValueError('Repaired actual GPU query implementation differs from successful diagnostic')
    state=subprocess.check_output(['sacct','-nP','-j',receipt['job'],'-X','--format=JobID,State,ExitCode,UID'],text=True,timeout=30).strip()
    if state!=receipt['job']+'|COMPLETED|0:0|14090':raise ValueError('Actual diagnostic has not completed successfully')
    return dict(receipt=receipt,accepted=proof,actual_terminal=state)


def read_terminal_job(preserved_detail):
    result=subprocess.run(['scontrol','show','job','-d',JOB],capture_output=True,text=True,timeout=30)
    if result.returncode==0:
        return result.stdout,dict(kind='live_terminal_job',checked_unix=time.time())
    # Completed jobs age out of slurmctld quickly. Its missing record is not
    # terminal evidence: require fresh accounting plus the preserved actual
    # owner's detailed job birth, rather than treating invisibility as failure.
    accounting=subprocess.check_output(['sacct','-nP','-j',JOB,'-X',
        '--format=JobID,State,ExitCode,User,UID,Account,QOS,Start,End,NodeList'],text=True,timeout=30)
    rows=[line.split('|') for line in accounting.splitlines() if line.strip()]
    selected=[row for row in rows if row[0]==JOB]
    if len(selected)!=1:raise ValueError('Actual terminal accounting row missing or duplicated')
    row=selected[0];old=fields(preserved_detail)
    expected=[JOB,'FAILED',old['ExitCode'],'jxu957','14090','jxu957','formal-user',old['StartTime'],old['EndTime'],old['NodeList']]
    if row!=expected:raise ValueError('Fresh actual terminal accounting differs from preserved own job')
    return preserved_detail,dict(kind='fresh_accounting_and_preserved_owner_birth',
        accounting=accounting,scontrol_exit=result.returncode,scontrol_stderr=result.stderr,checked_unix=time.time())


def terminal():
    if os.getuid()!=14090:raise ValueError('Actual jxu957 owner required')
    evidence=FAILED/('failed-job-'+JOB)
    saved=evidence/'terminal-and-files.json'
    if not saved.is_file() or saved.is_symlink() or saved.stat().st_uid!=14090:
        raise ValueError('Actual owner failure evidence required before recovery')
    preserved=json.loads(saved.read_text())
    detail,readback=read_terminal_job(preserved['terminal']['stdout'])
    actual=fields(detail)
    expected=dict(JobId=JOB,JobState='FAILED',UserId='jxu957(14090)',Account='jxu957',
        QOS='formal-user',WorkDir=str(FAILED),Restarts='0',
        Command=str(ROOT/'source'/FAILED_COMMIT/'reproduction/hpc3_qprobe/train.sh'))
    if any(actual.get(k)!=v for k,v in expected.items()):raise ValueError('Not the actual failed own pretraining job')
    prior_path=ROOT/'preparation/account-transition-submission.json' if JOB=='717554' else FAILED/'preparation/training-submission.json'
    prior=json.loads(prior_path.read_text())
    validate_original_submission(prior)
    run=json.loads((FAILED/'logs/run-manifest.json').read_text())
    if run['source_commit']!=FAILED_COMMIT or run['completed_updates']!=0 or run['training_world_size']!=4:
        raise ValueError('Actual failed run differs or already trained')
    forbidden=['training-launch-claim.json','search-g1-common8-training.log',
               'actual-updates.jsonl','actual-attempts.jsonl']
    if JOB not in ('717698','717765'):forbidden+=['model-initial.log']
    if JOB not in ('717637','717698','717765'):forbidden+=['process-ledger.json','native-allocation-admission.json','pipeline-status.json']
    for name in forbidden:
        if (FAILED/'logs'/name).exists():raise ValueError('Pipeline work already claimed; use full-state recovery instead')
    if JOB in ('717637','717698','717765'):
        cleanup=json.loads((FAILED/'logs/cleanup-status.json').read_text())
        ledger=json.loads((FAILED/'logs/process-ledger.json').read_text())
        stage=json.loads((FAILED/'logs/pipeline-status.json').read_text())
        if cleanup['cleaned'] is not True or cleanup['run_id']!=run['run_id'] or ledger['run_id']!=run['run_id']:
            raise ValueError('Actual own synthetic preflight cleanup not accepted')
        names={'gpu-preflight'} if JOB=='717637' else ({'gpu-preflight','model-initial'} if JOB=='717698' else {'gpu-preflight','model-initial','retrieval-stage','retriever'})
        if set(ledger['jobs'])!=names or any(ledger['jobs'][name]['state']!='STOPPED' for name in names):
            raise ValueError('Other pipeline work or live owned process exists')
        expected_error="AssertionError('retriever exited during endpoint readiness')" if JOB=='717765' else "AssertionError('"+('gpu-preflight' if JOB=='717637' else 'model-initial')+" failed; see project log')"
        if stage['stage']!='FAILED' or stage['error']!=expected_error:
            raise ValueError('Not the observed startup failure')
        if JOB=='717637':
            preflight=(FAILED/'logs/gpu-preflight.log').read_text()
            if 'FileNotFoundError' not in preflight or '/cache/triton/' not in preflight:
                raise ValueError('Actual shared-cache compilation failure absent')
        elif JOB=='717698':
            if ledger['jobs']['gpu-preflight'].get('exit_code')!=0 or ledger['jobs']['model-initial'].get('exit_code')!=1:
                raise ValueError('Actual successful mechanics and failed model-only smoke required')
            smoke=(FAILED/'logs/model-initial.log').read_text()
            if 'COMMON8_MODEL_PARITY_REJECTED=' not in smoke or 'Actual initial actor/reference/vLLM parity rejected' not in smoke:
                raise ValueError('Not the actual pre-Adam cross-backend model smoke failure')
            if run['behavior_probability_policy']!='sampling_journal_no_extra_hf_forward' or run['precision_profile']!='native4_bf16_varlen_fp32_master':
                raise ValueError('Failed model smoke does not have the original bound BF16 journal policy')
        else:
            for name in ('gpu-preflight','model-initial','retrieval-stage'):
                if ledger['jobs'][name].get('exit_code')!=0:raise ValueError('Previous actual completed startup components differ')
            for rank in range(4):
                model=json.loads((FAILED/'logs/model-smoke-initial'/('rank-'+str(rank)+'.json')).read_text())
                if model['accepted'] is not True or model['actual_optimizer_writes']!=0 or model['actual_grpo_updates']!=0:
                    raise ValueError('Actual zero-update model evidence required before reset')
            retrieval=(FAILED/'logs/retriever.log').read_text()
            if 'actual_gpu_uuids=cuda_uuids()' not in retrieval or 'Cannot obtain actual CUDA device properties' not in retrieval:
                raise ValueError('Not the actual retriever default-eight CUDA probe failure')
    log=FAILED/'logs'/('training-'+JOB+'.log')
    text=log.read_text()
    component={'717637':'gpu-preflight failed','717698':'model-initial failed','717765':'retriever exited during endpoint readiness'}.get(JOB)
    known=('validate_native_four' in text and 'AssertionError' in text) if JOB=='717554' else (component in text if component else ('--query-gpu=index,uuid,memory.total,minor_number' in text and 'CalledProcessError' in text))
    if not known:
        raise ValueError('Known actual allocation failure absent')
    for name,record in preserved['files'].items():
        if name.startswith('logs/') and hashlib.sha256((FAILED/name).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Preserved terminal log changed')
    return dict(actual=actual,detail=detail,terminal_readback=readback,prior=prior,failed_run_id=run['run_id'],checked_unix=time.time())


def fresh(commit):
    actual=terminal()
    diagnostic=diagnostic_gate(commit)
    intent=ROOT/'preparation'/('recovery'+JOB+'-prepare-intent.json')
    if intent.exists() or TARGET.exists():raise ValueError('Recovery preparation already attempted; inspect instead of replaying')
    proof=json.loads((ROOT/'preparation'/('source-'+commit+'-verification.json')).read_text())
    source=ROOT/'source'/commit
    if proof['status']!='PUBLISHED_SOURCE_FULL_SHA_PASS' or proof['commit']!=commit:
        raise ValueError('Repair must be published and verified')
    for name,sha in proof['files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=sha:raise ValueError('Repair bytes differ: '+name)
    os.umask(0o007)
    publish(intent,dict(commit=commit,target=str(TARGET),failure=actual,epoch=time.time()))
    TARGET.mkdir(mode=0o770)
    prepare(TARGET);link(TARGET/'repo',source)
    (TARGET/'source').mkdir();link(TARGET/'source'/commit,source)
    link(TARGET/'question-features',ROOT/'question-features')
    for name in ('features-ready.json','official-model-manifest.json'):
        link(TARGET/'preparation'/name,ROOT/'preparation'/name)
    policy=json.loads((FAILED/'preparation/personal-storage-authorization.json').read_text())
    if policy['budget_bytes']!=2*10**12 or policy['scope']!='jxu957_native4_qprobe_only':raise ValueError('Actual user budget differs')
    publish(TARGET/'preparation/personal-storage-authorization.json',policy)
    publish(TARGET/'preparation/recovery-source-binding.json',dict(source_commit=commit,source=str(source),
        target=str(TARGET),failed_job=JOB,failure=actual,source_files=len(proof['files']),
        training_world_size=4,actor_micro_batch_per_rank=4,feature_cache_reused=True,actual_gpu_query_diagnostic=diagnostic,
        optimizer_writes_before_recovery=0,teacher_labels_before_recovery=0))
    return dict(status='FRESH_FAILED_PRETRAINING_RECOVERY_PREPARED',target=str(TARGET),commit=commit)


def submit(commit):
    actual=terminal()
    diagnostic_gate(commit)
    binding=json.loads((TARGET/'preparation/recovery-source-binding.json').read_text())
    source=ROOT/'source'/commit
    if binding['source_commit']!=commit or (TARGET/'repo').resolve()!=source or (TARGET/'source'/commit).resolve()!=source:
        raise ValueError('Fresh recovery source binding differs')
    proof=json.loads((ROOT/'preparation'/('source-'+commit+'-verification.json')).read_text())
    if proof['commit']!=commit or proof['status']!='PUBLISHED_SOURCE_FULL_SHA_PASS':raise ValueError('Repair publication differs')
    for name,sha in proof['files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=sha:raise ValueError('Actual submitted repair bytes differ: '+name)
    claim=TARGET/'preparation/training-submit-claim.json';receipt=TARGET/'preparation/training-submission.json'
    if claim.exists() or receipt.exists():raise ValueError('Recovery submission already attempted; never duplicate')
    jobs=subprocess.check_output(['squeue','-h','-u','jxu957','-o','%i|%j|%T'],text=True,timeout=30)
    if any('qprobe' in line for line in jobs.splitlines()):raise ValueError('Existing own qprobe job requires inspection')
    from search_g1.qprobe_features import FeatureCache
    features=FeatureCache(TARGET/'question-features');features.verify_all()
    if features.manifest['count']!=169028:raise ValueError('Actual feature cache incomplete')
    command=['sbatch','--parsable','--account=jxu957','--qos=formal-user','--partition=acd_ue',
        '--nodes=1','--ntasks=1','--gres=gpu:4','--cpus-per-task=48','--mem=512G','--time=7-00:00:00',
        '--job-name=qprobe-warmup200','--export=NONE','--chdir='+str(TARGET),
        '--output='+str(TARGET/'logs/training-%j.log'),'--error='+str(TARGET/'logs/training-%j.log'),
        str(source/'reproduction/hpc3_qprobe/train.sh'),str(TARGET),commit]
    test=subprocess.run(command[:1]+['--test-only']+command[1:],capture_output=True,text=True,timeout=30)
    publish(TARGET/'preparation/training-test-only.json',dict(command=command,exit=test.returncode,stdout=test.stdout,stderr=test.stderr,epoch=time.time()))
    if test.returncode:raise ValueError('Actual recovery scheduler preflight failed')
    publish(claim,dict(commit=commit,command=command,recovery_of_failed_job=JOB,failure=actual,epoch=time.time()))
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    value=dict(commit=commit,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,epoch=time.time(),recovery_of_failed_job=JOB)
    if result.returncode==0:
        value['job']=result.stdout.strip().split(';')[0]
        if not value['job'].isdecimal():raise ValueError('Invalid actual recovery job')
    publish(receipt,value)
    if result.returncode:raise RuntimeError('Recovery submission failed; inspect saved receipt')
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','submit'));parser.add_argument('--commit',required=True)
    parser.add_argument('--failed-job',choices=('717554','717607','717637','717698','717765'),default='717554')
    args=parser.parse_args();select_failed_job(args.failed_job)
    print(json.dumps({'prepare':fresh,'submit':submit}[args.mode](args.commit)),flush=True)
