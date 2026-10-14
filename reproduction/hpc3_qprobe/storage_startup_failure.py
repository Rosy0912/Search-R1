"""Archive/reverify only the actual zero-pipeline storage failure 718288."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from reproduction.hpc3_qprobe.control import publish

ROOT = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
BASE = ROOT.parent/'qprobe-teacher-warmup200-4gpu-jxu957-recovery717938-20261014'
SOURCE = 'e715462c55af0b836aee47432ca051cc02f5a5d9'
JOB = '718288'
TERMINAL = [JOB,'FAILED','1:0','jxu957','14090','jxu957','formal-user','emergency_acd','2026-10-14T18:16:10','2026-10-14T18:19:44','ACD1-15']
DENIED = [str(ROOT.parent/'topd-20261008/runtime/logs/topd-hpc3-713717/evidence/checkpoints'/part/'global_step_200') for part in ('algorithm','actor')]
NAMES = ['logs/training-'+JOB+'.log','preparation/training-submit-claim.json',
         'preparation/training-submission.json','preparation/training-test-only.json',
         'preparation/recovery-source-binding.json','preparation/personal-storage-authorization.json']


def validate_terminal(raw):
    rows = [line.split('|') for line in raw.splitlines() if line.strip()]
    if rows != [TERMINAL]:
        raise ValueError('Exact owned terminal storage failure required')


def verify():
    if os.getuid() != 14090:
        raise ValueError('Actual JXU owner required')
    command = ['sacct','-nP','-j',JOB,'-X','--format=JobID,State,ExitCode,User,UID,Account,QOS,Partition,Start,End,NodeList']
    raw = subprocess.check_output(command,text=True,timeout=30)
    validate_terminal(raw)
    claim = json.loads((BASE/NAMES[1]).read_text())
    prior = json.loads((BASE/NAMES[2]).read_text())
    binding = json.loads((BASE/NAMES[4]).read_text())
    entry = str(ROOT/'source'/SOURCE/'reproduction/hpc3_qprobe/train.sh')
    launch = claim['command']
    if prior.get('job') != JOB or prior.get('returncode') != 0 or prior.get('commit') != SOURCE or claim.get('commit') != SOURCE or binding.get('source_commit') != SOURCE or binding.get('failed_job') != '717938':
        raise ValueError('Original immutable storage-failure submission differs')
    if launch[0] != 'sbatch' or launch[-3:] != [entry,str(BASE),SOURCE] or any(value not in launch for value in ('--account=jxu957','--qos=formal-user','--partition=emergency_acd','--gres=gpu:4','--cpus-per-task=48','--mem=512G','--exclude=ACD1-67','--chdir='+str(BASE))):
        raise ValueError('Original native4 entry/owner/resources differ')
    for name in ('run-manifest.json','pipeline-status.json','process-ledger.json','model-initial.log','native-allocation-admission.json','training-launch-claim.json','search-g1-common8-training.log','actual-updates.jsonl','actual-attempts.jsonl','storage-budget-admission.json'):
        if (BASE/'logs'/name).exists():
            raise ValueError('Pipeline work exists; zero-startup recovery forbidden')
    for name in ('teacher-warmup-cache','checkpoints','artifacts/question-probe','logs/model-smoke-initial'):
        path = BASE/name
        if path.exists() and any(path.iterdir()):
            raise ValueError('Actual actor/teacher state requires full-state recovery')
    text = (BASE/NAMES[0]).read_text()
    if 'PERSONAL_STORAGE_USAGE_NOT_VERIFIED' not in text or 'storage_admission(base,model)' not in text or any("du: cannot read directory '"+path+"': Permission denied" not in text for path in DENIED):
        raise ValueError('Not the actual pre-model storage permission failure')
    if text.count('Permission denied') != 2:
        raise ValueError('Unexpected extra storage error requires diagnosis')
    proof = json.loads((ROOT/'preparation'/('source-'+SOURCE+'-verification.json')).read_text())
    if proof['status'] != 'PUBLISHED_SOURCE_FULL_SHA_PASS' or proof['commit'] != SOURCE:
        raise ValueError('Original full source publication missing')
    for name, sha in proof['files'].items():
        path = ROOT/'source'/SOURCE/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError('Original immutable source changed')
    files = {}
    for name in NAMES:
        path = BASE/name
        if path.is_symlink() or not path.is_file() or path.stat().st_uid != 14090:
            raise ValueError('Original owned evidence missing or aliased')
        files[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,uid=14090)
    return dict(kind='ACTUAL_OWNER_PRE_PIPELINE_STORAGE_FAILURE',job=JOB,base=str(BASE),source_commit=SOURCE,
                actual_accounting=raw,actual_command=command,original_claim=claim,prior=prior,
                files=files,pipeline_manifest_absent=True,actor_teacher_state_absent=True,
                verified_by_uid=os.getuid(),checked_unix=time.time())


def archive():
    value = verify()
    directory = BASE/('failed-job-'+JOB)
    if directory.exists():
        raise ValueError('Archive already attempted; inspect rather than replay')
    directory.mkdir(mode=0o770)
    for name, record in value['files'].items():
        target = directory/name
        target.parent.mkdir(mode=0o770,parents=True,exist_ok=True)
        shutil.copyfile(BASE/name,target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Failure copy SHA differs')
    publish(directory/'terminal-and-files.json',value)
    return value


def preserved():
    path = BASE/('failed-job-'+JOB)/'terminal-and-files.json'
    if path.is_symlink() or not path.is_file() or path.stat().st_uid != 14090:
        raise ValueError('Actual owner archive required before recovery')
    saved = json.loads(path.read_text())
    now = verify()
    for name in ('kind','job','base','source_commit','actual_accounting','original_claim','prior','files','pipeline_manifest_absent','actor_teacher_state_absent','verified_by_uid'):
        if saved[name] != now[name]:
            raise ValueError('Archived exact failure changed')
    for name,record in now['files'].items():
        if hashlib.sha256((path.parent/name).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Preserved failure copy changed')
    return now


if __name__ == '__main__':
    print(json.dumps(archive()),flush=True)
