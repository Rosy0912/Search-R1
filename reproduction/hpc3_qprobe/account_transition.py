"""Human-authorized transfer of one unstarted qprobe job to jxu957.

Keep the submitted training source immutable. A replacement is initially held;
it is released only after the old, still-unstarted job is terminal CANCELLED.
Receipts make every submission and cancellation once-only and inspectable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from reproduction.hpc3_qprobe.schedule_pending import scheduler_fields, once

BASE = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
OLD = '716883'
ORIGINAL_SOURCE = 'd4cc3dcbb4f2b2b93ac42fba11f72a4009480bd3'
SOURCE = None
OWNER_OBSERVATION = None
TARGET = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-20261014')
ORIGIN_UID = 21930
TARGET_UID = 14090
TARGET_ACCOUNT = "jxu957"
SLURM_ACCOUNT = "d_huiqimai_team"
TARGET_GROUPS = {21919, 10034}
AUTHORIZATION = dict(
    original_request='ok，换一个账号，我记得本地还有另一个账号吧？',
    clarification='HPC3 登录账号（解决训练资源额度）',
    group_access_authorization='chmod -R 770 /data/user/jxu957/experiments/；mou176和jxu957共同读写',
    message_timestamp='not exposed by client; explicit human messages in current turn',
    latest_hardware_request='不用8卡啊，这个4卡就行啊；每张卡处理的样本多一倍',
    scope='replace only unstarted qprobe 716883 with fresh native4 jxu957; warmup plus 200 outers; per-rank micro4, unchanged global512/G5/mini128',
    running_jobs_in_scope=False, goal_cancelled=False)


def cli(args):
    r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, timeout=60)
    if r.returncode:
        raise RuntimeError(json.dumps(dict(command=args, exit=r.returncode,
                                          stdout=r.stdout, stderr=r.stderr)))
    return r.stdout


def job(job_id):
    # HPC3 hides other users' jobs. Each side supplies an actual fresh readback
    # from its own authenticated login; never infer cancellation from invisibility.
    if job_id==OLD and os.getuid()!=ORIGIN_UID:
        path=Path(OWNER_OBSERVATION) if OWNER_OBSERVATION else BASE/'preparation/account-transition-retired.json'
        if not path.is_file() or path.is_symlink() or path.stat().st_uid!=ORIGIN_UID:
            raise ValueError('Actual original-owner readback required')
        if path.parent.resolve()!=BASE/'preparation':
            raise ValueError('Owner observation must remain in this run preparation namespace')
        row=json.loads(path.read_text())
        if time.time()-row['epoch']>300:
            raise ValueError('Original-owner job readback is stale')
        detail=row['actual'];value=scheduler_fields(detail)
        if value.get('JobId')!=OLD or value.get('UserId')!='mou176('+str(ORIGIN_UID)+')':
            raise ValueError('Owner job readback differs')
        return value,detail
    if job_id!=OLD and os.getuid()!=TARGET_UID:
        path=BASE/'preparation/account-transition-submission.json'
        if path.stat().st_uid!=TARGET_UID:
            raise ValueError('Replacement must be observed by its actual owner')
        row=json.loads(path.read_text())
        if row['job']!=job_id or time.time()-row['epoch']>120:
            raise ValueError('Replacement owner observation is stale or differs')
        return scheduler_fields(row['actual']),row['actual']
    detail = cli(['scontrol', 'show', 'job', '-d', job_id])
    return scheduler_fields(detail), detail


def untouched():
    if (BASE/'repo').resolve() != BASE/'source'/ORIGINAL_SOURCE:
        raise ValueError('Original training source binding differs')
    submission = json.loads((BASE/'preparation/training-submission.json').read_text())
    if submission['job'] != OLD or submission['commit'] != ORIGINAL_SOURCE or submission['returncode'] != 0:
        raise ValueError('Original successful submission receipt differs')
    if (BASE/'logs/run-manifest.json').exists() or (BASE/'logs'/('training-'+OLD+'.log')).exists():
        raise ValueError('Original native training already began; leave it alone')


def validate_old(value, uid=ORIGIN_UID):
    expected = dict(JobId=OLD, JobState='PENDING', AllocTRES='(null)',
                    JobName='qprobe-warmup200', UserId='mou176('+str(uid)+')',
                    Account='mou176', QOS='normal', NumCPUs='96', NumTasks='1',
                    MinMemoryNode='512G', TresPerNode='gres/gpu:8',
                    TimeLimit='7-00:00:00', OverSubscribe='NO', Restarts='0',
                    Command=str(BASE/'source'/ORIGINAL_SOURCE/'reproduction/hpc3_qprobe/train.sh'))
    if any(value.get(k) != v for k, v in expected.items()) or value.get('RunTime') != '00:00:00':
        raise ValueError('Old job is not this never-started immutable pending job')


def needed(path, bits):
    st = path.stat()
    if st.st_uid == TARGET_UID:
        available = (st.st_mode >> 6) & 7
    elif st.st_gid in TARGET_GROUPS:
        available = (st.st_mode >> 3) & 7
    else:
        available = st.st_mode & 7
    return available & bits != bits


def access():
    if os.getuid() != ORIGIN_UID:
        raise ValueError('Access preparation requires original file owner')
    untouched(); validate_old(job(OLD)[0])
    receipt = BASE/'preparation/account-transition-access.json'
    if receipt.exists():
        return json.loads(receipt.read_text())
    intent = BASE/'preparation/account-transition-access-intent.json'
    if intent.exists():
        raise ValueError('Inspect incomplete access preparation before continuing')
    reused = Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008')
    topd = Path('/data/user/jxu957/experiments/topd-20261008/runtime/assets')
    roots = [BASE/'source'/SOURCE, BASE/'env', BASE/'data', BASE/'preparation',
             BASE/'question-features', topd/'qwen2.5-3b-instruct-aa8e725',
             topd/'common-instruct-0b002b', reused/'source/env/lib/python3.10/site-packages',
             reused/'runtime/retriever-python310-candidate', reused/'runtime/gpu-faiss-alignment-candidate',
             reused/'source/assets-root/assets/e5-base-v2',
             reused/'source/assets-root/assets/wiki/e5_Flat.index',
             reused/'source/assets-root/assets/wiki/wiki-18.jsonl',
             Path('/share/anaconda3/envs/video_dit/include/python3.10')]
    requirements = {}
    for root in roots:
        root = root.resolve(strict=True)
        for parent in root.parents:
            requirements[parent] = requirements.get(parent, 0) | 1
        if root.is_file():
            requirements[root] = requirements.get(root, 0) | 4
            continue
        for directory, directories, files in os.walk(str(root), followlinks=False):
            p = Path(directory)
            requirements[p] = requirements.get(p, 0) | 5
            for name in files:
                original=Path(directory)/name
                if not original.exists():continue  # Preserve unused dangling environment links.
                p = original.resolve(strict=True)
                requirements[p] = requirements.get(p, 0) | (5 if p.stat().st_mode & 0o111 else 4)
    # The target must measure the shared 1TB experiment budget with du. Grant
    # directory metadata access only; private files outside the reused assets
    # receive no read/write ACL and no ownership/content changes.
    for directory, _, _ in os.walk('/data/user/jxu957/experiments',followlinks=False):
        p=Path(directory);requirements[p]=requirements.get(p,0)|5
    writable = set()
    for name in ('', 'logs', 'preparation', 'tmp', 'cache'):
        root = BASE/name
        root.mkdir(parents=True, exist_ok=True)
        writable.update(Path(d) for d, _, _ in os.walk(str(root), followlinks=False)) if name else writable.add(BASE)
    # Only the run's output directories receive write/default ACLs. Shared
    # models, environment, retrieval and completed features remain read-only.
    changes = []
    for p, bits in sorted(requirements.items(), key=lambda item: str(item[0])):
        if p in writable or not needed(p, bits):
            continue
        if p.stat().st_uid != ORIGIN_UID:
            raise ValueError('Required inaccessible path is not owned: '+str(p))
        perms = ''.join(c if bits & b else '-' for c, b in zip('rwx', (4, 2, 1)))
        changes.append(dict(path=str(p), acl='u:'+str(TARGET_UID)+':'+perms,
                            before=cli(['getfacl', '-cp', str(p)])))
    for p in sorted(writable):
        if p.stat().st_uid != ORIGIN_UID:
            raise ValueError('Writable run directory belongs to another user')
        changes.append(dict(path=str(p), acl='u:'+str(TARGET_UID)+':rwx,d:u:'+str(TARGET_UID)+':rwx,d:u:'+str(ORIGIN_UID)+':rwx',
                            before=cli(['getfacl', '-cp', str(p)])))
    once(intent, dict(authorization=AUTHORIZATION, observed_unix=time.time(), changes=changes,
                      shared_file_content_changes=False, ssh_auth_changes=False))
    for item in changes:
        cli(['setfacl', '-m', item['acl'], item['path']])
        item['after'] = cli(['getfacl', '-cp', item['path']])
    out = dict(status='TARGET_ACCESS_PREPARED', epoch=time.time(), changes=changes,
               authorization=AUTHORIZATION, target_uid=TARGET_UID)
    once(receipt, out)
    return dict(status=out['status'], changed_paths=len(changes), target_uid=TARGET_UID)


def command():
    return ['sbatch', '--parsable', '--hold', '--account='+SLURM_ACCOUNT, '--qos=normal',
            '--partition=acd_ue', '--nodes=1', '--ntasks=1', '--gres=gpu:4',
            '--cpus-per-task=48', '--mem=512G', '--time=7-00:00:00',
            '--job-name=qprobe-warmup200', '--export=NONE', '--chdir='+str(TARGET),
            '--output='+str(TARGET/'logs/training-%j.log'), '--error='+str(TARGET/'logs/training-%j.log'),
            str(BASE/'source'/SOURCE/'reproduction/hpc3_qprobe/train.sh'), str(TARGET), SOURCE]


def validate_replacement(value):
    expected=dict(JobState='PENDING',Priority='0',AllocTRES='(null)',
        UserId=TARGET_ACCOUNT+'('+str(TARGET_UID)+')',Account=SLURM_ACCOUNT,QOS='normal',
        NumCPUs='48',NumTasks='1',MinMemoryNode='512G',TresPerNode='gres/gpu:4',
        TimeLimit='7-00:00:00',Command=str(BASE/'source'/SOURCE/'reproduction/hpc3_qprobe/train.sh'),
        WorkDir=str(TARGET),Restarts='0',RunTime='00:00:00')
    if any(value.get(k)!=v for k,v in expected.items()):
        raise ValueError('Replacement is not the actual held immutable native4 request')


def prepare_target():
    if os.getuid()!=TARGET_UID or SOURCE is None:
        raise ValueError('Explicit target identity and published commit required')
    untouched();validate_old(job(OLD)[0])
    proof=json.loads((BASE/'preparation'/('source-'+SOURCE+'-verification.json')).read_text())
    if proof['status']!='PUBLISHED_SOURCE_FULL_SHA_PASS' or proof['commit']!=SOURCE:
        raise ValueError('New native4 source is not published and verified')
    source=BASE/'source'/SOURCE
    for name,sha in proof['files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('New native4 source file differs: '+name)
    if TARGET.exists():raise ValueError('Inspect existing fresh native4 target; never recreate')
    os.umask(0o007)
    TARGET.mkdir(mode=0o770)
    # The human explicitly selected shared-group770 after this filesystem
    # rejected per-user ACLs. Keep new ordinary output directories group-writable.
    if TARGET_UID!=ORIGIN_UID:
        group=(BASE.parent.stat().st_gid)
        if group not in os.getgroups() or TARGET.stat().st_gid!=group:
            raise ValueError('New target does not share the authorized group')
    os.chmod(TARGET,0o770)
    from reproduction.hpc3_qprobe.control import prepare,link
    prepare(TARGET)
    link(TARGET/'repo',source)
    link(TARGET/'question-features',BASE/'question-features')
    for name in ('features-ready.json','official-model-manifest.json'):
        link(TARGET/'preparation'/name,BASE/'preparation'/name)
    once(TARGET/'preparation/account-source-binding.json',dict(source_commit=SOURCE,
        original_source_commit=ORIGINAL_SOURCE,source=str(source),target_base=str(TARGET),
        original_base=str(BASE),source_files=len(proof['files']),training_world_size=4,
        actor_micro_batch_per_rank=4,feature_cache_reused=True,authorization=AUTHORIZATION))
    return dict(status='FRESH_NATIVE4_TARGET_PREPARED',target_base=str(TARGET),source_commit=SOURCE)


def preflight():
    if os.getuid() != TARGET_UID:
        raise ValueError('Target account identity differs')
    untouched(); validate_old(job(OLD)[0])
    queue = cli(['squeue', '-u', TARGET_ACCOUNT, '-h', '-o', '%i|%T|%j'])
    if any(line.split('|')[0]!=OLD and 'qprobe' in line for line in queue.splitlines()):
        raise ValueError('Inspect existing target jobs before submitting any replacement')
    assoc = cli(['sacctmgr', '-nP', 'show', 'assoc', 'where', 'user='+TARGET_ACCOUNT,
                 'format=Account,User,Partition,QOS,DefaultQOS'])
    if not any(line.startswith(SLURM_ACCOUNT+'|'+TARGET_ACCOUNT+'||normal|') for line in assoc.splitlines()):
        raise ValueError('Target normal-QOS association absent')
    from search_g1.qprobe_features import FeatureCache
    features = FeatureCache(BASE/'question-features'); features.verify_all()
    if features.manifest['count'] != 169028:
        raise ValueError('Canonical completed feature count differs')
    for name in ('', 'logs', 'preparation', 'tmp', 'cache'):
        if not os.access(str(TARGET/name), os.W_OK | os.X_OK):
            raise ValueError('Target output directory is not writable')
    test = [v for v in command() if v != '--hold']
    test.insert(1, '--test-only')
    estimate = cli(test)
    out = dict(status='TARGET_PREFLIGHT_PASS', epoch=time.time(), target_uid=os.getuid(),
               account=SLURM_ACCOUNT, qos='normal', queue=queue, association=assoc,
               features_count=features.manifest['count'], features_spec_sha=features.spec_sha,
               scheduler_test=estimate, source_commit=SOURCE, authorization=AUTHORIZATION)
    once(BASE/'preparation/account-transition-preflight.json', out)
    return out


def submit():
    if os.getuid() != TARGET_UID:
        raise ValueError('Replacement must be submitted by the target account')
    untouched(); validate_old(job(OLD)[0])
    ready = json.loads((BASE/'preparation/account-transition-preflight.json').read_text())
    if ready['status'] != 'TARGET_PREFLIGHT_PASS' or time.time()-ready['epoch'] > 600:
        raise ValueError('Fresh actual target preflight required')
    claim = BASE/'preparation/account-transition-submit-claim.json'
    receipt = BASE/'preparation/account-transition-submission.json'
    if claim.exists() or receipt.exists():
        raise ValueError('Submission already attempted; inspect saved result, never repeat')
    if any(line.split('|')[0]!=OLD and 'qprobe' in line for line in cli(['squeue','-u',TARGET_ACCOUNT,'-h','-o','%i|%j']).splitlines()):
        raise ValueError('Target queue changed')
    cmd = command(); once(claim, dict(command=cmd, epoch=time.time(), authorization=AUTHORIZATION))
    output = cli(cmd); new = output.strip().split(';')[0]
    if not new.isdecimal():
        raise ValueError('Inspect unknown submission outcome; never repeat')
    value, detail = job(new)
    validate_replacement(value)
    out = dict(status='REPLACEMENT_HELD', job=new, epoch=time.time(), actual=detail,
               source_commit=SOURCE, old_job=OLD)
    once(receipt, out)
    return out


def cancel_old():
    if os.getuid() != ORIGIN_UID:
        raise ValueError('Only the old job owner may retire it')
    untouched(); validate_old(job(OLD)[0])
    replacement = json.loads((BASE/'preparation/account-transition-submission.json').read_text())
    value, detail = job(replacement['job'])
    validate_replacement(value)
    claim = BASE/'preparation/account-transition-retire-intent.json'
    if claim.exists():
        raise ValueError('Retirement already attempted; inspect actual state')
    once(claim, dict(authorization=AUTHORIZATION, epoch=time.time(), old=job(OLD)[1], replacement=detail))
    cli(['scancel', '--state=PENDING', OLD])
    value, detail = job(OLD)
    if value['JobState'] != 'CANCELLED' or value['RunTime'] != '00:00:00':
        raise ValueError('Old job not actually retired; do not release replacement')
    out = dict(status='OLD_UNSTARTED_JOB_RETIRED', old_job=OLD, replacement=replacement['job'],
               epoch=time.time(), actual=detail, authorization=AUTHORIZATION)
    once(BASE/'preparation/account-transition-retired.json', out)
    return out


def release():
    if os.getuid() != TARGET_UID:
        raise ValueError('Only target owner can release the replacement')
    # No earlier pending observation can be used in place of terminal proof.
    path=BASE/'preparation/account-transition-retired.json'
    if path.stat().st_uid!=ORIGIN_UID:raise ValueError('Retirement must be produced by actual original owner')
    terminal=json.loads(path.read_text())
    old=scheduler_fields(terminal['actual'])
    if old['JobState'] != 'CANCELLED' or old['RunTime'] != '00:00:00':
        raise ValueError('Old job was not cancelled before any native execution')
    retirement = json.loads((BASE/'preparation/account-transition-retired.json').read_text())
    new = retirement['replacement']; value, detail = job(new)
    validate_replacement(value)
    claim = BASE/'preparation/account-transition-release-intent.json'
    if claim.exists():
        raise ValueError('Release already attempted; inspect actual job instead')
    once(claim, dict(job=new, epoch=time.time(), before=detail))
    cli(['scontrol', 'release', new])
    value, detail = job(new)
    out = dict(status='ACCOUNT_SWITCH_RELEASED', job=new, old_job=OLD,
               target_uid=TARGET_UID, epoch=time.time(), actual=detail,
               native_training_started=value['JobState'] in ('RUNNING', 'CONFIGURING', 'COMPLETING'),
               source_commit=SOURCE, target_training_outers=200,training_world_size=4,actor_micro_batch_per_rank=4,target_base=str(TARGET))
    once(BASE/'preparation/account-transition-released.json', out)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=('access', 'prepare-target', 'preflight', 'submit', 'cancel-old', 'release'))
    p.add_argument('--commit',required=True)
    p.add_argument('--target-account',choices=('jxu957','mou176'),default='jxu957')
    p.add_argument('--owner-observation')
    args = p.parse_args();SOURCE=args.commit
    OWNER_OBSERVATION=args.owner_observation
    TARGET_ACCOUNT=args.target_account
    TARGET_UID=14090 if TARGET_ACCOUNT=='jxu957' else ORIGIN_UID
    SLURM_ACCOUNT='d_huiqimai_team' if TARGET_ACCOUNT=='jxu957' else 'mou176'
    TARGET=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-'+TARGET_ACCOUNT+'-20261014')
    if len(SOURCE)!=40 or any(c not in '0123456789abcdef' for c in SOURCE):raise ValueError('Invalid source commit')
    print(json.dumps({'access': access, 'prepare-target':prepare_target, 'preflight': preflight, 'submit': submit,
                      'cancel-old': cancel_old, 'release': release}[args.mode]()), flush=True)
