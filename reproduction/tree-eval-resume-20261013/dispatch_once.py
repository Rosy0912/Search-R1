"""One reviewed eval-only submission, then explicit user-authorized old-eval handoff."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from eval_merge import digest, atomic_immutable


def main():
    p=argparse.ArgumentParser()
    for name in ('tools','prefix','manifest','release','score-output','journal','review'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--stop-old-evaluation-3360',action='store_true')
    a=p.parse_args()
    if not a.stop_old_evaluation_3360:
        raise RuntimeError('Explicit current user eval-only handoff required')
    root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
    if os.uname().nodename.split('.')[0]!='h100-10':
        raise RuntimeError('CPU dispatch only on own h100-10')
    journal=Path(a.journal)
    if journal.exists():
        raise RuntimeError('Existing submit journal; inspect outcome instead of submitting again')
    release=json.loads(Path(a.release).read_text())
    review=json.loads(Path(a.review).read_text())
    if review['review_status']!='ACCEPTED' or release['review_status']!='ACCEPTED':
        raise RuntimeError('Independent source review missing')
    for name,sha in release['files'].items():
        if digest(Path(a.tools)/name)!=sha or review['files'][name]!=sha:
            raise RuntimeError('Review/deployment source differs: '+name)
    if digest(a.prefix)!=release['prefix_sha256'] or digest(a.manifest)!=release['manifest_sha256']:
        raise RuntimeError('Preserved prefix or assets changed')
    done=json.loads(Path(a.prefix+'.done.json').read_text())
    if done['sha256']!=release['prefix_sha256'] or done['actual_trained_outer']!=200:
        raise RuntimeError('Completed200 prefix binding missing')
    env={k:v for k,v in os.environ.items() if k not in ('SLURM_CLUSTERS','SLURM_CONF','SLURM_CONF_SERVER','SLURM_FEDERATION','SLURM_FEDERATION_CLUSTER','SLURM_CLUSTER_NAME') and not k.startswith(('SBATCH_','SQUEUE_','SCANCEL_','SCONTROL_','SACCT_'))}
    old=subprocess.check_output(['scontrol','--local','show','job','3360','-o'],text=True,env=env)
    if 'JobId=3360 ' not in old or 'JobName=tree-branpo-fresh200 ' not in old or 'NodeList=h100-10 ' not in old or 'JobState=RUNNING ' not in old:
        raise RuntimeError('Expected exact old evaluation job identity changed; inspect instead of cancel')
    active=subprocess.check_output(['squeue','--local','-h','-n','tree-final-eval-resume','-o','%i|%T|%R'],text=True,env=env)
    if active.strip():
        raise RuntimeError('Existing replacement evaluator; no duplicate submission')
    env.update(TREE_EVAL_TOOLS=a.tools,TREE_EVAL_PREFIX=a.prefix,TREE_EVAL_MANIFEST=a.manifest,
               TREE_EVAL_RELEASE=a.release,TREE_EVAL_SCORE_OUTPUT=a.score_output)
    value=dict(status='SUBMISSION_UNKNOWN',epoch=time.time(),old_job=3360,prefix_sha256=digest(a.prefix),
               prefix_count=done['count'],release_sha256=digest(a.release),review_sha256=digest(a.review),
               old_slurm_before=old,training_updates=200,scope='User-authorized evaluation-only handoff; no training restart')
    atomic_immutable(journal,value)
    def update():
        temp=journal.with_name(journal.name+'.update')
        with temp.open('x') as f:
            json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(temp,journal)
        fd=os.open(journal.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    r=subprocess.run(['sbatch','--parsable','--dependency=afterany:3360','--export=ALL',
            '--output',str(root/'instruct300/node-move-logs/eval-resume-%j.out'),
            '--error',str(root/'instruct300/node-move-logs/eval-resume-%j.err'),str(Path(a.tools)/'launch.sbatch')],env=env,text=True,capture_output=True,timeout=30)
    value.update(submit_returncode=r.returncode,submit_stdout=r.stdout,submit_stderr=r.stderr)
    update()
    match=re.fullmatch(r'([1-9][0-9]*)(?:;wuhan-h100)?\s*',r.stdout)
    if r.returncode or not match:
        raise RuntimeError('Submission not confirmed; old evaluator left running; inspect journal/queue')
    new=match.group(1)
    value.update(status='SUBMITTED_OLD_EVAL_STILL_RUNNING',replacement_job=new);update()
    target=subprocess.check_output(['scontrol','--local','show','job',new,'-o'],text=True,env=env)
    if ('JobName=tree-final-eval-resume ' not in target or 'ReqNodeList=h100-10 ' not in target
            or 'afterany:3360' not in target or 'TresPerNode=gres:gpu:8' not in target):
        raise RuntimeError('Replacement job allocation/dependency differs; old job not canceled')
    value['replacement_slurm_before_stop']=target;update()
    # Save the last real accounting/resource/log evidence before changing old state.
    oldrun=root/'runs/tree-branpo-fresh200-3360'
    value['old_stage_size_before_stop']=(oldrun/'stage.log').stat().st_size
    with (oldrun/'resource-samples.jsonl').open('rb') as f:
        f.seek(0,2);f.seek(max(0,f.tell()-65536));lines=f.read().decode().splitlines()
        value['old_last_resource_before_stop']=json.loads(lines[-1])
    value['status']='OLD_EVAL_STOP_REQUEST_PENDING';value['stop_requested_epoch']=time.time();update()
    r=subprocess.run(['scancel','3360'],env=env,text=True,capture_output=True,timeout=30)
    value.update(cancel_returncode=r.returncode,cancel_stdout=r.stdout,cancel_stderr=r.stderr,
                 status='HANDOFF_REQUESTED' if r.returncode==0 else 'STOP_UNCONFIRMED');update()
    if r.returncode:
        raise RuntimeError('Old stop unconfirmed; preserve submission and inspect, never resubmit')
    print(json.dumps(value,sort_keys=True))


if __name__=='__main__':
    main()
