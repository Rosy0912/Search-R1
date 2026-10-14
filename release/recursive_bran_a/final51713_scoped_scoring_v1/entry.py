"""Continue preserved final scores in an independent scoped-scoring process."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

B=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013')
ORIGINAL=B/'final51713-supervisor-recovery-v1'
PREVIOUS=B/'final51713-worker-recovery-v1'
FOLDER=B/'final51713-scoped-scoring-v1'
EXPECTED=dict(pid=1441964,uid=0,tick=675104567)


def main():
    sys.path.insert(0,str(ORIGINAL))
    import boundary_supervisor as infrastructure
    FOLDER.mkdir(exist_ok=True)
    lock=(FOLDER/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (FOLDER/'evaluation-launch.json').exists(), 'Never duplicate an existing final continuation'
    old=json.loads((PREVIOUS/'execution-plan.json').read_text())
    assert infrastructure.matches(infrastructure.process(EXPECTED['pid']),EXPECTED)
    assert not (PREVIOUS/'evaluation-actual-exit.json').exists()
    assert infrastructure.sha(old['checkpoint_manifest'])==old['checkpoint_manifest_sha256']
    manifest=json.loads(Path(old['checkpoint_manifest']).read_text())
    infrastructure.validate_checkpoint(manifest,old['config_sha256'])
    # Capture only this final Ray runtime and its explicit srun launcher.
    own={EXPECTED['pid']:EXPECTED}
    prefix='/tmp/rf-'+old['run_id'][:8]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            env=dict(item.split(b'=',1) for item in (p/'environ').read_bytes().split(b'\0') if b'=' in item)
            if prefix not in command and env.get(b'E1_RUN_ID')!=old['run_id'].encode():continue
            assert 'slurmstepd' not in command, 'Never signal infrastructure'
            record=infrastructure.process(int(p.name));assert record['uid']==0
            own[record['pid']]=record
        except FileNotFoundError:pass
    assert len(own)>=9, 'Expected current launcher and all current final workers'
    infrastructure.write(FOLDER/'own-before-transition.json',dict(time=time.time(),records=list(own.values()),
        previous_run=old['run_id'],reason='Independent scoped scoring; preserve completed final batches'))
    fds=[]
    for record in own.values():
        if not infrastructure.matches(infrastructure.process(record['pid']),record):continue
        fd=os.pidfd_open(record['pid'])
        assert infrastructure.matches(infrastructure.process(record['pid']),record)
        fds.append((record,fd))
    try:
        for record,fd in reversed(fds):
            try:signal.pidfd_send_signal(fd,signal.SIGTERM)
            except ProcessLookupError:pass
        for _ in range(30):
            remaining=[r for r,fd in fds if infrastructure.matches(infrastructure.process(r['pid']),r)
                       and infrastructure.process(r['pid'])['state']!='Z']
            if not remaining:break
            time.sleep(1)
        for record,fd in fds:
            current=infrastructure.process(record['pid'])
            if infrastructure.matches(current,record) and current['state']!='Z':
                signal.pidfd_send_signal(fd,signal.SIGKILL)
    finally:
        for record,fd in fds:os.close(fd)
    for _ in range(20):
        remaining=[r for r in own.values() if infrastructure.matches(infrastructure.process(r['pid']),r)
                   and infrastructure.process(r['pid'])['state']!='Z']
        if not remaining:break
        time.sleep(1)
    assert not remaining
    release=infrastructure.write(FOLDER/'previous-owned-release.json',dict(time=time.time(),released=True,
        records=[dict(original=r,current=infrastructure.process(r['pid'])) for r in own.values()],
        cause='INTENTIONAL_SCORING_PERFORMANCE_CONTINUATION',natural_exit0=False))
    previous_output=Path(old['output'])
    progress=json.loads((previous_output/'progress.json').read_text())
    count=progress['completed'];assert 0<count<51713 and count%512==0
    rows=[]
    with (previous_output/'question-metrics.jsonl').open() as stream:
        for index in range(count):
            row=json.loads(next(stream));assert row['final_index']==index;rows.append(row)
    prefix_path=FOLDER/'accepted-prefix-question-metrics.jsonl'
    with prefix_path.open('x') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    prefix_receipt=infrastructure.write(FOLDER/'accepted-prefix.json',dict(time=time.time(),count=count,
        path=str(prefix_path),sha256=infrastructure.sha(prefix_path),previous_run=old['run_id'],
        previous_output=str(previous_output),checkpoint_sha256=old['checkpoint_manifest_sha256']))
    # Release only the private task retriever's stage GPU child after the client exits.
    import urllib.request
    token=(B/'retriever/receipts/private-probe-token').read_text().strip()
    request=urllib.request.Request('http://127.0.0.1:18086/release-gpu',method='POST',
        headers={'X-E1-Probe-Token':token})
    with urllib.request.urlopen(request,timeout=120) as response:scanner=json.load(response)
    assert scanner['released'] and scanner['gpu_worker_pid'] is None
    resource=infrastructure.resource_receipt(FOLDER)
    assert not json.loads((FOLDER/'resource-admission.json').read_text())['compute_processes'].strip()
    plan=dict(old);plan['run_id']=uuid.uuid4().hex;plan['output']=str(B/'final51713'/plan['run_id'])
    plan.update(resume_rows=str(prefix_path),resume_rows_sha256=infrastructure.sha(prefix_path),resume_count=count,
                previous_evaluation_plan=str(PREVIOUS/'execution-plan.json'),
                previous_evaluation_plan_sha256=infrastructure.sha(PREVIOUS/'execution-plan.json'))
    plan['admission_receipts']=old['admission_receipts'][:2]+[release,prefix_receipt,resource]
    plan_path=FOLDER/'execution-plan.json';infrastructure.write(plan_path,plan)
    command,env=infrastructure.evaluator_command(plan_path,FOLDER)
    env['RAY_TMPDIR']='/tmp/rf-'+plan['run_id'][:8]
    infrastructure.write(FOLDER/'runtime-recovery.json',dict(time=time.time(),ray_tmpdir=env['RAY_TMPDIR'],
        scoring_change='Shared checked tokenizer context per batch; no scorer formula or generation changes'))
    with (FOLDER/'evaluation.log').open('ab') as log:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=log)
        infrastructure.write(FOLDER/'evaluation-launch.json',dict(time=time.time(),pid=child.pid,command=command,run_id=plan['run_id']))
        with (FOLDER/'gpu-device-observer.jsonl').open('a') as observer:
            while child.poll() is None:
                sample=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
                observer.write(json.dumps(dict(time=time.time(),rc=sample.returncode,device_sample=sample.stdout,
                    scope='Whole device utilization; cohost activity may contribute'))+'\n');observer.flush();time.sleep(10)
        rc=child.wait()
    infrastructure.write(FOLDER/'evaluation-actual-exit.json',dict(time=time.time(),pid=child.pid,rc=rc,run_id=plan['run_id']))
    if rc:raise RuntimeError('Final continuation exited '+str(rc)+'; preserve and repair independently')


if __name__=='__main__':main()
