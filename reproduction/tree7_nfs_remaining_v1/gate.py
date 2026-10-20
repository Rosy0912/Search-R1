"""Only publish an exact saved UID. No submission, automatic retry, or GPU work."""
import json
import os
from pathlib import Path
import sys
import time
from reproduction.official_eval_v2.driver import once
from .contract import sha,validate_gate,old_for
from reproduction.trained_tree7b_v2.scheduler import cluster,actual


def run(path):
    path=Path(path);plan=json.loads(path.read_text());root=Path(plan['root']);OLD_JOB,OLD_UID=old_for(plan['repeat'])
    sub=json.loads((root/'submission.json').read_text());assert sub['plan_sha256']==sha(path)
    once(root/'publisher-once.json',dict(at=time.time(),pid=os.getpid(),uid=sub['uid']))
    client=cluster().client;start=time.monotonic()
    while time.monotonic()-start<21600:
        old=client.get_training_job(OLD_JOB);assert old['uid']==OLD_UID and old['status']['state']=='FAILED'
        job=client.get_training_job(plan['job']);assert job['uid']==sub['uid'] and job['name']==plan['job']
        state=job['status']['state'];assert state in ('PENDING','RUNNING'), 'Terminal before release; inspect, never retry'
        if state=='RUNNING':
            gate=dict(at=time.time(),source=plan['source_commit'],plan_sha256=sha(path),actual=actual(job),superseded_uid=OLD_UID,entry_source_commit=plan['recovery']['entry_source_commit'])
            validate_gate(plan,sub,gate)
            assert not (root/'entry-once.json').exists()
            tmp=root/'scheduler-release.tmp';once(tmp,gate)
            os.link(tmp,root/'scheduler-release-once.json');tmp.unlink()
            print('GATE='+json.dumps(gate),flush=True);return
        time.sleep(1)
    raise TimeoutError('Six-hour CPU publisher expired; inspect instead of restarting')


if __name__=='__main__':run(sys.argv[1])
