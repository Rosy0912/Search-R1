"""External exact-UID CPU gate for one isolated tiny native GPU probe."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from .contract import read_plan,sha,validate_tasks


def main():
    pp=Path(sys.argv[1]);plan=read_plan(pp);root=Path(plan['root'])
    with (root/'publisher-once.json').open('x') as f:json.dump(dict(at=time.time(),pid=os.getpid(),plan_sha256=sha(pp)),f)
    submission=json.loads((root/'submission.json').read_text());assert submission['job']==plan['job'] and submission['plan_sha256']==sha(pp)
    spec=importlib.util.spec_from_file_location('pt','/plm-shared/zhangjunming/Workspace/mmsearch-hybrid-explore/runs/msjev-pt-probe-20260924/pt_client.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);client=m.get_cluster().client
    deadline=time.monotonic()+1800
    while True:
        actual=client.get_training_job(plan['job']);assert actual['name']==plan['job'] and actual['uid']==submission['uid']
        state=actual['status']['state'];assert state in ['PENDING','RUNNING'],'Unexpected terminal probe; no retry'
        entries=[root/f'node-{i}/entry-once.json' for i in range(2)]
        if state=='RUNNING' and all(p.exists() for p in entries):
            rows=[json.loads(p.read_text()) for p in entries]
            assert {r['node'] for r in rows}=={0,1} and len({r['hostname'] for r in rows})==2
            assert len({(r['master'],r['port']) for r in rows})==1 and all(r['plan_sha256']==sha(pp) for r in rows)
            break
        assert time.monotonic()<deadline,'CPU gate expired; do not auto restart';time.sleep(5)
    validate_tasks(actual['spec']['vc_job']['tasks'])
    # Auth annotations are irrelevant to allocation validation and not retained.
    for k in list(actual.get('annotations',{})):
        if any(x in k.lower() for x in ['secret','signature']):actual['annotations'][k]='REDACTED_NOT_NEEDED_FOR_AUDIT'
    value=dict(at=time.time(),job=plan['job'],uid=actual['uid'],state=state,nodes=2,total_gpus=16,plan_sha256=sha(pp),actual_scheduler=actual)
    temp=root/'scheduler-release-once.pending'
    with temp.open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.link(temp,root/'scheduler-release-once.json');temp.unlink()


if __name__=='__main__':main()
