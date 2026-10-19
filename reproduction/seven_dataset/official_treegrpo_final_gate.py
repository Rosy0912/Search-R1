"""Unique external CPU exactUID publisher; no submission or GPU work."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from .contracts import sha
from .official_treegrpo_final import validate_gate


def run(planfile):
    plan=json.loads(Path(planfile).read_text());root=Path(plan['output']).parent
    submission=json.loads((root/'submission.json').read_text());assert submission['plan_sha256']==sha(planfile)
    with (root/'scheduler-publisher.once').open('x') as f:f.write(str(os.getpid()));f.flush();os.fsync(f.fileno())
    s=importlib.util.spec_from_file_location('pt','/plm-shared/zhangjunming/Workspace/mmsearch-hybrid-explore/runs/msjev-pt-probe-20260924/pt_client.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);client=m.get_cluster().client
    start=time.monotonic()
    while time.monotonic()-start<21600:
        j=client.get_training_job(plan['job']);assert j['uid']==submission['uid'] and j['name']==plan['job']
        state=j['status']['state'];assert state in ('PENDING','RUNNING'),'Terminal before gate; inspect, never retry'
        if state=='RUNNING':
            tasks=j['spec']['vc_job']['tasks'];assert len(tasks)==1 and tasks[0]['replicas']==1
            r=tasks[0]['resource_spec'];actual=dict(name=j['name'],uid=j['uid'],state=state,resources=dict(
                gpus=r['accelerate_device_count'],cpus=r['cpu_count'],memory_gib=r['memory_gib'],machine_type=r['machine_types'][0],
                gpu_model=r['accelerate_device_model'],gpu_memory_gib=r['accelerate_device_memory_gib']))
            gate=dict(at=time.time(),publisher_pid=os.getpid(),plan_sha256=sha(planfile),actual=actual,source=plan['evaluation_code_commit'],no_gpu_submission=True)
            validate_gate(plan,submission,gate)
            assert not (root/'entry-once.json').exists()
            tmp=root/'scheduler-release.tmp'
            with tmp.open('x') as f:json.dump(gate,f,indent=2);f.flush();os.fsync(f.fileno())
            os.link(tmp,root/'scheduler-release-once.json');tmp.unlink();print('GATE='+json.dumps(gate),flush=True);return
        time.sleep(1)
    raise TimeoutError('Bounded6h publisher expired; no automatic retry')


if __name__=='__main__':run(sys.argv[1])
