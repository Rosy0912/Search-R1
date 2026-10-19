"""External login-CPU exact allocation publisher; never submit or retry a job."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from .ablations import validate_gate, sha


def run(planfile):
    plan = json.loads(Path(planfile).read_text()); root = Path(plan['root'])
    sub = json.loads((root / 'submission.json').read_text())
    assert sub['plan_sha256'] == sha(planfile)
    with (root / 'scheduler-publisher.once').open('x') as f:
        f.write(str(os.getpid())); f.flush(); os.fsync(f.fileno())
    spec = importlib.util.spec_from_file_location('pt', '/plm-shared/zhangjunming/Workspace/mmsearch-hybrid-explore/runs/msjev-pt-probe-20260924/pt_client.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    client = module.get_cluster().client
    end = time.monotonic() + 43200
    while time.monotonic() < end:
        j = client.get_training_job(plan['job']); assert j['uid'] == sub['uid']
        state = j['status']['state']
        if state not in ['PENDING', 'RUNNING']:
            raise RuntimeError('Terminal allocation before release: ' + state)
        if state == 'RUNNING':
            tasks = j['spec']['vc_job']['tasks']; assert len(tasks) == 1 and tasks[0]['replicas'] == 1
            r = tasks[0]['resource_spec']
            actual = dict(name=j['name'], uid=j['uid'], state=state, resources=dict(gpus=r['accelerate_device_count'], cpus=r['cpu_count'], memory_gib=r['memory_gib'], machine_type=r['machine_types'][0], gpu_model=r['accelerate_device_model'], gpu_memory_gib=r['accelerate_device_memory_gib']))
            gate = dict(at=time.time(), plan_sha256=sha(planfile), actual=actual, publisher_pid=os.getpid(), no_gpu_submission=True)
            validate_gate(plan, sub, gate)
            assert not (root / 'entry-once.json').exists()
            tmp = root / 'scheduler-release.pending'
            with tmp.open('x') as f:
                json.dump(gate, f, indent=2); f.flush(); os.fsync(f.fileno())
            os.link(tmp, root / 'scheduler-release-once.json'); tmp.unlink()
            print('RELEASED=' + json.dumps(gate), flush=True)
            return
        time.sleep(1)
    raise TimeoutError('Bounded CPU publisher ended; no automatic restart')


if __name__ == '__main__':
    run(sys.argv[1])
