"""Exact-UID external CPU scheduler gate; no submission, restart or GPU action."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from .contract import read_plan, sha, validate_actual_tasks


def main():
    pp = Path(sys.argv[1]); plan = read_plan(pp); root = Path(plan['root'])
    with (root / 'publisher-once.json').open('x') as f:
        json.dump(dict(at=time.time(), pid=os.getpid(), plan_sha256=sha(pp)), f)
    submission = json.loads((root / 'submission.json').read_text())
    assert submission['plan_sha256'] == sha(pp) and submission['job'] == plan['job']
    spec = importlib.util.spec_from_file_location('pt',
        '/plm-shared/zhangjunming/Workspace/mmsearch-hybrid-explore/runs/msjev-pt-probe-20260924/pt_client.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    client = module.get_cluster().client
    deadline = time.monotonic() + 1800
    while True:
        j = client.get_training_job(plan['job'])
        assert j['name'] == plan['job'] and j['uid'] == submission['uid']
        state = j['status']['state']
        assert state in ('PENDING', 'RUNNING'), 'Unexpected terminal probe; never retry'
        if state == 'RUNNING': break
        assert time.monotonic() < deadline, 'Bounded CPU gate expired; inspect without replay'
        time.sleep(10)
    tasks = j['spec']['vc_job']['tasks']; validate_actual_tasks(tasks)
    gate = dict(at=time.time(), job=plan['job'], uid=j['uid'], state=state,
                nodes=2, total_gpus=16, plan_sha256=sha(pp), actual_scheduler=j)
    temp = root / 'scheduler-release-once.pending'
    with temp.open('x') as f:
        json.dump(gate, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.link(temp, root / 'scheduler-release-once.json'); temp.unlink()


if __name__ == '__main__': main()
