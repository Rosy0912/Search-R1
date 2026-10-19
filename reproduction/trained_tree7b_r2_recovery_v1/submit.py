"""One explicitly inspected replacement for repeat 2; no automatic retry."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from reproduction.official_eval_v2.driver import once
from reproduction.trained_tree7b_v2 import submit as original
from reproduction.trained_tree7b_v2.scheduler import cluster, exact, zero, actual
from .contract import ROOT, BASE, JOB, OLD_JOB, OLD_UID, UID, ALLOCATION, sha, verify


def command(plan, path):
    text = original.command(plan, path)
    source = shlex.quote(plan['source'])
    entry_source = shlex.quote(plan['recovery']['entry_source'])
    text = text.replace('PYTHONPATH=' + source, 'PYTHONPATH=' + entry_source, 1)
    text = text.replace('cd ' + source + '\n', 'cd ' + entry_source + '\n', 1)
    return text.replace('reproduction.trained_tree7b_v2.launch',
                        'reproduction.trained_tree7b_r2_recovery_v1.launch', 1)


def params(plan, path):
    value = original.params(plan, path)
    value.update(cmd=command(plan, path), tmpdir=str(ROOT / 'submit-r2'))
    return value


def inspected(client):
    control = exact(client, 'zhangjunming-ttt-b20-150-r1-54c0')
    assert len(control) == 1 and control[0]['uid'] == '33ac6bbf-7e6a-4a66-bcc6-261b66371975'
    old = client.get_training_job(OLD_JOB)
    assert old['uid'] == OLD_UID and old['status']['state'] == 'FAILED'
    assert actual(old)['resources'] == ALLOCATION
    old_root = original.ROOT / 'r2'
    assert (old_root / 'exit.code').read_text().strip() == '1'
    assert not (old_root / 'output/entry-exit.json').exists()
    training = client.get_training_job('zhangjunming-7b-treegrpo-pj-v1')
    assert training['uid'] == UID and training['status']['state'] == 'COMPLETED'
    assert (BASE / 'runs/treegrpo-v1/exit.code').read_text().strip() == '0'
    zero(client, JOB)


def run(execute=False):
    if not (ROOT / 'prepared.json').exists():
        print(json.dumps(dict(status='WAIT_ISOLATED_INSPECTED_RECOVERY_PREPARE', no_gpu_action=True)))
        return
    prepared = json.loads((ROOT / 'prepared.json').read_text())
    assert prepared['uid'] == UID and len(prepared['plans']) == 3
    item = next(p for p in prepared['plans'] if p['job'] == JOB)
    path = Path(item['plan'])
    assert sha(path) == item['plan_sha256']
    plan = json.loads(path.read_text())
    _, root = verify(plan)
    c = cluster()
    inspected(c.client)
    assert not any(root.iterdir()), 'Existing intent/partial: inspect, never replay'
    assert not (ROOT / 'submit-r2-once.json').exists()
    value = params(plan, path)
    assert value['retry'] == 0 and not value.get('include')
    if not execute:
        print(json.dumps(dict(status='READY_EXACT_ZERO_REPEAT2_ONLY', no_gpu_action=True,
                              command_sha256=__import__('hashlib').sha256(value['cmd'].encode()).hexdigest())))
        return
    from clusterx.launcher.ssp.ssp import SSPRunParams
    once(ROOT / 'submit-r2-once.json', dict(at=time.time(), superseded_uid=OLD_UID, no_auto_retry=True))
    inspected(c.client)
    once(root / 'submission-intent.json', dict(at=time.time(), job=JOB, plan_sha256=sha(path),
                                              params=value, repeat_index=2, superseded_uid=OLD_UID))
    c.run(SSPRunParams(**value))
    job = c.client.get_training_job(JOB)
    assert job['name'] == JOB and job['uid'] and job['uid'] != OLD_UID
    receipt = dict(at=time.time(), job=JOB, uid=job['uid'], source=plan['source_commit'],
                   plan=str(path), plan_sha256=sha(path), root=str(root), repeat_index=2,
                   model_identity=plan['model_identity'], superseded_uid=OLD_UID,
                   entry_source_commit=plan['recovery']['entry_source_commit'])
    once(root / 'submission.json', receipt)
    env = dict(os.environ, PYTHONPATH=plan['recovery']['entry_source'], PYTHONDONTWRITEBYTECODE='1')
    with (root / 'publisher.stdout').open('x') as out, (root / 'publisher.stderr').open('x') as err:
        child = subprocess.Popen([sys.executable, '-m', 'reproduction.trained_tree7b_r2_recovery_v1.gate', str(path)],
                                 env=env, cwd=plan['recovery']['entry_source'], stdout=out, stderr=err,
                                 start_new_session=True)
    once(root / 'publisher-launch.json', dict(at=time.time(), pid=child.pid, uid=job['uid'], no_auto_restart=True))
    print('SUBMISSION=' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    run(parser.parse_args().execute)
