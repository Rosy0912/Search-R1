"""Resume final scoring after pre-model Ray socket failure; unchanged evaluator payload."""
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

B = Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013')
OLD = B/'final51713-supervisor-recovery-v1'
FAILED = B/'final51713-tokenizer-recovery-v1'
FOLDER = B/'final51713-worker-recovery-v1'


def main():
    sys.path.insert(0, str(OLD))
    import boundary_supervisor as original
    FOLDER.mkdir(exist_ok=True)
    lock=(FOLDER/'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (FOLDER/'evaluation-launch.json').exists(), 'Inspect existing run before any recovery'
    previous=json.loads((FAILED/'evaluation-actual-exit.json').read_text())
    previous_launch=json.loads((FAILED/'evaluation-launch.json').read_text())
    assert previous['rc']==1 and previous['pid']==previous_launch['pid']
    assert original.process(previous['pid']) is None
    assert 'ActorRolloutRefWorker.__init__() missing 2 required positional arguments' in (FAILED/'evaluation.log').read_text()
    old_plan=json.loads((FAILED/'execution-plan.json').read_text())
    assert not (Path(old_plan['output'])/'question-metrics.jsonl').exists(), 'No formal predictions may be duplicated'
    old_publication=json.loads((OLD/'launch.json').read_text())
    assert all(original.sha(OLD/name)==digest for name,digest in old_publication['payload_manifest'].items())
    released=json.loads((OLD/'owned-terminal-release.json').read_text())
    assert released['released'] and not released['natural_exit0']
    assert all(not original.matches(original.process(r['original']['pid']), r['original'])
               or original.process(r['original']['pid'])['state']=='Z' for r in released['records'])
    manifest=json.loads(Path(old_plan['checkpoint_manifest']).read_text())
    assert original.sha(old_plan['checkpoint_manifest'])==old_plan['checkpoint_manifest_sha256']
    original.validate_checkpoint(manifest, old_plan['config_sha256'])
    resource=original.resource_receipt(FOLDER)
    assert not json.loads((FOLDER/'resource-admission.json').read_text())['compute_processes'].strip()
    binding=json.loads((OLD/'binding.json').read_text())
    assert original.matches(original.process(binding['retriever_owner']['pid']), binding['retriever_owner'])
    plan=dict(old_plan)
    plan.update(run_id=uuid.uuid4().hex)
    plan['output']=str(B/'final51713'/plan['run_id'])
    plan['admission_receipts']=old_plan['admission_receipts'][:2]+[resource]
    plan['recovery_from']=dict(run_id=old_plan['run_id'],terminal_sha256=original.sha(FAILED/'evaluation-actual-exit.json'),
                               reason='Preserve native Worker direct base for old Ray colocation interface')
    plan_path=FOLDER/'execution-plan.json'
    original.write(plan_path,plan)
    assert original.sha(FOLDER/'final_eval.py')==original.sha(OLD/'final_eval.py')
    command,env=original.evaluator_command(plan_path,FOLDER)
    env['RAY_TMPDIR']='/tmp/rf-'+plan['run_id'][:8]
    assert len(env['RAY_TMPDIR'])<24
    original.write(FOLDER/'runtime-recovery.json',dict(time=time.time(),ray_tmpdir=env['RAY_TMPDIR'],
        unchanged_payload_manifest=old_publication['payload_manifest'],model_sampling_scoring_unchanged=True))
    with (FOLDER/'evaluation.log').open('ab') as log:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=log)
        original.write(FOLDER/'evaluation-launch.json',dict(time=time.time(),pid=child.pid,command=command,run_id=plan['run_id']))
        with (FOLDER/'gpu-device-observer.jsonl').open('a') as observer:
            while child.poll() is None:
                sample=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
                observer.write(json.dumps(dict(time=time.time(),rc=sample.returncode,device_sample=sample.stdout,
                    scope='Whole device utilization; cohost activity may contribute'))+'\n')
                observer.flush()
                time.sleep(10)
        rc=child.wait()
    original.write(FOLDER/'evaluation-actual-exit.json',dict(time=time.time(),pid=child.pid,rc=rc,run_id=plan['run_id']))
    if rc: raise RuntimeError('Final scoring exited '+str(rc)+'; preserved for independent recovery')


if __name__=='__main__':
    main()
