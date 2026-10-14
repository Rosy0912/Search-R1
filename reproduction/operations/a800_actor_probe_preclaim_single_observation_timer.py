"""Wait locally, then make one bounded resource read for terminal new3ecc.

No launch, reservation, signal, evaluation, or automatic retry. A changed
authoritative run cancels this obsolete observer locally before any SSH call.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_RUN='3ecc1a1afd9948d18d1d5d670c547b51'
HANDLE='actor-probes-ten-production-after-A9-OOM-fresh-deployment.json'
READER='a800_actor_probe_preclaim_failure_metadata_read.py'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--not-before',type=float,required=True)
    args=parser.parse_args();root=args.root.resolve()
    handle=root/'preparation'/HANDLE;intent=json.loads(handle.read_text())
    assert intent['deployment_id']=='76398c90bc554e7cbd81d465a547eb71'
    assert intent['source_commit']=='c4d156de124de7a94652d75858bcbabc8c56083c'
    print(json.dumps(dict(role='single_local_new3ecc_resource_observation_waiter',
        pid=os.getpid(),created_unix=time.time(),not_before_unix=args.not_before,
        run_id=ROOT_RUN,remote_query_performed=False)),flush=True)
    while True:
        remaining=args.not_before-time.time()
        if remaining<=0:break
        time.sleep(min(30,remaining))
    assert json.loads(handle.read_text())==intent,'Immutable handle changed during wait'
    current=json.loads((root/'preparation/a800-common-current-deployment.json').read_text())
    if current['run_id']!=ROOT_RUN or current.get('stage')!='PRETRAINER_EXITED_RESOURCE_OCCUPIED':
        print(json.dumps(dict(role='obsolete_new3ecc_observer_cancelled_locally',
            run_id=ROOT_RUN,remote_query_performed=False,checked_unix=time.time())),flush=True)
        return 0
    output=b'';error_output=b'';exit_code=None;failure_type=None;result=None
    try:
        process=subprocess.run([sys.executable,str(root/'repo/reproduction/operations'/READER),
            '--root',str(root),'--handle',str(root/'preparation/actor-probes-ten-production-fresh-resource-retry-deployment.json'),
            '--observation',str(root/'preparation/actor-probes-ten-production-fresh-resource-retry-deployment.observe-1791941917586750000.json')],
            capture_output=True,timeout=125)
        output=process.stdout;error_output=process.stderr;exit_code=process.returncode
    except (subprocess.TimeoutExpired,OSError) as error:
        failure_type=type(error).__name__;output=getattr(error,'stdout',None) or b''
        error_output=getattr(error,'stderr',None) or b''
    if len(output)<=65536:
        try:result=json.loads(output)
        except (ValueError,UnicodeDecodeError):pass
    target=root/'preparation'/('actor-probes-ten-3ecc-resource-observer-completion-'+str(time.time_ns())+'.json')
    value=dict(role='single_new3ecc_resource_observer_completion',run_id=ROOT_RUN,
        exit_code=exit_code,failure_type=failure_type,result=result,checked_unix=time.time(),
        stdout_bytes=len(output),stdout_sha256=hashlib.sha256(output).hexdigest(),
        stderr_bytes=len(error_output),stderr_sha256=hashlib.sha256(error_output).hexdigest(),
        no_remote_mutation=True,no_automatic_retry=True)
    with target.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(receipt=str(target),**value)),flush=True)
    return 0 if exit_code==0 and isinstance(result,dict) and result.get('status')=='observed' else 1


if __name__=='__main__':raise SystemExit(main())
