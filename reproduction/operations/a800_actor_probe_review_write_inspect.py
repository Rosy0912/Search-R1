"""One read-only reconciliation of an uncertain independent review write.

Uses the existing physical route and reviewed runtime environment. Does not
write remotely, allocate CUDA, start a trainer, or repeat an uncertain write.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REMOTE = r'''
import hashlib,json,os,stat,sys,time,traceback
from pathlib import Path
p=json.load(sys.stdin);intent=p['intent'];base=Path(intent['base']);repo=base/'repo'
result=dict(role='actual_readonly_independent_review_write_reconciliation',base=str(base),
 checked_unix=time.time(),remote_write_attempted=False,trainer_started=False,
 native_algorithm_accepted=False)
try:
 assert repo==repo.resolve() and repo.stat().st_uid==os.getuid()
 name='reproduction/a800_common/prepare_actor_probe_production.py'
 assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==intent['source_sha256'][name]
 sys.path.insert(0,str(repo))
 from reproduction.a800_common.prepare_actor_probe_production import validate_intent,environment
 validate_intent(intent)
 from reproduction.a800_common.actor_probe_production_review import read,verify_documents,verify,PREPARED,REVIEW
 assert read(base,'preparation/deployment-intent.json')[0]==intent
 run,_=read(base,'logs/run-manifest.json');prepared,pfile=read(base,PREPARED)
 result['identity']=prepared['identity'];result['prepared_receipt_sha256']=pfile['sha256']
 result['deployment_stage']=read(base,'logs/deployment-status.json')[0]['stage']
 result['launch_artifact_exists']={n:(base/'logs'/n).exists() or (base/'logs'/n).is_symlink()
  for n in ('prepared-launch-claim.json','deployment-launch-receipt.json','pipeline-launch-claim.json',
   'training-launch-claim.json','process-ledger.json','pipeline-status.json')}
 path=base/REVIEW
 result['review_exists']=path.exists();result['review_symlink']=path.is_symlink()
 if result['review_exists'] or result['review_symlink']:
  info=path.lstat();assert stat.S_ISREG(info.st_mode) and info.st_uid==os.getuid() and info.st_size<=65536
  review,rfile=read(base,REVIEW)
  result['review_file']=rfile
  result['expected_review_sha256']=p['review_sha256']
  result['review_hash_matches_expected']=rfile['sha256']==p['review_sha256']
  assert result['review_hash_matches_expected']
  verify_documents(review,prepared,run,pfile['sha256'])
  result['review_documents_accepted']=True
  env=environment(base,Path(run['ray_tmp_dir']));os.environ.clear();os.environ.update(env)
  os.chdir(repo)
  result['runtime_environment']=dict(training_world_size=os.environ['SEARCH_G1_TRAINING_WORLD_SIZE'],
   base=os.environ['SEARCH_G1_BASE'],python_prefix=sys.prefix,ray_tmp_dir=os.environ['RAY_TMPDIR'])
  result['actual_preparation_verification']=verify(base,run)
  result['accepted']=True
 else:
  result['accepted']=False
  result['reason']='review_absent_at_this_actual_read; no automatic write retry'
except Exception as error:
 result['accepted']=False;result['error_type']=type(error).__name__;result['error']=str(error)
 result['traceback']=traceback.format_exc()[-12000:]
print(json.dumps(result,allow_nan=False),flush=True)
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--handle',type=Path,required=True)
    parser.add_argument('--review',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.root/'actor-probes-ten'))
    from reproduction.a800_common.deploy import ssh_command
    from reproduction.a800_common.prepare_actor_probe_production import validate_intent
    intent=validate_intent(json.loads(args.handle.read_text()))
    review_raw=args.review.read_bytes();assert len(review_raw)<=65536
    payload=dict(intent=intent,review_sha256=hashlib.sha256(review_raw).hexdigest())
    bootstrap='import os;os.environ["PYTHONDONTWRITEBYTECODE"]="1";os.execv("/data/search-g1/env/bin/python",["/data/search-g1/env/bin/python","-c",'+repr(REMOTE)+'])'
    receipt=args.root/'preparation'/('actor-probes-ten-review-write-readonly-'+str(time.time_ns())+'.json')
    errors=receipt.with_suffix('.stderr')
    output=b'';exit_code=None;failure_type=None
    with errors.open('xb') as stream:
        try:
            result=subprocess.run(ssh_command(bootstrap),input=json.dumps(payload,allow_nan=False).encode(),
                stdout=subprocess.PIPE,stderr=stream,timeout=240)
            output=result.stdout;exit_code=result.returncode
        except (subprocess.TimeoutExpired,OSError) as error:
            failure_type=type(error).__name__;output=getattr(error,'stdout',None) or b''
        stream.flush();os.fsync(stream.fileno())
    observation=None
    if len(output)<=65536:
        try:observation=json.loads(output)
        except (ValueError,UnicodeDecodeError):pass
    value=dict(role='local_readonly_uncertain_review_reconciliation_receipt',handle=str(args.handle),
        exit_code=exit_code,failure_type=failure_type,observation=observation,
        stdout_bytes=len(output),stdout_sha256=hashlib.sha256(output).hexdigest(),
        transport_stderr=str(errors),no_remote_mutation=True,checked_unix=time.time())
    with receipt.open('x') as target:
        json.dump(value,target,indent=2,sort_keys=True,allow_nan=False);target.write('\n')
        target.flush();os.fsync(target.fileno())
    print(json.dumps(dict(receipt=str(receipt),exit_code=exit_code,failure_type=failure_type,observation=observation)))
    return 0 if exit_code==0 and isinstance(observation,dict) and observation.get('accepted') is True else 1


if __name__=='__main__':raise SystemExit(main())
