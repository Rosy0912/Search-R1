"""Write exactly one independently produced prepared-deployment review."""
import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REMOTE = r'''
import base64,hashlib,json,os,sys,time
from pathlib import Path
p=json.load(sys.stdin);intent=p['intent'];base=Path(intent['base'])
assert str(base)=='/data/shaoshitong/Search-G1-a800-fresh200-'+intent['deployment_id']
repo=base/'repo';assert repo==repo.resolve() and repo.stat().st_uid==os.getuid()
sys.path.insert(0,str(repo))
from reproduction.a800_common.prepare_actor_probe_production import validate_intent,environment
from reproduction.a800_common.actor_probe_production_review import read,verify_documents,verify,actual_preparation,PREPARED,REVIEW
validate_intent(intent)
assert read(base,'preparation/deployment-intent.json')[0]==intent
run,_=read(base,'logs/run-manifest.json');prepared,pfile=read(base,PREPARED)
env=environment(base,Path(run['ray_tmp_dir']));os.environ.clear();os.environ.update(env)
os.chdir(repo)
raw=base64.b64decode(p['review_base64'],validate=True);assert len(raw)<=65536
assert hashlib.sha256(raw).hexdigest()==p['review_sha256']
review=json.loads(raw);verify_documents(review,prepared,run,pfile['sha256'])
assert review['independent_initialization_without_parent_context'] is True
assert read(base,'logs/deployment-status.json')[0]['stage']=='PREPARED_FOR_INDEPENDENT_DEPLOYMENT_REVIEW'
assert prepared['preparation']==actual_preparation(base,run), 'Actual preparation must match before writing review'
path=base/REVIEW
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'wb') as target:target.write(raw);target.flush();os.fsync(target.fileno())
fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
try:os.fsync(fd)
finally:os.close(fd)
assert hashlib.sha256(path.read_bytes()).hexdigest()==p['review_sha256']
verified=verify(base,run)
print(json.dumps(dict(role='actual_independent_prepared_review_exclusive_write',identity=review['identity'],
 review_sha256=p['review_sha256'],bytes=len(raw),exclusive=True,native_algorithm_accepted=False,
 actual_preparation_still_matches=True,verified=verified,checked_unix=time.time())))
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
    raw=args.review.read_bytes();assert len(raw)<=65536
    review=json.loads(raw)
    assert review['accepted'] is True and review['native_algorithm_accepted'] is False
    assert review['identity']['source_commit']==intent['source_commit']
    claim=args.review.with_name(args.review.stem+'.write-once.json')
    with claim.open('x') as target:
        json.dump(dict(claimed_unix=time.time(),base=intent['base'],review_sha256=hashlib.sha256(raw).hexdigest(),no_auto_retry=True),target)
        target.write('\n');target.flush();os.fsync(target.fileno())
    payload=dict(intent=intent,review_base64=base64.b64encode(raw).decode(),review_sha256=hashlib.sha256(raw).hexdigest())
    # Bootstrap uses the already configured remote runtime before parsing stdin.
    bootstrap='import os;os.environ["PYTHONDONTWRITEBYTECODE"]="1";os.execv("/data/search-g1/env/bin/python",["/data/search-g1/env/bin/python","-c",'+repr(REMOTE)+'])'
    path=args.root/'preparation'/('actor-probes-ten-review-write-'+str(time.time_ns())+'.json')
    stdout=path.with_suffix('.stdout');stderr=path.with_suffix('.stderr')
    exit_code=None;failure_type=None
    with stdout.open('xb') as out,stderr.open('xb') as errors:
        try:
            result=subprocess.run(ssh_command(bootstrap),input=json.dumps(payload,allow_nan=False).encode(),
                stdout=out,stderr=errors,timeout=240)
            exit_code=result.returncode
        except (subprocess.TimeoutExpired,OSError) as failure:
            failure_type=type(failure).__name__
        out.flush();os.fsync(out.fileno());errors.flush();os.fsync(errors.fileno())
    value=None;output_bytes=stdout.stat().st_size
    if output_bytes<=65536:
        try:value=json.loads(stdout.read_bytes())
        except (ValueError,UnicodeDecodeError):pass
    with path.open('x') as target:
        json.dump(dict(result=value,transport_exit_code=exit_code,failure_type=failure_type,
            transport_stdout=str(stdout),transport_stderr=str(stderr),stdout_bytes=output_bytes,
            transport_stderr_bytes=stderr.stat().st_size,checked_unix=time.time(),no_automatic_retry=True),
            target,indent=2,sort_keys=True,allow_nan=False)
        target.write('\n');target.flush();os.fsync(target.fileno())
    assert exit_code==0 and failure_type is None, \
        'Review write uncertain/failed: diagnostics saved; preserve claim and inspect, never repeat blindly'
    assert value is not None and output_bytes<=65536, \
        'Review write response uncertain: diagnostics saved; inspect the original handle'
    print(json.dumps(dict(receipt=str(path),review_sha256=value['review_sha256'],identity=value['identity'],exclusive=True,native_algorithm_accepted=False)))

if __name__=='__main__':main()
