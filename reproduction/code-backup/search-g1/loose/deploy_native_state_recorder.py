#!/usr/bin/env python3
"""Deploy once for the saved actual submission; never submit or change a job."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time


HERE = Path(__file__).resolve().parent
ATTEMPT = HERE / "actual-stop-retry-3320"
RECORDER = HERE / "native_state_recorder.py"

DEPLOY = r'''
import base64,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin)
os.umask(0o077)
b=Path(p['base'])
assert b.is_absolute() and b.resolve()==b and b.is_dir()
assert os.getuid()==p['uid']
prep=b/'preparation';assert prep.is_dir() and prep.resolve()==prep
d=prep/'native-state-recorder'/p['job_id'];d.mkdir(parents=True,mode=0o700,exist_ok=True)
assert d.resolve()==d
handle=d/'launch-handle.json'
assert not (d/'claim.json').exists(), 'Recorder already claimed; refuse duplicate'

def write(path,value,replace=False):
 target=path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp') if replace else path
 with target.open('x',encoding='utf-8') as f:
  json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 if replace:os.replace(target,path)
 fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

reserved=dict(state='LAUNCH_RESERVED',job_id=p['job_id'],job_name=p['job_name'],
 base=p['base'],script_sha256=p['script_sha256'],submit_receipt_sha256=p['submit_receipt_sha256'],
 reserved_unix=time.time(),host=socket.gethostname(),uid=os.getuid())
write(handle,reserved)  # Exclusive, durable reservation before any launch.
source=base64.b64decode(p['source'],validate=True)
assert hashlib.sha256(source).hexdigest()==p['script_sha256']
path=prep/'native-state-recorder.py'
with path.open('xb') as f:f.write(source);f.flush();os.fsync(f.fileno())
env=os.environ.copy()
for key in ('SEARCH_G1_BASE','SEARCH_G1_PROFILE'):env.pop(key,None)
stdout=d/'stdout.log';stderr=d/'stderr.log'
with stdout.open('xb') as out,stderr.open('xb') as err:
 child=subprocess.Popen([sys.executable,'-u',str(path),p['job_id'],p['job_name'],p['base']],
  stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=env,start_new_session=True,
  cwd=str(prep),close_fds=True)
process=Path('/proc')/str(child.pid)
row=(process/'stat').read_text().rsplit(')',1)[1].split()
launched=dict(reserved,state='LAUNCHED',launched_unix=time.time(),pid=child.pid,
 start_ticks=int(row[19]),sid=int(row[3]),uid=process.stat().st_uid,
 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 script_path=str(path),stdout_path=str(stdout),stderr_path=str(stderr),
 recorder_directory=str(d),trainer_namespace_variables_unset=True)
assert launched['sid']==child.pid
write(handle,launched,replace=True)
print(json.dumps(launched,sort_keys=True))
'''


def main():
    spec = importlib.util.spec_from_file_location("saved_direct_retry", ATTEMPT / "run_direct_retry.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    receipt_bytes = (ATTEMPT / runner.native.NAMES["submitted"]).read_bytes()
    receipt = json.loads(receipt_bytes)
    job_id, job_name, base = (receipt[key] for key in ("job_id", "job_name", "base"))
    if not isinstance(job_id, str) or re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise ValueError("Actual receipt must contain one exact numeric job ID")
    if not isinstance(job_name, str) or not job_name or not isinstance(base, str) or not Path(base).is_absolute():
        raise ValueError("Actual receipt must contain the job name and absolute job base")
    if any(ord(char) < 32 for char in job_name + base):
        raise ValueError("Receipt identity contains control characters")
    if (receipt.get("submission_authentication") != "actual_sbatch_receipt" or
            type(receipt.get("returncode")) is not int or receipt["returncode"] != 0 or
            receipt.get("stdout", "").strip().split(";")[0] != job_id):
        raise ValueError("A saved actual successful sbatch receipt is required")
    source = RECORDER.read_bytes()
    payload = dict(job_id=job_id, job_name=job_name, base=base, uid=receipt["uid"],
                   script_sha256=hashlib.sha256(source).hexdigest(),
                   submit_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
                   source=base64.b64encode(source).decode("ascii"))
    intent = {key: value for key, value in payload.items() if key != "source"}
    intent["reserved_unix"] = time.time()
    runner.native.save(ATTEMPT / "native-state-recorder-deploy-intent.json", intent)
    inputs = json.loads((ATTEMPT / "direct-runtime-inputs.json").read_text())
    try:
        result = runner.native.remote("h100-8", DEPLOY, payload, timeout=45,
                                      producer_sha256=inputs["cpu_producer_sha256"])
    except Exception as error:
        runner.native.save(ATTEMPT / "native-state-recorder-deploy-unknown.json",
                           dict(intent, outcome="UNKNOWN", error_type=type(error).__name__,
                                recorded_unix=time.time(), instruction="Retain handle; never relaunch"))
        raise
    runner.native.save(ATTEMPT / "native-state-recorder-deploy-receipt.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
