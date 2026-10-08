#!/usr/bin/env python3
"""Replace only the saved job-3324 observer; never alter a Slurm job."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import time


HERE = Path(__file__).resolve().parent
ATTEMPT = HERE / "actual-stop-retry-3320"
BOOTLOADER = (
    "import json,sys;from pathlib import Path;"
    "c=json.loads(Path(sys.argv[1]).read_text());"
    "sys.argv=c['argv'];s=c['script'];"
    "exec(compile(Path(s).read_bytes(),s,'exec'),"
    "{'__name__':'__main__','__file__':s})"
)

REPAIR = r'''
import base64,hashlib,json,os,select,signal,socket,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);os.umask(0o077)
old=p['old_handle'];b=Path(p['base'])
assert b.is_absolute() and b.resolve()==b and b.is_dir()
assert os.getuid()==p['uid']==old['uid']
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
old_dir=Path(old['recorder_directory']);old_source=Path(old['script_path'])
assert old_dir==b/'preparation/native-state-recorder'/p['job_id']
assert old_source==b/'preparation/native-state-recorder.py'
saved=json.loads((old_dir/'launch-handle.json').read_text())
assert saved==old, 'Original recorder handle changed; refuse unrelated stop'
original=old_source.read_bytes()
assert hashlib.sha256(original).hexdigest()==old['script_sha256']==p['original_sha256']
claim=json.loads((old_dir/'claim.json').read_text())
expected=dict(JobId=p['job_id'],JobName=p['job_name'],
 Command=str(b/'repo/reproduction/common8/launch_native.sh'),WorkDir=str(b/'repo'))
assert claim['expected_job']==expected
for k in ('pid','start_ticks','sid','uid','boot_id','script_sha256'):
 assert claim['recorder'][k]==old[k], 'Original recorder claim identity changed'

watchers=Path('/mnt/data/shaoshitong/g1-native-watchers')
watchers.mkdir(mode=0o700,exist_ok=True);assert watchers.resolve()==watchers
d=watchers/(p['job_id']+'-v2')
d.mkdir(mode=0o700,exist_ok=True);assert d.resolve()==d
assert not (d/'launch-handle.json').exists(), 'External recorder handle exists'
new_records=b/'preparation/native-state-recorder'/(p['job_id']+'-external-v2')
assert not (new_records/'claim.json').exists(), 'External recorder claim exists'

def write(path,value,replace=False):
 target=path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp') if replace else path
 with target.open('x',encoding='utf-8') as f:
  json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 if replace:os.replace(target,path)
 fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

intent=dict(state='EXTERNAL_REPAIR_RESERVED',job_id=p['job_id'],job_name=p['job_name'],
 base=p['base'],old_handle=old,script_sha256=p['script_sha256'],
 submit_receipt_sha256=p['submit_receipt_sha256'],reserved_unix=time.time(),
 scope='Stop only the exact old observer; preserve original evidence; no Slurm mutation')
write(d/'launch-intent.json',intent)  # Exclusive reservation, never reclaimed.
assert hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal')
stop=dict(old_pid=old['pid'],old_start_ticks=old['start_ticks'],old_sid=old['sid'],
 old_uid=old['uid'],old_script_sha256=old['script_sha256'],recorded_unix=time.time())
try:
 fd=os.pidfd_open(old['pid'],0)
except ProcessLookupError:
 stop['outcome']='ALREADY_ABSENT'
else:
 try:
  proc=Path('/proc')/str(old['pid'])
  try:
   row=(proc/'stat').read_text().rsplit(')',1)[1].split()
   argv=(proc/'cmdline').read_bytes().split(b'\0')
   argv=[x.decode('utf-8') for x in argv if x]
   assert (int(row[19]),int(row[3]),proc.stat().st_uid)==(
    old['start_ticks'],old['sid'],old['uid']), 'PID birth/SID/UID mismatch; refuse stop'
   assert len(argv)==6 and argv[1:]==['-u',str(old_source),p['job_id'],p['job_name'],p['base']], 'Old observer argv mismatch; refuse stop'
   assert Path(argv[0]).resolve()==Path(sys.executable).resolve()
   assert (proc/'exe').resolve()==Path(sys.executable).resolve()
  except (FileNotFoundError,ProcessLookupError):
   poll=select.poll();poll.register(fd,select.POLLIN)
   assert poll.poll(0), 'Process evidence vanished without pidfd exit'
   stop['outcome']='ALREADY_EXITED'
  else:
   stop['checked_os_argv']=argv
   signal.pidfd_send_signal(fd,signal.SIGTERM,None,0)
   poll=select.poll();poll.register(fd,select.POLLIN)
   assert poll.poll(10000), 'Old observer did not exit; retain reservation and do not relaunch'
   stop['outcome']='EXACT_OBSERVER_STOPPED_WITH_PIDFD_SIGTERM'
 finally:os.close(fd)
write(d/'old-recorder-stop.json',stop)

source=base64.b64decode(p['source'],validate=True)
assert hashlib.sha256(source).hexdigest()==p['script_sha256']
script=d/'recorder.py'
with script.open('xb') as f:f.write(source);f.flush();os.fsync(f.fileno())
config=d/'config.json'
write(config,dict(script=str(script),argv=[str(script),p['job_id'],p['job_name'],p['base']]))
env=os.environ.copy()
for key in ('SEARCH_G1_BASE','SEARCH_G1_PROFILE'):env.pop(key,None)
command=[sys.executable,'-c',p['bootloader'],str(config)]
assert not any(x==p['base'] or x.startswith(p['base']+'/') for x in command)
handle=d/'launch-handle.json'
write(handle,dict(intent,state='EXTERNAL_LAUNCH_RESERVED',old_stop=stop))
stdout=d/'stdout.log';stderr=d/'stderr.log'
with stdout.open('xb') as out,stderr.open('xb') as err:
 child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
  env=env,start_new_session=True,cwd=str(d),close_fds=True)
proc=Path('/proc')/str(child.pid)
row=(proc/'stat').read_text().rsplit(')',1)[1].split()
argv=[x.decode('utf-8') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
assert argv==command
assert not any(x==p['base'] or x.startswith(p['base']+'/') for x in argv)
launched=dict(state='EXTERNAL_LAUNCHED',job_id=p['job_id'],job_name=p['job_name'],
 base=p['base'],host=socket.gethostname(),launched_unix=time.time(),pid=child.pid,
 start_ticks=int(row[19]),sid=int(row[3]),uid=proc.stat().st_uid,
 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 script_sha256=p['script_sha256'],original_script_sha256=old['script_sha256'],
 submit_receipt_sha256=p['submit_receipt_sha256'],script_path=str(script),
 config_path=str(config),canonical_os_argv=argv,cwd=str(d),old_stop=stop,
 old_handle_preserved=str(old_dir/'launch-handle.json'),
 old_claim_preserved=str(old_dir/'claim.json'),
 old_logs_preserved=[old['stdout_path'],old['stderr_path']],
 recorder_directory=str(new_records),stdout_path=str(stdout),stderr_path=str(stderr),
 trainer_namespace_variables_unset=True)
assert launched['sid']==child.pid and launched['uid']==p['uid']
write(handle,launched,replace=True)
print(json.dumps(launched,sort_keys=True))
'''


def main():
    spec = importlib.util.spec_from_file_location("saved_direct_retry", ATTEMPT / "run_direct_retry.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    receipt_bytes = (ATTEMPT / runner.native.NAMES["submitted"]).read_bytes()
    receipt = json.loads(receipt_bytes)
    old = json.loads((ATTEMPT / "native-state-recorder-deploy-receipt.json").read_text())
    assert receipt["job_id"] == old["job_id"] == "3324"
    assert receipt["job_name"] == old["job_name"]
    assert receipt["base"] == old["base"]
    assert receipt["uid"] == old["uid"]
    assert (old["pid"], old["start_ticks"], old["sid"]) == (3890315, 2230000684, 3890315)
    assert receipt["submission_authentication"] == "actual_sbatch_receipt" and receipt["returncode"] == 0
    original = (HERE / "native_state_recorder.py").read_bytes()
    original_sha = hashlib.sha256(original).hexdigest()
    assert original_sha == old["script_sha256"]
    anchor = b'directory = base / "preparation/native-state-recorder" / args.job_id'
    replacement = b'directory = base / "preparation/native-state-recorder" / (args.job_id + "-external-v2")'
    assert original.count(anchor) == 1
    source = original.replace(anchor, replacement, 1)
    payload = dict(job_id=receipt["job_id"], job_name=receipt["job_name"], base=receipt["base"],
                   uid=receipt["uid"], old_handle=old, original_sha256=original_sha,
                   script_sha256=hashlib.sha256(source).hexdigest(),
                   submit_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
                   source=base64.b64encode(source).decode("ascii"), bootloader=BOOTLOADER)
    intent = {key: value for key, value in payload.items() if key != "source"}
    intent["reserved_unix"] = time.time()
    runner.native.save(ATTEMPT / "external-recorder-repair-intent.json", intent)
    inputs = json.loads((ATTEMPT / "direct-runtime-inputs.json").read_text())
    try:
        result = runner.native.remote("h100-8", REPAIR, payload, timeout=45,
                                      producer_sha256=inputs["cpu_producer_sha256"])
    except Exception as error:
        runner.native.save(ATTEMPT / "external-recorder-repair-unknown.json",
                           dict(intent, outcome="UNKNOWN", error_type=type(error).__name__,
                                recorded_unix=time.time(), instruction="Retain all handles; never relaunch"))
        raise
    runner.native.save(ATTEMPT / "external-recorder-repair-receipt.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
