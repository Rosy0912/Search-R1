"""Stop only exact owned preclaim v2 management receiver; never touches training or foreign processes."""
import json, pathlib, shlex, subprocess, sys, hashlib
if sys.flags.optimize or not sys.flags.no_site or not sys.flags.ignore_environment: raise RuntimeError('Use -E -B -S')
sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control
OUT=pathlib.Path.cwd()/'outputs/gigpo-e1-instruct200'
record=json.loads((OUT/'FRESH200-STARTUP3318-TRANSPORT-V2-POSTTIMEOUT-RECONCILIATION-v1.json').read_bytes())
if not (record['hostname']=='h100-8' and record['state_exists'] is False and record['entries']==[] and record['SUBMIT-INTENT.json'] is None and record['SUBMIT-RECEIPT.json'] is None and record['startup_error'] is None and record['queue']['returncode']==record['gpu']['returncode']==0 and record['queue']['stdout']==record['gpu']['stdout']=='' and [x['pid'] for x in record['matching_receivers']]==[3883617]): raise RuntimeError('Unknown outcome; no abort')
remote = r"""import pathlib,os,signal,time,json,hashlib,socket
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006');state=root/'logs/fresh200-startuprepair3318-once-v1';pid=3883617
if socket.gethostname().split('.')[0]!='h100-8' or state.exists() or state.is_symlink(): raise RuntimeError('Wrong node or actual claimed outcome; no abort')
p=pathlib.Path('/proc')/str(pid)
def identity():
 args=(p/'cmdline').read_bytes().split(b'\0');args=[x for x in args if x]
 if len(args)!=6 or args[:5]!=[b'/usr/bin/python3',b'-E',b'-B',b'-S',b'-c'] or hashlib.sha256(args[5]).hexdigest()!='CODEHASH': raise RuntimeError('Foreign or different management receiver; no abort')
 fields=(p/'stat').read_text().rsplit(')',1)[1].split();return {'pid':pid,'start_ticks':int(fields[19]),'code_sha256':hashlib.sha256(args[5]).hexdigest(),'cmdline_sha256':hashlib.sha256((p/'cmdline').read_bytes()).hexdigest(),'wchan':(p/'wchan').read_text(),'io':(p/'io').read_text()}
before=identity();fd=os.pidfd_open(pid,0)
try:
 after=identity()
 if before['start_ticks']!=after['start_ticks'] or before['cmdline_sha256']!=after['cmdline_sha256'] or state.exists():raise RuntimeError('Changed identity or claimed outcome; no abort')
 signal.pidfd_send_signal(fd,signal.SIGTERM,None,0)
 deadline=time.monotonic()+15
 while p.exists() and time.monotonic()<deadline:time.sleep(.2)
 gone=not p.exists()
 if not gone or state.exists():raise RuntimeError('Receiver remains or actual claim exists; no replay')
 print(json.dumps({'status':'EXACT_OWNED_PRECLAIM_RECEIVER_TERMINATED','hostname':socket.gethostname(),'before':before,'pidfd_signal':'SIGTERM','receiver_gone':gone,'state_exists':state.exists(),'no_training_or_foreign_process_touched':True}))
finally:os.close(fd)
""".replace('CODEHASH','1f11e5e4fe48af2791c3020a344bad99209fa4493dcf482ee3189c62d0a85252')
inner='env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /usr/bin/python3 -E -B -S -c '+shlex.quote(remote)
mid='env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 '+shlex.quote(inner)
route=control.ssh('mgmt-host')+['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 -o HostName=10.33.0.16 -o HostKeyAlias=100.108.173.4 -o StrictHostKeyChecking=yes h100-16 '+shlex.quote(mid)]
with (OUT/'FRESH200-STARTUP3318-OWNED-PRECLAIM-ABORT-CLAIM-v1.json').open('x') as f:json.dump({'pid':3883617,'owned_code_sha256':'1f11e5e4fe48af2791c3020a344bad99209fa4493dcf482ee3189c62d0a85252','one_invocation_only':True},f)
r=subprocess.run(route,capture_output=True,text=True,timeout=60)
with (OUT/'FRESH200-STARTUP3318-OWNED-PRECLAIM-ABORT-RECEIPT-v1.json').open('x') as f:f.write(r.stdout or json.dumps({'returncode':r.returncode,'stderr':r.stderr}))
if r.returncode:raise RuntimeError('Inspect actual abort outcome before any action')
print(r.stdout)
