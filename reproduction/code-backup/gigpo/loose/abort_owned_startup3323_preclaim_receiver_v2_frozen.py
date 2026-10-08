"""Stop only exact owned preclaim v1 management receiver; never touches training or foreign processes."""
import json, pathlib, shlex, subprocess, sys, hashlib
if sys.flags.optimize or not sys.flags.no_site or not sys.flags.ignore_environment: raise RuntimeError('Use -E -B -S')
sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control
OUT=pathlib.Path.cwd()/'outputs/gigpo-e1-instruct200'
record=json.loads((OUT/'FRESH200-STARTUP3323-ACTUAL-RECEIVER-DIAGNOSTIC-v1.json').read_bytes())
if not (record['hostname']=='h100-8' and record['state_exists'] is False and record['state_entries']==[] and record['metadata']=={} and record['queue']['returncode']==0 and len(record['owned_exact_receiver_processes'])==1 and record['owned_exact_receiver_processes'][0]['pid']==3958361 and record['owned_exact_receiver_processes'][0]['code_sha256']=='306b983e2cfd629e304993362e16693d7bc700b18c0b1967af98f6330e492023'):raise RuntimeError('Unknown outcome; no abort')
reviewp=OUT/'FRESH200-STARTUP3323-OWNED-PRECLAIM-ABORT-INDEPENDENT-REVIEW-v2-frozen.json'
review=json.loads(reviewp.read_bytes())
if not (review['status']=='PASS' and review['independent_review'] is True and review['blocking_findings']==[] and review['adapter_sha256']==hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()):raise RuntimeError('Missing exact abort review')
remote = r"""import pathlib,os,signal,time,json,hashlib,socket
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006');state=root/'logs/fresh200-startuprepair3323-once-v1';pid=3958361
if socket.gethostname().split('.')[0]!='h100-8' or state.exists() or state.is_symlink(): raise RuntimeError('Wrong node or actual claimed outcome; no abort')
p=pathlib.Path('/proc')/str(pid)
def identity():
 args=(p/'cmdline').read_bytes().split(b'\0');args=[x for x in args if x]
 if len(args)!=7 or args[:5]!=[b'/usr/bin/python3',b'-E',b'-B',b'-S',b'-c'] or hashlib.sha256(args[5]).hexdigest()!='CODEHASH' or args[6]!=b'5627002': raise RuntimeError('Foreign or different management receiver; no abort')
 fields=(p/'stat').read_text().rsplit(')',1)[1].split();return {'pid':pid,'start_ticks':int(fields[19]),'code_sha256':hashlib.sha256(args[5]).hexdigest(),'cmdline_sha256':hashlib.sha256((p/'cmdline').read_bytes()).hexdigest(),'wchan':(p/'wchan').read_text(),'io':(p/'io').read_text()}
before=identity()
if before['start_ticks']!=2230242979 or before['wchan']!='pipe_read':raise RuntimeError('Different receiver lifetime/phase')
fd=os.pidfd_open(pid,0);frozen=False
try:
 after=identity()
 if before['start_ticks']!=after['start_ticks'] or before['cmdline_sha256']!=after['cmdline_sha256'] or state.exists() or state.is_symlink():raise RuntimeError('Changed identity or actual claimed outcome')
 signal.pidfd_send_signal(fd,signal.SIGSTOP,None,0);frozen=True
 deadline=time.monotonic()+15
 while p.exists() and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='T' and time.monotonic()<deadline:time.sleep(.02)
 if not p.exists() or (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='T':raise RuntimeError('Owned receiver not confirmed frozen')
 stopped=identity()
 if stopped['start_ticks']!=before['start_ticks'] or stopped['cmdline_sha256']!=before['cmdline_sha256'] or state.exists() or state.is_symlink():raise RuntimeError('Frozen receiver identity changed or already claimed')
 io=dict(line.split(': ',1) for line in stopped['io'].splitlines())
 if int(io['wchar'])!=0 or int(io['syscw'])!=0:raise RuntimeError('Frozen receiver already reached stderr/staging; preserve and resume')
 signal.pidfd_send_signal(fd,signal.SIGTERM,None,0)
 signal.pidfd_send_signal(fd,signal.SIGCONT,None,0);frozen=False
 deadline=time.monotonic()+15
 while p.exists() and time.monotonic()<deadline:time.sleep(.2)
 gone=not p.exists()
 if not gone or state.exists() or state.is_symlink():raise RuntimeError('Receiver remains or actual claim exists; no replay')
 print(json.dumps({'status':'EXACT_OWNED_PRECLAIM_RECEIVER_TERMINATED','hostname':socket.gethostname(),'before':before,'frozen_before_termination':stopped,'frozen_state_checked':True,'pidfd_signals':['SIGSTOP','SIGTERM','SIGCONT'],'receiver_gone':gone,'state_exists':state.exists(),'no_training_or_foreign_process_touched':True}))
finally:
 if frozen:
  try:signal.pidfd_send_signal(fd,signal.SIGCONT,None,0)
  except ProcessLookupError:pass
 os.close(fd)
""".replace('CODEHASH','306b983e2cfd629e304993362e16693d7bc700b18c0b1967af98f6330e492023')
inner='env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /usr/bin/python3 -E -B -S -c '+shlex.quote(remote)
mid='env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 '+shlex.quote(inner)
route=control.ssh('mgmt-host')+['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 -o HostName=10.33.0.16 -o HostKeyAlias=100.108.173.4 -o StrictHostKeyChecking=yes h100-16 '+shlex.quote(mid)]
with (OUT/'FRESH200-STARTUP3323-OWNED-PRECLAIM-ABORT-CLAIM-v2-frozen.json').open('x') as f:json.dump({'pid':3958361,'owned_code_sha256':'306b983e2cfd629e304993362e16693d7bc700b18c0b1967af98f6330e492023','one_invocation_only':True},f)
r=subprocess.run(route,capture_output=True,text=True,timeout=60)
with (OUT/'FRESH200-STARTUP3323-OWNED-PRECLAIM-ABORT-RECEIPT-v2-frozen.json').open('x') as f:f.write(r.stdout or json.dumps({'returncode':r.returncode,'stderr':r.stderr}))
if r.returncode:raise RuntimeError('Inspect actual abort outcome before any action')
print(r.stdout)
