"""Latest human stop: signal only the identified own job3338 launcher via pidfd."""
import datetime, json, os, pathlib, shlex, subprocess

BASE=pathlib.Path(__file__).resolve().parent.parent
OUT=BASE/'outputs/gigpo-e1-instruct200'
CODE=r'''
import os,sys,json,pathlib,subprocess,socket,hashlib,signal,time
ROOT=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006')
RUN=ROOT/'runs/gigpo-e1-fresh200-live-job3338-1791440187570483460'
BIND=ROOT/'logs/fresh200-cudasynchronous-repair3337-once-v1/BINDING.json'
SHA='d73886efdd76bd50fbb34f3e53c665c42f371bc200adf1d7bdc6920f8bd55844'
HELPER=ROOT/'tools/fresh200-continuation-v19-cudasynchronous1/calibration_run_v1.py'
STATE=ROOT/'logs/human-stop3338-for-xformers-once-v1'
def req(ok,msg):
 if not ok:raise RuntimeError(msg)
def once(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def proc(pid):
 p=pathlib.Path('/proc')/str(pid);stat=(p/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
 env={}
 for item in (p/'environ').read_bytes().split(b'\0'):
  if b'=' in item:
   k,v=item.split(b'=',1)
   if k in (b'SEARCH_R1_BASELINE_OUTPUT',b'SEARCH_R1_BASELINE_COMMIT',b'GIGPO_POLICY_NATIVE_BINDING_SHA256'):env[k.decode()]=v.decode()
 return dict(pid=pid,ppid=int(tail[1]),start_ticks=int(tail[19]),command=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v],cgroup=(p/'cgroup').read_text(),env=env)
req(socket.gethostname().split('.')[0]=='h100-8','Wrong host')
req(hashlib.sha256(BIND.read_bytes()).hexdigest()==SHA,'Own binding changed')
STATE.mkdir(exist_ok=False)
once(STATE/'STOP-INTENT.json',dict(status='HUMAN_AUTHORIZED_STOP_OWN3338_NEVER_REPLAY',job=3338,run=str(RUN),binding_sha256=SHA,observed_instruction_utc=HUMAN_TIME,reason='Switch to documented XFORMERS workaround; not a claim that job3338 has failed'))
s=subprocess.run(['scontrol','show','job','3338','-o'],capture_output=True,text=True,timeout=20)
req(s.returncode==0 and 'JobId=3338 ' in s.stdout and 'JobState=RUNNING ' in s.stdout and 'NodeList=h100-8 ' in s.stdout and SHA in s.stdout,'Exact own live scheduler identity missing')
d=proc(1984437)
req(d['env'].get('SEARCH_R1_BASELINE_OUTPUT')==str(RUN) and d['env'].get('SEARCH_R1_BASELINE_COMMIT')=='ebe91bfebd56c3bd445219dcd94342d815f899f0' and d['env'].get('GIGPO_POLICY_NATIVE_BINDING_SHA256')==SHA,'Driver identity changed')
req('job_3338/step_0' in d['cgroup'],'Driver outside own Slurm cgroup')
parent=d;launcher=None
for _ in range(30):
 if parent['command'][1:]==[str(HELPER),str(BIND),SHA]:launcher=parent;break
 req(parent['ppid']>1,'Launcher ancestor absent; do not signal')
 parent=proc(parent['ppid'])
req(launcher is not None and 'job_3338/step_0' in launcher['cgroup'],'Own launcher identity missing')
progress=json.loads((RUN/'live-progress.json').read_text()) if (RUN/'live-progress.json').exists() else None
receipt=dict(status='SIGNALLED_OWN_LAUNCHER_TERMINAL_NOT_YET_CONFIRMED',job=3338,run=str(RUN),human_authorized=True,human_observed_utc=HUMAN_TIME,classification='HUMAN_STOP_TO_APPLY_XFORMERS_WORKAROUND',scheduler_before=dict(returncode=s.returncode,stdout=s.stdout,stderr=s.stderr),driver=d,launcher=launcher,progress_before_signal=progress,checkpoint_directories=[str(p) for p in sorted((RUN/'checkpoints/actor').glob('global_step_*'))],signal='SIGTERM',signal_sent=False)
fd=os.pidfd_open(launcher['pid'])
try:
 live=proc(launcher['pid']);req(live['start_ticks']==launcher['start_ticks'] and live['command']==launcher['command'] and live['cgroup']==launcher['cgroup'],'Launcher changed before pidfd signal')
 signal.pidfd_send_signal(fd,signal.SIGTERM,None,0);receipt['signal_sent']=True
finally:os.close(fd)
once(STATE/'STOP-RECEIPT.json',receipt)
print(json.dumps(receipt))
'''

def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    intent=OUT/'JOB3338-HUMAN-STOP-XFORMERS-LOCAL-INTENT-v1.json'
    with intent.open('x') as f:
        json.dump(dict(job=3338,observed_utc=now,original_instruction='Stop current3338 and apply XFORMERS workaround; preserve history and add fault-table row',never_replay=True),f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    code=CODE.replace('HUMAN_TIME',repr(now))
    worker=shlex.join(['env','-u','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python','-B','-c',code])
    mgmt=r'''import json,shlex,subprocess,sys
from pathlib import Path
cfg=json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())
target='h100-8';ip=cfg['targets'][target];gateway=cfg['gateway_ip'];user=cfg['user']
assert gateway=='10.33.0.18' and ip=='10.33.0.8' and cfg['management_identity_file']=='/home/lichen/.ssh/id_ed25519'
prefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ConnectionAttempts=1']
args=prefix+['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',user+'@'+gateway])]
keys=cfg.get('trusted_target_host_keys',{}).get(target)
if keys:args+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\n'.join(keys)])]
r=subprocess.run(args+[user+'@'+ip,WORKER],capture_output=True,text=True,timeout=50)
print(json.dumps({'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}));sys.exit(r.returncode)
'''.replace('WORKER',repr(worker))
    outer=['env','-u','LD_LIBRARY_PATH','/usr/bin/ssh','-T','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host','env -u LD_LIBRARY_PATH /usr/bin/python3 -c '+shlex.quote(mgmt)]
    try:
        r=subprocess.run(outer,capture_output=True,text=True,timeout=65)
        record=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,observed_utc=now)
    except BaseException as e:record=dict(status='UNCERTAIN_NEVER_REPLAY',error=repr(e),observed_utc=now)
    p=OUT/'JOB3338-HUMAN-STOP-XFORMERS-TRANSPORT-v1.json'
    with p.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(record))
    if record.get('returncode')!=0:raise RuntimeError('Read original stop intent/receipt; never replay')

if __name__=='__main__':main()
