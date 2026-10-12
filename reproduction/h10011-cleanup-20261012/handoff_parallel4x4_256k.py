import pathlib,subprocess,shlex
mgmt='''import os,sys,json,time,pathlib,signal,fcntl,subprocess
r=pathlib.Path('/home/lichen/h100-cleanup-20261012-bapo');token=sys.stdin.readline().strip();assert token
c=json.loads((r/'controller.json').read_text());pid=c['pid'];cmd=pathlib.Path('/proc/%d/cmdline'%pid).read_bytes();assert b'/home/lichen/h100-cleanup-20261012-bapo/uploader_parallel4x4_gated.py' in cmd;assert os.getpgid(pid)==pid
members=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  fields=(p/'stat').read_text().split(') ')[1].split()
  if int(fields[2])==pid:members.append(int(p.name))
 except (OSError,ValueError):pass
progressfiles=[json.loads(p.read_text()) for p in r.glob('progress-*.json')];verifiedpaths={json.loads(p.read_text())['path'] for p in r.glob('file-*-verified.json')}; prefix=json.loads((r/'plan.json').read_text())['prefix'];progress={'read_position':sum(p['read_position'] for p in progressfiles if prefix+p['file'] not in verifiedpaths)}
archive=r/'parallel4x4-gated-progress-before-256k-handoff';archive.mkdir(exist_ok=True)
for p in list(r.glob('progress-*.json'))+list(r.glob('file-*-error.json')):p.replace(archive/p.name)
os.killpg(pid,signal.SIGTERM)
for attempt in range(100):
 live=[]
 for member in members:
  try:
   fields=pathlib.Path('/proc/%d/stat'%member).read_text().split(') ')[1].split()
   if fields[0]!='Z':live.append(member)
  except OSError:pass
 if not live:break
 time.sleep(.1)
assert not live,'Old source uploader processes still alive'
l=(r/'uploader.lock').open('w');fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(l,fcntl.LOCK_UN);l.close()
receipt={'epoch':time.time(),'old_pid':pid,'old_group_members':members,'all_old_members_terminated':True,'old_lock_released':True,'uncommitted_retransmit_bytes':progress['read_position'],'prior_verified_receipts':len(list(r.glob('file-*-verified.json'))),'generation':'parallel4x4-256k'}
(r/'handoff4x4-256k.json').write_text(json.dumps(receipt))
f=(r/'upload-parallel4x4-256k.log').open('a');p=subprocess.Popen(['python3',str(r/'uploader_parallel4x4_256k.py')],stdin=subprocess.PIPE,stdout=f,stderr=f,start_new_session=True);p.stdin.write((token+'\\n').encode());p.stdin.close();receipt['new_pid']=p.pid;(r/'handoff4x4-256k.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
'''
p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ProxyCommand=none','-o','ProxyJump=none','mgmt-host','python3 -c '+shlex.quote(mgmt)],input=pathlib.Path('/Users/shaoshitong/.cache/huggingface/token').read_text().strip()+'\n',capture_output=True,text=True,timeout=40)
pathlib.Path('outputs/cleanup-h10011/upload-handoff4x4-256k.json').write_text(p.stdout)
print(p.returncode,p.stdout,p.stderr)
