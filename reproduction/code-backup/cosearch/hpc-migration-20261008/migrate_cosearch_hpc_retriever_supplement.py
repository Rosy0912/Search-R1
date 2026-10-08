"""Copy immutable CoSearch recovery assets through existing SSH routes; no GPU/job/auth changes."""
import fcntl, hashlib, importlib.util, json, os, shlex, subprocess, time, threading
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
E=BASE/'outputs/evidence'
MANIFEST=E/'native-x-hpc-retriever-supplement-manifest.jsonl'
STATE=E/'native-x-hpc-retriever-supplement-progress.json'
SOURCE='/dataset/from_h100-16/shaoshitong/search-r1-rl/cosearch'
TARGET='/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/source'
spec=importlib.util.spec_from_file_location('existing_control','/Users/shaoshitong/Documents/remote_codex_control/control.py')
ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
lock=(E/'native-x-hpc-retriever-supplement.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status={'pid':os.getpid(),'started_epoch':time.time(),'status':'waiting_source_manifest','source':SOURCE,'target':TARGET,'verified_files':0,'verified_bytes':0,'training_started':False}
def save(**kw):
 status.update(kw,updated_epoch=time.time());p=STATE.with_suffix('.tmp');p.write_text(json.dumps(status,indent=2)+'\n');p.replace(STATE)
def query(host,code):
 r=subprocess.run(ctl.route(host,'python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(host+': '+r.stderr[-1000:])
 return json.loads(r.stdout)
def pipeline(source_command,target_command,input_bytes=None):
 with (E/'native-x-hpc-retriever-supplement-transport.error.log').open('ab') as err:
  a=subprocess.Popen(ctl.route('h100-11',source_command),stdin=subprocess.PIPE if input_bytes else subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=err)
  b=subprocess.Popen(ctl.route('hpc3',target_command),stdin=subprocess.PIPE,stdout=err,stderr=err)
  try:
   writer=None
   if input_bytes:
    def feed():
     try:a.stdin.write(input_bytes)
     finally:a.stdin.close()
    writer=threading.Thread(target=feed);writer.start()
   while True:
    data=a.stdout.read(4*1024*1024)
    if not data:break
    b.stdin.write(data);status['current_stream_bytes']=status.get('current_stream_bytes',0)+len(data)
    if time.time()-status.get('updated_epoch',0)>15:save()
   b.stdin.close()
   if writer:writer.join(timeout=30);assert not writer.is_alive()
   ra=a.wait();rb=b.wait()
   if ra or rb:raise RuntimeError(f'transport exits source={ra},target={rb}')
  finally:
   for p in [a,b]:
    if p.poll() is None:p.terminate();p.wait(timeout=15)
try:
 save(status='waiting_primary_transfer')
 while True:
  primary=json.loads((E/'native-x-hpc-migration-progress.json').read_text())
  if primary['status']=='asset_transfer_complete_all_sha_verified':break
  if primary['status']=='transfer_failed_partial_preserved':raise RuntimeError('Primary transfer failed; supplement held until repaired')
  time.sleep(60)
 while True:
  rows=[json.loads(x) for x in MANIFEST.read_text().splitlines()]
  if rows and rows[-1].get('kind')=='complete':break
  time.sleep(10)
 files=[x for x in rows if x['kind']=='file'];links=[x for x in rows if x['kind']=='symlink']
 assert len(files)==rows[-1]['files']
 assert all(not x['path'].startswith('/') and '..' not in Path(x['path']).parts for x in files+links)
 total=sum(x['bytes'] for x in files);save(status='transferring',total_files=len(files),total_bytes=total)
 query('hpc3',"from pathlib import Path;import json;p=Path("+repr(TARGET)+");p.mkdir(parents=True,exist_ok=True);print(json.dumps({'created':True}))")
 large=sorted([x for x in files if x['bytes']>=64*1024*1024],key=lambda x:(0 if '/global_step_203/' in x['path'] else 1,x['path']))
 for rec in large:
  src=SOURCE+'/'+rec['path'];dst=TARGET+'/'+rec['path'];part=dst+'.migration-partial'
  info=query('hpc3',"from pathlib import Path;import hashlib,json;p=Path("+repr(dst)+");q=Path("+repr(part)+");h=hashlib.sha256();\nif p.exists():\n with p.open('rb') as f:\n  for b in iter(lambda:f.read(16777216),b''):h.update(b)\nprint(json.dumps({'existing_sha':h.hexdigest() if p.exists() else None,'partial_bytes':q.stat().st_size if q.exists() else 0}))")
  if info['existing_sha']!=rec['sha256']:
   offset=info['partial_bytes'];assert offset<=rec['bytes']
   save(current_file=rec['path'],current_stream_bytes=offset)
   reader="import sys;f=open("+repr(src)+",'rb');f.seek("+str(offset)+");\nfor b in iter(lambda:f.read(4194304),b''):sys.stdout.buffer.write(b)"
   receiver="from pathlib import Path;import sys;p=Path("+repr(part)+");p.parent.mkdir(parents=True,exist_ok=True);f=p.open('ab');\nfor b in iter(lambda:sys.stdin.buffer.read(4194304),b''):f.write(b)\nf.close()"
   pipeline('python3 -c '+shlex.quote(reader),'python3 -c '+shlex.quote(receiver))
   verifier="from pathlib import Path;import hashlib,json;p=Path("+repr(part)+");h=hashlib.sha256();\nwith p.open('rb') as f:\n for b in iter(lambda:f.read(16777216),b''):h.update(b)\nassert p.stat().st_size=="+str(rec['bytes'])+";assert h.hexdigest()=="+repr(rec['sha256'])+";p.replace("+repr(dst)+");print(json.dumps({'verified':True}))"
   query('hpc3',verifier)
  save(verified_files=status['verified_files']+1,verified_bytes=status['verified_bytes']+rec['bytes'])
 small=[x for x in files if x['bytes']<64*1024*1024]
 names=[x['path'] for x in small+links]
 save(status='transferring_small_bundle',current_file='small files and symlinks',current_stream_bytes=0)
 pipeline('tar -C '+shlex.quote(SOURCE)+' --null -T - -cf -','mkdir -p '+shlex.quote(TARGET)+' && tar -C '+shlex.quote(TARGET)+' -xf -',b''.join(n.encode()+b'\0' for n in names))
 manifest=json.dumps({'files':files,'symlinks':links})
 command='mkdir -p '+shlex.quote(TARGET)+' && cat > '+shlex.quote(TARGET+'/migration-retriever-manifest.json')
 subprocess.run(ctl.route('hpc3',command),input=manifest.encode(),check=True,timeout=120)
 save(status='verifying_all_destination_files')
 code="""from pathlib import Path
import hashlib,json,os
r=Path(TARGET);m=json.loads((r/'migration-retriever-manifest.json').read_text())
for v in m['files']:
 p=r/v['path'];assert p.stat().st_size==v['bytes'],v['path'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16777216),b''):h.update(b)
 assert h.hexdigest()==v['sha256'],v['path']
for v in m['symlinks']:assert os.readlink(r/v['path'])==v['target'],v['path']
print(json.dumps({'verified_files':len(m['files']),'verified_bytes':sum(v['bytes'] for v in m['files']),'symlinks':len(m['symlinks'])}))
""".replace('TARGET',repr(TARGET))
 r=subprocess.run(ctl.route('hpc3','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=1800)
 assert r.returncode==0,r.stderr[-1000:]
 result=json.loads(r.stdout);save(status='asset_transfer_complete_all_sha_verified',**result,source_originals_retained=True,runtime_relocation_pending=True,pure_load_performed=False)
except BaseException as exc:
 save(status='transfer_failed_partial_preserved',error=repr(exc));raise
