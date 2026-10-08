import json,hashlib,subprocess,shlex,sys,time,re
from pathlib import Path
manifest=json.loads(Path('work/topd/reproduction/h20/assets.json').read_text())
root=Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/topd-large-assets-20261003');root.mkdir(parents=True,exist_ok=True)
def remote(code):
 r=subprocess.run([sys.executable,'work/remote_topd.py'],input=code,text=True,capture_output=True)
 if r.returncode:raise RuntimeError(r.stderr)
 return r.stdout.strip()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
records=[]
for a in sorted(manifest,key=lambda a:a['size'] or 0):
 if a['size']<=8*1024**3:continue
 target='/dataset/shaoshitong/search-r1-rl/topd/assets-root/assets/'+a['path'];stage=target+'.topd-relay'
 remote('mkdir -p '+shlex.quote(str(Path(target).parent))+'\ntest -f '+shlex.quote(stage)+' || touch '+shlex.quote(stage)+'\n')
 offset=int(remote('stat -c %s '+shlex.quote(stage)+'\n'))
 url='https://hf-mirror.com/datasets/'+a['repo']+'/resolve/'+a['revision']+'/'+a['file']
 while offset<a['size']:
  end=min(offset+1024**3,a['size'])-1;p=root/(Path(a['path']).name+'.'+str(offset)+'.chunk')
  print('CHUNK_DOWNLOAD',a['path'],offset,end,flush=True)
  failures=0
  while not p.exists() or p.stat().st_size < end-offset+1:
   have=p.stat().st_size if p.exists() else 0
   begin=offset+have;stop=min(begin+64*1024**2-1,end)
   fragment=p.with_name(p.name+'.fragment');headers=p.with_name(p.name+'.headers')
   r=subprocess.run(['curl','--proxy','http://127.0.0.1:7892','--http1.1','-fLsS','--connect-timeout','15','--speed-time','60','--speed-limit','1024','--max-time','600','--max-filesize',str(stop-begin+1),'--range',str(begin)+'-'+str(stop),'-D',str(headers),'-o',str(fragment),url])
   ranges=re.findall(r'content-range:\s*bytes (\d+)-(\d+)/(\d+)',headers.read_text(errors='replace').lower()) if headers.exists() else []
   good=bool(ranges) and tuple(map(int,ranges[-1]))==(begin,stop,a['size']) and fragment.exists() and 0<fragment.stat().st_size<=stop-begin+1
   if good:
    with p.open('ab') as out,fragment.open('rb') as source:
     for block in iter(lambda:source.read(8*1024**2),b''):out.write(block)
    fragment.unlink();headers.unlink();failures=0
    print('VERIFIED_HTTP_RANGE_PREFIX',a['path'],p.stat().st_size,flush=True)
   else:
    failures+=1;print('RANGE_RETRY_RETAINED',a['path'],begin,'curl',r.returncode,flush=True)
    if failures>=6:raise RuntimeError('range unavailable; retained '+str(p))
    time.sleep(3)
  if p.stat().st_size!=end-offset+1:raise RuntimeError('chunk size mismatch; retained '+str(p))
  digest=sha(p);chunk=target+'.chunk-'+str(offset)
  subprocess.run([sys.executable,'work/transfer_file.py',str(p),chunk],check=True)
  code='set -e\ntest "$(stat -c %s '+shlex.quote(stage)+')" = '+str(offset)+'\ncat '+shlex.quote(chunk)+' >> '+shlex.quote(stage)+'\ntest "$(stat -c %s '+shlex.quote(stage)+')" = '+str(end+1)+'\nrm '+shlex.quote(chunk)+'\n'
  remote(code)
  records.append({'asset':a['path'],'start':offset,'end':end,'sha256':digest,'target_verified':True})
  (root/'chunk-manifest.json').write_text(json.dumps(records,indent=2));p.unlink();offset=end+1
  print('CHUNK_TARGET_VERIFIED_LOCAL_CLEANED',a['path'],offset,flush=True)
 observed=remote('sha256sum '+shlex.quote(stage)+'\n').split()[0]
 if observed!=a['sha256']:raise RuntimeError('upstream hash mismatch; retained '+stage)
 remote('mv '+shlex.quote(stage)+' '+shlex.quote(target)+'\n')
 print('FULL_UPSTREAM_SHA256_VERIFIED',a['path'],observed,flush=True)
