#!/usr/bin/env python3
"""Own-node official transport only. No model loading or algorithm acceptance."""
import argparse, concurrent.futures, fcntl, hashlib, json, os, pathlib, socket, subprocess, threading, time, signal
REVISION='1818d35814b8319459f4bd55ed1ac8709630f003'
METADATA_SHA='e546ba207f643dbe1a098323e28d45ff07942a6390bc73b8391c0acfc20d84d1'
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
SMALL=('config.json','generation_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','model.safetensors.index.json')
NAMES=SMALL+tuple('model-%05d-of-00017.safetensors'%i for i in range(1,18))
PROXY='http://127.0.0.1:18089'
RESERVE=2*1024**3

def inventory(path):
    raw=pathlib.Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=METADATA_SHA: raise ValueError('official metadata bytes mismatch')
    data=json.loads(raw)
    if data['sha']!=REVISION or data['id']!='Qwen/Qwen2.5-32B': raise ValueError('official identity mismatch')
    siblings={x['rfilename']:x for x in data['siblings']}
    items={n:siblings[n] for n in NAMES}
    for name,item in items.items():
        if item['size']<=0 or len(item['blobId'])!=40: raise ValueError('bad official inventory')
        if name.endswith('.safetensors') and (len(item.get('lfs',{}).get('sha256',''))!=64 or item['lfs']['size']!=item['size']): raise ValueError('missing LFS binding')
    return items

def check_regular(path):
    path=pathlib.Path(path)
    if path.is_symlink(): raise ValueError('symlink forbidden: '+str(path))
    if path.exists() and not path.is_file(): raise ValueError('not regular file')

def verify(path,item):
    check_regular(path)
    sha=hashlib.sha256(); git=hashlib.sha1(); git.update(('blob %d\0'%item['size']).encode())
    with open(path,'rb') as f:
        before=os.fstat(f.fileno())
        if before.st_size!=item['size']: raise ValueError('official size mismatch')
        while True:
            block=f.read(8*1024**2)
            if not block: break
            sha.update(block); git.update(block)
        after=os.fstat(f.fileno())
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns): raise ValueError('changed while verifying')
    digest=sha.hexdigest()
    if item.get('lfs'):
        if digest!=item['lfs']['sha256']: raise ValueError('official LFS SHA mismatch')
    elif git.hexdigest()!=item['blobId']: raise ValueError('official Git blob mismatch')
    return {'remote_sha_verified':True,'official_lfs_sha256':item.get('lfs',{}).get('sha256'),'size':after.st_size,'sha256':digest,'git_blob_sha1':git.hexdigest(),'official_verified':True}

def atomic_json(path,data):
    check_regular(path); tmp=pathlib.Path(str(path)+'.tmp')
    check_regular(tmp)
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump(data,f,sort_keys=True,indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY); os.fsync(fd); os.close(fd)

def curl_command(name,partial,size):
    return ['curl','--disable','--http1.1','--proxy',PROXY,'--noproxy','','--proto','=https','--proto-redir','=https','--location','--fail','--show-error','--silent','--connect-timeout','45','--speed-time','120','--speed-limit','1024','--retry','0','--max-filesize',str(size),'--continue-at','-','--output',str(partial),'https://huggingface.co/Qwen/Qwen2.5-32B/resolve/'+REVISION+'/'+name]

def clean_env():
    env=dict(os.environ)
    for key in list(env):
        if key.lower() in ('http_proxy','https_proxy','all_proxy','no_proxy'): del env[key]
    env['CUDA_VISIBLE_DEVICES']=''; return env

class Downloader:
    def __init__(self,target,state,items,runner=None,sleeper=time.sleep):
        self.target=pathlib.Path(target); self.state=pathlib.Path(state); self.items=items
        self.runner=runner or self.run_curl; self.sleeper=sleeper; self.mu=threading.Lock(); self.stop=threading.Event(); self.active=set()
        self.data={'target_canonical':str(self.target.resolve()),'scope':'OFFICIAL_NATIVE_H16PROXY_TRANSPORT_NO_TRAINING','revision':REVISION,'metadata_sha256':METADATA_SHA,'proxy':PROXY,'host':socket.gethostname(),'pid':os.getpid(),'max_parallel':3,'complete':False,'status':'VERIFYING_EXISTING','files':{},'errors':{},'active':[],'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'parent_pid':os.getppid(),'no_gpu_api_used':True}
    def save(self):
        self.data['updated_epoch']=time.time(); atomic_json(self.state,self.data)
    def run_curl(self,name,partial):
        # Only this worker's direct curl child is waited/reaped; no global cleanup.
        logpath=self.state.parent/(name+'.h16proxy.curl.log'); check_regular(logpath)
        logfd=os.open(logpath,os.O_WRONLY|os.O_CREAT|os.O_APPEND|os.O_NOFOLLOW,0o600)
        with os.fdopen(logfd,'ab') as log:
            process=subprocess.Popen(curl_command(name,partial,self.items[name]['size']),env=clean_env(),stdout=subprocess.DEVNULL,stderr=log)
            with self.mu: self.data.setdefault('curl_children',[]).append({'pid':process.pid,'name':name,'started':time.time()}); self.save()
            while True:
                try: rc=process.wait(timeout=1); break
                except subprocess.TimeoutExpired:
                    if self.stop.is_set():
                        process.terminate()
                        try: rc=process.wait(timeout=10)
                        except subprocess.TimeoutExpired: process.kill(); rc=process.wait()
                        break
            with self.mu:
                for rec in self.data['curl_children']:
                    if rec['pid']==process.pid: rec.update(returncode=rc,finished=time.time())
                self.save()
            if rc: raise RuntimeError('curl failed rc=%d; partial retained'%rc)
    def one(self,name):
        final=self.target/name; partial=self.target/(name+'.h16proxy.partial'); item=self.items[name]
        check_regular(final); check_regular(partial)
        if final.exists(): return verify(final,item)
        if partial.exists() and partial.stat().st_size>item['size']: raise ValueError('oversized partial retained')
        # Full-size partial is independently verified; failed verification never discards it.
        for attempt in range(64):
            if self.stop.is_set(): raise RuntimeError('admissions stopped; partial retained')
            if not partial.exists() or partial.stat().st_size<item['size']:
                try: self.runner(name,partial)
                except RuntimeError as error:
                    current=partial.stat().st_size if partial.exists() else 0
                    with self.mu:
                        self.data.setdefault('retries',[]).append({'name':name,'attempt':attempt+1,'partial_size':current,'error':str(error),'epoch':time.time()}); self.save()
                    if current>item['size']: raise ValueError('oversized partial retained')
                    if current==item['size']: break
                    if attempt==63 or self.stop.is_set(): raise
                    self.sleeper(3)
                    continue
            if partial.stat().st_size==item['size']: break
        receipt=verify(partial,item)
        with open(partial,'rb') as f: os.fsync(f.fileno())
        try: os.link(partial,final) # atomic no-replace: cannot overwrite any prior target
        except FileExistsError: verify(final,item)
        partial.unlink()
        fd=os.open(self.target,os.O_RDONLY); os.fsync(fd); os.close(fd)
        receipt=verify(final,item)
        receipt.update(official_lfs_sha256=item.get('lfs',{}).get('sha256'),remote_sha_verified=True)
        return receipt
    def worker(self,name):
        with self.mu:
            if self.stop.is_set(): return
            self.active.add(name); self.data['active']=sorted(self.active); self.save()
        try:
            rec=self.one(name)
            with self.mu: self.data['files'][name]=rec; self.save()
        except Exception as error:
            self.stop.set()
            with self.mu: self.data['errors'][name]=str(error); self.save()
        finally:
            with self.mu: self.active.remove(name); self.data['active']=sorted(self.active); self.save()
    def execute(self):
        # Existing official small files and first shard are verified before any network admission.
        for name in self.items:
            final=self.target/name
            if final.exists(): self.data['files'][name]=verify(final,self.items[name])
        self.data['status']='DOWNLOADING'; self.save()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(self.worker,[n for n in self.items if n not in self.data['files']]))
        if self.stop.is_set() and not self.data['errors']: self.data['errors']['interrupted']='owned stop received; no further admission'
        if self.data['errors']: self.data['status']='FAILED_PARTIALS_PRESERVED'
        else:
            # Never trust old state, rehash every final file before COMPLETE.
            self.data['files']={n:verify(self.target/n,item) for n,item in self.items.items()}
            self.data['status']='VERIFIED_MANIFEST_PENDING'
            atomic_json(pathlib.Path(str(self.state)+'.manifest.json'),self.data)
            self.data['manifest_sha256']=hashlib.sha256(pathlib.Path(str(self.state)+'.manifest.json').read_bytes()).hexdigest()
            self.data['status']='COMPLETE_OFFICIAL_TRANSPORT_ONLY'; self.data['complete']=True
        self.save(); return not self.data['errors']

def handoff_check(path,target,proc=pathlib.Path('/proc')):
    raw=pathlib.Path(path).read_bytes(); rec=json.loads(raw)
    if rec.get('scope')!='OWN_H11_OFFICIAL_WRITER_HANDOFF' or rec.get('target')!=str(target) or not rec.get('process_identities'): raise ValueError('real writer identity handoff required')
    for identity in rec['process_identities']:
        if not all(k in identity for k in ('pid','ppid','starttime')): raise ValueError('incomplete writer identity')
        if (proc/str(int(identity['pid']))).exists(): raise ValueError('old writer PID still exists')
    # Actual native FD scan, independent of a Boolean in the handoff receipt.
    for process in proc.iterdir():
        if not process.name.isdigit(): continue
        try:
            if process.stat().st_uid!=os.getuid(): continue
            for fd in (process/'fd').iterdir():
                try: destination=os.readlink(fd)
                except FileNotFoundError: continue
                destination=destination.removesuffix(' (deleted)')
                if any(destination.startswith(prefix+'/') for prefix in (str(target),str(target.resolve()))) and destination.endswith(('.transfer','.partial')): raise ValueError('old asset partial FD still open: '+process.name)
        except FileNotFoundError: continue
    return {'handoff_sha256':hashlib.sha256(raw).hexdigest(),'checked_process_identities':rec['process_identities'],'actual_old_partial_fds':[]}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--metadata',required=True); parser.add_argument('--state',required=True); parser.add_argument('--handoff',required=True); args=parser.parse_args()
    os.umask(0o077)
    if socket.gethostname() not in ('h10011','h100-11'): raise ValueError('own h11 only')
    if not ROOT.is_dir(): raise ValueError('own ROOT missing')
    target=ROOT/'assets-root/assets/qwen2.5-32b'; target.mkdir(parents=True,exist_ok=True)
    state=pathlib.Path(args.state)
    if not state.parent.resolve().is_relative_to((ROOT/'logs').resolve()): raise ValueError('state must be own ROOT/logs')
    items=inventory(args.metadata)
    handoff=handoff_check(args.handoff,target)
    lock=target/'.h16proxy-download.lock'; check_regular(lock)
    fd=os.open(lock,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # New partials need their own capacity; legacy partials are retained, never reused.
        missing=sum(i['size']-(target/(n+'.h16proxy.partial')).stat().st_size if (target/(n+'.h16proxy.partial')).exists() else i['size'] for n,i in items.items() if not (target/n).exists())
        free=os.statvfs(target).f_bavail*os.statvfs(target).f_frsize
        if free<missing+RESERVE: raise ValueError('insufficient target capacity preserving old partials')
        downloader=Downloader(target,state,items); downloader.data['handoff']=handoff
        signal.signal(signal.SIGTERM,lambda *_:downloader.stop.set())
        signal.signal(signal.SIGINT,lambda *_:downloader.stop.set())
        if not downloader.execute(): raise SystemExit(1)
    finally: os.close(fd)
if __name__=='__main__': main()
