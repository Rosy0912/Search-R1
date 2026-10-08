"""Read verified H20 gzip in independent small chunks; retain local files until final verification."""
import concurrent.futures,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
workspace=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-4')
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
source_args=[v.replace('StrictHostKeyChecking=no','StrictHostKeyChecking=yes') for v in control.ssh('jump_h20-1')]
source_args+=['-o','StrictHostKeyChecking=yes'] # options must precede host; corrected below
source_args=source_args[:-3]+['-o','StrictHostKeyChecking=yes',source_args[-3]]
root='/mnt/data/shaoshitong/search-r1-rl/opsd'
stage=Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/opsd-20261003-h20-relay/gzip-chunks');stage.mkdir(exist_ok=True)
manifest=json.loads((workspace/'work/opsd/reproduction/h20/assets.json').read_text())
a=next(a for a in manifest if a['path']=='wiki/wiki-18.jsonl.gz');width=16*1024*1024
remote_dir=root+'/assets-root/relay-gzip-20261003'
def remote(cmd,stdin=None):
    env=dict(os.environ,OPSD_REMOTE_TIMEOUT='180')
    return subprocess.run([sys.executable,str(workspace/'work/remote.py'),cmd],stdin=stdin,env=env,timeout=190,check=True,capture_output=True)
remote('mkdir -p '+remote_dir)
def one(index):
    size=min(width,a['size']-index*width);p=stage/f'{index:04d}.bin'
    for attempt in range(3):
        try:
            if not p.exists() or p.stat().st_size!=size:
                assert __import__('shutil').disk_usage(stage).free > size+2*1024**3, 'Retain stage; local disk needs space'
                with p.with_suffix('.partial').open('wb') as f:
                    subprocess.run(source_args+[f'dd if=/data/search-r1/assets/wiki/wiki-18.jsonl.gz bs=4M skip={index*4} count=4 status=none'],stdout=f,stderr=subprocess.PIPE,check=True,timeout=150)
                assert p.with_suffix('.partial').stat().st_size==size
                p.with_suffix('.partial').replace(p)
            sha=hashlib.sha256(p.read_bytes()).hexdigest();dest=remote_dir+'/'+p.name
            command='cat > '+dest+'.partial && printf '+shlex.quote(sha+'  '+dest+'.partial\n')+' | sha256sum -c - && mv '+dest+'.partial '+dest
            with p.open('rb') as f: remote(command,stdin=f)
            result=dict(index=index,bytes=size,sha256=sha,verified_target=dest)
            (stage/f'{index:04d}.json').write_text(json.dumps(result))
            print('CHUNK_VERIFIED',index,size,flush=True)
            return result
        except (subprocess.SubprocessError,OSError,AssertionError) as e:
            print('CHUNK_RETRY',index,attempt,type(e).__name__,flush=True)
            if attempt==2: raise
            time.sleep(3)
start=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    results=list(pool.map(one,range((a['size']+width-1)//width)))
(stage/'manifest.json').write_text(json.dumps(results,indent=2))
# Assemble only exact numbered chunks generated above; atomically install only fixed LFS hash.
files=[remote_dir+'/'+f'{r["index"]:04d}.bin' for r in results];dest=root+'/assets-root/assets/wiki/wiki-18.jsonl.gz'
cmd='cat '+shlex.join(files)+' > '+dest+'.relay.partial && printf '+shlex.quote(a['sha256']+'  '+dest+'.relay.partial\n')+' | sha256sum -c - && test "$(stat -c %s '+dest+'.relay.partial)" = '+str(a['size'])+' && mv '+dest+'.relay.partial '+dest
remote(cmd)
print('GZIP_REMOTE_FULL_SHA_VERIFIED',time.monotonic()-start,flush=True)
(stage/'completed.json').write_text(json.dumps(dict(sha256=a['sha256'],bytes=a['size'],seconds=time.monotonic()-start)))
# All targets for every local chunk and the complete artifact are now verified.
for p in stage.glob('*.bin'):p.unlink()
remote('touch '+root+'/logs/wiki-relay.ready')
