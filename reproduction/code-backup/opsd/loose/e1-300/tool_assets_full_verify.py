"""CPU-only streamed original-content proof; never imports model/GPU libraries."""
import argparse, datetime, gzip, hashlib, json, os, pathlib, socket, stat, sys, time
MANIFEST_SHA = 'a033b9d6e5b4453374c25d8d9a33d70254c1a79ea0d71677f21afb083d94b3fd'
ROOT = pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd/assets-root/assets')
COMMIT = '04f8cc5a777f8dfafbb9e48f877673c2c5751f33'
CHUNK = 8 * 1024 * 1024

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def metadata(s): return {k: getattr(s, k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_uid')}
def stable_open(path):
    path = pathlib.Path(path)
    if path.resolve() != path.absolute(): raise ValueError('symlink/path alias: '+str(path))
    f = open(path, 'rb'); before = metadata(os.fstat(f.fileno()))
    if not stat.S_ISREG(os.fstat(f.fileno()).st_mode): f.close(); raise ValueError('not regular')
    if metadata(path.stat()) != before: f.close(); raise ValueError('open identity changed')
    return f, before

def finish(f, path, before):
    after = metadata(os.fstat(f.fileno()))
    if before != after or metadata(pathlib.Path(path).stat()) != before: raise ValueError('file changed: '+str(path))

def hash_stream(f, segments=None, label='stream'):
    h=hashlib.sha256(); total=0; next_progress=5*1024**3
    parts=[hashlib.sha256() for _ in (segments or [])]; cursor=0; used=0
    while True:
        b=f.read(CHUNK)
        if not b: break
        h.update(b); total+=len(b)
        if segments:
            offset=0
            while offset<len(b):
                if cursor>=len(segments): raise ValueError('index oversized')
                take=min(len(b)-offset,segments[cursor]-used)
                parts[cursor].update(b[offset:offset+take]); offset+=take; used+=take
                if used==segments[cursor]: cursor+=1; used=0
        if total>=next_progress:
            print(json.dumps({'utc':utc(),'phase':label,'bytes_read':total}),flush=True); next_progress+=5*1024**3
    if segments and (cursor!=len(segments) or used): raise ValueError('index truncated')
    return {'bytes':total,'sha256':h.hexdigest(),'segment_sha256':[p.hexdigest() for p in parts]}

def hash_file(path, segments=None):
    f,before=stable_open(path)
    try:
        result=hash_stream(f,segments,str(path)); finish(f,path,before)
        return dict(result,path=str(path),metadata=before)
    finally: f.close()

def atomic_json(path, value):
    payload=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode(); temp=path.with_suffix('.tmp')
    with open(temp,'xb') as f: f.write(payload); f.flush(); os.fsync(f.fileno())
    os.link(temp,path); temp.unlink()

def verify(manifest, root):
    e5=[r for r in manifest if r['repo']=='intfloat/e5-base-v2']
    if len(e5)!=8 or {r['revision'] for r in e5}!={'f52bf8ec8c7124536f0efb74aca902b2995e5bcd'}: raise ValueError('E5 schema')
    records=[]
    for r in e5:
        p=root/r['path']; a=hash_file(p)
        if a['bytes']!=r['size']: raise ValueError('E5 size: '+str(p))
        if r['sha256'] and a['sha256']!=r['sha256']: raise ValueError('E5 SHA: '+str(p))
        if r['git_blob_sha1']:
            f,before=stable_open(p)
            try:
                h=hashlib.sha1(b'blob '+str(r['size']).encode()+b'\0')
                while True:
                    b=f.read(CHUNK)
                    if not b: break
                    h.update(b)
                finish(f,p,before)
                if before!=a['metadata']: raise ValueError('E5 changed across reads')
                a['git_blob_sha1']=h.hexdigest()
                if h.hexdigest()!=r['git_blob_sha1']: raise ValueError('E5 blob hash: '+str(p))
            finally: f.close()
        a['original']=r; records.append(a); print(json.dumps({'utc':utc(),'asset_verified':str(p)}),flush=True)
    gz=next(r for r in manifest if r['path']=='wiki/wiki-18.jsonl.gz')
    z=hash_file(root/gz['path'])
    if (z['bytes'],z['sha256'])!=(gz['size'],gz['sha256']): raise ValueError('wiki gzip original SHA/size')
    pieces=[next(r for r in manifest if r['path']=='wiki/'+part) for part in ('part_aa','part_ab')]
    index=hash_file(root/'wiki/e5_Flat.index',[r['size'] for r in pieces])
    if index['segment_sha256']!=[r['sha256'] for r in pieces]: raise ValueError('index original segments SHA')
    corpus=hash_file(root/'wiki/wiki-18.jsonl')
    f,before=stable_open(root/gz['path'])
    try:
        with gzip.GzipFile(fileobj=f,mode='rb') as decomp: decoded=hash_stream(decomp,label='gzip_decompressed')
        finish(f,root/gz['path'],before)
        if before!=z['metadata']: raise ValueError('gzip changed across reads')
    finally: f.close()
    if (decoded['bytes'],decoded['sha256'])!=(corpus['bytes'],corpus['sha256']): raise ValueError('decompressed corpus mismatch')
    # Recheck all source identities at completion, including files read earlier.
    for r in records+[z,index,corpus]:
        if metadata(pathlib.Path(r['path']).stat())!=r['metadata']: raise ValueError('asset changed before completion')
    return {'e5':records,'wiki_gzip':dict(z,original=gz),'index':dict(index,original_segments=pieces),'corpus':corpus,'gzip_decompressed':decoded}

def main():
    a=argparse.ArgumentParser(); a.add_argument('--manifest',type=pathlib.Path,required=True); a.add_argument('--output',type=pathlib.Path,required=True); args=a.parse_args()
    if socket.gethostname().split('.')[0]!='h100-10': raise ValueError('wrong node')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('CPU visibility required')
    raw=args.manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA: raise ValueError('fixed manifest mismatch')
    if ROOT.resolve()!=ROOT or not ROOT.is_dir(): raise ValueError('root alias')
    args.output.mkdir(exist_ok=False)
    identity={'hostname':socket.gethostname(),'pid':os.getpid(),'uid':os.getuid(),'proc_start_ticks':pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'started_utc':utc(),'frozen_source_commit':COMMIT,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'original_manifest_sha256':MANIFEST_SHA,'assets_root':str(ROOT),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'scope':'CPU original asset bytes only; no GPU or training acceptance'}
    atomic_json(args.output/'identity.json',identity)
    try:
        result=verify(json.loads(raw),ROOT)
        atomic_json(args.output/'manifest.json',dict(identity,status='FULL_TOOL_ASSET_CONTENT_VERIFIED',completed_utc=utc(),results=result))
        print(json.dumps({'utc':utc(),'status':'FULL_TOOL_ASSET_CONTENT_VERIFIED'}),flush=True)
    except BaseException as e:
        atomic_json(args.output/'failure.json',dict(identity,status='FAILED',ended_utc=utc(),error=repr(e))); raise
if __name__=='__main__': main()
