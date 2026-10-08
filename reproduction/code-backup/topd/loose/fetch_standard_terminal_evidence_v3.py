"""Read-only, bounded, complete terminal evidence transport. Never submits jobs."""
import argparse,base64,gzip,hashlib,importlib.util,json,os,pathlib,re,time
ROOT='/dataset/shaoshitong/search-r1-rl/topd'
CHUNK=8*1024**2
TRANSPORT=pathlib.Path(__file__).parent/'g1-native-cpu-deploy-v6/stream_transport.py'
TRANSPORT_SHA='8c28ce8e119df18ed359ba2accc7ce33ed82c167f97e0ddb1236a496cb6865dd'
def digest(b):return hashlib.sha256(b).hexdigest()
def terminal(record,job):
 if record.get('returncode')!=0 or int(record.get('job_id',0))!=job:raise ValueError('Actual terminal record identity/command failure')
 raw=record['stdout']
 if not re.search(r'\bJobId='+str(job)+r'\b',raw) or not re.search(r'\bJobState=(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)\b',raw) or not re.search(r'\bExitCode=\d+:\d+\b',raw):raise ValueError('No captured actual Slurm terminal/exit code')
 if 'NodeList=h100-11' not in raw:raise ValueError('Wrong actual node')
def decode_chunk(row,expected_offset):
 if row['offset']!=expected_offset:raise ValueError('Wrong chunk offset')
 compressed=base64.b64decode(row['data'],validate=True)
 if digest(compressed)!=row['compressed_sha256']:raise ValueError('Compressed chunk drift')
 # gzip decompression bounded even for malicious compressed output.
 import zlib
 obj=zlib.decompressobj(16+zlib.MAX_WBITS);raw=obj.decompress(compressed,CHUNK+1)
 if len(raw)>CHUNK or obj.unconsumed_tail or not obj.eof or obj.unused_data:raise ValueError('Oversized or malformed gzip chunk')
 if len(raw)!=row['size'] or digest(raw)!=row['sha256']:raise ValueError('Original chunk drift')
 return raw
REMOTE=r'''
import pathlib,json,hashlib,os,sys,base64,gzip,time,re
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');CHUNK=8*1024**2

def local(s):
 p=pathlib.Path(s);assert not p.is_symlink() and ROOT in p.resolve().parents;return p.resolve()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(CHUNK),b''):h.update(b)
 return h.hexdigest()
def stat(p):
 s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def checked_ref(r):
 p=local(r['path']);assert sha(p)==r['sha256'];return p,json.loads(p.read_text())
def preflight(q):
 tp,t=checked_ref(q['terminal']);assert t['returncode']==0 and int(t['job_id'])==q['job']
 text=t['stdout'];assert re.search(r'\bJobId='+str(q['job'])+r'\b',text) and re.search(r'\bJobState=(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)\b',text) and re.search(r'\bExitCode=\d+:\d+\b',text) and 'NodeList=h100-11' in text
 rp,r=checked_ref(q['result']);cp,c=checked_ref(q['cleanup']);assert int(r['job'])==q['job'] and r['status'] in ('COMPLETE_UNREVIEWED','FAILED')
 assert c['cleanup']['actual_cgroup_empty'] is True and c['own_processes_remaining']==[] and c['own_gpu_records_remaining']==[] and c['parent_gpu_records']==[] and c['parent_cuda_initialized'] is False
 assert c['baseline_identity_missing']==[] and c['original_gpu_baseline_missing']==[]
 root=local(q['root']);assert root.is_dir() and ROOT/'logs' in root.parents
 return root
q=PAYLOAD
root=preflight(q)
if q['op']=='inventory':
 paths=[];tree=[]
 for p in sorted(root.rglob('*')):
  assert not p.is_symlink()
  if p.is_file():
   n=str(p.relative_to(root));s=stat(p);tree.append({'name':n,'stat':s})
   if pathlib.PurePosixPath(n).parts[0]!='cache' and p.suffix in ('.json','.jsonl','.log','.txt','.yaml','.yml'):
    h=sha(p);assert stat(p)==s;paths.append({'name':n,'path':str(p),'stat':s,'size':s[2],'sha256':h})
 assert paths
 before={x['path']:x['stat'] for x in paths};time.sleep(.5)
 assert all(stat(pathlib.Path(p))==s for p,s in before.items())
 print(json.dumps({'files':paths,'tree':tree,'terminal':q['terminal'],'result':q['result'],'cleanup':q['cleanup'],'epoch':time.time(),'excluded_cache_scope':'RUNTIME_DERIVED_CACHE_TREE_METADATA_ONLY_NOT_RAW_RUN_EVIDENCE'}))
elif q['op']=='chunk':
 p=local(q['file']['path']);assert root in p.parents and stat(p)==q['file']['stat'];offset=q['offset'];assert 0<=offset<q['file']['size']
 with p.open('rb') as f:f.seek(offset);b=f.read(CHUNK)
 assert stat(p)==q['file']['stat'];z=gzip.compress(b,compresslevel=3,mtime=0)
 print(json.dumps({'offset':offset,'size':len(b),'sha256':hashlib.sha256(b).hexdigest(),'compressed_sha256':hashlib.sha256(z).hexdigest(),'data':base64.b64encode(z).decode()}))
elif q['op']=='final':
 assert all(stat(local(f['path']))==f['stat'] and sha(local(f['path']))==f['sha256'] and stat(local(f['path']))==f['stat'] for f in q['files'])
 current=[{'name':str(p.relative_to(root)),'stat':stat(p)} for p in sorted(root.rglob('*')) if p.is_file()];assert current==q['tree']
 print(json.dumps({'status':'FULL_ORIGINAL_BYTES_STABLE_SHA_VERIFIED_NOT_RUNTIME_APPROVAL','epoch':time.time()}))
else:raise ValueError('Unknown read-only operation')
'''
def sender():
 if digest(TRANSPORT.read_bytes())!=TRANSPORT_SHA:raise ValueError('Transport source drift')
 spec=importlib.util.spec_from_file_location('explicit_readonly_stream',TRANSPORT);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.send

def fetch(request,out,send):
 def call(op,**extra):
  q=dict(request,op=op,**extra);source=REMOTE.replace('q=PAYLOAD','q='+repr(q))
  result=send(source,attempts=3,retry_safe=True,timeout=180)
  if result.returncode:raise RuntimeError('Read-only remote failure: '+result.stderr.decode(errors='replace')[-4000:])
  return json.loads(result.stdout)
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=False)
 inventory=call('inventory');(out/'remote-original-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
 for entry in inventory['files']:
  relative=pathlib.PurePosixPath(entry['name'])
  if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe remote relative path')
  final=out.joinpath(*relative.parts);final.parent.mkdir(parents=True,exist_ok=True);private=final.with_name(final.name+'.fetch-private')
  h=hashlib.sha256();offset=0;chunks=[]
  with private.open('xb') as f:
   while offset<entry['size']:
    row=call('chunk',file=entry,offset=offset);raw=decode_chunk(row,offset)
    if not raw:raise ValueError('Empty nonterminal chunk')
    f.write(raw);h.update(raw);chunks.append({k:v for k,v in row.items() if k!='data'});offset+=len(raw)
   f.flush();os.fsync(f.fileno())
  if offset!=entry['size'] or h.hexdigest()!=entry['sha256']:raise ValueError('Full original file drift')
  os.link(private,final);private.unlink();entry['chunks']=chunks
  (out/'fetch-progress.json').write_text(json.dumps({'completed':entry['name'],'size':offset,'sha256':h.hexdigest(),'epoch':time.time()})+'\n')
 verified=call('final',files=inventory['files'],tree=inventory['tree'])
 inventory['verified']=verified;(out/'full-original-fetch-refs.json').write_text(json.dumps(inventory,indent=2)+'\n');return inventory

def main():
 p=argparse.ArgumentParser();p.add_argument('--job',required=True,type=int);p.add_argument('--evidence-root',required=True);p.add_argument('--terminal-record',required=True);p.add_argument('--terminal-sha256',required=True);p.add_argument('--result',required=True);p.add_argument('--result-sha256',required=True);p.add_argument('--cleanup',required=True);p.add_argument('--cleanup-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 if a.job<=0:raise ValueError('Positive actual job required')
 request={'job':a.job,'root':a.evidence_root,'terminal':{'path':a.terminal_record,'sha256':a.terminal_sha256},'result':{'path':a.result,'sha256':a.result_sha256},'cleanup':{'path':a.cleanup,'sha256':a.cleanup_sha256}}
 result=fetch(request,a.output,sender());print(json.dumps({'status':result['verified']['status'],'files':len(result['files']),'original_bytes':sum(f['size'] for f in result['files'])}))
if __name__=='__main__':main()
