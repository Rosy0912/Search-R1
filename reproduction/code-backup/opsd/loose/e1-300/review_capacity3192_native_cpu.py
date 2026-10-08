"""CPU-only original remote files/hashes/HF header audit, never model or CUDA loads."""
import json,pathlib,shlex,subprocess
b=pathlib.Path(__file__).resolve().parent
code=r'''import pathlib,hashlib,json,os,socket,math,struct,subprocess
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004');run=root/'runs/capacity-3192';tools=root/'tools/capacity-495f3320314c';repo=root/'repo-60cbea97d61f';model=root/'assets/qwen2.5-3b-instruct-aa8e725'
assert socket.gethostname().split('.')[0]=='h100-10' and os.environ['CUDA_VISIBLE_DEVICES']==''
def digest(p):
 p=pathlib.Path(p);assert p.resolve()==p and p.is_file();s=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for v in iter(lambda:f.read(8388608),b''):h.update(v)
 t=p.stat();assert (s.st_ino,s.st_dev,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(t.st_ino,t.st_dev,t.st_size,t.st_mtime_ns,t.st_ctime_ns)
 return dict(path=str(p),bytes=s.st_size,sha256=h.hexdigest(),uid=s.st_uid,ino=s.st_ino,device=s.st_dev)
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='60cbea97d61ff95a8f7dfa58de7b79962e79bd10'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
manifest=digest(tools/'manifest.json');assert manifest['sha256']=='495f3320314c6078ef0a25b7ca306a74cbc91bd7e333e9aef18172ec39a27d4d'
m=json.loads((tools/'manifest.json').read_bytes());toolpins={n:digest(tools/n) for n in m['tools']};sourcepins={n:digest(repo/n) for n in m['source_files']}
for n,p in toolpins.items():assert p['sha256']==m['tools'][n]
for n,p in sourcepins.items():assert p['sha256']==m['source_files'][n]
files={p.name:digest(p) for p in sorted(run.iterdir()) if p.is_file()}
assert (run/'job-exit.txt').read_text().strip()=='0' and (run/'torchrun-exit.txt').read_text().strip()=='0'
faiss=json.loads((run/'actual-faiss-runtime.json').read_text());faisspins={n:digest(v['path']) for n,v in faiss['modules'].items()}
for n,p in faisspins.items():assert p['sha256']==faiss['modules'][n]['sha256']
modelmanifest=digest(model/'VERIFIED-MODEL-MANIFEST.json');assert modelmanifest['sha256']=='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
mm=json.loads((model/'VERIFIED-MODEL-MANIFEST.json').read_text());assert mm['revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1' and len(mm['files'])==11
modelpins={r['file']:digest(model/r['file']) for r in mm['files']}
for r in mm['files']:assert (modelpins[r['file']]['sha256'],modelpins[r['file']]['bytes'])==(r['sha256'],r['bytes'])
config=json.loads((model/'config.json').read_text());index=json.loads((model/'model.safetensors.index.json').read_text());headers={}
for name in sorted(set(index['weight_map'].values())):
 with (model/name).open('rb') as f:
  length=struct.unpack('<Q',f.read(8))[0];assert 0<length<64*1024**2;h=json.loads(f.read(length))
 for k,v in h.items():
  if k=='__metadata__':continue
  assert k not in headers and index['weight_map'][k]==name;headers[k]=v
assert set(headers)==set(index['weight_map'])
asset=root/'tools/tool-assets-full-verify-04f8-d1f36d3d5578/evidence/manifest.json';assetpin=digest(asset);assert assetpin['sha256']=='87b5e85df12cbf4635cb787b08161950ddc23e2e5b850e68fb7249668573cc87'
print(json.dumps(dict(status='ACTUAL_CPU_REMOTE_CONTENT_AND_HEADER_CHECK_PASS',host=socket.gethostname(),uid=os.getuid(),pid=os.getpid(),cuda_visible_devices='',scope='No torch/model/GPU load; original files and serialized HF headers only',runfiles=files,tool_manifest=manifest,tool_files=toolpins,source_files=sourcepins,faiss_module_files=faisspins,model_manifest=modelmanifest,model_files=modelpins,asset_manifest=assetpin,hf_headers=dict(tensor_count=len(headers),shards=sorted(set(index['weight_map'].values())),serialized_numel=sum(math.prod(v['shape']) for v in headers.values()),dtypes=sorted(set(v['dtype'] for v in headers.values())),tie_word_embeddings=config.get('tie_word_embeddings'),embed=headers.get('model.embed_tokens.weight'),lm_head=headers.get('lm_head.weight'),config=config)),sort_keys=True))
'''
command='env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /usr/bin/python3 -c '+shlex.quote(code)
r=subprocess.run(['python3',str(b.parent/'remote.py'),command],capture_output=True,text=True)
(b/'job3192-independent-native-cpu-original.txt').write_text(r.stdout+r.stderr+'\nactual_exit='+str(r.returncode)+'\n');assert r.returncode==0,r.stderr
obj=json.loads(r.stdout);local=b/'job3192-actual-evidence'
import hashlib
assert set(obj['runfiles'])=={p.name for p in local.iterdir() if p.is_file()}
for n,p in obj['runfiles'].items():assert hashlib.sha256((local/n).read_bytes()).hexdigest()==p['sha256'],n
(b/'job3192-independent-native-content-pins.json').write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:obj[k] for k in ('status','host','pid','uid','scope','hf_headers')},indent=2))
