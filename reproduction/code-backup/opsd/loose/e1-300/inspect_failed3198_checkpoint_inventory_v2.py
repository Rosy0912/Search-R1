"""Read-only original failed job inventory; no model load, GPU or remote mutation."""
import json,pathlib,shlex,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent
code=r'''import pathlib,json,hashlib,struct,subprocess,os,socket
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004');run=root/'runs/formal-3198';repo=root/'repo-89beac273f78';actor=run/'checkpoints/actor/global_step_5'
out={'scope':'READ_ONLY_FAILED3198_ACTUAL_FILE_INVENTORY_NOT_HF_LOAD_OR_ACCEPTANCE','host':socket.gethostname(),'uid':os.getuid(),'job_id':'3198','source':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'source_status':subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True),'slurm_current_query':(lambda q:{'stdout':q.stdout,'stderr':q.stderr,'returncode':q.returncode})(subprocess.run(['scontrol','show','job','3198','-o'],text=True,capture_output=True)),'files':{},'HF_headers':{},'directories':{}}
for d in [actor,actor/'training_state',run/'full_state/global_step_5']:
 out['directories'][str(d)]={'exists':d.exists(),'children':sorted(p.name for p in d.iterdir()) if d.exists() else []}
for p in sorted(actor.rglob('*')):
 if not p.is_file():continue
 assert p.resolve()==p;st=p.stat();rec={'bytes':st.st_size,'uid':st.st_uid}
 if st.st_size<1024*1024:rec['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 out['files'][str(p)]=rec
index=actor/'model.safetensors.index.json'
if index.is_file():
 indexv=json.loads(index.read_bytes());mapping=indexv['weight_map'];actual={};out['index_keys']=len(mapping);out['index_shards']=sorted(set(mapping.values()));out['index_sha256']=hashlib.sha256(index.read_bytes()).hexdigest()
 for name in out['index_shards']:
  assert pathlib.Path(name).name==name;p=actor/name
  with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];assert 0<n<64*1024**2;h=json.loads(f.read(n))
  fields={k:v for k,v in h.items() if k!='__metadata__'};assert not(set(fields)&set(actual));actual.update(fields)
  out['HF_headers'][name]={'keys':len(fields),'dtypes':sorted({v['dtype'] for v in fields.values()}),'serialized_numel':sum(__import__('math').prod(v['shape']) for v in fields.values())}
 out['all_actual_header_keys']=len(actual);out['index_header_sets_exact']=set(actual)==set(mapping)
 out['serialized_numel']=sum(__import__('math').prod(v['shape']) for v in actual.values());out['embedding']=actual.get('model.embed_tokens.weight');out['lm_head']=actual.get('lm_head.weight')
print(json.dumps(out,sort_keys=True))'''
p=subprocess.run([sys.executable,str(HERE.parent/'remote.py'),'python3 -c '+shlex.quote(code)],capture_output=True,timeout=80)
(HERE/'failed3198-checkpoint-inventory-v2-original.txt').write_bytes(p.stdout);(HERE/'failed3198-checkpoint-inventory-v2-stderr.txt').write_bytes(p.stderr)
print('actual remote exit',p.returncode)
if p.returncode:sys.exit(p.returncode)
v=json.loads(p.stdout);print(json.dumps({k:v[k] for k in ['scope','source','source_status','directories','index_keys','index_shards','all_actual_header_keys','serialized_numel'] if k in v},indent=2))
