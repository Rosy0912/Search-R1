import pathlib,json,hashlib,fcntl,os,time,socket
assert socket.gethostname().split('.')[0]=='h100-9'
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
src=pathlib.Path('/mnt/shared/workspace/training_data_node_5/shaoshitong/search-r1-rl/bapo-e1-instruct-300-20261004/assets/qwen2.5-3b-instruct-aa8e725')
dst=root/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
expected=json.loads((root/'instruct300/candidates/tree-cbb74d0-a42c2f86/manifest.json').read_bytes())['model_identity']
lock=(dst.parent/(dst.name+'.tree-upload.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
done=set();deadline=time.monotonic()+7200
while len(done)<len(expected['files']):
 for name,value in expected['files'].items():
  if name in done:continue
  source=src/name;target=dst/name
  if target.exists():assert sha(target)==value;done.add(name);continue
  if not source.exists():continue
  assert source.is_file() and sha(source)==value,name
  partial=target.with_name(name+'.part');offset=partial.stat().st_size if partial.exists() else 0
  assert offset<=source.stat().st_size
  with source.open('rb') as a,partial.open('ab' if offset else 'xb') as b:
   a.seek(offset)
   for chunk in iter(lambda:a.read(8388608),b''):b.write(chunk)
   b.flush();os.fsync(b.fileno())
  assert sha(partial)==value,name
  os.link(partial,target);partial.unlink();done.add(name)
  print(json.dumps({'status':'SHARED_H100_TO_LOCAL_FILE_SHA_PASS','file':name,'sha256':value,'bytes':target.stat().st_size}),flush=True)
 if len(done)==len(expected['files']):break
 assert time.monotonic()<deadline,'Source delivery wait timed out; preserve part; do not resubmit'
 print(json.dumps({'status':'WAITING_PUBLISHED_SHARED_SOURCE','missing':sorted(set(expected['files'])-done),'epoch':time.time()}),flush=True)
 time.sleep(60)
actual={str(p.relative_to(dst)):sha(p) for p in dst.rglob('*') if p.is_file()};assert actual==expected['files']
identity=hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest();assert identity==expected['sha256']
print(json.dumps({'status':'H1009_COMPLETE_OFFICIAL_MODEL_SHA_PASS','identity':identity,'files':len(actual)}),flush=True)
