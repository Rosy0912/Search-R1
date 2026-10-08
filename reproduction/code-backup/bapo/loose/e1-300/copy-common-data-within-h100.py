import subprocess,shlex,json,fcntl,pathlib
lock=pathlib.Path('/dev/shm/bapo-tree-instruct300-common-data-copy.lock')
with lock.open('w')as handle:
 fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 destination_code=r'''import pathlib,tempfile,tarfile,hashlib,json,os,sys
parent=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300');parent.mkdir(parents=True,exist_ok=True);target=parent/'common-data'
assert not target.exists(),'Target already exists: inspect rather than overwrite'
stage=pathlib.Path(tempfile.mkdtemp(prefix='common-data-transfer-',dir=parent))
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|')as archive:
 for item in archive:
  p=pathlib.PurePosixPath(item.name);assert not p.is_absolute()and '..'not in p.parts
  dst=stage/p
  if item.isdir():dst.mkdir(parents=True,exist_ok=True)
  else:
   assert item.isfile()and not item.issym()and not item.islnk(),item.name
   dst.parent.mkdir(parents=True,exist_ok=True)
   with archive.extractfile(item)as src,dst.open('xb')as out:
    while True:
     b=src.read(1048576)
     if not b:break
     out.write(b)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
expected='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24';assert sha(stage/'manifest.json')==expected
m=json.loads((stage/'manifest.json').read_text());records=list(m['splits'].values())+m['preparation']['raw_sources']+[m['exclusion_audit']]
for r in records:assert sha(stage/r['path'])==r['sha256'],r['path']
os.rename(stage,target);print(json.dumps(dict(state='COPIED_VERIFIED',node='h100-9',target=str(target),manifest_sha256=expected,verified_data_files=len(records),rows={k:v['rows']for k,v in m['splits'].items()})),flush=True)
'''
 opts=['-T','-o','BatchMode=yes','-o','ConnectTimeout=15']
 producer=subprocess.Popen(['ssh',*opts,'h100-5','tar -C /dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-20261004/common-data -cf - .'],stdout=subprocess.PIPE)
 consumer=subprocess.run(['ssh',*opts,'root@h100-9','python3 -c '+shlex.quote(destination_code)],stdin=producer.stdout,timeout=180)
 producer.stdout.close();assert producer.wait(timeout=30)==0;assert consumer.returncode==0
