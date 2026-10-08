import pathlib,json,hashlib,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');run=root/'runs/bapo-instruct300-formal300-3292'
assert (run/'source-commit.txt').read_text().strip()=='a2568d0df306ead6e059760e28b5bd53ee3f6876'
rows=[]
for folder in sorted((run/'checkpoints').glob('*/global_step_*')):
 members=[dict(name=p.name,bytes=p.stat().st_size)for p in folder.iterdir()if p.is_file()]
 small={}
 for name in ('checkpoint-complete.json','checkpoint-manifest.json'):
  p=folder/name
  if p.is_file():
   b=p.read_bytes();small[name]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),json=json.loads(b))
 rows.append(dict(path=str(folder),members=members,markers=small))
print(json.dumps(dict(job_id=3292,time_unix=time.time(),scope='DIRECTORY_AND_SMALL_METADATA_ONLY_NO_PAYLOAD_HASH_OR_HF_LOAD_NOT_CHECKPOINT_ACCEPTANCE',checkpoints=rows)))
