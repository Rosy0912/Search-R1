import pathlib,json,hashlib,base64,time
paths=['/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-C1/runs/outers/outer-00000/training/outer-complete.json', '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-C1/runs/outers/outer-00001/training/outer-complete.json', '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-C1/runs/outers/outer-00002/training/outer-complete.json', '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-C1/runs/outers/outer-00003/training/outer-complete.json']
rows=[]
for text in paths:
 p=pathlib.Path(text)
 try:
  a=p.stat();raw=p.read_bytes();b=p.stat()
 except FileNotFoundError:
  rows.append(dict(path=text,exists=False));continue
 keys=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
 rows.append(dict(path=text,exists=True,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),metadata={k:getattr(b,k) for k in keys},stable_during_read=all(getattr(a,k)==getattr(b,k) for k in keys),raw_base64=base64.b64encode(raw).decode()))
print(json.dumps(dict(observed_unix=time.time(),files=rows,no_node_writes=True,no_model_or_optimizer_read=True)))
