import pathlib,json,hashlib,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004/runs/bapo-instruct300-formal300-3177');mp=r/'checkpoints/full_state/global_step_5/manifest.json';raw=mp.read_bytes();m=json.loads(raw);assert hashlib.sha256(raw).hexdigest()=='e491cbb1dd1063db418f32759b16f88b9a2bff5ffeac4a205e52ef31d067d899';start=time.monotonic()
for x in m['files']:
 p=pathlib.Path(x['path']);assert p.is_file()and not p.is_symlink()and r in p.resolve().parents;h=hashlib.sha256();n=0
 with p.open('rb')as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block);n+=len(block)
 assert n==x['bytes']and h.hexdigest()==x['sha256'];print(json.dumps({'path':str(p),'bytes':n,'sha256':h.hexdigest(),'same_manifest':True}),flush=True)
assert mp.read_bytes()==raw;print(json.dumps({'status':'INDEPENDENT_STREAM_FULL33_MEMBERS_PASS','count':len(m['files']),'seconds':time.monotonic()-start,'manifest_sha256':hashlib.sha256(raw).hexdigest(),'checkpoint_bytes_downloaded':False}),flush=True)
