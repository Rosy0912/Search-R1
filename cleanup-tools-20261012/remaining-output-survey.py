import pathlib,json,subprocess,time
base=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006/runs')
result={'epoch':time.time(),'roots':[]}
for p in sorted(base.glob('gigpo-e1-fresh200-live-job33*')):
 if not any(n in p.name for n in ['job3340-','job3344-','job3347-']):continue
 r={'path':str(p),'du':subprocess.check_output(['du','--max-depth=2','-B1',str(p)],text=True),'large_files':[]}
 for d,ds,fs in __import__('os').walk(p):
  ds[:]=[x for x in ds if x not in ['checkpoints','.git','env','site-packages']]
  for n in fs:
   q=pathlib.Path(d)/n
   if not q.is_symlink() and q.stat().st_size>1024**3:r['large_files'].append({'path':str(q),'bytes':q.stat().st_size})
 result['roots'].append(r)
print(json.dumps(result))
