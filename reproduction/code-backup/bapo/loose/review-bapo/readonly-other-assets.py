import concurrent.futures,json,subprocess
nodes=[('h100-8','/mnt/data/shaoshitong/search-r1-rl/gigpo'),('h100-9','/dataset/shaoshitong/search-r1-rl/tree-grpo'),('h100-10','/mnt/data/shaoshitong/search-r1-rl/opsd'),('h100-11','/dataset/shaoshitong/search-r1-rl/topd')]
probe='''import json,sys\nfrom pathlib import Path\nr=Path(sys.argv[1]); names={'part_aa','part_ab','wiki-18.jsonl.gz'}\nout={'root':str(r),'exists':r.exists(),'manifests':{},'files':[],'evidence':{}}\nfor p in [r/'assets-root/assets.json',r/'assets.json',r/'repo/reproduction/h20/assets.json']:\n if p.is_file():\n  try: out['manifests'][str(p)]=[a for a in json.loads(p.read_text()) if Path(a.get('path','')).name in names]\n  except Exception as e:out['manifests'][str(p)]=str(e)\nfor sub in ['assets-root/assets/wiki','assets/wiki','assets-root/wiki']:\n for name in names:\n  for suffix in ['', '.partial']:\n   p=r/sub/(name+suffix)\n   if p.is_file():out['files'].append({'path':str(p),'size':p.stat().st_size,'mtime':p.stat().st_mtime})\nlogs=r/'logs'\nif logs.is_dir():\n for p in logs.iterdir():\n  if p.is_file() and any(k in p.name.lower() for k in ['hash','verif','canonical','manifest','ready']):\n   out['evidence'][p.name]=p.read_text(errors='replace')[:12000] if p.stat().st_size<50000 else 'large evidence file; size='+str(p.stat().st_size)\nprint(json.dumps(out))\n'''
def one(item):
 node,root=item
 try:
  p=subprocess.run(['env','-u','LD_LIBRARY_PATH','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',node,'python3 - '+root],input=probe,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=22)
  return {'node':node,'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr[-1500:]}
 except Exception as e:return {'node':node,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for result in pool.map(one,nodes):print(json.dumps(result),flush=True)
