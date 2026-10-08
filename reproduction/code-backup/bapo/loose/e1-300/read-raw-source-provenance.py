import pathlib,json,hashlib
base=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo/assets-root');sources=['nq','triviaqa','popqa','hotpotqa','2wikimultihopqa','musique','bamboogle'];a=json.loads((base/'assets.json').read_text());out=[]
for source in sources:
 for split in (['train','test'] if source=='nq' else ['train','dev'] if source=='hotpotqa' else ['test'] if source in ['triviaqa','popqa','bamboogle'] else ['dev']):
  p=base/'assets'/source/(split+'.jsonl');b=p.read_bytes();lines=b.splitlines();matching=[r for r in a if r['path']==source+'/'+split+'.jsonl'];out.append(dict(source=source,split=split,path=str(p),sha256=hashlib.sha256(b).hexdigest(),rows=len(lines),provenance=matching,first=json.loads(lines[0])))
print(json.dumps(out))
