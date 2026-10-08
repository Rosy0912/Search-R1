import json,hashlib,urllib.request
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/assets-root/assets');rows=[]
for source in ['nq','triviaqa','popqa','hotpotqa','2wikimultihopqa','musique','bamboogle']:
 for split in ['train','test','dev']:
  p=root/source/(split+'.jsonl')
  if p.is_file():
   h=hashlib.sha256();n=0
   with p.open('rb') as f:
    for line in f:h.update(line);n+=1
   rows.append({'path':str(p),'source':source,'split':split,'rows':n,'sha256':h.hexdigest(),'bytes':p.stat().st_size})
print(json.dumps({'raw_inputs':rows}),flush=True)
try:
 req=urllib.request.Request('https://huggingface.co/api/models/Qwen/Qwen2.5-3B-Instruct',headers={'User-Agent':'Tree-Instruct300-authorized-model-inspection'})
 with urllib.request.urlopen(req,timeout=20) as r:d=json.load(r)
 print(json.dumps({'model_repo':d['id'],'model_revision':d['sha'],'siblings':[x['rfilename'] for x in d.get('siblings',[]) if x['rfilename'].endswith(('.json','.safetensors','.txt','.tiktoken'))]}),flush=True)
except Exception as e:print(json.dumps({'official_model_metadata_error':str(e)}),flush=True)
