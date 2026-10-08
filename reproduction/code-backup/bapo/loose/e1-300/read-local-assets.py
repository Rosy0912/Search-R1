import pathlib,json,subprocess,os
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo');a=json.loads((root/'assets-root/assets.json').read_text());print('assets',a);print('space',subprocess.check_output(['df','-h',str(root)],text=True))
for base in [root/'cache/hf',root/'assets-root/data/paper-grpo',pathlib.Path('/root/.cache/huggingface')]:
 print('BASE',str(base),base.exists())
 if base.exists():
  print('FILES',[str(p) for p in base.rglob('*') if p.name in ['dataset_info.json','state.json','refs','train.parquet','revision.txt']][:35])
import pyarrow.parquet as pq
p=root/'assets-root/data/paper-grpo/train.parquet';t=pq.read_table(p);print('TRAIN',len(t),t.schema);print('FIRST',t.slice(0,1).to_pylist());print('sources',set(t.column('data_source').to_pylist()))
try:
 import requests
 r=requests.get('https://huggingface.co/api/models/Qwen/Qwen2.5-3B-Instruct',timeout=15);print('HF_MODEL_STATUS',r.status_code);d=r.json();print('HF_MODEL_SHA',d.get('sha'));print('HF_FILES',[s['rfilename'] for s in d.get('siblings',[])])
except Exception as e:print('HF_MODEL_ERROR',type(e).__name__,str(e)[:150])
