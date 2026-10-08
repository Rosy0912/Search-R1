# -*- coding: utf-8 -*-
from pathlib import Path
import json,hashlib,importlib.util,numpy as np
chat=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2');root=chat/'work/tree-grpo';ev=root/'reproduction/baselines/tree-validation-20261002/evidence';source=chat/'outputs/evidence'
a=json.loads((source/'retrieval-cpu-2854.json').read_text());b=json.loads((source/'retrieval-gpu-index-cpu-embeddings.json').read_text());s=json.loads((source/'retrieval-index-only-difference.json').read_text())
assert b['reused_embeddings'] and a['embeddings_sha256']==b['embeddings_sha256']
assert a['asset_identity']==b['asset_identity'] and b['asset_identity_verified']
assert a['recipe']==b['recipe'] and a['queries']==b['queries']
assert a['source']['code_commit']==b['source']['code_commit'] and a['source']['slurm_job_id']==b['source']['slurm_job_id']=='2854'
rows=[]
for i,q in enumerate(a['queries']):
 r={'query_index':i+1,'query':q}
 for k in ('3','50'):
  l=a['results'][i][k];t=b['results'][i][k];lm=dict(zip(l['ids'],zip(l['scores'],l['contents_sha256'])));rm=dict(zip(t['ids'],zip(t['scores'],t['contents_sha256'])))
  assert set(lm)==set(rm) and all(lm[x][1]==rm[x][1] for x in lm)
  moved=[j+1 for j,(x,y) in enumerate(zip(l['ids'],t['ids'])) if x!=y]
  r[k]={'changed_rank_positions':moved,'max_delta':max(abs(lm[x][0]-rm[x][0]) for x in lm)}
  if moved:r[k]['changed_entries']=[{'id':x,'cpu_rank':l['ids'].index(x)+1,'gpu_rank':t['ids'].index(x)+1,'cpu_score':lm[x][0],'gpu_score':rm[x][0]} for x in lm if l['ids'].index(x)!=t['ids'].index(x)]
 assert not r['3']['changed_rank_positions'];rows.append(r)
for k in ('3','50'):
 assert s['summary'][k]['exact_rank_queries']==sum(not r[k]['changed_rank_positions'] for r in rows)
 assert s['summary'][k]['max_matched_id_score_abs_delta']==max(r[k]['max_delta'] for r in rows)
path=root/'verl/utils/logger/aggregate_logger.py';sp=importlib.util.spec_from_file_location('aggregate_logger_review',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
vals={'loss':0.12345678901234568,'grad':1e-12,'large':np.float64(1.234567890123456e18),'float32':np.float32(0.123456789),'tiny32':np.float32(1e-12),'updates':500,'npint':np.int64(500),'negative':-1.234567891234567e-24}
before=dict(vals);rendered=m.concat_dict_to_str(vals,500);parsed={k:float(v) for k,v in (t.split(':',1) for t in rendered.split(' - ')[1:])}
assert all(parsed[k]==float(v) for k,v in vals.items()) and vals==before and parsed['grad']>0
(ev/'ablation-and-logger-independent.json').write_text(json.dumps({'rows':rows,'summary':s['summary'],'logger_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'logger_roundtrip':{'rendered':rendered,'all_values_exact_to_float':True,'input_unchanged':True},'input_sha256':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ['retrieval-gpu-index-cpu-embeddings.json','retrieval-index-only-difference.json']}},indent=2))
print(json.dumps({'changed_queries':[r for r in rows if r['50']['changed_rank_positions']],'summary':s['summary'],'rendered':rendered},indent=2))
