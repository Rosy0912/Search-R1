import json,pathlib,sys,hashlib,time,importlib.util
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');r=root/'runs/author-native-phase-aware-warm-five-m'
sys.path.insert(0,str(root/'repo/verl'))
from transformers import AutoTokenizer
spec=importlib.util.spec_from_file_location('native_checker',r/'tools/check_native_five.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
rows=[];identities=[]
for p in list((r/'evidence').glob('controller-*.jsonl'))+list((r/'evidence/workers').glob('*.jsonl')):
 b=p.read_bytes();lines=b.splitlines(keepends=True);complete=b''.join(l for l in lines if l.endswith(b'\n'));identities.append(dict(path=str(p),complete_bytes=len(complete),sha256=hashlib.sha256(complete).hexdigest(),incomplete_tail_bytes=len(b)-len(complete)))
 rows.extend(json.loads(l) for l in complete.splitlines())
launch=next(x for x in rows if x.get('kind')=='launch');tokenizer=AutoTokenizer.from_pretrained(launch['config']['reranker_actor_rollout_ref']['model']['path'],local_files_only=True)
inferences=[x for x in rows if x.get('kind')=='ranker_ranking_result'];result=[]
for x in rows:
 if x.get('kind')=='global_pre_update' and x.get('role')=='reranker':
  try:result.append(dict(step=x['step'],strict_semantic_informative_groups=c.semantic_groups(x,tokenizer,inferences)))
  except Exception as e:result.append(dict(step=x['step'],error_type=type(e).__name__,error=str(e)))
import torch
print(json.dumps(dict(checked_at=time.time(),run=r.name,checker_sha256=hashlib.sha256((r/'tools/check_native_five.py').read_bytes()).hexdigest(),results=result,source_prefix_identities=identities,CUDA_initialized=torch.cuda.is_initialized(),RL_updates_performed=0,full_acceptance=False)))
