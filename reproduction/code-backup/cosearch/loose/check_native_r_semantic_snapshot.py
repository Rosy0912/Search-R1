import json,pathlib,sys,hashlib,time,importlib.util,os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');r=root/'runs/author-native-continue-300-r'
out=root/'author-native-continue-300-r-semantic-snapshot6-10.json'
assert not out.exists(), 'Snapshot already exists; read it rather than repeating'
checker=r/'tools/check_native_five.py';sha=hashlib.sha256(checker.read_bytes()).hexdigest()
assert sha=='424aef4a162ddbacbfdceac5857a18302edef77b30d7b622ee6160c659efba0d'
sys.path.insert(0,str(root/'repo/verl'))
from transformers import AutoTokenizer
spec=importlib.util.spec_from_file_location('native_checker',checker);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
rows=[];identities=[];worker_rows=[]
for p in sorted(list((r/'evidence').glob('controller-*.jsonl'))+list((r/'evidence/workers').glob('*.jsonl'))):
 b=p.read_bytes();end=b.rfind(b'\n')+1;complete=b[:end]
 identities.append(dict(path=str(p),complete_bytes=end,sha256=hashlib.sha256(complete).hexdigest(),incomplete_tail_bytes=len(b)-end))
 parsed=[json.loads(l) for l in complete.splitlines()];rows.extend(parsed)
 if p.parent.name=='workers':worker_rows.append((str(p),parsed))
launch=c.one(rows,'launch');tokenizer=AutoTokenizer.from_pretrained(launch['config']['reranker_actor_rollout_ref']['model']['path'],local_files_only=True)
results=[]
for step in range(6,11):
 outer=c.one(rows,'outer',step=step);assert c.all_finite(outer['metrics'])
 attempts=[v for v in rows if v.get('kind')=='rollout_attempt_return' and v.get('step')==step and not v.get('validation')]
 accepted=[v for v in attempts if not v['discarded_before_both_PPO']];assert len(accepted)==1
 ret=accepted[0];start=c.one(rows,'rollout_attempt_start',step=step,attempt=ret['attempt'],validation=False)
 assert start['input_sha256']==ret['input_sha256'] and ret['qualified_groups']>0
 inferences=[];bindings=[]
 for path,values in worker_rows:
  pending=[]
  for v in values:
   if v.get('kind')=='ranker_ranking_result':pending.append(v)
   if v.get('kind')=='raw_rollout_payload' and v.get('stage')=='pre-postprocess':
    mono=int(pathlib.Path(v['path']).stem.rsplit('-',1)[1])
    if start['monotonic_ns']<=mono<=ret['monotonic_ns']:
     inferences.extend(pending);bindings.append(dict(worker_prefix=path,raw_payload=v,raw_monotonic_ns=mono,inference_records=len(pending),request_ids_sha256=hashlib.sha256(json.dumps([q['request_id'] for q in pending],sort_keys=True).encode()).hexdigest()))
    pending=[]
 assert len(bindings)==16 and len({b['raw_payload']['pid'] for b in bindings})==8,(step,len(bindings))
 assert all(sum(b['raw_payload']['pid']==pid for b in bindings)==2 for pid in {b['raw_payload']['pid'] for b in bindings})
 row=c.one(rows,'global_pre_update',step=step,role='reranker')
 count=c.semantic_groups(row,tokenizer,inferences)
 results.append(dict(step=step,strict_semantic_informative_groups=count,accepted_attempt=ret['attempt'],input_sha256=ret['input_sha256'],attempt_start=start,attempt_return=ret,inference_count=len(inferences),worker_attempt_bindings=bindings))
import torch
assert not torch.cuda.is_initialized()
value=dict(checked_at=time.time(),run=r.name,checker_sha256=sha,results=results,source_prefix_identities=identities,CUDA_initialized=False,RL_updates_performed=0,full_acceptance=False,attempt_binding='Per-worker sequential inference records ending at original pre-postprocess payload; payload monotonic filename is inside accepted controller attempt window. Eight distinct workers, each with two original nested pre-postprocess payload observations, required per accepted attempt; inference chunks are consumed once.')
b=json.dumps(value,allow_nan=False);out.write_text(b+'\n');print(b)
