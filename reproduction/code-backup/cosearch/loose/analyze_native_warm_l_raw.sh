set -euo pipefail
CUDA_VISIBLE_DEVICES='' /dataset/shaoshitong/search-r1-rl/cosearch/author-native/env/bin/python - <<'ANALYZE'
import sys,os,json,hashlib,re,collections
from pathlib import Path
sys.path[:0]=['/dataset/shaoshitong/search-r1-rl/cosearch/author-native/repo','/dataset/shaoshitong/search-r1-rl/cosearch/author-native/repo/verl']
import torch
from transformers import AutoTokenizer
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native/runs/author-native-warm-five-l')
d=root/'evidence/raw-rollouts';out=root/'evidence/raw-diagnosis';out.mkdir(exist_ok=True)
tok=AutoTokenizer.from_pretrained('/dataset/shaoshitong/search-r1-rl/cosearch/assets-root/assets/qwen2.5-3b',local_files_only=True)
summary=dict(run=root.name,job=3100,real_joint_updates=0,files=[],examples=[])
for p in sorted(d.glob('*.pt')):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 try:value=torch.load(p,map_location='cpu',weights_only=False)
 except RuntimeError as error:
  summary['files'].append(dict(name=p.name,bytes=p.stat().st_size,sha256=h.hexdigest(),load_failed=str(error),incomplete_payload_not_counted=True));continue
 rec=dict(name=p.name,bytes=p.stat().st_size,sha256=h.hexdigest(),type=type(value).__name__)
 if p.name.startswith('pre-postprocess'):
  rec['rows']=len(value);counts=collections.Counter();lengths=[]
  with (out/(p.stem+'.jsonl')).open('w') as stream:
   for i,v in enumerate(value):
    ids=v.response_ids.reshape(-1);mask=v.response_mask.reshape(-1);active=ids[mask.bool()].tolist();text=tok.decode(active,skip_special_tokens=False)
    full=tok.decode(ids.tolist(),skip_special_tokens=False)
    lengths.append(len(active));flags={k:bool(re.search(pattern,text)) for k,pattern in dict(reason=r'<reason>',answer=r'<answer>',tool_call=r'<tool_call>',search_tag=r'<search>',think=r'<think>',json_search=r'"name"\s*:\s*"search"').items()}
    counts.update(k for k,b in flags.items() if b)
    line=dict(row=i,response_text=text,full_response_text=full,active_response_ids=active,response_mask=mask.tolist(),flags=flags,extra_fields_keys=list(v.extra_fields),num_turns=v.num_turns)
    stream.write(json.dumps(line,ensure_ascii=False,default=str)+'\n')
    if len(summary['examples'])<8:summary['examples'].append(dict(file=p.name,row=i,text=text[:2400],flags=flags,active_tokens=len(active),extra_fields_keys=list(v.extra_fields)))
  rec['format_counts']=dict(counts);rec['active_tokens_minmax']=[min(lengths),max(lengths)] if lengths else[];rec['active_tokens_total']=sum(lengths)
 elif p.name.startswith('pre-ranker-filter'):
  args=value['args'];nested=args[0];rec.update(local_main_rows=args[1] if len(args)>1 else None,nested_groups=len(nested),raw_ranker_outputs=sum(len(v) for v in nested))
 elif p.name.startswith('post-ranker-filter'):rec.update(retained_ranker_outputs=len(value[0]),metrics=value[1])
 summary['files'].append(rec)
assert not torch.cuda.is_initialized()
(out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str)+'\n')
print(json.dumps(summary,ensure_ascii=False,default=str))
ANALYZE
