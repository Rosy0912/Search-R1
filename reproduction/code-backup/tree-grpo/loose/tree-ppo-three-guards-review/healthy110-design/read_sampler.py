import json,pathlib,hashlib
p=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/env/lib/python3.10/site-packages/vllm/model_executor/layers/sampler.py')
s=p.read_text();lines=s.splitlines();wanted=[i for i,line in enumerate(lines) if any(k in line for k in ['log_softmax','_apply_top_k_top_p','logits.div_','logprobs =','_get_logprobs('])];indexes=set()
for i in wanted:indexes.update(range(max(0,i-5),min(len(lines),i+6)))
print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'selected_lines':[{'line':i+1,'text':lines[i]} for i in sorted(indexes)]},indent=2))
