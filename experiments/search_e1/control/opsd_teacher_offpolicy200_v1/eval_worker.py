import argparse,json,sys,time,random,hashlib,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--checkpoint');p.add_argument('--checkpoint-sha256');p.add_argument('--rank',type=int);p.add_argument('--output');a=p.parse_args()
cfg=json.loads(Path(a.config).read_text());raw=Path(a.checkpoint).read_bytes();assert hashlib.sha256(raw).hexdigest()==a.checkpoint_sha256
cp=json.loads(raw);assert cp['accepted'] and cp['run_id']==cfg['run_id'] and cp['complete_rank_count']==8
sys.path.insert(0,cfg['paths']['runtime'])
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel,get_peft_model_state_dict
from search_e1.rollout import SearchAgent
from search_e1.core import reward
base=AutoModelForCausalLM.from_pretrained(cfg['paths']['model'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
model=PeftModel.from_pretrained(base,cp['adapter']['path'],is_trainable=False)
values=get_peft_model_state_dict(model);h=hashlib.sha256()
for name,t in sorted(values.items()):
    t=t.detach().cpu().contiguous();assert t.dtype==torch.float32 and torch.isfinite(t).all()
    h.update(json.dumps([name,str(t.dtype),list(t.shape)],separators=(',',':')).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
assert len(values)==504 and h.hexdigest()==cp['all_rank_adapter_sha256'][a.rank]
assert all(t.dtype==torch.bfloat16 for n,t in model.named_parameters() if 'lora_' not in n)
assert model.generation_config.repetition_penalty==1.05 and model.generation_config.eos_token_id==[151645,151643]
model.eval();tokenizer=AutoTokenizer.from_pretrained(cfg['paths']['model'],use_fast=True);agent=SearchAgent(model,tokenizer,cfg['paths']['retriever_url'],max_new_tokens=500,max_context=8192)
random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
rawdata=Path(cfg['paths']['validation']).read_bytes();rows=[json.loads(x) for x in rawdata.splitlines()];assert len(rows)==512
out=Path(a.output);started=time.time()
with (out/f'results-{a.rank}.jsonl').open('x') as sink:
    for index in range(a.rank,512,8):
        row=rows[index]
        with torch.no_grad():result=agent.run(row['question'],temperature=0)
        em=int(reward(result['response'],row['answers'],result['policy_spans']))
        item=dict(index=index,question_id=row['question_id'],dataset=row['data_source'],em=em,result=result)
        sink.write(json.dumps(item,ensure_ascii=False)+'\n');sink.flush()
        print(json.dumps(dict(index=index,em=em,elapsed_s=time.time()-started)),flush=True)
print('COMPLETE rank64',flush=True)
