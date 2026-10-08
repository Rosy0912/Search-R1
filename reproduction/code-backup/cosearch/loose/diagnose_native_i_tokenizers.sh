set -euo pipefail
CUDA_VISIBLE_DEVICES='' /dataset/shaoshitong/search-r1-rl/cosearch/author-native/env/bin/python - <<'PY'
import json,hashlib,sys
from pathlib import Path
from transformers import AutoTokenizer,AutoConfig
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch');native=root/'author-native';sys.path[:0]=[str(native/'repo'),str(native/'repo/verl')]
from verl.utils import hf_tokenizer
models={'base':root/'assets-root/assets/qwen2.5-3b','warm_actor':native/'runs/author-native-grammar-warm-start-v2/warm-start/actor','warm_ranker':native/'runs/author-native-grammar-warm-start-v2/warm-start/ranker'}
result={}
for name,path in models.items():
 for loader,func in [('Auto',AutoTokenizer.from_pretrained),('native',hf_tokenizer)]:
  t=func(str(path),trust_remote_code=False);c=AutoConfig.from_pretrained(path)
  result[name+'_'+loader]=dict(vocab_size=t.vocab_size,len_tokenizer=len(t),config_vocab_size=c.vocab_size,eos=t.eos_token_id,pad=t.pad_token_id,special_tokens=t.special_tokens_map,added_vocab=t.get_added_vocab(),tags={s:dict(ids=t.encode(s,add_special_tokens=False),decoded=t.decode(t.encode(s,add_special_tokens=False),skip_special_tokens=False),skip_special=t.decode(t.encode(s,add_special_tokens=False),skip_special_tokens=True)) for s in ['<tool_call>','</tool_call>','<reason>','</reason>','<answer>']})
corpus=json.loads((native/'native-grammar-warm-start-fulltext-v2/corpus.json').read_text());row=corpus['actor'][0];t=AutoTokenizer.from_pretrained(models['warm_actor'])
result['actual_first_sft_response']=dict(ids=row['input_ids'][-row['response_length']:],decoded=t.decode(row['input_ids'][-row['response_length']:],skip_special_tokens=False),target=row['target'])
result['cuda_initialized']=False
print(json.dumps(result,ensure_ascii=False))
PY
