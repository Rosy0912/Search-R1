set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
/dataset/shaoshitong/search-r1-rl/cosearch/author-native/env/bin/python - <<'PY'
import json,hashlib,sys,os,time
from pathlib import Path
import torch,pandas as pd
from transformers import AutoTokenizer
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')
sys.path[:0]=[str(root/'repo'),str(root/'repo/verl')]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
destination=root/'native-format-t1-next-cohort-v4.json';assert not destination.exists()
assert not (root/'runs/author-native-grammar-actor-continue-v4').exists()
k_path=root/'runs/author-native-format-t1-probe-k/result.json';k=json.loads(k_path.read_text())
excluded={q['sha256'] for q in k['question_prefixes']}
corpus_file=root/'native-grammar-warm-start-fulltext-v2/corpus.json';corpus=json.loads(corpus_file.read_text())
assert sha(corpus_file)==json.loads((corpus_file.parent/'manifest.json').read_text())['corpus_sha256']
for row in corpus['actor'][::2]:excluded.add(hashlib.sha256(json.dumps(row['input_ids'][:-row['response_length']]).encode()).hexdigest())
tok=AutoTokenizer.from_pretrained(root/'runs/author-native-grammar-actor-continue-v3/warm-start/actor',local_files_only=True)
train_file=root/'data/train.parquet';val_file=root/'data/validation-64.parquet'
train={list(p)[-1]['content'] for p in pd.read_parquet(train_file,columns=['prompt']).prompt};val={list(p)[-1]['content'] for p in pd.read_parquet(val_file,columns=['prompt']).prompt}
raw_file=sorted((root/'runs/author-native-warm-five-i/evidence/raw-rollouts').glob('pre-postprocess-696092-*.pt'))[0]
rows=torch.load(raw_file,map_location='cpu',weights_only=False);questions=[]
for row in rows:
 ids=row.prompt_ids.reshape(-1).tolist()
 while ids and ids[0]==tok.pad_token_id:ids.pop(0)
 digest=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
 if digest in excluded:continue
 text=tok.decode(ids,skip_special_tokens=False);content=text.split('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0]
 assert content in train and content not in val
 excluded.add(digest);questions.append(dict(prefix_ids=ids,prefix_text=text,sha256=digest))
 if len(questions)==16:break
assert len(questions)==16 and not torch.cuda.is_initialized()
record=dict(questions=questions,created_before_v4_training=True,created_at=time.time(),TRAIN_development_only_not_protocol_holdout=True,warm128_and_K16_excluded=True,corpus_sha256=sha(corpus_file),K_result_sha256=sha(k_path),raw_source_sha256=sha(raw_file),train_sha256=sha(train_file),val64_sha256=sha(val_file),supervised_updates=0,real_rl_updates=0)
with destination.open('x') as f:json.dump(record,f,indent=2,ensure_ascii=False)
print(json.dumps(dict(path=str(destination),sha256=sha(destination),questions=16,real_rl_updates=0,supervised_updates=0)))
PY
