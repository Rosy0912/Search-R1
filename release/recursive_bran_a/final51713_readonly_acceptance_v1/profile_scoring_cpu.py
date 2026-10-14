"""CPU-only comparison on 16 already saved training trajectories, no generation."""
import json
import os
import subprocess

CODE=r'''
import cProfile,gzip,hashlib,io,json,pstats,time
from pathlib import Path
from types import SimpleNamespace
import torch
from omegaconf import OmegaConf
from verl.utils import hf_tokenizer
from search_r1.baselines import tree_raw_tokens as raw
from search_r1.baselines.evaluation import batch_answer_metrics
B=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013')
paths=list(B.glob('**/outer-000200*/trajectories.jsonl.gz'))
assert len(paths)==1, 'Expected one preserved outer200 trajectory file'
path=paths[0];before=path.stat();records=[]
with gzip.open(path,'rt') as stream:
    for _ in range(16):records.append(json.loads(next(stream)))
config=OmegaConf.load(B/'receipts/resolved.yaml')
tok=hf_tokenizer(config.actor_rollout_ref.model.path)
identity=raw.register_validation_tokenizer(tok)
items=[]
for r in records:
    v=r['metadata'][raw.FIELD];prompt=v['prompt_token_ids'];response=v['policy_response_token_ids'];mask=v['policy_model_mask']
    batch=dict(prompts=torch.tensor(prompt),responses=torch.tensor(response),
        attention_mask=torch.ones(len(prompt)+len(response),dtype=torch.long),
        info_mask=torch.tensor([1]*len(prompt)+mask))
    items.append(SimpleNamespace(batch=batch,non_tensor_batch=r['metadata'],meta_info={}))
result={};output={}
for label in ('original_per_item_validation','shared_explicit_batch_scope'):
    profile=cProfile.Profile();start=time.perf_counter();profile.enable()
    if label=='original_per_item_validation':value=batch_answer_metrics(items,tok)
    else:
        with raw.tokenizer_validation_scope(tok,identity):value=batch_answer_metrics(items,tok)
    profile.disable();seconds=time.perf_counter()-start;buf=io.StringIO()
    pstats.Stats(profile,stream=buf).sort_stats('cumulative').print_stats(12)
    output[label]=value
    result[label]=dict(seconds=seconds,rows=len(value),metrics_sha256=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest(),profile=buf.getvalue())
assert output['original_per_item_validation']==output['shared_explicit_batch_scope']
after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(time=time.time(),source=str(path),sample_rows=16,metrics_exact_equal=True,
    conditions=result,cuda_initialized=False,scope='Existing outer200 CPU scoring only; no new rollout, GPU, model weights, or production RNG')))
'''
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
         PYTHONPATH='/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013/source-resume100-v1')
env.pop('LD_LIBRARY_PATH',None)
p=subprocess.run(['/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009/env/bin/python','-B','-c',CODE],
                 env=env,capture_output=True,text=True,timeout=120)
print(json.dumps(dict(rc=p.returncode,stdout=p.stdout,stderr=p.stderr[-3000:])))
raise SystemExit(p.returncode)
