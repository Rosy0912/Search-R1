"""CPU preparation only: fixed data, all model SHA, six exclusive namespaces."""
import json
from pathlib import Path
import time
from .protocol import PROFILE,SCIENCE,sha,digest,prompt,question_from_old_prompt,validate_rows
from .driver import once,search

BASE=Path('/home/jmzhang/Workspace/Recursive-Tree-GRPO-7B-20261018-v1')
OLD=Path('/home/jmzhang/Workspace/Recursive-Tree-GRPO-20261014-v1')
ROOT=BASE/'reevaluation-official-v2c-20261019'

def retriever_endpoint(base_url):
 from urllib.parse import urlsplit
 parsed=urlsplit(base_url)
 assert parsed.scheme=='http' and parsed.hostname=='node1' and parsed.port==29147
 assert parsed.path in ('','/','/retrieve') and not parsed.query and not parsed.fragment
 return base_url.rstrip('/') if parsed.path=='/retrieve' else base_url.rstrip('/')+'/retrieve'

def prepare():
 import importlib.metadata as metadata
 import pyarrow.parquet as pq
 from transformers import AutoConfig,AutoTokenizer
 from vllm.engine.arg_utils import EngineArgs
 from vllm.platforms.cuda import device_id_to_physical_device_id
 import os,ctypes
 old_mask=os.environ.get('CUDA_VISIBLE_DEVICES')
 try:
  for index in range(8):
   os.environ['CUDA_VISIBLE_DEVICES']=str(index)
   assert device_id_to_physical_device_id(0)==index
 finally:
  if old_mask is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
  else:os.environ['CUDA_VISIBLE_DEVICES']=old_mask
 assert ctypes.CDLL('libcudart.so.12').cudaDeviceGetUuid
 deployment=json.loads((ROOT/'deployment.json').read_text())
 source=Path(deployment['source']);assert source==Path(__file__).resolve().parents[2]
 for name,h in deployment['source_files'].items():assert sha(source/name)==h,name
 assert not (ROOT/'prepare-once.json').exists()
 assert not (ROOT/'plans').exists() and not (ROOT/'data').exists()
 once(ROOT/'prepare-once.json',dict(at=time.time(),source=deployment['commit']))
 versions={n:metadata.version(n) for n in ['torch','transformers','vllm','pyarrow']}
 assert versions['vllm']=='0.6.3' and versions['torch'].split('+')[0]=='2.4.0'
 original_manifest=OLD/'common-data/manifest.json'
 assert sha(original_manifest)=='7b6369e17ebd497ea159d57dc159647eed8adad852b7468cc600b7f203e80f43'
 dm=json.loads(original_manifest.read_text());original=OLD/'common-data/final.parquet'
 assert sha(original)==dm['splits']['final']['sha256']
 rows=[]
 for i,r in enumerate(pq.read_table(original).to_pylist()):
  question=question_from_old_prompt(r['prompt']);gold=r['reward_model']['ground_truth']['target']
  rows.append(dict(final_index=i,question_id=r['extra_info']['index'],source=r['data_source'],
   gold_answers=[gold] if isinstance(gold,str) else list(gold),question=question,prompt=prompt(question)))
 validate_rows(rows)
 (ROOT/'data').mkdir();data=ROOT/'data/final.jsonl'
 with data.open('x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 identity=digest([dict(final_index=r['final_index'],question_id=r['question_id'],source=r['source'],gold_answers=r['gold_answers']) for r in rows])
 receipt=OLD/'receipts/gpu-retriever-ready.json';rr=json.loads(receipt.read_text())
 endpoint=retriever_endpoint(rr['url']);probe,probe_sha=search(endpoint,['Beijing capital'])
 assert len(probe)==1 and len(probe[0])==3
 once(ROOT/'retriever-health-preflight.json',dict(at=time.time(),endpoint=endpoint,response_sha256=probe_sha,documents=3,no_new_gpu_allocation=True))
 models=json.loads((ROOT/'models.json').read_text())['models'];(ROOT/'plans').mkdir();plans=[]
 for kind in ['base','searchr1']:
  model=BASE/'models'/kind/models[kind]['revision'];files={v['file']:v['sha256'] for v in models[kind]['files']}
  for name,h in files.items():assert sha(model/name)==h,name
  c=AutoConfig.from_pretrained(model,local_files_only=True)
  assert (c.hidden_size,c.num_hidden_layers,c.num_attention_heads,c.num_key_value_heads)==(3584,28,28,4)
  tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
  assert tok.eos_token_id==151645 and tok.pad_token_id==151643
  args=EngineArgs(model=str(model),tokenizer=str(model),dtype='bfloat16',tensor_parallel_size=1,
   enforce_eager=True,gpu_memory_utilization=.85,max_model_len=4596,
   max_num_batched_tokens=8192,max_num_seqs=128,seed=0,disable_log_stats=True)
  assert args.max_model_len==4596 and args.tensor_parallel_size==1 and args.seed==0
  max_len=max(len(tok.apply_chat_template([dict(role='user',content=r['prompt'])],add_generation_prompt=True,tokenize=True)) for r in rows)
  assert max_len<=2048
  for repeat in [1,2,3]:
   root=ROOT/(kind+'-r'+str(repeat));assert not root.exists();root.mkdir()
   plan=dict(profile=PROFILE,science=SCIENCE,kind=kind,repeat=repeat,repeat_count=3,root=str(root),
    job='zhangjunming-7b-v2c-'+kind+'-r'+str(repeat),source=str(source),source_commit=deployment['commit'],
    source_files=deployment['source_files'],model=str(model),model_revision=models[kind]['revision'],
    model_files=files,model_identity=digest(dict(repo=models[kind]['repo'],revision=models[kind]['revision'],files=files)),
    data=str(data),data_sha256=sha(data),id_gold_identity=identity,original_data_sha256=sha(original),
    retriever_receipt=str(receipt),retriever_receipt_sha256=sha(receipt),retriever_url=endpoint,
    gpus=8,cpus=64,memory_gib=600,no_auto_retry=True)
   path=ROOT/'plans'/(kind+'-r'+str(repeat)+'.json');once(path,plan)
   once(root/'preflight.json',dict(at=time.time(),plan_sha256=sha(path),versions=versions,
    actual_all_model_data_source_sha_verified=True,all51713_prompt_tokenization_verified=True,max_prompt_tokens=max_len))
   plans.append(str(path))
 once(ROOT/'prepared.json',dict(at=time.time(),plans=plans,profile=PROFILE,data_sha256=sha(data),
  id_gold_identity=identity,rows=len(rows),versions=versions,no_gpu_action=True))
 print(json.dumps(dict(status='SIX_PLANS_CPU_PREPARED',plans=plans)),flush=True)

if __name__=='__main__':prepare()
