"""CPU actual retained 16-row QA equality/timing; no logits, RNG or training writes."""
import argparse,json,os,time
from pathlib import Path
from types import SimpleNamespace as NS
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--model',required=True);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
from transformers import AutoTokenizer
from verl import DataProto
from verl.trainer.main_ppo import RewardManager
from search_r1.baselines.candidate_qa_reward import candidate_qa_rewards
from search_r1.baselines.tree_raw_tokens import FIELD,tokenizer_identity
saved=torch.load(a.input,map_location='cpu',weights_only=False)
tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
data=DataProto.from_dict(tensors=saved['batch'],non_tensors=saved['non_tensor_batch'],meta_info=saved['meta_info'])[:16]
identity=data.non_tensor_batch[FIELD][0]['raw_ledger'][0]['action']['tokenizer_identity']
assert tokenizer_identity(tok,identity['tokenizer_json_sha256'])==identity
engine=NS(tokenizer=tok,tokenizer_identity=identity,token_preservation={},config=NS(max_prompt_length=4096))
reward=RewardManager(tok,0)
start=time.perf_counter();old=reward(data).sum(-1).tolist();oldwall=time.perf_counter()-start
states=[NS(source=i,response=p['policy_response_token_ids'],raw_ledger=p['raw_ledger']) for i,p in enumerate(data.non_tensor_batch[FIELD])]
start=time.perf_counter();new=candidate_qa_rewards(engine,data,states,reward);newwall=time.perf_counter()-start
actual=data.batch['token_level_scores'].sum(-1).tolist()
assert new==old==actual
assert torch.cuda.is_initialized() is False
print(json.dumps(dict(status='ACTUAL_RETAINED_16_QA_EQUAL',rows=16,new_equals_original_and_actual=True,oldwall=oldwall,newwall=newwall,cuda_initialized=False,scope='16 actual retained rows, not whole-index probability or training throughput')))
