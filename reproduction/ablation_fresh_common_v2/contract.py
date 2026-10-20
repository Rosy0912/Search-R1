"""All23 retained commonv2 slots, bound to independently accepted new models."""
from pathlib import Path
import json,re
from reproduction.seven_dataset.contracts import require
from reproduction.official_eval_v2.protocol import PROFILE,SCIENCE,digest,sha,validate_rows
from reproduction.cgrts150_common_v2.contract import CANONICAL,CANONICAL_FILES,DATA_MANIFEST,SCIENCE_FILES
from .scope import BASE,SOURCE,slot,repeat_count,model_binding,DATA_SHA,ID_GOLD
from .lineage import EVIDENCE,validate,fresh_boundary
COMMON='/home/jmzhang/Workspace/Recursive-Tree-GRPO-20261014-v1'

def checkpoint_root(group,step):return Path(slot(group,step,1)['root']).parent
def job(group,step,index):return slot(group,step,index)['job_name']
def root(group,step,index):return Path(slot(group,step,index)['root'])

def validate_plan(plan,values,hashes):
 group,step,index=plan['group'],plan['actual_training_iterations'],plan['repeat']
 declared=slot(group,step,index);cp=checkpoint_root(group,step)
 b=validate(values,hashes,group,step);hf=values['hf.json'];sub=values['training-submission.json']
 identity=model_binding(declared,sub,hf,hashes['hf.json'])
 require(plan['profile']==PROFILE and plan['science']==SCIENCE and plan['kind']=='trained-ablation-fresh','Wrong common protocol/method')
 require(type(plan['repeat_count']) is int and plan['repeat_count']==declared['repeat_count'] and plan['job']==declared['job_name'] and plan['root']==declared['root'],'Wrong retained slot')
 require(plan['training_job_id']==sub['job_id'] and plan['training_source']==SOURCE and plan['training_science_source']==sub['science_source'] and plan['training_limit_remains']==150 and plan['native_boundary']=='post-evaluation','Wrong new training identity')
 require(plan['accepted_model']==declared['model'] and plan['hf_files']==hf['hf_identity']['files'],'Wrong accepted model files')
 require(plan['model']==str(cp/'model-view') and plan['model_revision'] is None and plan['model_identity']==identity,'Wrong canonical model view')
 expected={n:h for n,h in plan['hf_files'].items() if n.endswith('.safetensors') or n in ('model.safetensors.index.json','config.json','generation_config.json')};expected.update(CANONICAL_FILES)
 require(plan['model_files']==expected,'Weight or canonical tokenizer changed')
 require(plan['data']==str(cp/'final.jsonl') and plan['data_sha256']==DATA_SHA and plan['id_gold_identity']==ID_GOLD and plan['data_manifest_sha256']==DATA_MANIFEST,'Wrong full51713 inputs')
 require(plan['retriever_receipt']==COMMON+'/receipts/gpu-retriever-ready.json','Wrong protected retriever')
 require((plan['gpus'],plan['cpus'],plan['memory_gib'])==(8,64,600) and plan['no_auto_retry'] is True,'Wrong allocation/retry')
 require(plan['engine_seed']==0 and plan['greedy'] is True and plan['questions']==51713,'Changed seed/decoding or subset')
 require(re.fullmatch('[0-9a-f]{40}',plan['source_commit']) and plan['source']==str(BASE/'sources'/plan['source_commit']),'Unfrozen inference source')
 for n,h in SCIENCE_FILES.items():require(plan['source_files'][n]==h,'Changed common scientific source')
 return Path(declared['root'])

def evidence(plan):
 cp=checkpoint_root(plan['group'],plan['actual_training_iterations']);values={};hashes={}
 require(set(plan['evidence'])==set(EVIDENCE),'Missing new model proof')
 for n in EVIDENCE:
  row=plan['evidence'][n];f=cp/'inputs'/n
  require(row['path']==str(f) and sha(f)==row['sha256'],'Accepted evidence changed')
  values[n]=json.loads(f.read_text());hashes[n]=row['sha256']
 return values,hashes

def verify(plan):
 values,hashes=evidence(plan);r=validate_plan(plan,values,hashes)
 fresh_boundary(values,hashes,plan['group'],plan['actual_training_iterations'])
 for n,h in plan['source_files'].items():require(sha(Path(plan['source'])/n)==h,'Source changed: '+n)
 for n,h in plan['hf_files'].items():require(sha(Path(plan['accepted_model'])/n)==h,'Accepted HF changed: '+n)
 for n,h in plan['model_files'].items():require(sha(Path(plan['model'])/n)==h,'Canonical view changed: '+n)
 require(sha(plan['data'])==DATA_SHA and sha(plan['retriever_receipt'])==plan['retriever_receipt_sha256'],'Data/retriever changed')
 rows=[json.loads(line) for line in Path(plan['data']).open()];validate_rows(rows)
 require(digest([{k:v[k] for k in ('final_index','question_id','source','gold_answers')} for v in rows])==ID_GOLD,'ID/gold changed')
 from reproduction.official_eval_v2.prepare import retriever_endpoint
 ready=json.loads(Path(plan['retriever_receipt']).read_text());require(plan['retriever_url']==retriever_endpoint(ready['url']),'Retriever URL changed')
 return rows,r
