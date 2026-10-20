"""Accepted pre-evaluation100 boundary and the unique correctedcommonv2 n1 slot."""
import json,re
from pathlib import Path
from reproduction.seven_dataset.contracts import require
from reproduction.official_eval_v2.protocol import PROFILE,SCIENCE,digest,sha,validate_rows
from . import lineage
BASE='/home/jmzhang/Workspace/Recursive-Tree-GRPO-20261014-v1'
ROOT='/home/jmzhang/Workspace/Recursive-Tree-GRPO-7B-20261018-v1/evaluations/original100-commonv2-20261020-v1'
TRAIN_SOURCE=lineage.SOURCE
MODEL=lineage.MODEL
VIEW=ROOT+'/model-view'
CURRENT_METADATA=lineage.ROOT+'/checkpoints/actor/global_step_50'
CANONICAL='/home/jmzhang/Workspace/Recursive-Tree-GRPO-7B-20261018-v1/models/base/a09a35458c702b33eeacc393d103063234e8bc28'
DATA_MANIFEST='7b6369e17ebd497ea159d57dc159647eed8adad852b7468cc600b7f203e80f43'
FINAL_JSONL='a1b95766beecb66753b5ee4c3dd2552d6a47e375c4c117200248dae6b6fc08d5'
ID_GOLD='4953085d4fdaf823fe355972d8dceae9e167e27d264f0f6b7f8bcf64473c6845'
EVIDENCE=('native.json','native-manifest.json','training-plan.json','training-submission.json')
CANONICAL_FILES={
 'tokenizer.json':'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
 'tokenizer_config.json':'5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
 'merges.txt':'599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3',
 'vocab.json':'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}
SCIENCE_FILES={
 'reproduction/official_eval_v2/driver.py':'f3b81d83ba60bca1b3d5bd5415b34b8ee80ffc7df2c418c7797096014a8a3f93',
 'reproduction/official_eval_v2/protocol.py':'4da23b50ad4e80b4da17c2a17c677ebbd74b6b8eacd5e9740e76d4bc46ed92ed',
 'reproduction/official_eval_v2/port_recovery.py':'ec586e53ca75b9f2e21cd7f68001445234768e81acd967fb7dbe4238d63776d5',
 'search_r1/baselines/preflight.py':'2c918d0ef82281d0cfc8a16a198d9338348197f33886300b1098ba34d8b7d7d4'}

def job(index):
 require(type(index) is int and index==1,'Only the authorized100 n1 slot')
 return 'zhangjunming-cgrts7-100-v2-n1-oct20'

def root(index):job(index);return Path(ROOT)/'r1'

def identity(native):
 return digest(dict(method='CGRTS',size='7b',actual_updates=100,training_limit=200,
  boundary='pre-evaluation',training_job=lineage.JOB,training_source=TRAIN_SOURCE,
  native_manifest_sha256=native['manifest_sha256'],hf_files=native['hf_identity']['files'],data=FINAL_JSONL,profile=PROFILE))

def validate_lineage(native,manifest,training_plan,submission):
 return lineage.validate(native,manifest,training_plan,submission)

def validate_plan(plan,native,manifest,training_plan,submission):
 validate_lineage(native,manifest,training_plan,submission)
 require(plan['profile']==PROFILE and plan['science']==SCIENCE and plan['kind']=='trained-cgrts','Changed corrected commonv2/method')
 require(type(plan['repeat_count']) is int and plan['repeat_count']==1 and plan['job']==job(plan['repeat']) and plan['root']==str(root(plan['repeat'])),'Wrong n1 slot')
 require(plan['training_job_id']==lineage.JOB and plan['training_source']==TRAIN_SOURCE and plan['actual_training_iterations']==100 and plan['training_limit_remains']==200 and plan['native_boundary']=='pre-evaluation','Wrong100/200 training boundary')
 require(plan['accepted_model']==MODEL and plan['hf_files']==native['hf_identity']['files'],'Accepted HF bytes differ')
 require(plan['model']==VIEW and plan['model_revision'] is None and plan['model_identity']==identity(native),'Wrong canonical model view')
 files={n:h for n,h in plan['hf_files'].items() if n.endswith('.safetensors') or n in ('model.safetensors.index.json','config.json','generation_config.json')};files.update(CANONICAL_FILES)
 require(plan['model_files']==files,'View must preserve accepted weights and canonical tokenizer')
 require(plan['data']==ROOT+'/final.jsonl' and plan['data_sha256']==FINAL_JSONL and plan['id_gold_identity']==ID_GOLD and plan['data_manifest_sha256']==DATA_MANIFEST,'Wrong all51713 inputs')
 require(plan['retriever_receipt']==BASE+'/receipts/gpu-retriever-ready.json','Wrong protected retriever')
 require((plan['gpus'],plan['cpus'],plan['memory_gib'])==(8,64,600) and plan['no_auto_retry'] is True,'Wrong allocation/retry')
 require(re.fullmatch('[0-9a-f]{40}',plan['source_commit']) and plan['source']==BASE+'/sources/'+plan['source_commit'],'Unfrozen inference source')
 for n,h in SCIENCE_FILES.items():require(plan['source_files'][n]==h,'Changed common scientific source')
 return root(plan['repeat'])

def verify(plan):
 evidence={}
 require(set(plan['evidence'])==set(EVIDENCE),'Incomplete C15 evidence bundle')
 for name in EVIDENCE:
  row=plan['evidence'][name];require(row['path']==ROOT+'/inputs/'+name and sha(row['path'])==row['sha256'],'Lineage proof changed')
  evidence[name]=json.loads(Path(row['path']).read_text())
 native=evidence['native.json']
 require(plan['evidence']['native-manifest.json']['sha256']==native['manifest_sha256'] and plan['evidence']['training-plan.json']['sha256']==lineage.PLAN,'Original manifest/plan physical SHA differs')
 require(plan['evidence']['training-submission.json']['sha256']==native['files']['submission.json']['sha256'],'Original submission physical SHA differs')
 r=validate_plan(plan,*[evidence[n] for n in EVIDENCE])
 for n,h in plan['source_files'].items():require(sha(Path(plan['source'])/n)==h,'Source changed: '+n)
 for n,h in plan['hf_files'].items():require(sha(Path(MODEL)/n)==h,'Accepted HF changed: '+n)
 for n,h in plan['model_files'].items():require(sha(Path(VIEW)/n)==h,'Common view changed: '+n)
 require(sha(plan['data'])==FINAL_JSONL and sha(plan['retriever_receipt'])==plan['retriever_receipt_sha256'],'Data/retriever receipt changed')
 rows=[json.loads(line) for line in Path(plan['data']).open()];validate_rows(rows)
 require(digest([{k:v[k] for k in ('final_index','question_id','source','gold_answers')} for v in rows])==ID_GOLD,'ID/gold identity changed')
 from reproduction.official_eval_v2.prepare import retriever_endpoint
 ready=json.loads(Path(plan['retriever_receipt']).read_text());require(plan['retriever_url']==retriever_endpoint(ready['url']),'Retriever URL changed')
 return rows,r
