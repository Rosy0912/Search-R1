"""Final150 BranPO7B lineage and corrected commonv2 fixed-three slots."""
import json,re
from pathlib import Path
from reproduction.seven_dataset.contracts import require
from reproduction.official_eval_v2.protocol import PROFILE,SCIENCE,digest,sha,validate_rows
BASE='/home/jmzhang/Workspace/Recursive-Tree-GRPO-20261014-v1'
BR=BASE+'/official-branpo'
ROOT=BR+'/evaluation7b-final150-commonv2-v1'
TRAIN_SOURCE='37cd0777a39ea3af077cb230d49ed56bb0630275'
MODEL=BR+'/exports/7b-150-final-cpu-v1/huggingface'
VIEW=ROOT+'/model-view'
CANONICAL='/home/jmzhang/Workspace/Recursive-Tree-GRPO-7B-20261018-v1/models/base/a09a35458c702b33eeacc393d103063234e8bc28'
DATA_MANIFEST='7b6369e17ebd497ea159d57dc159647eed8adad852b7468cc600b7f203e80f43'
FINAL_JSONL='a1b95766beecb66753b5ee4c3dd2552d6a47e375c4c117200248dae6b6fc08d5'
ID_GOLD='4953085d4fdaf823fe355972d8dceae9e167e27d264f0f6b7f8bcf64473c6845'
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
 require(type(index) is int and index in (1,2,3),'Fixed three repeats only')
 return 'zhangjunming-bran7-150-v2-r'+str(index)

def root(index):job(index);return Path(ROOT)/('r'+str(index))

def identity(native,hf):
 return digest(dict(method='BranPO',size='7b',actual_updates=150,label=250,training_job=native['job_id'],training_source=TRAIN_SOURCE,hf_files={n:v['sha256'] for n,v in hf['files'].items()},data=FINAL_JSONL,profile=PROFILE))

def validate_lineage(cold,native,hf,intent):
 jid=native['job_id'];require(jid.isdigit() and jid not in ('139135','139136'),'Must use accepted continuation')
 require(cold['job_id']==hf['job_id']==jid and cold['parent_job_id']==native['parent_job_id']=='139136','Training chain differs')
 require(native['source']==hf['source']==cold['source']==TRAIN_SOURCE,'Wrong continuation source')
 require(all(v['local_raw_sha_verified'] for v in (cold,native,hf)),'Local original evidence SHA not accepted')
 require(cold['actual_loaded_model_adam_scheduler_saved_rng_exact'] and cold['actual_first_batch_ids_and_driver_exact'] and cold['actual_complete_first_resumed_iteration'],'Cold restore incomplete')
 require(native['status']=='FINAL250_ACTUAL150_ALL24_SHA_MODEL_MEASURED_ADAM_RNG_SCHEDULER_DATA_CPU_ACCEPTED' and native['actual_updates']==150 and native['label']==250 and native['size']=='7b','Final150 native incomplete')
 n=native['native'];require(n['world_size']==8 and n['actual_updates']==150 and n['label']==250 and len(n['files'])==24 and len(n['workers'])==8,'All8 native absent')
 require(n['cuda_initialized'] is False and len(native['shapes'])==339,'Native architecture differs')
 for rank,w in enumerate(n['workers']):
  require(w['rank']==rank and w['scheduler']['last_epoch']==150 and w['scheduler']['_last_lr']==[1e-6],'Scheduler differs')
  require(len(w['optimizer']['state'])==29 and all(v['step']==w['actual_adam_calls'] for v in w['optimizer']['state'].values()),'Actual Adam incomplete')
  old=cold['rank_adam'][rank]['preserved_actual_adam_calls'];require(w['actual_adam_calls']>=old+10,'Adam reset or insufficient resumed updates')
 require(native['dataloader']['next_label']==251 and len(native['dataloader']['next_ids'])==256,'Final data cursor incomplete')
 require(native['scheduler'][3:5]==['COMPLETED','0:0'],'Training terminal differs')
 require(hf['status']=='OFFICIAL_CPU_MERGE_EXACT_ALL_LOGICAL_BF16_FULL_HF_RELOAD_ACCEPTED' and hf['actual_updates']==150 and hf['label']==250 and hf['size']=='7b' and hf['target']==MODEL,'Final HF identity differs')
 p=hf['projection'];require(p['exact_native_bf16_projection_all_parameters'] and p['logical_parameters']==339 and p['rank_parameter_comparisons']==2712 and p['parameters_different_from_initialization'],'HF projection incomplete')
 require(hf['actual_complete_hf_cpu_reload'] and hf['no_cuda_initialized'],'Full HF reload absent')
 require(hf['semantics']['complete_vocabulary_special_tokens_and_chat_template_equal'] and hf['semantics']['eos_token_ids']==[151645,151643],'Tokenizer/EOS differs')
 require(intent['sub']['job_id']==jid and intent['sub']['source']==TRAIN_SOURCE and intent['actual_updates']==150 and intent['label']==250,'Official export binding differs')
 require(intent['native_identity']=={n:r['sha256'] for n,r in native['native']['files'].items()},'HF export not bound to this native')
 return jid

def validate_plan(plan,cold,native,hf,intent):
 validate_lineage(cold,native,hf,intent)
 require(plan['profile']==PROFILE and plan['science']==SCIENCE and plan['kind']=='trained-branpo','Changed corrected commonv2/method')
 require(plan['repeat_count']==3 and plan['job']==job(plan['repeat']) and plan['root']==str(root(plan['repeat'])),'Wrong repeat slot')
 require(plan['training_job_id']==native['job_id'] and plan['training_source']==TRAIN_SOURCE and plan['actual_training_iterations']==150 and plan['final_directory_label']==250,'Wrong training chain')
 require(plan['accepted_model']==MODEL and plan['hf_files']=={n:r['sha256'] for n,r in hf['files'].items()},'Accepted HF bytes differ')
 require(plan['model']==VIEW and plan['model_revision'] is None and plan['model_identity']==identity(native,hf),'Wrong canonical model view')
 files={n:h for n,h in plan['hf_files'].items() if n.endswith('.safetensors') or n in ('model.safetensors.index.json','config.json','generation_config.json')}
 files.update(CANONICAL_FILES)
 require(plan['model_files']==files,'View does not preserve accepted weights and canonical tokenizer')
 require(plan['data']==ROOT+'/final.jsonl' and plan['data_sha256']==FINAL_JSONL and plan['id_gold_identity']==ID_GOLD and plan['data_manifest_sha256']==DATA_MANIFEST,'Wrong all51713 inputs')
 require(plan['retriever_receipt']==BASE+'/receipts/gpu-retriever-ready.json','Wrong protected retriever')
 require((plan['gpus'],plan['cpus'],plan['memory_gib'])==(8,64,600) and plan['no_auto_retry'] is True,'Wrong allocation/retry')
 require(re.fullmatch('[0-9a-f]{40}',plan['source_commit']) and plan['source']==BASE+'/sources/'+plan['source_commit'],'Unfrozen source')
 for n,h in SCIENCE_FILES.items():require(plan['source_files'][n]==h,'Changed common scientific source')
 return root(plan['repeat'])

def verify(plan):
 evidence={}
 for name in ('cold.json','native.json','hf.json','export-intent.json'):
  row=plan['evidence'][name];require(row['path']==ROOT+'/inputs/'+name and sha(row['path'])==row['sha256'],'Lineage proof changed')
  evidence[name]=json.loads(Path(row['path']).read_text())
 r=validate_plan(plan,*[evidence[n] for n in ('cold.json','native.json','hf.json','export-intent.json')])
 for n,h in plan['source_files'].items():require(sha(Path(plan['source'])/n)==h,'Source changed: '+n)
 for n,h in plan['hf_files'].items():require(sha(Path(MODEL)/n)==h,'Accepted HF changed: '+n)
 for n,h in plan['model_files'].items():require(sha(Path(VIEW)/n)==h,'Common view changed: '+n)
 require(sha(plan['data'])==FINAL_JSONL and sha(plan['retriever_receipt'])==plan['retriever_receipt_sha256'],'Data/retriever receipt changed')
 rows=[json.loads(line) for line in Path(plan['data']).open()];validate_rows(rows)
 require(digest([{k:v[k] for k in ('final_index','question_id','source','gold_answers')} for v in rows])==ID_GOLD,'ID/gold identity changed')
 from reproduction.official_eval_v2.prepare import retriever_endpoint
 ready=json.loads(Path(plan['retriever_receipt']).read_text());require(plan['retriever_url']==retriever_endpoint(ready['url']),'Retriever URL changed')
 return rows,r
