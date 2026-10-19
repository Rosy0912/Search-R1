"""Method-specific lineage and fixed-three namespaces; no torch or submission."""
import re
from pathlib import Path
from reproduction.seven_dataset.contracts import DATA_SHA,IDS_SHA,digest,require,sha
BASE='/home/jmzhang/Workspace/Recursive-Tree-GRPO-20261014-v1'
BR=BASE+'/official-branpo'
ROOT=BR+'/evaluation3b-final150-v1'
TRAIN_SOURCE='37cd0777a39ea3af077cb230d49ed56bb0630275'
TEMPLATE=BASE+'/receipts/seven-new-original-200-e8211b2abe28.plan.json'
TEMPLATE_SHA='bedc2d11cdbd2550024466606f0de19f5fb9a023ecef8ebd0265e76ef7a4b09d'
CONFIG_SHA='274b6801a78591ca8b28dadcbe68dd0ab46ff5c7b0e4e6be4fc34a82964ceaa0'
MODEL=BR+'/exports/3b-150-final-cpu-v1/huggingface'
SCIENCE={
 'reproduction/seven_dataset/worker.py':'7dad706109c9ee08a9d83bd7dd8ffb099f5a768378b9cca4da9888916b39d31c',
 'reproduction/seven_dataset/runner.py':'7004276ae17ce61fbd8b8a95b14f8008a8021d313f22d49d9f08c92d5f0c8fc4',
 'reproduction/seven_dataset/entry.py':'0788c0a261cd9e738e95ee6395c0b84069c052b4a3f272282b23db018570413e',
 'reproduction/seven_dataset/bootstrap.py':'240d666b5974a9470d183fe197944d8bcb82f6d25e764ddf3c031b19b0207cfb',
 'search_r1/baselines/local_runtime.py':'7712a8802f63f148680d59fc8dfa00869f78e3986cdb3fd10004635c8c465a5b'}
INHERITED=('base','backend','data_root','data_manifest_sha256','final_content_identity','ordered_ids_sha256','retriever_receipt','retriever_receipt_sha256','canonical_tokenizer','canonical_tokenizer_files')
FIELDS=set(INHERITED)|{'protocol','group','outer','model','model_files','training_job_id','training_source','checkpoint_identity','template_plan','template_plan_sha256','protocol_template_config_sha256','evidence','inference_model','evaluation_code_commit','evaluation_run_id','job','output','repeat_index','repeat_count','no_auto_retry','allocation'}

def job(index):
 require(type(index) is int and index in (1,2,3),'Fixed three repeats only')
 return 'zhangjunming-bran3-150-r'+str(index)

def root(index):job(index);return Path(ROOT)/('r'+str(index))

def identity(native,hf):
 return digest(dict(method='BranPO',size='3b',actual_updates=150,label=250,training_job=native['job_id'],training_source=TRAIN_SOURCE,hf_files={n:v['sha256'] for n,v in hf['files'].items()},data=DATA_SHA))

def validate_lineage(cold,native,hf,intent):
 jid=native['job_id'];require(jid.isdigit() and jid not in ('139135','139136'),'Must use accepted continuation')
 require(cold['job_id']==hf['job_id']==jid and cold['parent_job_id']==native['parent_job_id']=='139135','Training chain differs')
 require(native['source']==hf['source']==cold['source']==TRAIN_SOURCE,'Wrong continuation source')
 require(all(v['local_raw_sha_verified'] for v in (cold,native,hf)),'Local original evidence SHA not accepted')
 require(cold['actual_loaded_model_adam_scheduler_saved_rng_exact'] and cold['actual_first_batch_ids_and_driver_exact'] and cold['actual_complete_first_resumed_iteration'],'Cold restore incomplete')
 require(native['status']=='FINAL250_ACTUAL150_ALL24_SHA_MODEL_MEASURED_ADAM_RNG_SCHEDULER_DATA_CPU_ACCEPTED' and native['actual_updates']==150 and native['label']==250 and native['size']=='3b','Final150 native incomplete')
 n=native['native'];require(n['world_size']==8 and n['actual_updates']==150 and n['label']==250 and len(n['files'])==24 and len(n['workers'])==8,'All8 native absent')
 require(n['cuda_initialized'] is False and len(native['shapes'])==435,'Native architecture differs')
 for rank,w in enumerate(n['workers']):
  require(w['rank']==rank and w['scheduler']['last_epoch']==150 and w['scheduler']['_last_lr']==[1e-6],'Scheduler differs')
  require(len(w['optimizer']['state'])==37 and all(v['step']==w['actual_adam_calls'] for v in w['optimizer']['state'].values()),'Actual Adam incomplete')
  old=cold['rank_adam'][rank]['preserved_actual_adam_calls'];require(w['actual_adam_calls']>=old+10,'Adam reset or insufficient resumed updates')
 require(native['dataloader']['next_label']==251 and len(native['dataloader']['next_ids'])==256,'Final data cursor incomplete')
 require(native['scheduler'][3:5]==['COMPLETED','0:0'],'Training terminal differs')
 require(hf['status']=='OFFICIAL_CPU_MERGE_EXACT_ALL_LOGICAL_BF16_FULL_HF_RELOAD_ACCEPTED' and hf['actual_updates']==150 and hf['label']==250 and hf['size']=='3b' and hf['target']==MODEL,'Final HF identity differs')
 p=hf['projection'];require(p['exact_native_bf16_projection_all_parameters'] and p['logical_parameters']==435 and p['rank_parameter_comparisons']==3480 and p['parameters_different_from_initialization'],'HF projection incomplete')
 require(hf['actual_complete_hf_cpu_reload'] and hf['no_cuda_initialized'],'Full HF reload absent')
 require(hf['semantics']['complete_vocabulary_special_tokens_and_chat_template_equal'] and hf['semantics']['eos_token_ids']==[151645,151643],'Tokenizer/EOS differs')
 require(intent['sub']['job_id']==jid and intent['sub']['source']==TRAIN_SOURCE and intent['actual_updates']==150 and intent['label']==250,'Official export binding differs')
 require(intent['native_identity']=={n:r['sha256'] for n,r in native['native']['files'].items()},'HF export not bound to this native')
 return jid

def validate_plan(plan,template,native,hf):
 require(set(plan)==FIELDS,'Unknown/missing plan field')
 require(plan['protocol']=='branpo3b_final150_common_v1' and plan['group']=='BranPO' and type(plan['outer']) is int and plan['outer']==150,'Wrong final method')
 require(plan['backend']=='slurm_lumia' and plan['base']==BASE and all(plan[k]==template[k] for k in INHERITED),'Changed common data/runtime')
 require(plan['template_plan']==TEMPLATE and plan['template_plan_sha256']==TEMPLATE_SHA and plan['protocol_template_config_sha256']==CONFIG_SHA,'Changed protocol template')
 require(plan['model']==MODEL and plan['model_files']=={n:r['sha256'] for n,r in hf['files'].items()},'Model bytes differ')
 require(plan['training_job_id']==native['job_id'] and plan['training_source']==TRAIN_SOURCE and plan['checkpoint_identity']==identity(native,hf),'Unbound training/model')
 require(plan['inference_model']==ROOT+'/model-view','Unexpected immutable canonical tokenizer view')
 require(plan['job']==job(plan['repeat_index']) and Path(plan['output'])==root(plan['repeat_index'])/'output' and type(plan['repeat_count']) is int and plan['repeat_count']==3,'Duplicate or changed repeat')
 require(plan['data_manifest_sha256']==DATA_SHA and plan['ordered_ids_sha256']==IDS_SHA,'Different data')
 require(plan['allocation']==dict(gpus=8,cpus=64,memory_gib=600) and plan['no_auto_retry'] is True,'Allocation/retry differs')
 require(re.fullmatch('[0-9a-f]{40}',plan['evaluation_code_commit']) and re.fullmatch('[0-9a-f]{32}',plan['evaluation_run_id']),'Unfrozen source/execution')
 return root(plan['repeat_index'])
