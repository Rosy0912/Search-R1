"""Native CPU concrete G1 recipe and fixed300 calibration; never submits GPU."""
import argparse,hashlib,importlib.util,json,os,pathlib,socket,sys
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
COMMON=ROOT/'outputs/common-data/59f547830de6916f0d20'
CONTENT='fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524'
TRAIN_SHA='e5e774fd1f5777a8be91103facaccb711cfd9593d7563e9d5afe04e6b1a52bea'
HOLDOUT_SHA='c4c42dc200de0b0561f7e9b0d2bc067c732cfe7e158f1718882bb0a8f2c84fbb'
QUESTION_SOURCE='33fd089db355f770f3491b2ee6b0f0b8ec86d6051f404d250c30079ce9a44116'
QUALITY_REVIEW='19d172818cc07cc46a030b70afdbeee2d7fdc6a9cce62a33b140b2919000e569'
MANIFESTS={'candidate':'8f6582c4cb31ccbdbef62cfbe602981d1cf7032b69a01655604908d64fb783ff','producer':'0c5669eea865d8f1494e3aa752f99b657b4a2a5a294fe65e09fae5b737b1f9c5','pipeline':'d9f9b063f64495f05119eccd54ae050b6edf45d647d9cbb02cef01c2c0557e0b'}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def ref(p):return {'path':str(p),'sha256':sha(p)}
def atomic(p,d):
 with open(p,'x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def select_calibration(holdout,train,final,val,key):
 trainkeys=set(train.prompt.map(key));finalkeys=set(final.prompt.map(key))|set(val.prompt.map(key))
 hk=holdout.prompt.map(key)
 need(not (set(hk)&trainkeys) and not(set(hk)&finalkeys),'Common holdout question contamination')
 selected=holdout.loc[holdout.data_source=='nq'].copy();selected['_question_key']=selected.prompt.map(key)
 selected=selected.sort_values('_question_key',kind='mergesort').drop_duplicates('_question_key').iloc[:300]
 need(len(selected)==300,'Need300 unique heldout NQ questions')
 keys=selected.pop('_question_key').tolist();need(not(set(keys)&trainkeys) and not(set(keys)&finalkeys),'Calibration leakage')
 return selected,keys

def command(config,steps=101,save=25):
 overrides=dict(data__train_files=config['train_file'],data__val_files=config['val_file'],data__train_data_num='null',data__val_data_num='64',data__train_batch_size='128',data__val_batch_size='64',data__max_prompt_length='4096',data__max_response_length='500',data__max_start_length='2048',data__max_obs_length='500',data__shuffle_train_dataloader='True',algorithm__adv_estimator='grpo',actor_rollout_ref__model__path=config['base_model'],actor_rollout_ref__model__enable_gradient_checkpointing='true',actor_rollout_ref__model__use_remove_padding='False',actor_rollout_ref__actor__optim__lr='1e-6',actor_rollout_ref__actor__optim__lr_warmup_steps_ratio='0.285',actor_rollout_ref__actor__use_kl_loss='true',actor_rollout_ref__actor__kl_loss_coef='0.001',actor_rollout_ref__actor__kl_loss_type='low_var_kl',actor_rollout_ref__actor__ppo_mini_batch_size='256',actor_rollout_ref__actor__ppo_micro_batch_size='64',actor_rollout_ref__actor__fsdp_config__param_offload='true',actor_rollout_ref__actor__fsdp_config__grad_offload='true',actor_rollout_ref__actor__fsdp_config__optimizer_offload='true',actor_rollout_ref__rollout__log_prob_micro_batch_size='16',actor_rollout_ref__rollout__tensor_model_parallel_size='8',actor_rollout_ref__rollout__name='vllm',actor_rollout_ref__rollout__gpu_memory_utilization='0.60',actor_rollout_ref__rollout__n_agent='5',actor_rollout_ref__rollout__temperature='1',actor_rollout_ref__ref__log_prob_micro_batch_size='16',actor_rollout_ref__ref__fsdp_config__param_offload='true',actor_rollout_ref__actor__state_masking='true',algorithm__no_think_rl='false',trainer__logger="['console']",trainer__default_hdfs_dir='null',trainer__n_gpus_per_node='8',trainer__nnodes='1',trainer__save_freq=str(save),trainer__test_freq='25',trainer__project_name='Search-G1',trainer__experiment_name=config['experiment'],trainer__total_epochs='15',trainer__total_training_steps=str(steps),trainer__default_local_dir=config['checkpoint_root'],max_turns='4',retriever__url=config['retriever_url'],retriever__topk='3')
 return [str(ROOT/'env/bin/python'),'-m','verl.trainer.main_ppo_eco']+[k.replace('__','.')+'='+v for k,v in overrides.items()]+['+actor_rollout_ref.actor.optim.foreach=false','+trainer.val_only=false','+trainer.val_before_train=true','+trainer.g1_refresh_freq=25']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-directory',required=True);args=ap.parse_args()
 need(socket.gethostname()=='h100-11' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Native h11 CPU-only required')
 out=pathlib.Path(args.output_directory).resolve();need(ROOT/'logs' in out.parents and not out.exists(),'Fresh ownlogs directory required');out.mkdir(mode=0o700)
 need(sha(COMMON/'train.parquet')==TRAIN_SHA and sha(COMMON/'stage-validation.parquet')==HOLDOUT_SHA,'Common98/2 file identity drift')
 manifest=json.loads((COMMON/'manifest.json').read_text());need(manifest['content_identity']['sha256']==CONTENT and manifest['train_rows']==166222 and manifest['holdout_rows']==3393,'Common split mismatch')
 keyfile=ROOT/'repo-terminal-fixed413/search_r1/baselines/holdout.py';need(sha(keyfile)==QUESTION_SOURCE,'Original question identity helper drift')
 spec=importlib.util.spec_from_file_location('common_holdout',keyfile);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
 import pandas as pd
 train=pd.read_parquet(COMMON/'train.parquet');holdout=pd.read_parquet(COMMON/'stage-validation.parquet');final=pd.read_parquet(ROOT/'assets-root/data/paper-grpo/test.parquet');val=pd.read_parquet(ROOT/'assets-root/data/paper-grpo/validation-64.parquet')
 calibration,keys=select_calibration(holdout,train,final,val,h.question_key);calibration_path=out/'g1-calibration300.parquet';calibration.to_parquet(calibration_path,index=False)
 with open(calibration_path,'rb') as f:os.fsync(f.fileno())
 atomic(out/'calibration-selection.json',dict(status='ACTUAL_CPU_DATA_SELECTION_NOT_CALIBRATION',source=ref(COMMON/'stage-validation.parquet'),common_content_identity=CONTENT,train=ref(COMMON/'train.parquet'),train_rows=166222,original_holdout_rows=3393,selected_rows=300,consumed_calibration_questions=300,remaining_uncalibrated_holdout_questions=3093,rule='NQ unique question_key ascending first300; original common holdout; no train/periodic/final overlap',question_keys_sha256=hashlib.sha256(json.dumps(keys,ensure_ascii=False).encode()).hexdigest(),calibration=ref(calibration_path),question_key_source=ref(keyfile)))
 quality_path=ROOT/'logs/qwen25-32b-quality64-3077-independent-review.json';need(sha(quality_path)==QUALITY_REVIEW,'Actual independent64 review drift');quality=json.loads(quality_path.read_text());need(quality['status']=='PASS' and quality['correct']==17 and quality['denominator']==64 and quality['next_path']=='SEARCH_G1_GRPO100' and not quality['base_teacher_approved'],'Only actual G1 route required')
 qr=json.loads((ROOT/'logs/quality64-3077-source-recomputed.json').read_text());raw=qr['raw_files'];base64={'result_path':raw['result.json']['path'],'result_sha256':raw['result.json']['sha256'],'qa_path':raw['qa.jsonl']['path'],'qa_sha256':raw['qa.jsonl']['sha256']}
 source_roots={'candidate':ROOT/'tools/g1-32b-fallback-candidate-v3','producer':ROOT/'tools/g1-staged-calibration-producer-v3','pipeline':ROOT/'tools/g1-production-pipeline-v3-r2'}
 bindings=[]
 for kind,directory in source_roots.items():
  p=directory/('reproduction/conditional-teacher100/files.json' if kind=='candidate' else 'files.json');need(sha(p)==MANIFESTS[kind],'Source manifest mismatch '+kind);bindings.append({'kind':kind,'root':str(directory),'manifest':ref(p)})
 sys.path.insert(0,str(source_roots['pipeline']));from checkpoint_inventory import inventory
 model=ROOT/'assets-root/assets/qwen2.5-32b';inv=inventory(model);need(inv['tensor_count']==771 and len(inv['shards'])==17,'Actual complete official32B required');atomic(out/'base-hf-inventory.json',inv)
 binding=json.loads((ROOT/'logs/qwen25-32b-terminal-fixed413-binding.json').read_text());need(set(inv['files'])==set(binding['model_files']) and {n:v['sha256'] for n,v in inv['files'].items()}==binding['model_files'],'Official base bytes differ from actual64 source')
 overlay=ROOT/'tools/g1-head-dependencies-v2';receipt=overlay/'install-receipt.json';need(sha(receipt)=='21b345df2b5a278cfc9e982ae866d813cbe2a83bf67dfbf5d935d42cb62cba40','Native overlay receipt drift')
 exp='search_g1_32b_fallback100';work=out/'calibration-work';manifests=out/'probe-manifests';checkpoint=out/'checkpoints';rargs={'work_dir':str(work),'manifest_root':str(manifests),'data':str(calibration_path),'tag':'nq','nq':300,'holdout_offset':0,'group_size':5,'layers':'21,24,27','gpu_mem':.55,'commit_position':'Xa','commit_label':'flip'}
 config=dict(experiment=exp,base_model=str(model),candidate=str(source_roots['candidate']),producer=str(source_roots['producer']),source_bindings=bindings,train_file=str(COMMON/'train.parquet'),val_file=str(ROOT/'assets-root/data/paper-grpo/validation-64.parquet'),checkpoint_root=str(checkpoint),final_checkpoint=str(checkpoint/'actor/global_step_100'),actual_outer_receipt_path=str(out/'actual-outer100.json'),evidence_root=str(out/'initial-allocation-evidence'),refit_args=rargs,overlay=str(overlay/'overlay'),overlay_receipt=ref(receipt),base64_quality_independent_review=ref(quality_path),base_model_inventory=ref(out/'base-hf-inventory.json'),base_model_inventory_sha256=inv['sha256'],base64_result_sha256=base64['result_sha256'],input_bindings=[ref(p) for p in [COMMON/'train.parquet',ROOT/'assets-root/data/paper-grpo/validation-64.parquet',calibration_path,COMMON/'manifest.json']],preflight_source=ref(ROOT/'repo-terminal-fixed413/search_r1/baselines/preflight.py'),retriever_url='http://127.0.0.1:18085/retrieve',retriever_health_url='http://127.0.0.1:18085/health',probe_url='http://127.0.0.1:18086/judge_probe',probe_health_url='http://127.0.0.1:18086/health',owner_context={'support_sha256':'d4de70880869abbec64f81ccd86f6f8dd8755b01d9db2cb93e2ec112d3031a09','config':{'stage_timeout_seconds':86400,'base64_binding':base64,'job':None,'leases':[],'uuids':[],'cgroup':None}},independent_gate={'path':str(out/'initial-source-allocation-gate.json'),'sha256':None},telemetry_policy={'directory':str(ROOT/'tools/qwen25-32b-window-safety-candidate-v4'),'runtime_sha256':'9e41464d49054a1765fed29f5839f9fd8c568f0040aedbf6cf36b3c096adb3c6','guard_sha256':'d340c77f54c8c744816da416c91016e59bd5a21bd42262b2ae672a3432f6588b','manifest_sha256':'9835bc499bcb10da1959ae0e3bce88cf7a7966ad51ed21dc80fbe03368eaba8c','actual_independent_native_review':{'path':str(ROOT/'logs/qwen25-32b-window-v4-3092-observed-paths-independent-review.json'),'sha256':'a8e3668e142b39d773ac6ac3267e5d183f688ef10c2fa30f1f84a11d8bb1129b'}})
 config.update({k+'_manifest_sha256':v for k,v in MANIFESTS.items()});config['training_command']=command(config)
 server=ROOT/'repo/search_r1/search/retrieval_server.py';config['retriever_command']=[str(ROOT/'env/bin/python'),'-c',"import torch,runpy,sys; server=sys.argv.pop(1); sys.argv[0]=server; runpy.run_path(server,run_name='__main__')",str(server),'--index_path',str(ROOT/'assets-root/assets/wiki/e5_Flat.index'),'--corpus_path',str(ROOT/'assets-root/assets/wiki/wiki-18.jsonl'),'--retriever_model',str(ROOT/'assets-root/assets/e5-base-v2'),'--retriever_name','e5','--topk','3','--device','cuda','--faiss_gpu','--threads','8','--host','127.0.0.1','--port','18085']
 config['retrieval_asset_binding']={'path':str(ROOT/'logs/qwen25-32b-terminal-fixed413-binding.json'),'sha256':'7ded91b8c81a18bd56d6123b1fb5c1dc6426368766a34b2e29383388fa1304a2'}
 config['probe_command']=[str(ROOT/'env/bin/python'),str(source_roots['candidate']/'probe_server.py'),'--manifest',str(manifests/('live_'+exp)/'manifest.json'),'--port','18086','--outer-barrier']
 atomic(out/'initial-only-static-config.json',config)
 import torch
 need(not torch.cuda.is_initialized(),'CPU assembler CUDA context forbidden')
 atomic(out/'assembly-receipt.json',dict(status='ACTUAL_NATIVE_CPU_STATIC_RECIPE_NOT_GPU_RESOURCE_PROOF',hostname=socket.gethostname(),pid=os.getpid(),slurm_job=os.environ.get('SLURM_JOB_ID'),cuda_initialized=False,correct=17,calibration_questions=300,train_rows=166222,recipe=ref(out/'initial-only-static-config.json'),source_bindings=bindings,calibration_selection=ref(out/'calibration-selection.json'),base_hf_inventory=ref(out/'base-hf-inventory.json'),quality_independent_review=ref(quality_path),reviewed_static_recipe_sha256=hashlib.sha256(json.dumps({k:v for k,v in config.items() if k!='independent_gate'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()))
 print(json.dumps({'status':'CONCRETE_NATIVE_RECIPE_SOURCE_AUDIT_PENDING_NO_GPU','config':str(out/'initial-only-static-config.json'),'correct':17,'calibration_questions':300,'source_manifest_refs':bindings,'guardian_policy_scope':'V4 source/native3091/3092 observed paths only; NOTFOUND unobserved and3092overallFAILED preserved; newcalibration resource unknown'}))
if __name__=='__main__':main()
