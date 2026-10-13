"""Separate CPU final acceptance wrapper; never imports or resumes training.

Run only after authorized original cancellation and complete replacement cleanup.
Never synthesizes an original normal terminal or resumes optimizer/training state.
Calls unchanged frozen full-HF verifier; all weights remain on retained18.
"""
import argparse,hashlib,json,math,os,re,socket,stat,subprocess,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
SOURCE='8b6f99d1b3353145200eb3fa7c16a579a79a987c'
CONFIG='c9415c0096853f78d870e205fc956f20045cde985836de5225f609eaf091d4b3'
PACKAGE_SHA='e2503977c8746edbc83b722d8224548f060909df9fc25d91a9cb88a947a79a21'
VERIFIER_REL='source/repo/reproduction/baselines/tree-validation-20261002/evidence/runtime/check_real_checkpoint.py'
VERIFIER_SHA='b3075d7b9346bb7682e0ea05962de0624c117b354590b728ed8a18b7e8ba3c97'
RUN=ROOT/'runs/tree-branpo-fresh200-3360'
PACKAGE=ROOT/('instruct300/candidates/branpo-fresh200-'+SOURCE)
BASE=ROOT/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
CHECKPOINT=RUN/'checkpoints/actor/global_step_200'
NATIVE=RUN/'checkpoints/training_state/global_step_200/manifest.json'

def require(condition,message):
 if not condition:raise ValueError(message)
def file_sha(path):
 require(stat.S_ISREG(path.lstat().st_mode),'Evidence member must be ordinary file: '+str(path))
 digest=hashlib.sha256()
 with path.open('rb') as stream:
  before=os.fstat(stream.fileno())
  for data in iter(lambda:stream.read(8*1024*1024),b''):digest.update(data)
  after=os.fstat(stream.fileno())
 require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Evidence changed while hashing')
 return digest.hexdigest()
def read_json(path,expected=None):
 digest=file_sha(path);require(expected is None or expected==digest,'JSON SHA changed: '+str(path))
 value=json.loads(path.read_text());require(file_sha(path)==digest,'JSON changed while reading');return value,digest
def identity(path):
 files={str(p.relative_to(path)):file_sha(p) for p in sorted(path.rglob('*')) if p.is_file()}
 return dict(files=files,sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())
def write_once(path,value):
 with path.open('x') as out:json.dump(value,out,indent=2,allow_nan=False);out.flush();os.fsync(out.fileno())
def normalize_tokenizer_value(value):
 # AddedToken behavior must be retained, not reduced to its string content.
 fields=('content','single_word','lstrip','rstrip','normalized','special')
 if all(hasattr(value,key) for key in fields):return {key:getattr(value,key) for key in fields}
 if isinstance(value,dict):return {str(key):normalize_tokenizer_value(item) for key,item in value.items()}
 if isinstance(value,(list,tuple)):return [normalize_tokenizer_value(item) for item in value]
 require(value is None or isinstance(value,(str,int,float,bool)),'Unexpected tokenizer property type')
 return value

def normalized_tokenizer_config(path):
 value,_=read_json(path/'tokenizer_config.json')
 # These fields locate serialized files/revisions; they do not alter tokenization.
 ignored=('name_or_path','_name_or_path','tokenizer_file','_commit_hash')
 return normalize_tokenizer_value({key:item for key,item in value.items() if key not in ignored})

def tokenizer_semantics(tokenizer):
 require(hasattr(tokenizer,'backend_tokenizer'),'Fast tokenizer backend unavailable')
 options=('padding_side','truncation_side','model_max_length','clean_up_tokenization_spaces',
          'split_special_tokens','add_bos_token','add_prefix_space','errors')
 return dict(backend=json.loads(tokenizer.backend_tokenizer.to_str()),
     full_special_roles=normalize_tokenizer_value(tokenizer.special_tokens_map_extended),
     added_tokens=normalize_tokenizer_value(tokenizer.added_tokens_decoder),
     runtime_options={key:normalize_tokenizer_value(getattr(tokenizer,key,None)) for key in options})

EVAL_SOURCE='0e00998fdc645281986250cd8632bb70c6e2fa9b'
UNIT='sst-tree-eval-shared-0e00998.service'
RELEASE_SHA='fb7c69b764de008b2d597fc90840c3f6527ef431827b1be7fd32d6125a93b070'
MANIFEST_SHA='6e1b088c100f36fb168e4a5bb33660f3af0143bb5ef7504d607df3dd34432dc1'
PREFIX_SHA='cbc70dc4138eae5c79c0cf61e6b63793936802383fba42aae543e1ff54882435'
EXPECTED_COUNTS={'nq':3610,'triviaqa':11313,'popqa':14267,'hotpotqa':7405,'2wikimultihopqa':12576,'musique':2417,'bamboogle':125}
SCORES=RUN/'output/eval-resume/scores'
PREFIX=RUN/'output/eval-resume/bound-prefix-30880.json'
EVAL_RUN=ROOT/'runs'/UNIT.removesuffix('.service')


def canonical(value):
 return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()

def plain(value):
 if hasattr(value,'tolist'):return plain(value.tolist())
 if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
 if isinstance(value,(tuple,list)):return [plain(v) for v in value]
 return value

def question_sha(row,prompt_key):
 return hashlib.sha256(canonical(plain({k:row[k] for k in (prompt_key,'data_source','reward_model')}))).hexdigest()

def check_rank_numerics(record,outer,rank):
 """Consume actual published norms and fixed-coordinate samples, never full-shard claims."""
 require(record['outer_step']==outer and record['rank']==rank and record['successful_optimizer_calls']==20,'Numerical audit actual rank/outer/writes differ')
 norms=record['existing_grad_norms']
 require(type(norms) is list and len(norms)==20 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in norms),'Successful writes require observed finite positive gradient norms')
 require(type(record['finite_grad_norms']) is int and record['finite_grad_norms']==len(norms) and type(record['nonzero_grad_norms']) is int and record['nonzero_grad_norms']==sum(v!=0 for v in norms),'Published gradient finite/nonzero counters differ from actual norms')
 require(record['local_shard_sample_layout_unchanged'] is True,'Fixed sample layout changed')
 sample=record['local_shard_sample_delta'];count=sample['count'];nonzero=sample['nonzero']
 require(type(count) is int and 0<count<=8192 and type(sample['finite']) is int and sample['finite']==count and type(nonzero) is int and 0<=nonzero<=count,'Actual sample finite/cardinality/nonzero proof invalid')
 low,high=sample['minimum'],sample['maximum']
 require(all(type(v) in (int,float) and math.isfinite(v) for v in (low,high)) and low<=high,'Sample extrema are nonfinite or reversed')
 require((nonzero==0 and low==high==0) or (nonzero>0 and (low!=0 or high!=0)),'Sample extrema and observed nonzero count disagree')
 if nonzero<count:require(low<=0<=high,'Partial zero sample lacks zero-bounding extrema')
 if nonzero==count:require(low!=0 and high!=0,'All-nonzero sample has a zero extreme')
 if count==1:require(low==high,'Single-coordinate sample has unequal extrema')
 layout=record['local_shard_sample_layout'];require(type(layout) is list and layout,'Actual fixed sample layout absent')
 names=set();coordinates=0;selected_parameter_numel=0
 for entry in layout:
  name,shape,indices=entry['name'],entry['local_shape'],entry['indices']
  require(type(name) is str and name and name not in names,'Duplicate/invalid sampled parameter name');names.add(name)
  require(type(shape) is list and all(type(v) is int and v>0 for v in shape),'Invalid sampled local shape')
  size=math.prod(shape)
  selected_parameter_numel+=size
  require(type(indices) is list and indices and all(type(i) is int and 0<=i<size for i in indices) and indices==sorted(set(indices)),'Sample coordinate layout invalid')
  coordinates+=len(indices)
 require(coordinates==count,'Published sample cardinality differs from actual coordinate layout')
 if selected_parameter_numel>=8192:require(count==8192,'Large actual local parameter layout lacks the full8192-coordinate sample')
 before,after=record['local_shard_sample_before_sha256'],record['local_shard_sample_after_sha256']
 require(all(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) for value in (before,after)),'Sample before/after SHA format invalid')
 if outer==1:
  require(nonzero==0 and before==after,'Outer1 zero-LR sample must remain unchanged')
 else:
  require(nonzero>0 and before!=after,'Nonzero-LR outer must have actual observed sampled parameter movement')
 return dict(outer=outer,rank=rank,observed_finite_positive_norms=len(norms),sampled_coordinates=count,sampled_nonzero_delta_coordinates=nonzero,sampled_delta_minimum=low,sampled_delta_maximum=high,learning_rate_before=1e-6*min((outer-1)/57,1),zero_learning_rate_outer=outer==1,sample_before_sha256=before,sample_after_sha256=after)

def exact_attempt_inventory(run,accounting):
 """The frozen consumer stops at200; reject any unconsumed later physical record."""
 directory=run/'checkpoints/training_state/attempt-results'
 actual=sorted(directory.glob('attempt-*.json'))
 consumed={Path(path).resolve() for path in accounting['record_sha256'] if Path(path).parent.resolve()==directory.resolve() and Path(path).name.startswith('attempt-')}
 require({path.resolve() for path in actual}==consumed and len(actual)==accounting['training_attempts'],'Unaccounted attempt record before/after completed200 boundary')
 require([path.name for path in actual]==['attempt-%06d.json'%i for i in range(1,accounting['training_attempts']+1)],'Actual attempt inventory is not exact consumed sequence')
 events=accounting['credited']+accounting['all_skipped']
 require(events and max(x['attempt'] for x in events)==accounting['training_attempts'] and accounting['credited'][-1]['outer']==200 and accounting['credited'][-1]['attempt']==accounting['training_attempts'],'Outer200 must be final actual consumed attempt')
 inventory={str(path.relative_to(directory)):file_sha(path) for path in sorted(directory.rglob('*')) if path.is_file()}
 require(set(inventory)=={path.name for path in actual},'Unconsumed pending/fault/nested attempt artifact exists at completed200 boundary')
 return inventory


def complete_ledger(manifest,bindings,checked):
 """Read and recompute actual immutable scores; no worker/actor imports or writes."""
 identities={key:manifest[key] for key in ('config_sha256','dataset_sha256','model_sha256')}
 def bound(name,path,expected=None):
  value,digest=read_json(path,expected);bindings[name]=Path(path);checked[name]=digest;return value
 prefix=bound('preserved_prefix',PREFIX,PREFIX_SHA)
 done=bound('preserved_prefix_done',Path(str(PREFIX)+'.done.json'))
 require(done['sha256']==PREFIX_SHA and done['count']==30880 and done['actual_trained_outer']==200 and done['original_job']==3360 and done['identity']==identities,'Prefix durability/training identity differs')
 require(all(prefix[k]==v for k,v in identities.items()) and prefix['original_job']==3360 and prefix['actual_trained_outer']==200 and prefix['actual_adam_per_rank']=={str(i):4000 for i in range(8)},'Bound original prefix/Adam counters differ')
 rows=prefix['rows'];require(len(rows)==30880,'Exact preserved30880 required')
 rest={};execution=None
 provenance=sorted(SCORES.glob('rank-*-provenance.json'))
 require({p.name for p in provenance}=={'rank-%d-provenance.json'%i for i in range(8)},'Eight rank provenance records required')
 expected_execution=dict(local_batch_size=64,world_size=8,tensor_parallel_size=1,resident_weights=True,free_cache_engine=False,rollout_padding_divisor=1,backend='XFORMERS',dtype='bfloat16',native_emission=True,resource_protocol='memory_headroom_shared_v1',allocation_kind='shared_systemd',physical_gpus=8,slurm_reserved_gpus=0,gpu_memory_utilization=.20)
 for path in provenance:
  value=bound('provenance/'+path.name,path)
  require(value['identity']==identities and value['execution']==expected_execution,'Actual reviewed shared execution differs')
  execution=value['execution']
 batch_files=sorted((SCORES/'batches').glob('*.json'))
 done_paths=sorted((SCORES/'batches').glob('*.done.json'))
 require({p.name for p in batch_files}=={p.name for p in done_paths}|{p.name.replace('.done.json','.json') for p in done_paths},'Orphan/uncommitted score file at terminal')
 for path in done_paths:
  receipt=bound('batch_done/'+path.name,path);batchpath=path.with_name(path.name.replace('.done.json','.json'))
  value=bound('batch/'+batchpath.name,batchpath,receipt['sha256']);part=value['rows']
  require(value['identity']==identities and value['prefix_count']==30880 and value['execution']==execution and value['allocation_job']=='not-slurm;systemd:'+UNIT and type(value['rank']) is int and value['rank'] in range(8),'Committed batch provenance differs')
  require(part and len(part)==receipt['count'] and len(part)<=64 and [r['index'] for r in part]==list(range(part[0]['index'],part[0]['index']+len(part))),'Batch receipt/cardinality/contiguity differs')
  for row in part:
   index=row['index'];require(type(index) is int and 30880<=index<51713 and index not in rest,'Duplicate/outside replacement question index');rest[index]=row
 require(set(rest)==set(range(30880,51713)),'Replacement ledger has missing questions')
 rows=rows+[rest[i] for i in range(30880,51713)]
 fields=('f1','em','answer_format','format_rate','generated_tokens','search_calls')
 import pandas as pd
 frame=pd.concat([pd.read_parquet(x['path']) for x in manifest['dataset_files']],ignore_index=True)
 require(len(frame)==51713,'Actual dataset count differs')
 for index,row in enumerate(rows):
  require(row['index']==index and row['source']==frame.iloc[index]['data_source'] and row['question_sha256']==question_sha(frame.iloc[index],manifest['prompt_key']),'Saved question/source/order identity differs')
  for key in fields:
   value=row[key];require(type(value) in (int,float) and math.isfinite(value) and value>=0 and (key in ('generated_tokens','search_calls') or value<=1),'Invalid score '+key)
   if key in ('generated_tokens','search_calls'):require(int(value)==value,'Non-integral score count')
 require(dict(Counter(r['source'] for r in rows))==EXPECTED_COUNTS,'Exact seven dataset coverage differs')
 metrics={}
 for source in list(EXPECTED_COUNTS)+['all']:
  group=rows if source=='all' else [r for r in rows if r['source']==source]
  metrics['val/qa_count/'+source]=len(group)
  for key in fields:metrics['val/'+key+'/'+source]=sum(r[key] for r in group)/len(group)
  if source!='all':metrics['val/test_score/'+source]=metrics['val/em/'+source];metrics['val/evaluated_count/'+source]=len(group)
 final_done=bound('final_metrics_done',SCORES/'final-metrics.done.json')
 final=bound('final_metrics',SCORES/'final-metrics.json',final_done['sha256'])
 require(final['identity']==identities and final['evaluated_count']==51713 and final['metrics']==metrics and final['execution_records']==[execution],'Published final metrics do not reproduce immutable score ledger')
 require(final['numerical_caveat']=='HF weights unchanged; altered greedy batching/kernel shapes are not guaranteed bit-identical.','Numerical execution caveat missing')
 return final


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('original-job-terminal','handoff-journal','manifest','release','review'):p.add_argument('--'+key,required=True,type=Path)
 a=p.parse_args()
 require(socket.gethostname().split('.')[0]=='h100-10','Only asset-local retained18 CPU verification')
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU acceptance must explicitly hide all GPUs')
 require(sys.flags.optimize==0,'Frozen verifier assertions require unoptimized Python')
 available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
 require(available>=64*1024**3,'At least64GiB actual free CPU memory required; no GPU fallback')
 package,package_sha=read_json(PACKAGE/'manifest.json',PACKAGE_SHA)
 require(package['code_commit']==SOURCE and package['files'][VERIFIER_REL]==VERIFIER_SHA,'Frozen original package/verifier differs')
 bindings=dict(wrapper=Path(__file__).resolve(),resolved_config=RUN/'resolved.yaml',package_manifest=PACKAGE/'manifest.json',frozen_verifier=PACKAGE/VERIFIER_REL,native_manifest=NATIVE,original_job_terminal=a.original_job_terminal,handoff_journal=a.handoff_journal,eval_asset_manifest=a.manifest,eval_release=a.release,eval_review=a.review,replacement_terminal=EVAL_RUN/'run-result.json',replacement_allocation=EVAL_RUN/'allocation-identity.json')
 for name,path in bindings.items():require(path.resolve().is_relative_to(ROOT.resolve()),'Acceptance input outside retained task filesystem: '+name)
 checked={name:file_sha(path) for name,path in bindings.items()}
 for name,digest in package['files'].items():
  path=PACKAGE/name;require(file_sha(path)==digest,'Frozen original source changed: '+name);bindings['frozen_package/'+name]=path;checked['frozen_package/'+name]=digest
 release,_=read_json(a.release,RELEASE_SHA);review,_=read_json(a.review)
 require(release['source_commit']==EVAL_SOURCE and release['review_status']=='ACCEPTED' and review['review_status']=='ACCEPTED' and release['resource_protocol']=='memory_headroom_shared_v1' and release['training_updates']==200 and release['original_job']==3360,'Reviewed actual evaluation source/authority differs')
 require(Path(release['score_output']).resolve()==SCORES.resolve() and release['prefix_sha256']==PREFIX_SHA and release['manifest_sha256']==MANIFEST_SHA,'Shared release score/prefix/assets differ')
 tools=ROOT/('instruct300/eval-shared-'+EVAL_SOURCE)
 for name,digest in release['files'].items():
  path=tools/name;require(file_sha(path)==digest and review['files'][name]==digest,'Reviewed evaluator code changed: '+name);bindings['eval_source/'+name]=path;checked['eval_source/'+name]=digest
 for name,digest in release['external_files'].items():
  path=Path(name);require(file_sha(path)==digest,'Reviewed evaluator external source changed');bindings['eval_external/'+name]=path;checked['eval_external/'+name]=digest
 original,_=read_json(a.original_job_terminal)
 require(original['job_id']=='3360' and original['state']=='CANCELLED' and original['exit_code']=='0:15' and original['source']=='scontrol-preserved' and type(original['observed_epoch']) in (int,float) and math.isfinite(original['observed_epoch']) and original['observed_epoch']>0,'Truthful original cancellation observation required')
 require(file_sha(a.original_job_terminal)=='732e971af9260bff7afc3fabc18de6a7736a8d8e49f9fa3ba5892640cdf12221','Exact honest preserved cancellation snapshot required')
 raw_fields=dict(item.split('=',1) for item in original['raw'].split() if '=' in item)
 require(all(raw_fields.get(k)==v for k,v in dict(JobId='3360',JobState='CANCELLED',ExitCode='0:15',JobName='tree-branpo-fresh200',NodeList='h100-10').items()),'Actual original scheduler cancellation identity differs')
 original_source=Path(original['preserved_source_path']);require(original_source.resolve().is_relative_to((RUN/'output/eval-handoff-acceptance-inputs').resolve()),'Preserved scheduler snapshot outside inputs')
 require(original['preserved_source_sha256']=='af0762ee0af4a94a03138480d03d9e7ec5c2eb8fe7a795a537e72cf8b7cdcef0','Historical source snapshot pin differs')
 observation,observation_sha=read_json(original_source,original['preserved_source_sha256'])
 require(observation['old_slurm']==original['raw'] and observation['observed_epoch']==original['observed_epoch'],'Original historical scheduler evidence was rewritten')
 bindings['original_scontrol_observation']=original_source;checked['original_scontrol_observation']=observation_sha
 require(original['original_run_result_absent'] is True and original['original_train_terminal_absent'] is True and original['current_queue_absent'] is True,'Original absence facts missing')
 require(not (RUN/'run-result.json').exists() and not (RUN/'output/train200-terminal.json').exists(),'Original absent terminal status changed; never normalize exit0')
 queue=subprocess.check_output(['squeue','--local','-h','-o','%i'],text=True)
 require(not set(queue.split()).intersection({'3360','3392'}),'Original/obsolete own evaluator still queued or active')
 state_raw=subprocess.check_output(['systemctl','show',UNIT,'--property=ActiveState,SubState,MainPID,ExecMainStatus'],text=True)
 state=dict(line.split('=',1) for line in state_raw.splitlines() if '=' in line)
 require(state.get('MainPID')=='0' and state.get('ExecMainStatus')=='0' and state.get('ActiveState')=='inactive','Replacement owned unit must actually be terminal after cleanup')
 live_terminal_facts=dict(observed_epoch=time.time(),original_run_result_absent=True,original_train_terminal_absent=True,current_old_queue_absent=True,current_queue_raw=queue,replacement_systemd_raw=state_raw)
 handoff,_=read_json(a.handoff_journal)
 require(handoff['status']=='HANDOFF_REQUESTED' and handoff['old_job']==3360 and handoff['training_updates']==200 and handoff['cancel_returncode']==0 and handoff['prefix_sha256']==PREFIX_SHA and handoff['prefix_count']==30880 and handoff['scope']=='User-authorized evaluation-only handoff; no training restart','Actual authorized old-evaluator handoff receipt required')
 require(handoff==observation['journal'],'Supplied handoff journal is not the authenticated historical cancellation journal')
 old_before=dict(item.split('=',1) for item in handoff['old_slurm_before'].split() if '=' in item)
 require(all(old_before.get(k)==v for k,v in dict(JobId='3360',JobState='RUNNING',JobName='tree-branpo-fresh200',NodeList='h100-10').items()),'Historical pre-cancel observation does not identify own running3360')
 epochs=[handoff['epoch'],handoff['stop_requested_epoch'],original['observed_epoch']]
 require(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in epochs) and epochs==sorted(epochs),'Historical cancellation/request/observation epoch order differs')
 terminal,terminal_sha=read_json(EVAL_RUN/'run-result.json')
 require(terminal['status']=='COMPLETE' and terminal['unit']==UNIT and terminal['predecessor_job']==3360 and terminal['evaluation_only'] is True and terminal['training_updates']==200 and terminal['worker_exit_code']==0 and terminal['own_cleanup_residual']=={} and terminal['prefix_sha256']==PREFIX_SHA and terminal['allocation_kind']=='user_authorized_shared_systemd' and terminal['physical_gpus']==8 and terminal['slurm_reserved_gpus']==0 and terminal['resource_protocol']=='memory_headroom_shared_v1' and Path(terminal['score_output']).resolve()==SCORES.resolve(),'Replacement evaluation must complete and clean only own processes')
 require(terminal['end_epoch']>=terminal['start_epoch'] and terminal['wall_seconds']>0,'Replacement actual terminal time absent')
 allocation,_=read_json(EVAL_RUN/'allocation-identity.json')
 require(all(allocation[key]==terminal[key] for key in ('unit','allocation_kind','physical_gpus','slurm_reserved_gpus','score_output')) and allocation['release_sha256']==RELEASE_SHA and allocation['training_updates']==200,'Replacement allocation identity differs')
 manifest,_=read_json(a.manifest,MANIFEST_SHA)
 require(manifest['dataset_count']==51713 and manifest['expected_source_counts']==EXPECTED_COUNTS and manifest['config_sha256']==CONFIG,'Evaluation split/config differs')
 for entry in manifest['dataset_files']:
  path=Path(entry['path']);require(file_sha(path)==entry['sha256'],'Actual final split bytes changed');bindings['dataset/'+str(path)]=path;checked['dataset/'+str(path)]=entry['sha256']
 require(hashlib.sha256(canonical(manifest['dataset_files'])).hexdigest()==manifest['dataset_sha256'],'Final dataset inventory identity differs')
 sys.path.insert(0,str(PACKAGE))
 from production_contract import direct_rank_records,tree_and_safety_records,checkpoint_manifest
 rank=direct_rank_records(RUN,package,CONFIG,200)
 require(rank['completed_outer']==200 and rank['actual_writes_per_rank']=={str(i):4000 for i in range(8)} and rank['skipped_opportunities_per_rank']=={str(i):0 for i in range(8)},'Actual full200/eight-rank4000 Adam write accounting required')
 attempt_inventory_before=exact_attempt_inventory(RUN,rank['attempt_accounting'])
 numerical=[]
 for event in rank['attempt_accounting']['credited']:
  outer,attempt=event['outer'],event['attempt']
  for actual_rank in range(8):
   path=RUN/'output/tree-rank-audit'/('step-%06d-attempt-%06d-rank-%02d.json'%(outer,attempt,actual_rank))
   audit,audit_sha=read_json(path,rank['record_sha256'][str(path)])
   numerical.append(check_rank_numerics(audit,outer,actual_rank))
 require(len(numerical)==1600,'All200x8 actual numerical audit records required')
 numerical_report=dict(status='ACTUAL_TRAIN200_RANK_NUMERICAL_JSON_PROOFS_VERIFIED',records=numerical,rank_records=1600,finite_positive_gradient_norms=sum(x['observed_finite_positive_norms'] for x in numerical),zero_lr_outer1_rank_records=8,nonzero_lr_sample_movement_rank_records=1592,scope='Actual published finite successful-return norms and fixed up-to8192 local coordinates per rank/outer; samples do not prove full-shard parameter or gradient finiteness')
 tree=tree_and_safety_records(RUN,200,package,CONFIG)
 for path,digest in dict(rank['record_sha256'],**tree['evidence_sha256']).items():
  require(file_sha(Path(path))==digest,'Direct training/Tree evidence changed');bindings['training/'+path]=Path(path);checked['training/'+path]=digest
 native,native_sha=checkpoint_manifest(RUN,package,CONFIG,200)
 # JSON manifest verification and hash preservation, never PT deserialization.
 for record in native['worker_states']+[native['driver']]:
  path=Path(record['path']);require(file_sha(path)==record['sha256'],'Saved native200 shard changed');bindings['native_file/'+str(path)]=path;checked['native_file/'+str(path)]=record['sha256']
 base_before=identity(BASE);require(base_before==package['model_identity'],'Official immutable base files differ')
 before=identity(CHECKPOINT);require(before==native['hf_identity'],'HF200 file inventory differs from native checkpoint')
 assets=[dict(name=name,sha256=digest) for name,digest in before['files'].items()]
 require(manifest['model_files']==assets and manifest['model_sha256']==hashlib.sha256(canonical(assets)).hexdigest()==release['model_sha256'],'Replacement evaluation used different HF200 inventory')
 final=complete_ledger(manifest,bindings,checked)
 score_inventory={str(path.relative_to(SCORES)) for path in SCORES.rglob('*') if path.is_file()}
 # Bind every preexisting slow-evaluator cost file without equating wasted work to preserved scores.
 partial_cost_paths=sorted((RUN/'output/validation-cost').glob('validation-*'))
 partial_cost=[]
 for path in partial_cost_paths:
  if path.is_file():bindings['original_partial_cost/'+path.name]=path;checked['original_partial_cost/'+path.name]=file_sha(path);partial_cost.append(str(path))
 bindings_before={name:file_sha(path) for name,path in bindings.items()}
 require(bindings_before==checked,'Checked immutable inputs changed before CPU verification')
 require(bindings_before['resolved_config']==CONFIG and bindings_before['package_manifest']==PACKAGE_SHA and bindings_before['frozen_verifier']==VERIFIER_SHA,'Actual source/config/package binding differs')
 binding_paths=bindings
 out=RUN/'output/eval-handoff-hf-acceptance';require(not out.exists(),'Existing acceptance namespace: refuse overwrite or automatic reload retry');out.mkdir()
 write_once(out/'actual-terminal-observation.json',live_terminal_facts)
 write_once(out/'training-numerical-json-proof.json',numerical_report)
 write_once(out/'exact-attempt-inventory.json',attempt_inventory_before)
 numerical_report_sha=file_sha(out/'training-numerical-json-proof.json')
 attempt_inventory_sha=file_sha(out/'exact-attempt-inventory.json')
 env={k:v for k,v in os.environ.items() if not k.startswith('TREE_')}
 env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONOPTIMIZE='0',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(out/'hf-cache'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
 started=time.time()
 with (out/'verifier.log').open('x') as log:
  result=subprocess.run([sys.executable,str(PACKAGE/VERIFIER_REL),'--base',str(BASE),'--checkpoint',str(CHECKPOINT),'--output',str(out/'full-HF-cold-reload.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
 require(result.returncode==0,'Actual full CPU cold reload failed; preserve log/evidence, no automatic retry')
 full,full_sha=read_json(out/'full-HF-cold-reload.json')
 require(full['status']=='HF_RELOAD_PARAMETER_DELTA_ACCEPTED' and full['all_parameters_finite'] is True and full['cpu_forward_logits_finite'] is True and full['changed_parameters']>0 and full['total_parameters']>1_000_000_000,'Complete finite/nonzero cold model proof missing')
 require(full['max_abs_parameter_delta']>0 and not any(full['hf_loading_info'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),'Full loading keys/shapes/errors or parameter delta invalid')
 require(Path(full['checkpoint']).resolve()==CHECKPOINT.resolve() and Path(full['base']).resolve()==BASE.resolve(),'Cold loader used different weights')
 require(full['checkpoint_file_sha256']==before['files'],'Cold reload observed a different HF inventory')
 # Tokenizer-only extra checks: no second model load or generation batch.
 os.environ.update({k:v for k,v in env.items() if k in ('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HOME')})
 from transformers import AutoTokenizer
 base_tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
 tok=AutoTokenizer.from_pretrained(CHECKPOINT,local_files_only=True)
 require(normalized_tokenizer_config(BASE)==normalized_tokenizer_config(CHECKPOINT),'Tokenizer semantic config differs (only serialization paths ignored)')
 require(tokenizer_semantics(base_tok)==tokenizer_semantics(tok),'Full tokenizer backend/AddedToken/special-role/runtime behavior differs')
 special=lambda t:dict(bos=t.bos_token_id,eos=t.eos_token_id,pad=t.pad_token_id,unk=t.unk_token_id,all_special_tokens=t.all_special_tokens,all_special_ids=t.all_special_ids,chat_template=t.chat_template)
 require(base_tok.get_vocab()==tok.get_vocab() and special(base_tok)==special(tok),'Full tokenizer/special/PAD/chat identity differs')
 probes=['The capital of France is','你好，世界。','<|im_start|>user\nFind an answer.<|im_end|>']
 for text in probes:
  b=base_tok.encode(text,add_special_tokens=True);c=tok.encode(text,add_special_tokens=True)
  require(b==c and base_tok.decode(b)==tok.decode(c),'Actual cold tokenizer encode/decode differs')
 messages=[dict(role='system',content='Answer clearly.'),dict(role='user',content='你好，what is 2+2?')]
 for add_generation_prompt in (False,True):
  for tokenize in (False,True):
   require(base_tok.apply_chat_template(messages,tokenize=tokenize,add_generation_prompt=add_generation_prompt)==tok.apply_chat_template(messages,tokenize=tokenize,add_generation_prompt=add_generation_prompt),'Actual chat rendering/token IDs differ')
 tokenizer=dict(status='ACTUAL_COLD_TOKENIZER_IDENTITY_VERIFIED',vocab_equal=True,special_pad_bos_eos_unk_equal=True,all_special_tokens_ids_equal=True,chat_template_equal=True,actual_encode_decode_probes=len(probes),canonical_backend_equal=True,full_added_token_attributes_equal=True,full_special_roles_equal=True,normalized_semantic_config_equal=True,actual_chat_template_comparisons=4)
 write_once(out/'tokenizer-cold-reload.json',tokenizer)
 require(identity(BASE)==base_before and identity(CHECKPOINT)==before,'Base/checkpoint changed during cold verification')
 require(exact_attempt_inventory(RUN,rank['attempt_accounting'])==attempt_inventory_before,'Complete attempt inventory changed during CPU verification')
 require({str(path.relative_to(SCORES)) for path in SCORES.rglob('*') if path.is_file()}==score_inventory,'Terminal score inventory changed during CPU verification')
 require(not (RUN/'run-result.json').exists() and not (RUN/'output/train200-terminal.json').exists(),'Original absent receipt changed during CPU verification')
 bindings_after={name:file_sha(path) for name,path in binding_paths.items()}
 require(bindings_after==bindings_before,'Wrapper/config/package/verifier/native/terminal evidence changed during cold verification')
 require(bindings_after['resolved_config']==CONFIG and bindings_after['package_manifest']==PACKAGE_SHA and bindings_after['frozen_verifier']==VERIFIER_SHA,'Pinned config/package/verifier changed')
 require(read_json(out/'full-HF-cold-reload.json')[1]==full_sha,'Full cold report changed during final input checks')
 require(read_json(out/'training-numerical-json-proof.json',numerical_report_sha)[0]==numerical_report,'Actual numerical report changed during CPU verification')
 require(read_json(out/'exact-attempt-inventory.json',attempt_inventory_sha)[0]==attempt_inventory_before,'Actual attempt inventory report changed during CPU verification')
 receipt=dict(schema_version='tree_train200_eval_handoff_composite_v1',status='JOB3360_TRAIN200_AUTHORIZED_EVAL_HANDOFF_FULL_HF_ACCEPTED',training_job_id='3360',training_run_id=RUN.name,training_source_commit=SOURCE,evaluation_source_commit=EVAL_SOURCE,source_commit=SOURCE,config_sha256=CONFIG,predecessor_package_manifest_sha256=package_sha,native_manifest_sha256=native_sha,hf_identity_sha256=before['sha256'],split_identity_sha256=package['common_data_identity'],eval_dataset_identity_sha256=manifest['dataset_sha256'],actual_completed_outer=200,actual_writes_per_rank=rank['actual_writes_per_rank'],original_cancelled_state='CANCELLED',original_slurm_exit_code=original['exit_code'],original_training_body_exit_code=None,original_training_normal_exit_claimed=False,original_final_evaluation_completed=False,original_evaluation_stop_authorized=True,replacement_evaluation_complete=True,evaluated_count=51713,preserved_count=30880,replacement_count=20833,final_per_dataset=EXPECTED_COUNTS,final_metrics=final['metrics'],replacement_unit=UNIT,replacement_worker_exit_code=0,replacement_own_cleanup_residual={},replacement_terminal_sha256=terminal_sha,input_hashes_before={name:dict(path=str(binding_paths[name]),sha256=digest) for name,digest in bindings_before.items()},input_hashes_after={name:dict(path=str(binding_paths[name]),sha256=digest) for name,digest in bindings_after.items()},full_hf_report_sha256=full_sha,tokenizer_report_sha256=file_sha(out/'tokenizer-cold-reload.json'),wrapper_sha256=bindings_before['wrapper'],frozen_full_verifier_sha256=VERIFIER_SHA,all_parameters_finite=True,shape_keys_full_checked=True,actual_cpu_model_cold_reload=True,actual_tokenizer_cold_reload=True,total_parameters=full['total_parameters'],changed_parameters=full['changed_parameters'],max_abs_parameter_delta=full['max_abs_parameter_delta'],cpu_forward_logits_finite=True,wall_seconds=time.time()-started,weights_left_on_h100_10=True,training_or_optimizer_restored=False,independent_final_acceptance=False,original_sampler_actor_probability_gate_accepted=False,production_thresholds_accepted=False,acceptance_wrapper_files={'verify_handoff_hf.py':dict(path=str(binding_paths['wrapper']),sha256=bindings_before['wrapper'])},actual_terminal_observation_sha256=file_sha(out/'actual-terminal-observation.json'),training_numerical_report_sha256=numerical_report_sha,exact_attempt_inventory_sha256=attempt_inventory_sha,original_partial_cost_files=partial_cost,original_partial_cost_scope='Actual original slow-evaluator cost files retained separately; only30880 salvaged scores count in final metrics; further compute is not credited as preserved evaluation',exact_executor_resume=False,numerical_caveat=final['numerical_caveat'])
 write_once(out/'identity-bound-handoff-hf.json',receipt);print(json.dumps(receipt,allow_nan=False))
if __name__=='__main__':main()
