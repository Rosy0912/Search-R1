"""Opt-in corrected final evaluation. Lossless scoring view only, not a full turn trace."""
import hashlib,json,math,os,subprocess
from pathlib import Path
import numpy as np
SOURCES={'nq','triviaqa','popqa','hotpotqa','2wikimultihopqa','musique','bamboogle'}
TOTAL=51713

def canonical(value):
 if isinstance(value,np.ndarray):return canonical(value.tolist())
 if isinstance(value,np.generic):return canonical(value.item())
 if isinstance(value,dict):return {str(k):canonical(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [canonical(v) for v in value]
 if value is None or isinstance(value,(str,bool,int)):return value
 if isinstance(value,float) and math.isfinite(value):return value
 raise ValueError('Unsupported/nonfinite evidence value')
def encoded(value):return json.dumps(canonical(value),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def loads(raw):
 def unique(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError('Duplicate evidence JSON key')
   d[k]=v
  return d
 return canonical(json.loads(raw,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite evidence JSON'))))
def targets(reward):
 value=canonical(reward['ground_truth']['target']);value=[value] if isinstance(value,str) else value
 if not isinstance(value,list) or not all(isinstance(v,str) for v in value):raise ValueError('Invalid target aliases')
 return value
def row_identity(prompt,source,reward):
 value=canonical(prompt)
 if not isinstance(value,list) or not all(isinstance(turn,dict) and isinstance(turn.get('content'),str) for turn in value):raise ValueError('Missing raw question prompt')
 return dict(prompt=value,question_prompt_text='\n'.join(turn['content'] for turn in value),source=str(source),targets=targets(reward))
def prepare_inputs(files,output,prompt_key='prompt'):
 """CPU preparation before generation; ordered parquet rows, never output-derived."""
 import pandas as pd
 p=Path(output)
 if p.exists():raise ValueError('Input manifest is immutable')
 records=[];inputs=[]
 for filename in files:
  path=Path(filename).resolve();file_sha=digest(path);inputs.append(dict(path=str(path),sha256=file_sha))
  frame=pd.read_parquet(path)
  for index,(_,row) in enumerate(frame.iterrows()):
   identity=row_identity(row[prompt_key],row['data_source'],row['reward_model'])
   records.append(dict(ordinal=len(records),file_sha256=file_sha,file_row=index,identity=identity,identity_sha256=hashlib.sha256(encoded(identity)).hexdigest()))
 value=dict(version=1,scope='native-scoring-view',prompt_key=prompt_key,inputs=inputs,records=records)
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(encoded(value))
 return digest(p)
def load_inputs(path,expected,full_final=True):
 if digest(path)!=expected:raise ValueError('Input manifest differs from external pin')
 m=loads(Path(path).read_bytes())
 if m['version']!=1 or m['scope']!='native-scoring-view':raise ValueError('Wrong input manifest version/scope')
 counts={};file_rows={}
 for item in m['inputs']:
  if digest(item['path'])!=item['sha256']:raise ValueError('Evaluation parquet changed')
  if item['sha256'] in file_rows:raise ValueError('Duplicate input file')
  file_rows[item['sha256']]=0
 for ordinal,row in enumerate(m['records']):
  identity=row['identity'];source=identity['source']
  if row['ordinal']!=ordinal or row['identity_sha256']!=hashlib.sha256(encoded(identity)).hexdigest():raise ValueError('Input row identity corrupt')
  if row['file_sha256'] not in file_rows or row['file_row']!=file_rows[row['file_sha256']]:raise ValueError('Input file/row order corrupt')
  file_rows[row['file_sha256']]+=1;counts[source]=counts.get(source,0)+1
 if full_final and (len(m['records'])!=TOTAL or set(counts)!=SOURCES):raise ValueError('Corrected final evaluation requires seven sources and exactly51713 rows')
 import pandas as pd
 ordinal=0
 for item in m['inputs']:
  for file_row,(_,source_row) in enumerate(pd.read_parquet(item['path']).iterrows()):
   if ordinal>=len(m['records']):raise ValueError('Input manifest dropped original source rows')
   saved=m['records'][ordinal]
   identity=row_identity(source_row[m['prompt_key']],source_row['data_source'],source_row['reward_model'])
   if saved['file_sha256']!=item['sha256'] or saved['file_row']!=file_row or encoded(saved['identity'])!=encoded(identity):raise ValueError('Input manifest differs from actual original parquet rows')
   ordinal+=1
 if ordinal!=len(m['records']):raise ValueError('Input manifest invented source rows')
 return m

def enabled(config):
 settings=config.get('baselines',{}).get('corrected_evaluation')
 return bool(settings and settings.get('enabled',False))
def configure(config,split_manifest):
 """Called only after common split identity is bound; blocked until reviewed deployment."""
 if not enabled(config):return None
 s=config.baselines.corrected_evaluation
 if not config.trainer.get('val_only',False) or not config.trainer.val_before_train:raise ValueError('Corrected evaluation requires val_only and val_before_train')
 if not config.do_search:raise ValueError('Corrected evaluation requires native search-agent generation')
 if config.baselines.method!='opsd' or config.baselines.holdout_eval_only:raise ValueError('Corrected evaluation requires native OPSD validation')
 if config.data.get('val_data_num') is not None or config.data.val_batch_size!=64:raise ValueError('No subsampling; evaluation batch64 required')
 if not config.baselines.local_runtime_only or config.baselines.local_runtime_host!='h100-10':raise ValueError('Corrected evaluation must use same-node native retrieval')
 m=load_inputs(s.input_manifest_path,s.input_manifest_sha256)
 from .holdout import paths
 if [str(Path(p).resolve()) for p in paths(config.data.final_val_files)]!=[x['path'] for x in m['inputs']]:raise ValueError('Final files differ from pregeneration manifest')
 if s.split_content_identity!=split_manifest['content_identity']['sha256']:raise ValueError('Common split identity changed')
 provenance=load_stage(s)
 verify_budget(config,provenance)
 from omegaconf import open_dict
 with open_dict(config):
  config.data.val_files=config.data.final_val_files
  config.data.return_raw_chat=True
 return provenance

def load_stage(s):
 stage=Path(s.get('reviewed_stage_manifest','UNFILLED'))
 expected=os.environ.get('OPSD_CORRECTED_EVAL_STAGE_SHA256','')
 if len(expected)!=64 or not stage.is_file() or digest(stage)!=expected:raise ValueError('Blocked: externally reviewed corrected evaluation stage manifest required')
 provenance=loads(stage.read_bytes())
 verify_generation_sources(provenance)
 verify_runtime_sources(provenance)
 if provenance['input_manifest_sha256']!=s.input_manifest_sha256:raise ValueError('Stage data pin mismatch')
 if provenance['model_role'] not in ('base','hf500'):raise ValueError('Wrong model role')
 if not provenance.get('training_hf_receipt_path') or digest(provenance['training_hf_receipt_path'])!=provenance['training_hf_receipt_sha256']:raise ValueError('Actual training/HF completion receipt required')
 provenance['reviewed_stage_path']=str(stage.resolve());provenance['reviewed_stage_sha256']=expected
 verify_stage_prerequisites(provenance)
 return provenance

class ZeroUpdateGuard:
 def __init__(self,trainer):
  self.attempted_dispatches=0;self.guarded=[];self.groups=[];self.original_dispatches=[]
  for name,method in (('actor_rollout_wg','update_actor'),('critic_wg','update_critic')):
   group=getattr(trainer,name,None)
   if group is not None and callable(getattr(group,method,None)):
    original=getattr(group,method)
    self.original_dispatches.append(dict(group=name,type_module=type(group).__module__,type_name=type(group).__qualname__,dispatch_module=getattr(original,'__module__',None),dispatch_name=getattr(original,'__qualname__',None)))
    setattr(group,method,self.reject);self.guarded.append(name+'.'+method);self.groups.append((group,method))
  if 'actor_rollout_wg.update_actor' not in self.guarded:raise ValueError('Actual actor update dispatch is not guarded')
 def reject(self,*args,**kwargs):
  self.attempted_dispatches+=1
  raise RuntimeError('Corrected evaluation forbids optimizer update dispatch')
 def record(self):
  if self.attempted_dispatches:raise RuntimeError('An optimizer update was attempted')
  if any(getattr(group,method)!=self.reject for group,method in self.groups):raise RuntimeError('Actual update guard was replaced')
  return dict(attempted_update_dispatches=0,guarded_dispatches=self.guarded,original_dispatches=self.original_dispatches,guard_source_sha256=digest(__file__))

def validate_arrays(arrays):
 required={'responses','prompts','attention_mask','info_mask'}
 if not required<=set(arrays):raise ValueError('Missing raw scoring tensors')
 n,width=arrays['responses'].shape
 for name,array in arrays.items():
  if array.ndim!=2 or len(array)!=n or array.dtype.kind not in 'iu':raise ValueError('Invalid raw tensor dtype/shape')
  if (array<0).any():raise ValueError('Negative token/mask value')
 for name in ('attention_mask','info_mask'):
  if arrays[name].shape[1]<width or not np.isin(arrays[name],[0,1]).all():raise ValueError('Invalid provenance mask')
 if arrays['attention_mask'].shape!=arrays['info_mask'].shape or (arrays['info_mask']>arrays['attention_mask']).any():raise ValueError('Provenance mask escapes attention')
 if arrays['attention_mask'].shape[1]!=arrays['prompts'].shape[1]+width:raise ValueError('Prompt/response offsets mismatch')

def score_arrays(arrays,records,tokenizer):
 """Replay exact native scorer on real DataProto, never reconstruct text token IDs."""
 import torch
 from verl import DataProto
 from .evaluation import batch_answer_metrics
 validate_arrays(arrays)
 if any((arrays[name]>=len(tokenizer)).any() for name in ('responses','prompts')):raise ValueError('Raw token ID outside evaluated tokenizer vocabulary')
 data=DataProto.from_dict(tensors={k:torch.from_numpy(v.copy()) for k,v in arrays.items()},non_tensors={'data_source':np.array([r['identity']['source'] for r in records],dtype=object),'reward_model':np.array([dict(ground_truth=dict(target=r['identity']['targets'])) for r in records],dtype=object)})
 return batch_answer_metrics(data,tokenizer)

class ScoringViewWriter:
 def __init__(self,directory,manifest,tokenizer,provenance):
  self.directory=Path(directory)
  if self.directory.exists():raise ValueError('Transcript directory must be new')
  self.directory.mkdir(parents=True);self.manifest=manifest;self.tokenizer=tokenizer;self.provenance=canonical(provenance);self.cursor=0;self.files=[];self.scores=[];self.rewards=[]
 def append(self,data,online,reward_tensor):
  from verl.trainer.main_ppo import decode_model_response_spans
  arrays={k:v.detach().cpu().clone().numpy() for k,v in data.batch.items() if k in ('responses','prompts','attention_mask','info_mask','position_ids')}
  validate_arrays(arrays);n=len(data);records=self.manifest['records'][self.cursor:self.cursor+n]
  if n!=min(64,len(self.manifest['records'])-self.cursor):raise ValueError('Wrong native evaluation batch count')
  if len(records)!=n or len(online)!=n:raise ValueError('Evaluation row overrun/drop')
  for i,record in enumerate(records):
   item=data[i];identity=row_identity(item.non_tensor_batch['raw_prompt'],item.non_tensor_batch['data_source'],item.non_tensor_batch['reward_model'])
   if encoded(identity)!=encoded(record['identity']):raise ValueError('Online question/source/GT/order differs from pregeneration identity')
  replay=score_arrays(arrays,records,self.tokenizer)
  if encoded(replay)!=encoded(online):raise ValueError('Online and saved scoring view replay disagree')
  width=arrays['responses'].shape[1];rows=[]
  for i,record in enumerate(records):
   import torch
   mask=arrays['attention_mask'][i,-width:].astype(bool)&arrays['info_mask'][i,-width:].astype(bool)
   spans=decode_model_response_spans(self.tokenizer,torch.from_numpy(arrays['responses'][i].copy()),torch.from_numpy(mask))
   from verl.utils.reward_score.qa_em import compute_score_generated_em
   reward=float(reward_tensor[i].sum().item())
   expected_reward=compute_score_generated_em(spans,dict(target=record['identity']['targets']))
   if not math.isfinite(reward) or reward!=expected_reward:raise ValueError('Native reward/scoring-view disagreement')
   rows.append(dict(input_record=record,model_spans=spans,metrics=replay[i],reward_score=reward))
   self.scores.append(replay[i]);self.rewards.append(reward)
  stem=f'batch-{len(self.files):06d}';tensor=self.directory/(stem+'.npz');meta=self.directory/(stem+'.json')
  with tensor.open('xb') as f:np.savez_compressed(f,**arrays)
  with meta.open('xb') as f:f.write(encoded(rows))
  self.files.append(dict(tensors=tensor.name,tensors_sha256=digest(tensor),rows=meta.name,rows_sha256=digest(meta),start=self.cursor,count=n));self.cursor+=n
 def finish(self,guard,online_metrics):
  if self.cursor!=len(self.manifest['records']):raise ValueError('Incomplete corrected evaluation transcript')
  verify_model(self.provenance)
  expected_metrics=aggregate_with_rewards(self.scores,self.rewards)
  if encoded(expected_metrics)!=encoded(online_metrics):raise ValueError('Original online aggregate/replay disagrees')
  result=dict(input_manifest_sha256=self.provenance['input_manifest_sha256'],online_metrics=canonical(online_metrics),version=1,scope='native-scoring-view-not-full-turn-trace',count=self.cursor,files=self.files,provenance=self.provenance,zero_update_guard=guard.record())
  p=self.directory/'manifest.json'
  with p.open('xb') as f:f.write(encoded(result))
  return digest(p)

def replay(directory,expected_sha,input_manifest_path,input_manifest_sha256,tokenizer,full_final=True):
 """External completed manifest pin mandatory; refuse co-tampered files/manifest."""
 from .evaluation import aggregate_answer_metrics
 input_manifest=load_inputs(input_manifest_path,input_manifest_sha256,full_final=full_final)
 directory=Path(directory);p=directory/'manifest.json'
 if digest(p)!=expected_sha:raise ValueError('Transcript manifest changed from external pin')
 m=loads(p.read_bytes());cursor=0;all_rows=[];all_rewards=[];filenames=set()
 verify_generation_sources(m['provenance']);verify_runtime_sources(m['provenance']);verify_stage_prerequisites(m['provenance']);verify_model(m['provenance'])
 if m['version']!=1 or m['scope']!='native-scoring-view-not-full-turn-trace':raise ValueError('Wrong transcript format')
 if m['input_manifest_sha256']!=input_manifest_sha256:raise ValueError('Transcript input pin mismatch')
 for entry in m['files']:
  if entry['count']!=min(64,len(input_manifest['records'])-cursor):raise ValueError('Wrong formal evaluation batch shape/count')
  if entry['start']!=cursor or entry['count']<=0:raise ValueError('Transcript batch order corrupt')
  for field in ('tensors','rows'):
   name=entry[field];f=directory/name
   if Path(name).name!=name or name in filenames or not f.resolve().is_relative_to(directory.resolve()) or digest(f)!=entry[field+'_sha256']:raise ValueError('Transcript file corrupt or escapes own directory')
   filenames.add(name)
  with np.load(directory/entry['tensors'],allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
  records=loads((directory/entry['rows']).read_bytes());wanted=input_manifest['records'][cursor:cursor+entry['count']]
  if len(records)!=entry['count'] or encoded([r['input_record'] for r in records])!=encoded(wanted):raise ValueError('Raw scoring identities differ from input manifest')
  scores=score_arrays(arrays,wanted,tokenizer)
  if encoded(scores)!=encoded([r['metrics'] for r in records]):raise ValueError('Replayed metrics disagree')
  from verl.trainer.main_ppo import decode_model_response_spans
  from verl.utils.reward_score.qa_em import compute_score_generated_em
  import torch
  width=arrays['responses'].shape[1]
  for i,r in enumerate(records):
   mask=arrays['attention_mask'][i,-width:].astype(bool)&arrays['info_mask'][i,-width:].astype(bool)
   spans=decode_model_response_spans(tokenizer,torch.from_numpy(arrays['responses'][i].copy()),torch.from_numpy(mask))
   reward=compute_score_generated_em(spans,dict(target=wanted[i]['identity']['targets']))
   if encoded(spans)!=encoded(r['model_spans']) or reward!=r['reward_score']:raise ValueError('Saved model spans/reward disagree with raw tokens and masks')
   all_rewards.append(reward)
  all_rows.extend(scores);cursor+=entry['count']
 audit=m['zero_update_guard']
 if audit.get('guard_source_sha256')!=digest(__file__) or 'actor_rollout_wg.update_actor' not in audit.get('guarded_dispatches',[]) or not audit.get('original_dispatches'):raise ValueError('Actual dispatch guard/source audit missing')
 if cursor!=m['count'] or cursor!=len(input_manifest['records']) or m['zero_update_guard']['attempted_update_dispatches']!=0:raise ValueError('Incomplete/updated evaluation')
 result=aggregate_with_rewards(all_rows,all_rewards)
 if encoded(result)!=encoded(m['online_metrics']):raise ValueError('Transcript replay differs from original online aggregate')
 return result


def aggregate_with_rewards(rows,rewards):
 from .evaluation import aggregate_answer_metrics
 result=aggregate_answer_metrics(rows);group={}
 for row,reward in zip(rows,rewards):group.setdefault(row['source'],[]).append(reward)
 for source,values in group.items():
  result['val/test_score/'+source]=float(np.mean(values));result['val/evaluated_count/'+source]=len(values)
 return result

def start_capture(trainer):
 settings=trainer.config.baselines.corrected_evaluation
 if not trainer.config.trainer.get('val_only',False) or not trainer.config.trainer.val_before_train:raise ValueError('Corrected capture cannot run in training')
 manifest=load_inputs(settings.input_manifest_path,settings.input_manifest_sha256)
 if len(trainer.val_dataset)!=len(manifest['records']):raise ValueError('Evaluation filtering dropped input questions')
 if trainer.val_dataloader.drop_last or trainer.val_dataloader.batch_size!=64:raise ValueError('Wrong evaluation loader budget')
 from torch.utils.data import SequentialSampler
 if not isinstance(trainer.val_dataloader.sampler,SequentialSampler):raise ValueError('Evaluation order must be sequential')
 provenance=load_stage(settings)
 verify_budget(trainer.config,provenance)
 model=Path(trainer.config.actor_rollout_ref.model.path).resolve()
 if str(model)!=provenance['model_directory']:raise ValueError('Student model role/path mismatch')
 verify_model(provenance)
 guard=ZeroUpdateGuard(trainer)
 return ScoringViewWriter(settings.output_dir,manifest,trainer.tokenizer,provenance),guard


def verify_model(provenance):
 model=Path(provenance['model_directory'])
 files={p.name for p in model.iterdir() if p.is_file()}
 if files!=set(provenance['model_files']) or not {'config.json','tokenizer.json'}<=files or not any(n.endswith('.safetensors') for n in files):raise ValueError('Incomplete model content pins')
 for name,sha in provenance['model_files'].items():
  if Path(name).name!=name or not (model/name).resolve().is_relative_to(model.resolve()) or digest(model/name)!=sha:raise ValueError('Student HF model changed')



def verify_runtime_sources(provenance):
 import importlib
 repo=Path(__file__).resolve().parents[2]
 actual_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
 if provenance['eval_commit']!=actual_commit or provenance['training_commit']!='8e60976d364ab7d7ae51714415813c1359dafe9a':raise ValueError('Training/evaluation source mismatch')
 needed={'search_r1/baselines/corrected_evaluation.py','search_r1/baselines/evaluation.py','search_r1/baselines/bapo.py','search_r1/qa_protocol.py','verl/trainer/main_ppo.py','verl/utils/reward_score/qa_em.py','search_r1/baselines/main.py','verl/trainer/ppo/ray_trainer.py','search_r1/llm_agent/generation.py','verl/utils/dataset/rl_dataset.py'}
 if not needed<=set(provenance['source_files']):raise ValueError('Incomplete actual scoring runtime pins')
 for filename,sha in provenance['source_files'].items():
  path=repo/filename
  if not filename.endswith('.py') or not path.resolve().is_relative_to(repo.resolve()) or digest(path)!=sha:raise ValueError('Actual scoring source bytes differ')
  if filename in needed:
   module=importlib.import_module(filename[:-3].replace('/','.'))
   if Path(module.__file__).resolve()!=path.resolve() or digest(module.__file__)!=sha:raise ValueError('Actual imported scorer differs from source pin')

def budget_snapshot(config):
 from omegaconf import OmegaConf
 keys=('data.val_batch_size','data.max_start_length','data.max_prompt_length','data.max_response_length','data.max_obs_length','max_turns','force_final_answer','do_search','retriever.url','retriever.topk','baselines.seed','algorithm.no_think_rl','trainer.n_gpus_per_node','trainer.nnodes','actor_rollout_ref.rollout.tensor_model_parallel_size','actor_rollout_ref.rollout.gpu_memory_utilization','actor_rollout_ref.rollout.enforce_eager','actor_rollout_ref.rollout.free_cache_engine','actor_rollout_ref.rollout.dtype','actor_rollout_ref.rollout.temperature','actor_rollout_ref.rollout.top_k','actor_rollout_ref.rollout.top_p','actor_rollout_ref.rollout.ignore_eos','actor_rollout_ref.rollout.n','actor_rollout_ref.rollout.max_num_batched_tokens','actor_rollout_ref.rollout.max_num_seqs')
 values={key:canonical(OmegaConf.select(config,key,throw_on_missing=True)) for key in keys}
 if any(value is None for value in values.values()):raise ValueError('Missing resolved native generation/retrieval budget')
 values['native_validation_sampling']=dict(do_sample=False,temperature=0.0,n=1,best_of=1,top_p=1.0,top_k=-1,min_p=0.0)
 return values

def verify_budget(config,provenance):
 actual=budget_snapshot(config)
 if encoded(actual)!=encoded(provenance['generation_budget']) or hashlib.sha256(encoded(actual)).hexdigest()!=provenance['generation_budget_sha256']:raise ValueError('Resolved generation/retrieval budget differs from reviewed stage')

def verify_stage_prerequisites(provenance):
 """Requires a separately reviewed STRICT external training/HF verifier; not implemented here."""
 import sys
 stage=Path(provenance['reviewed_stage_path'])
 if digest(stage)!=provenance['reviewed_stage_sha256']:raise ValueError('External evaluation stage pin changed')
 original=loads(stage.read_bytes())
 for key,value in original.items():
  if encoded(provenance.get(key))!=encoded(value):raise ValueError('Runtime provenance differs from external stage')
 prerequisite=provenance.get('strict_training_hf_verifier')
 if not prerequisite or not Path(prerequisite['path']).is_file() or digest(prerequisite['path'])!=prerequisite['sha256']:raise ValueError('Blocked: strict actual training3007/HF500 verifier is not supplied')
 subprocess.run(['git','-C',str(Path(__file__).resolve().parents[2]),'diff','--quiet','HEAD'],check=True)
 # The externally pinned tool must run existing complete actual500+HF checks, not accept a self-reported flag.
 result=subprocess.run([sys.executable,prerequisite['path'],'--receipt',provenance['training_hf_receipt_path'],'--receipt-sha256',provenance['training_hf_receipt_sha256'],'--model-role',provenance['model_role'],'--model-directory',provenance['model_directory']],check=True,capture_output=True,text=True)
 proof=loads(result.stdout)
 expected=dict(kind='strict_actual_training3007_hf500_prerequisites',training_commit='8e60976d364ab7d7ae51714415813c1359dafe9a',formal_job='3007',actual_outer_updates=500,final_questions=51713,receipt_sha256=provenance['training_hf_receipt_sha256'],model_role=provenance['model_role'],model_directory=provenance['model_directory'])
 if proof!=expected:raise ValueError('Strict external training/HF verifier did not return exact bound proof')


def verify_generation_sources(provenance):
 # Worker-only GPU modules are byte-pinned, never imported into CPU driver/replay.
 needed={'search_r1/baselines/corrected_evaluation.py','search_r1/baselines/evaluation.py','search_r1/baselines/bapo.py','search_r1/qa_protocol.py','verl/trainer/main_ppo.py','verl/utils/reward_score/qa_em.py','search_r1/baselines/main.py','verl/trainer/ppo/ray_trainer.py','search_r1/llm_agent/generation.py','search_r1/baselines/actor.py','search_r1/baselines/distillation.py','verl/workers/rollout/vllm_rollout/vllm_rollout.py','verl/utils/dataset/rl_dataset.py'}
 if not needed<=set(provenance['source_files']):raise ValueError('Incomplete native generation/dataset/runtime source pins')

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
 prepare=commands.add_parser('prepare-inputs');prepare.add_argument('--files',nargs='+',required=True);prepare.add_argument('--output',required=True);prepare.add_argument('--prompt-key',default='prompt')
 rescore=commands.add_parser('replay');rescore.add_argument('--directory',required=True);rescore.add_argument('--manifest-sha256',required=True);rescore.add_argument('--inputs',required=True);rescore.add_argument('--inputs-sha256',required=True);rescore.add_argument('--tokenizer',required=True)
 args=parser.parse_args()
 if args.command=='prepare-inputs':print(prepare_inputs(args.files,args.output,args.prompt_key))
 else:
  from transformers import AutoTokenizer
  manifest=loads((Path(args.directory)/'manifest.json').read_bytes())
  if Path(args.tokenizer).resolve()!=Path(manifest['provenance']['model_directory']).resolve():raise ValueError('Tokenizer must match the actual evaluated HF content pins')
  tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
  print(encoded(replay(args.directory,args.manifest_sha256,args.inputs,args.inputs_sha256,tokenizer)).decode())
