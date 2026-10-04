"""CPU-only diagnostic contract tests, not real model/GPU acceptance."""
import importlib.util
from pathlib import Path
import json
import os
import numpy as np
import pytest
import torch

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('probe', HERE/'opsd_prompt_distribution_probe.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
SOURCE = HERE/'opsd/search_r1/baselines/distillation.py'

class ChatTokenizer:
 pad_token_id=0
 all_special_ids=[0]
 chat_template='mock CPU-only Qwen ChatML'
 def encode(self, s, add_special_tokens=False): return [ord(c)+1 for c in s]
 def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
  s=''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in msgs)
  return s+('<|im_start|>assistant\n' if add_generation_prompt else '')
 def decode(self, ids): return ''.join(chr(i-1) for i in ids)

@pytest.fixture
def functions(): return p.extract_functions(SOURCE)

def test_real_source_full_vocab_math_and_counterexample(functions):
 t=torch.tensor([.9,.1]).log(); s=torch.tensor([.1,.9]).log()
 result=p.objective_metrics(s,t,functions[1])
 assert result['gradient_cosine'] == -1
 assert result['clipped_step_unclipped_kl_delta'] > 0 > result['unclipped_step_unclipped_kl_delta']
 assert result['clipped_components']==1
 assert abs(result['unclipped_kl']-torch.sum(t.exp()*(t-s)).item()) < 1e-6

@pytest.mark.parametrize('bad',[np.array([np.nan,0]),np.array([np.inf,0]),np.ones(3),np.ones((1,2))])
def test_invalid_logit_views_rejected(functions,bad):
 with pytest.raises(ValueError): p.objective_metrics(bad,np.zeros(2),functions[1])

def test_equal_distributions_zero_gradient(functions):
 result=p.objective_metrics(np.zeros(4),np.zeros(4),functions[1])
 assert result['unclipped_kl']==0 and result['gradient_cosine'] is None
 assert result['clipped_components']==0

@pytest.mark.parametrize('mode',['legacy','user_context'])
def test_prediction_prefix_excludes_placeholder_and_private_student(mode,functions):
 tok=ChatTokenizer(); student=tok.apply_chat_template([{'role':'user','content':'Question?'}],add_generation_prompt=True)
 row={'student_prompt_ids':tok.encode(student),'targets':['PrivateGold']}
 teacher=tok.decode(p.teacher_prefix(tok,functions[0],row,mode,32768))
 assert 'PrivateGold' not in student and 'PrivateGold' in teacher
 assert teacher.endswith('<|im_start|>assistant\n') if mode=='user_context' else teacher.startswith(student)
 if mode=='user_context': assert teacher.index('PrivateGold') < teacher.rindex('<|im_end|>')
 assert row['student_prompt_ids']==tok.encode(student)

@pytest.mark.parametrize('raw',['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'])
def test_json_duplicate_and_nonfinite_reject(raw):
 with pytest.raises(ValueError): p.loads(raw)

def test_external_pin_path_and_sha(tmp_path):
 file=tmp_path/'raw'; file.write_bytes(b'actual')
 good={'path':str(file),'sha256':p.digest(file)}
 assert p.bound(good,tmp_path)==file
 with pytest.raises(ValueError): p.bound({**good,'sha256':'0'*64},tmp_path)
 with pytest.raises(ValueError): p.bound(good,tmp_path/'other')
 link=tmp_path/'link'; link.symlink_to('/etc/hosts')
 with pytest.raises(ValueError): p.bound({'path':str(link),'sha256':p.digest(link)},tmp_path)

def test_immutable_writer(tmp_path):
 path=tmp_path/'evidence.json';p.write_new(path,{'real':1})
 with pytest.raises(FileExistsError): p.write_new(path,{'real':2})
 assert p.loads(path.read_bytes())=={'real':1}

def test_cpu_without_own_fresh_slurm_reject(monkeypatch):
 monkeypatch.delenv('SLURM_JOB_ID',raising=False)
 with pytest.raises(ValueError): p.hardware_guard({'version':1,'job_id':'3007','uid':os.getuid(),'node':'h100-10','selected_gpu_uuid':'unfilled'},True)

def test_selected_train_rows_arrays_no_validation_gold(tmp_path):
 import pandas as pd
 tok=ChatTokenizer()
 rows=[{'data_source':'nq','prompt':[{'role':'user','content':f'question {i}'}], 'reward_model':{'ground_truth':{'target':['accepted']}},'extra_info':{'split':'train','index':i}} for i in range(64)]
 path=tmp_path/'train.parquet';pd.DataFrame(rows).to_parquet(path)
 result=p.selected_inputs(path,tok)
 assert len(result)==64 and result[-1]['file_row']==63
 assert all('accepted' not in r['student_prompt_text'] and r['targets']==['accepted'] for r in result)
 rows[0]['extra_info']['split']='test';pd.DataFrame(rows).to_parquet(path)
 with pytest.raises(ValueError): p.selected_inputs(path,tok)

def test_unknown_ast_import_reject(tmp_path):
 source=SOURCE.read_text().replace('from .rollout import padded','from os import system')
 path=tmp_path/'source.py';path.write_text(source)
 with pytest.raises(ValueError): p.extract_functions(path)

def test_no_optimizer_or_model_grad_dispatch_in_probe():
 import ast
 tree=ast.parse((HERE/'opsd_prompt_distribution_probe.py').read_text())
 calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
 assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('step','backward','generate') for n in calls)
 assert not any(isinstance(n,ast.Import) and any(a.name.startswith(('requests','ray','faiss')) for a in n.names) for n in ast.walk(tree))

def test_real_cpu_forward_saves_complete_candidate_raw_and_probability(tmp_path,monkeypatch):
 """Actual torch CPU forwards with a toy model; CUDA device mapping explicitly stubbed."""
 from types import SimpleNamespace
 tok=ChatTokenizer(); original=torch.tensor
 def cpu_tensor(*args,**kwargs):
  if kwargs.get('device')=='cuda:0': kwargs['device']='cpu'
  return original(*args,**kwargs)
 monkeypatch.setattr(torch,'tensor',cpu_tensor)
 class ToyModel:
  config=SimpleNamespace(vocab_size=256,max_position_embeddings=2048)
  def __call__(self,input_ids,attention_mask,use_cache):
   assert not torch.is_grad_enabled() and use_cache is False
   length=input_ids.shape[1]
   return SimpleNamespace(logits=torch.arange(256,dtype=torch.float32).reshape(1,1,-1).expand(1,length,-1))
 row={'ordinal':0,'targets':['Gold']}; prefix=tok.encode('Question')
 path=tmp_path/'view.npz';record=p.forward_case(ToyModel(),tok,prefix,path,'teacher_legacy',row)
 assert set(record['tags'])==set((*p.TAGS,'<answer>Gold</answer>'))
 with np.load(path,allow_pickle=False) as raw:
  assert raw['prompt_ids'].tolist()==prefix and raw['initial_logits'].shape==(256,)
  for i,tag in enumerate((*p.TAGS,'<answer>Gold</answer>')):
   ids=torch.from_numpy(raw[f'tag{i}_ids']);logits=torch.from_numpy(raw[f'tag{i}_logits'])
   assert logits.shape==(len(tok.encode(tag)),256)
   total=logits.log_softmax(-1).gather(-1,ids[:,None]).sum().item()
   assert total==record['tags'][tag]['conditional_log_probability']
 assert p.digest(path)==record['sha256']

def test_long_complete_candidate_is_explicit_skip_not_truncation(tmp_path,monkeypatch):
 from types import SimpleNamespace
 tok=ChatTokenizer(); original=torch.tensor
 monkeypatch.setattr(torch,'tensor',lambda *a,**k: original(*a,**{**k,**({'device':'cpu'} if k.get('device')=='cuda:0' else {})}))
 class ToyModel:
  config=SimpleNamespace(vocab_size=256,max_position_embeddings=32)
  def __call__(self,input_ids,attention_mask,use_cache):return SimpleNamespace(logits=torch.zeros(1,input_ids.shape[1],256))
 record=p.forward_case(ToyModel(),tok,tok.encode('Q'),tmp_path/'view.npz','teacher_legacy',{'ordinal':0,'targets':['Gold'*30]})
 candidate='<answer>'+'Gold'*30+'</answer>'
 assert record['tags'][candidate]['skipped'] and len(record['tags'][candidate]['tokens'])==len(tok.encode(candidate))
 with np.load(tmp_path/'view.npz',allow_pickle=False) as raw: assert 'tag3_logits' not in raw and raw['tag3_ids'].tolist()==tok.encode(candidate)

@pytest.mark.parametrize('pci',['0000:81:00.0','00000000:81:00.0','0000:81:00.0\n'])
def test_actual_pci_canonical_domain(pci): assert p.normalized_pci(pci)==(0,0x81,0,0)

@pytest.mark.parametrize('pci',['GPU-uuid','81:00.0','0000:81:00.8','../../other'])
def test_actual_pci_malformed_reject(pci):
 with pytest.raises(ValueError):p.normalized_pci(pci)

@pytest.fixture
def runtime_fixture():
 fields=['0']*20;fields[19]='123'
 proc={'raw_status':'Uid:\t42\t42\t42\t42\n','uids':[42]*4,'raw_cgroup':'0::/slurm/uid_42/job_4000/step_batch\n','start_ticks':'123','raw_stat':'9 (python) '+' '.join(fields)}
 resource={'job_id':'4000','gpu_uuid':'GPU-test','process':proc,'selected_processes':[]}
 stages=[]
 for role,cases,dtype,backend in [('base',['student_base'],'torch.bfloat16','flash_attention_2'),('hf5',['student_hf5'],'torch.bfloat16','flash_attention_2'),('base',['teacher_legacy','teacher_user_context'],'torch.float32','eager')]:
  stages.append({'role':role,'cases':cases,'dtype':dtype,'backend':backend,'parameter_device':'cuda:0','parameters_frozen':True,'parameter_count':3085938688,'parameters_finite':True,'tied_embeddings':True,'loading':dict.fromkeys(['missing_keys','unexpected_keys','mismatched_keys','error_msgs'],[]),'peak_reserved_bytes':12_000_000_000,'total_device_bytes':80_000_000_000,'cuda_mapping':{'uuid':'GPU-test'},'resource':resource})
 return {'runtime':{'model_optimizer_steps':0},'stages':stages,'final_resource':resource},{'job_id':'4000','selected_gpu_uuid':'GPU-test','uid':42}

def test_synthetic_cpu_runtime_contract_accept(runtime_fixture):p.validate_runtime(*runtime_fixture)

@pytest.mark.parametrize('change',['optimizer','missing','role','backend','dtype','freeze','uuid','pid','uid','cgroup','peak','paramcount','nonfinite','loadschema'])
def test_synthetic_cpu_runtime_faults_reject(runtime_fixture,change):
 m,a=runtime_fixture
 if change=='optimizer':m['runtime']['model_optimizer_steps']=1
 elif change=='missing':m['stages'][0]['loading']['missing_keys']=['weight']
 elif change in ('role','backend','dtype'):m['stages'][0][change]='substituted'
 elif change=='freeze':m['stages'][0]['parameters_frozen']=False
 elif change=='uuid':m['stages'][0]['cuda_mapping']['uuid']='GPU-foreign'
 elif change=='pid':m['final_resource']['process']['start_ticks']='124'
 elif change=='uid':m['final_resource']['process']['uids']=[99]*4
 elif change=='cgroup':m['final_resource']['process']['raw_cgroup']='0::/slurm/job_3007/'
 elif change=='paramcount':m['stages'][0]['parameter_count']=3_000_000_000
 elif change=='nonfinite':m['stages'][0]['parameters_finite']=False
 elif change=='loadschema':del m['stages'][0]['loading']['missing_keys']
 elif change=='peak':m['stages'][0]['peak_reserved_bytes']=79_000_000_000
 with pytest.raises(ValueError):p.validate_runtime(m,a)

def test_preexisting_rejected_output_not_mutated_by_failure(tmp_path,monkeypatch):
 from types import SimpleNamespace
 monkeypatch.setattr(p,'ROOT',tmp_path)
 old=tmp_path/'old';old.mkdir();(old/'original').write_bytes(b'preserved')
 before={f.name:f.read_bytes() for f in old.iterdir()}
 p.preserve_failure(SimpleNamespace(output=str(old)),ValueError('existing output rejected'))
 assert {f.name:f.read_bytes() for f in old.iterdir()}==before

def test_only_exclusively_created_own_current_job_failure(tmp_path,monkeypatch):
 from types import SimpleNamespace
 monkeypatch.setattr(p,'ROOT',tmp_path);monkeypatch.setenv('SLURM_JOB_ID','4000')
 own=tmp_path/'new';own.mkdir()
 args=SimpleNamespace(_owned_output=own,_owner_job='4001',_owner_uid=os.getuid())
 p.preserve_failure(args,RuntimeError('failed'));assert not (own/'failure.json').exists()
 args._owner_job='4000';p.preserve_failure(args,RuntimeError('failed'))
 assert p.loads((own/'failure.json').read_bytes())['job_id']=='4000'

@pytest.fixture
def full_synthetic_replay(tmp_path,monkeypatch,runtime_fixture):
 """Mock assets/allocation; real256 CPU raw cases and real source math/replay."""
 import sys
 from types import SimpleNamespace
 tok=ChatTokenizer(); teacher,_=p.extract_functions(SOURCE)
 original=torch.tensor
 def cpu_tensor(*args,**kwargs):
  if kwargs.get('device')=='cuda:0':kwargs['device']='cpu'
  return original(*args,**kwargs)
 monkeypatch.setattr(torch,'tensor',cpu_tensor)
 monkeypatch.setattr(torch,'autocast',lambda *a,**k:__import__('contextlib').nullcontext())
 model=SimpleNamespace(config=SimpleNamespace(vocab_size=256,max_position_embeddings=2048))
 class ToyModel:
  config=model.config
  def __call__(self,input_ids,attention_mask,use_cache):return SimpleNamespace(logits=torch.arange(256,dtype=torch.float32).reshape(1,1,-1).expand(1,input_ids.shape[1],-1))
 base=tmp_path/'base';base.mkdir();p.write_new(base/'config.json',{'vocab_size':256,'max_position_embeddings':2048})
 monkeypatch.setattr(p,'BASE',base);monkeypatch.setattr(p,'verify_binding',lambda binding:binding)
 monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:tok)))
 rows=[]
 for i in range(64):
  text=tok.apply_chat_template([{'role':'user','content':f'Question{i}?'}],add_generation_prompt=True)
  rows.append({'ordinal':i,'student_prompt_ids':tok.encode(text),'targets':['Gold']})
 monkeypatch.setattr(p,'selected_inputs',lambda *a:rows)
 binding={'source':{'path':str(SOURCE.resolve()),'sha256':p.digest(SOURCE)},'probe_sha256':p.digest(Path(p.__file__)),'models':{'base':{'directory':str(base)}}}
 allocation={'job_id':'4000','selected_gpu_uuid':'GPU-test','uid':42}
 m,_=runtime_fixture;m.update({'scope':'train-initial-state-distribution-only','binding':binding,'source':binding['source'],'probe_sha256':binding['probe_sha256'],'rows':[__import__('hashlib').sha256(p.encoded(r)).hexdigest() for r in rows],'cases':[],'limitations':['CPU MOCK assets/allocation, not GPU evidence']})
 for name,value in [('inputs',rows),('binding_file',binding),('allocation',allocation),('initial_resource',m['final_resource'])]:
  path=tmp_path/(name+'.json');p.write_new(path,value);m[name]={'path':str(path),'sha256':p.digest(path)}
 for row in rows:
  for case in p.CASE_NAMES:
   prefix=row['student_prompt_ids'] if case.startswith('student_') else p.teacher_prefix(tok,teacher,row,case.removeprefix('teacher_'),2048)
   m['cases'].append(p.forward_case(ToyModel(),tok,prefix,tmp_path/f"{row['ordinal']}-{case}.npz",case,row))
 return tmp_path,m

@pytest.mark.parametrize('fault',[None,'missing_case','truncated_vocab','swapped_prefix'])
def test_complete_synthetic_raw_replay_and_tamper_rejection(full_synthetic_replay,fault):
 directory,m=full_synthetic_replay
 if fault=='missing_case':m['cases'].pop()
 elif fault in ('truncated_vocab','swapped_prefix'):
  record=m['cases'][0];path=Path(record['path'])
  with np.load(path,allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
  if fault=='truncated_vocab':arrays['initial_logits']=arrays['initial_logits'][:-1]
  else:arrays['prompt_ids'][-1]=1
  with path.open('wb') as handle:np.savez(handle,**arrays)
  record['sha256']=p.digest(path)
 p.write_new(directory/'manifest.json',m)
 if fault:
  with pytest.raises(ValueError):p.replay(directory,p.digest(directory/'manifest.json'))
 else:
  result=p.replay(directory,p.digest(directory/'manifest.json'))
  assert len(result['comparison_rows'])==256 and len(result['conditional_candidate_rows'])==256
  assert all(r['metrics']['unclipped_kl']==0 for r in result['comparison_rows'])

@pytest.mark.parametrize('role',['base','hf5'])
def test_exact_role_specific_tied_state_index(role):
 # Construct the independently known fixed36layer Qwen architecture keys.
 keys={'model.embed_tokens.weight','model.norm.weight'}
 for i in range(36):
  for module in ['q_proj','k_proj','v_proj']:
   keys|={f'model.layers.{i}.self_attn.{module}.{kind}' for kind in ['weight','bias']}
  keys.add(f'model.layers.{i}.self_attn.o_proj.weight')
  keys|={f'model.layers.{i}.mlp.{module}.weight' for module in ['gate_proj','up_proj','down_proj']}
  keys|={f'model.layers.{i}.{module}.weight' for module in ['input_layernorm','post_attention_layernorm']}
 if role=='hf5':keys.add('lm_head.weight')
 index={key:('one' if n%2 else 'two') for n,key in enumerate(sorted(keys))}
 assert len(index)==(434 if role=='base' else 435)
 p.validate_role_index(role,index,{'one','two'})
 swapped=dict(index);swapped['wrong']=swapped.pop('model.norm.weight')
 with pytest.raises(ValueError):p.validate_role_index(role,swapped,{'one','two'})
 if role=='base':
  index['lm_head.weight']='one'
  with pytest.raises(ValueError):p.validate_role_index(role,index,{'one','two'})
 else:
  del index['lm_head.weight']
  with pytest.raises(ValueError):p.validate_role_index(role,index,{'one','two'})
