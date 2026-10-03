"""Real verl.DataProto/native scoring tests; no AST fallback and no GPU use."""
import copy,json,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from tokenizers import Tokenizer,models,decoders
from transformers import PreTrainedTokenizerFast
from verl import DataProto
from verl.trainer.main_ppo import RewardManager
from search_r1.baselines.evaluation import batch_answer_metrics
from search_r1.baselines import corrected_evaluation as ce

STRICT_STAGE_CHECK=ce.verify_stage_prerequisites
SOURCE_FILES=('search_r1/baselines/corrected_evaluation.py','search_r1/baselines/evaluation.py','search_r1/baselines/bapo.py','search_r1/qa_protocol.py','verl/trainer/main_ppo.py','verl/utils/reward_score/qa_em.py','search_r1/baselines/main.py','verl/trainer/ppo/ray_trainer.py','search_r1/llm_agent/generation.py','search_r1/baselines/actor.py','search_r1/baselines/distillation.py','verl/workers/rollout/vllm_rollout/vllm_rollout.py','verl/utils/dataset/rl_dataset.py')

@pytest.fixture(autouse=True)
def cpu_raw_view_scope(monkeypatch):
 # These are raw-view unit tests, NOT actual3007/500/HF acceptance tests.
 # Production replay retains a mandatory external strict prerequisite boundary.
 monkeypatch.setattr(ce,'verify_stage_prerequisites',lambda provenance:None)

@pytest.fixture
def tokenizer():
 vocab={'<pad>':0,'<unk>':1,'<answer>Paris</answer>':2,'<search><answer>Paris</answer></search>':3,'<think><answer>Paris</answer></think>':4,'<search>next</search>':5,'<answer>':6,'</answer>':7,'Paris':8,'observation':9,'<search>Paris</search>':10}
 backend=Tokenizer(models.WordLevel(vocab,unk_token='<unk>'));backend.decoder=decoders.Fuse()
 return PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='<pad>',unk_token='<unk>')

def dataset(tmp_path,n=65):
 path=tmp_path/'data.parquet'
 rows=[dict(prompt=[dict(role='user',content=f'question-{i}')],data_source='nq',reward_model=dict(ground_truth=dict(target=['Paris'])),extra_info=dict(index=0)) for i in range(n)]
 pd.DataFrame(rows).to_parquet(path);p=tmp_path/'inputs.json';sha=ce.prepare_inputs([str(path)],p)
 return p,sha,ce.load_inputs(p,sha,full_final=False)

def batch(manifest,start=0,count=1,response=(2,0,0),model_mask=(1,0,0)):
 records=manifest['records'][start:start+count];responses=torch.tensor([response]*count,dtype=torch.long);prompts=torch.tensor([[9]]*count,dtype=torch.long)
 attention=torch.cat((torch.ones((count,1),dtype=torch.long),(responses!=0).long()),dim=1)
 info=torch.tensor([[1]+list(model_mask)]*count,dtype=torch.long)
 return DataProto.from_dict(tensors={'responses':responses,'prompts':prompts,'attention_mask':attention,'info_mask':info},non_tensors={'raw_prompt':np.array([r['identity']['prompt'] for r in records],dtype=object),'data_source':np.array([r['identity']['source'] for r in records],dtype=object),'reward_model':np.array([dict(ground_truth=dict(target=r['identity']['targets'])) for r in records],dtype=object)})

class Group:
 def update_actor(self,*a):raise AssertionError('unguarded update')
 def generate_sequences(self,*a):return 'unchanged'
def guard():return ce.ZeroUpdateGuard(SimpleNamespace(actor_rollout_wg=Group()))
def provenance(tmp_path,input_sha):
 model=tmp_path/'model';model.mkdir()
 for name in ('config.json','tokenizer.json','model.safetensors'):(model/name).write_text('CPU fixture only, not a real HF export')
 return dict(input_manifest_sha256=input_sha,model_directory=str(model),model_files={p.name:ce.digest(p) for p in model.iterdir()},training_commit='8e60976d364ab7d7ae51714415813c1359dafe9a',eval_commit=subprocess.check_output(['git','-C',str(Path(ce.__file__).resolve().parents[2]),'rev-parse','HEAD'],text=True).strip(),source_files={name:ce.digest(Path(ce.__file__).resolve().parents[2]/name) for name in SOURCE_FILES})

def finish_capture(tmp_path,tokenizer):
 p,pin,m=dataset(tmp_path);writer=ce.ScoringViewWriter(tmp_path/'out',m,tokenizer,provenance(tmp_path,pin));all_rows=[];rewards=[]
 for start,count in ((0,64),(64,1)):
  data=batch(m,start,count);before={k:v.clone() for k,v in data.batch.items()};before_meta=ce.encoded(data.non_tensor_batch)
  online=batch_answer_metrics(data,tokenizer);reward=RewardManager(tokenizer,0)(data);writer.append(data,online,reward)
  assert all(torch.equal(before[k],data.batch[k]) for k in before);assert before_meta==ce.encoded(data.non_tensor_batch)
  all_rows.extend(online);rewards.extend(reward.sum(-1).tolist())
 metrics=ce.aggregate_with_rewards(all_rows,rewards);outpin=writer.finish(guard(),metrics)
 return p,pin,m,writer,outpin,metrics

def test_real_dataproto_writer_tail_one_readonly_replay(tmp_path,tokenizer):
 assert DataProto.__module__=='verl.protocol'
 p,pin,m,w,outpin,metrics=finish_capture(tmp_path,tokenizer)
 assert [x['count'] for x in w.files]==[64,1]
 assert ce.replay(w.directory,outpin,p,pin,tokenizer,full_final=False)==metrics

@pytest.mark.parametrize('response,mask,score',[( (2,0,0),(1,0,0),1),((3,0,0),(1,0,0),0),((4,0,0),(1,0,0),0),((2,9,5),(1,0,1),0),((6,8,7),(1,0,1),0),((9,2,0),(0,1,0),1)])
def test_observation_provenance_nested_nonterminal(tmp_path,tokenizer,response,mask,score):
 p,pin,m=dataset(tmp_path,1);data=batch(m,response=response,model_mask=mask)
 online=batch_answer_metrics(data,tokenizer);assert online[0]['em']==score
 reward=RewardManager(tokenizer,0)(data);writer=ce.ScoringViewWriter(tmp_path/'out',m,tokenizer,provenance(tmp_path,pin));writer.append(data,online,reward)
 writer.finish(guard(),ce.aggregate_with_rewards(online,reward.sum(-1).tolist()))

@pytest.mark.parametrize('mutation',['target','question','source','ordinal','mask','online'])
def test_identity_mask_online_tampering_rejected(tmp_path,tokenizer,mutation):
 p,pin,m=dataset(tmp_path,1);data=batch(m);online=batch_answer_metrics(data,tokenizer);reward=RewardManager(tokenizer,0)(data)
 writer=ce.ScoringViewWriter(tmp_path/'out',m,tokenizer,provenance(tmp_path,pin))
 if mutation=='target':data.non_tensor_batch['reward_model'][0]['ground_truth']['target']=['Berlin']
 if mutation=='question':data.non_tensor_batch['raw_prompt'][0][0]['content']='another-question'
 if mutation=='source':data.non_tensor_batch['data_source'][0]='hotpotqa'
 if mutation=='ordinal':writer.cursor=1
 if mutation=='mask':data.batch['info_mask'][0,-1]=2
 if mutation=='online':online[0]['em']=0
 with pytest.raises(ValueError):writer.append(data,online,reward)

def test_external_pin_rejects_manifest_and_file_cotampering(tmp_path,tokenizer):
 p,pin,m,w,outpin,metrics=finish_capture(tmp_path,tokenizer)
 manifest=w.directory/'manifest.json';value=ce.loads(manifest.read_bytes());f=w.directory/value['files'][0]['rows'];rows=ce.loads(f.read_bytes());rows[0]['metrics']['em']=0;f.write_bytes(ce.encoded(rows));value['files'][0]['rows_sha256']=ce.digest(f);manifest.write_bytes(ce.encoded(value))
 with pytest.raises(ValueError):ce.replay(w.directory,outpin,p,pin,tokenizer,full_final=False)

def test_input_manifest_pin_and_sourcefile_order_identity(tmp_path):
 p,pin,m=dataset(tmp_path,2);m['records'].reverse();p.write_bytes(ce.encoded(m))
 with pytest.raises(ValueError):ce.load_inputs(p,pin,full_final=False)
 with pytest.raises(ValueError):ce.load_inputs(p,ce.digest(p),full_final=False)

def test_actual_dispatch_guard_rejects_counts_and_detects_replaced_guard():
 trainer=SimpleNamespace(actor_rollout_wg=Group());before=trainer.actor_rollout_wg.generate_sequences;g=ce.ZeroUpdateGuard(trainer)
 assert trainer.actor_rollout_wg.generate_sequences==before and g.record()['attempted_update_dispatches']==0
 with pytest.raises(RuntimeError):trainer.actor_rollout_wg.update_actor(None)
 assert g.attempted_dispatches==1
 with pytest.raises(RuntimeError):g.record()
 trainer=SimpleNamespace(actor_rollout_wg=Group());g=ce.ZeroUpdateGuard(trainer);trainer.actor_rollout_wg.update_actor=lambda *args:None
 with pytest.raises(RuntimeError):g.record()

@pytest.mark.parametrize('n',[64,3393])
def test_final_entry_rejects_wrong_denominator(tmp_path,n):
 p,pin,m=dataset(tmp_path,n)
 with pytest.raises(ValueError):ce.load_inputs(p,pin)

def test_default_disabled_and_entry_training_rejected():
 config=OmegaConf.create(dict(baselines=dict(corrected_evaluation=dict(enabled=False))));before=OmegaConf.to_container(config)
 assert ce.configure(config,{}) is None and OmegaConf.to_container(config)==before
 config=OmegaConf.create(dict(baselines=dict(corrected_evaluation=dict(enabled=True)),trainer=dict(val_only=False,val_before_train=True)))
 with pytest.raises(ValueError):ce.configure(config,{})

def test_incomplete_transcript_and_original_aggregate_rejected(tmp_path,tokenizer):
 p,pin,m=dataset(tmp_path,65);data=batch(m,count=64);rows=batch_answer_metrics(data,tokenizer);reward=RewardManager(tokenizer,0)(data)
 writer=ce.ScoringViewWriter(tmp_path/'out',m,tokenizer,provenance(tmp_path,pin));writer.append(data,rows,reward)
 with pytest.raises(ValueError):writer.finish(guard(),ce.aggregate_with_rewards(rows,[1]*64))
 assert not (writer.directory/'manifest.json').exists()

def test_real_agent_dp_padding_and_unpad_single_tail(tmp_path,tokenizer):
 from search_r1.llm_agent.generation import LLMGenerationManager
 p,pin,m=dataset(tmp_path,1);original=batch(m)
 class Rollout:
  seen=0
  def generate_sequences(self,active):
   self.seen=len(active)
   return DataProto.concat([original]*len(active))
 manager=object.__new__(LLMGenerationManager);manager.config=SimpleNamespace(num_gpus=8);manager.actor_rollout_wg=Rollout()
 active=DataProto.from_dict(tensors={'input_ids':original.batch['prompts'].clone(),'attention_mask':torch.ones((1,1),dtype=torch.long),'position_ids':torch.zeros((1,1),dtype=torch.long)})
 trimmed=manager._generate_with_gpu_padding(active)
 assert manager.actor_rollout_wg.seen==8 and trimmed.batch['responses'].shape[0]==1
 actual=DataProto.from_dict(tensors=dict(trimmed.batch),non_tensors=original.non_tensor_batch)
 assert len(actual)==1 and torch.equal(actual.batch['responses'],original.batch['responses'])
 writer=ce.ScoringViewWriter(tmp_path/'out',m,tokenizer,provenance(tmp_path,pin));rows=batch_answer_metrics(actual,tokenizer);rewards=RewardManager(tokenizer,0)(actual);writer.append(actual,rows,rewards)
 writer.finish(guard(),ce.aggregate_with_rewards(rows,rewards.sum(-1).tolist()))

def test_rehashed_target_identity_cannot_disagree_with_actual_parquet(tmp_path):
 p,pin,m=dataset(tmp_path,1);m['records'][0]['identity']['targets']=['Berlin'];m['records'][0]['identity_sha256']=__import__('hashlib').sha256(ce.encoded(m['records'][0]['identity'])).hexdigest();p.write_bytes(ce.encoded(m))
 with pytest.raises(ValueError):ce.load_inputs(p,ce.digest(p),full_final=False)

def test_original_aggregate_mismatch_rejected(tmp_path,tokenizer):
 p,pin,m=dataset(tmp_path,1);data=batch(m);rows=batch_answer_metrics(data,tokenizer);reward=RewardManager(tokenizer,0)(data);writer=ce.ScoringViewWriter(tmp_path/'out',m,tokenizer,provenance(tmp_path,pin));writer.append(data,rows,reward)
 aggregate=ce.aggregate_with_rewards(rows,reward.sum(-1).tolist());aggregate['val/f1/all']=0
 with pytest.raises(ValueError):writer.finish(guard(),aggregate)
 assert not (writer.directory/'manifest.json').exists()

def test_rehashed_saved_spans_are_reconstructed_from_raw_tokens(tmp_path,tokenizer):
 p,pin,m,w,outpin,metrics=finish_capture(tmp_path,tokenizer);manifest=w.directory/'manifest.json';value=ce.loads(manifest.read_bytes());f=w.directory/value['files'][0]['rows'];rows=ce.loads(f.read_bytes());rows[0]['model_spans']=['invented'];f.write_bytes(ce.encoded(rows));value['files'][0]['rows_sha256']=ce.digest(f);manifest.write_bytes(ce.encoded(value))
 with pytest.raises(ValueError):ce.replay(w.directory,ce.digest(manifest),p,pin,tokenizer,full_final=False)

def test_out_of_vocabulary_raw_tokens_rejected(tmp_path,tokenizer):
 p,pin,m=dataset(tmp_path,1);data=batch(m);data.batch['responses'][0,0]=len(tokenizer)+100
 arrays={k:v.numpy().copy() for k,v in data.batch.items()}
 with pytest.raises(ValueError):ce.score_arrays(arrays,m['records'],tokenizer)

def test_missing_provenance_mask_rejected(tmp_path,tokenizer):
 p,pin,m=dataset(tmp_path,1);data=batch(m);arrays={k:v.numpy().copy() for k,v in data.batch.items() if k!='info_mask'}
 with pytest.raises(ValueError):ce.score_arrays(arrays,m['records'],tokenizer)

@pytest.mark.parametrize('name',SOURCE_FILES)
def test_each_native_source_pin_is_required(tmp_path,name):
 p,pin,m=dataset(tmp_path,1);value=provenance(tmp_path,pin);del value['source_files'][name]
 with pytest.raises(ValueError):
  ce.verify_generation_sources(value);ce.verify_runtime_sources(value)

@pytest.mark.parametrize('mutation',['commit','training','bytes','import'])
def test_actual_runtime_not_just_source_labels(tmp_path,monkeypatch,mutation):
 p,pin,m=dataset(tmp_path,1);value=provenance(tmp_path,pin)
 if mutation=='commit':value['eval_commit']='0'*40
 if mutation=='training':value['training_commit']='0'*40
 if mutation=='bytes':value['source_files']['search_r1/qa_protocol.py']='0'*64
 if mutation=='import':
  import search_r1.qa_protocol as protocol
  monkeypatch.setattr(protocol,'__file__',str(tmp_path/'fake.py'))
 with pytest.raises(ValueError):ce.verify_runtime_sources(value)

def test_actual_runtime_source_pins_pass(tmp_path):
 p,pin,m=dataset(tmp_path,1);value=provenance(tmp_path,pin)
 ce.verify_generation_sources(value);ce.verify_runtime_sources(value)

def test_actual_resolved_budget_not_self_reported():
 config=OmegaConf.merge(OmegaConf.load(Path(ce.__file__).resolve().parents[2]/'verl/trainer/config/ppo_trainer.yaml'),OmegaConf.load(Path(ce.__file__).resolve().parents[2]/'verl/trainer/config/ppo_baselines.yaml'));value=ce.budget_snapshot(config)
 provenance=dict(generation_budget=value,generation_budget_sha256=__import__('hashlib').sha256(ce.encoded(value)).hexdigest())
 ce.verify_budget(config,provenance)
 config.retriever=dict(url='http://wrong-node:8000',topk=3)
 with pytest.raises(ValueError):ce.verify_budget(config,provenance)

def test_stage_missing_strict_actual_prerequisite_is_blocked(tmp_path):
 stage=tmp_path/'stage.json';stage.write_bytes(ce.encoded(dict(model_role='hf500')))
 provenance=dict(model_role='hf500',reviewed_stage_path=str(stage),reviewed_stage_sha256=ce.digest(stage))
 with pytest.raises(ValueError,match='strict actual training3007/HF500 verifier'):STRICT_STAGE_CHECK(provenance)

def test_stage_sha_and_runtime_provenance_are_not_output_self_attested(tmp_path):
 stage=tmp_path/'stage.json';stage.write_bytes(ce.encoded(dict(model_role='hf500')))
 provenance=dict(model_role='base',reviewed_stage_path=str(stage),reviewed_stage_sha256=ce.digest(stage))
 with pytest.raises(ValueError,match='Runtime provenance'):STRICT_STAGE_CHECK(provenance)
 provenance['model_role']='hf500';provenance['reviewed_stage_sha256']='0'*64
 with pytest.raises(ValueError,match='External evaluation stage'):STRICT_STAGE_CHECK(provenance)

def test_wrong_native_batch_shape_is_rejected(tmp_path,tokenizer):
 p,pin,m=dataset(tmp_path,65);data=batch(m,count=1);rows=batch_answer_metrics(data,tokenizer);reward=RewardManager(tokenizer,0)(data)
 writer=ce.ScoringViewWriter(tmp_path/'out',m,tokenizer,provenance(tmp_path,pin))
 with pytest.raises(ValueError,match='batch count'):writer.append(data,rows,reward)

def test_replay_refuses_rehashed_wrong_batch_count(tmp_path,tokenizer):
 p,pin,m,w,outpin,metrics=finish_capture(tmp_path,tokenizer);path=w.directory/'manifest.json';value=ce.loads(path.read_bytes());value['files'][0]['count']=63;path.write_bytes(ce.encoded(value))
 with pytest.raises(ValueError,match='batch shape/count'):ce.replay(w.directory,ce.digest(path),p,pin,tokenizer,full_final=False)

def test_replay_requires_actual_dispatch_guard_source(tmp_path,tokenizer):
 p,pin,m,w,outpin,metrics=finish_capture(tmp_path,tokenizer);path=w.directory/'manifest.json';value=ce.loads(path.read_bytes());value['zero_update_guard']['guard_source_sha256']='0'*64;path.write_bytes(ce.encoded(value))
 with pytest.raises(ValueError,match='dispatch guard'):ce.replay(w.directory,ce.digest(path),p,pin,tokenizer,full_final=False)

def test_cli_definitions_and_ray_environment_propagation():
 import ast
 tree=ast.parse(Path(ce.__file__).read_text());definitions={node.name:i for i,node in enumerate(tree.body) if isinstance(node,ast.FunctionDef)}
 cli=next(i for i,node in enumerate(tree.body) if isinstance(node,ast.If) and "__name__" in ast.unparse(node.test))
 assert all(index<cli for index in definitions.values())
 main_tree=ast.parse((Path(ce.__file__).parent/'main.py').read_text())
 assignments=[node for node in ast.walk(main_tree) if isinstance(node,ast.Assign)]
 assert any(ast.dump(node.targets[0])==ast.dump(ast.parse("audit_env['OPSD_CORRECTED_EVAL_STAGE_SHA256'] = None").body[0].targets[0]) and ast.dump(node.value)==ast.dump(ast.parse("os.environ['OPSD_CORRECTED_EVAL_STAGE_SHA256']",mode='eval').body) for node in assignments)
 calls=[node for node in ast.walk(main_tree) if isinstance(node,ast.Call)]
 assert any(isinstance(node.func,ast.Attribute) and node.func.attr=='options' and any(keyword.arg=='runtime_env' and ast.dump(keyword.value)==ast.dump(ast.parse("{'env_vars':audit_env}",mode='eval').body) for keyword in node.keywords) for node in calls)

def test_json_overflow_cannot_encode_nonfinite_evidence():
 with pytest.raises(ValueError):ce.loads('{"value":1e999}')

def test_resolved_budget_missing_fields_rejected():
 with pytest.raises(ValueError,match='Missing resolved'):ce.budget_snapshot(OmegaConf.create({}))

def test_cli_prepare_inputs_and_replay_missing_external_pin(tmp_path):
 import os,sys
 p,pin,m=dataset(tmp_path,1);prepared=tmp_path/'cli-inputs.json'
 env=dict(os.environ,PYTHONPATH=str(Path(ce.__file__).resolve().parents[2]))
 result=subprocess.run([sys.executable,'-m','search_r1.baselines.corrected_evaluation','prepare-inputs','--files',m['inputs'][0]['path'],'--output',str(prepared)],env=env,capture_output=True,text=True)
 assert result.returncode==0 and result.stdout.strip()==ce.digest(prepared)
 blocked=subprocess.run([sys.executable,'-m','search_r1.baselines.corrected_evaluation','replay','--directory',str(tmp_path)],env=env,capture_output=True,text=True)
 assert blocked.returncode!=0 and '--manifest-sha256' in blocked.stderr
