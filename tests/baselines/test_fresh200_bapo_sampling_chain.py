"""Real Controller -> SearchRollout -> DataProto CPU fixture; no native CUDA admission."""
from contextlib import contextmanager
import copy
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import pytest
import torch
from verl import DataProto
from search_r1.baselines.controller import Controller
from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
from search_r1.llm_agent.sampling_trace import capture_requests, sha
from search_r1.llm_agent.sampling_capture import capture_groups, neutral_sampler, sampler_values
from search_r1.llm_agent.sampling_tokens import audit_batch


def params():
    return NS(n=1,best_of=1,temperature=1.,top_p=1.,top_k=-1,min_p=0.,presence_penalty=0.,
              frequency_penalty=0.,repetition_penalty=1.,ignore_eos=False,min_tokens=0,max_tokens=10,
              stop=[],stop_token_ids=[],logprobs=1,truncate_prompt_tokens=None,logits_processors=[],
              logit_bias=None,allowed_token_ids=None,guided_decoding=None)


class Tokenizer:
    pad_token_id=0;eos_token_id=0;pad_token='<pad>'
    def decode(self,ids,**kwargs):
        return ''.join({2:'<search>x</search>',3:'<answer>y</answer>',0:'<eos>',4:'<special>'}[int(i)] for i in ids)
    def encode(self,text,**kwargs):return [7,8] if text else []


class Engine:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer;self.generation_config_fields={'eos_token_id':[0,9]};self.scheduler=[NS(waiting=[])]
    def add_request(self,request_id,request,caller):
        effective=copy.deepcopy(caller);effective.stop_token_ids=[9];effective.all_stop_token_ids={0,9}
        seq=NS(seq_id=len(self.scheduler[0].waiting),eos_token_id=0)
        group=NS(request_id=request_id,prompt_token_ids=request,sampling_params=effective,get_seqs=lambda:[seq])
        self.scheduler[0].waiting.append(group)


class Worker:
    def __init__(self,tokenizer):self.tokenizer=tokenizer;self.calls=[]
    def generate_sequences(self,call):
        requests=[row[mask.bool()].tolist() for row,mask in zip(call.batch['input_ids'],call.batch['attention_mask'])]
        self.calls.append(requests)
        evaluation=call.meta_info.get('fresh200_evaluation') is True
        ids=call.non_tensor_batch['fresh200_row_id'].tolist();caller=params();engine=Engine(self.tokenizer)
        caller.temperature=0. if evaluation else 1.
        native=[];raw=[]
        with capture_groups(engine,caller,requests,evaluation=evaluation) as (groups,values):
            for i,request in enumerate(requests):
                request_id=str(i);engine.add_request(request_id,request,caller)
                tokens=[2] if call.meta_info['fresh200_turn_index']==0 else [4,0,3,0]
                completion=NS(token_ids=tokens,logprobs=[{t:NS(logprob=-.125)} for t in tokens],finish_reason='stop',stop_reason=0)
                native.append(NS(prompt_token_ids=request,request_id=request_id,outputs=[completion]));raw.append(tokens)
            records=capture_requests(native,requests,ids,call.meta_info['policy_snapshot_id'],call.meta_info['fresh200_call_id'],call.meta_info['fresh200_turn_index'],values,0,0)
            for record in records:
                record['effective_stop_provenance']=groups[record['request_id']]
                record['native_sha256']=sha({k:v for k,v in record.items() if k!='native_sha256'})
        objects=np.empty(len(records),dtype=object);objects[:]=records
        return DataProto.from_dict({'responses':torch.tensor(raw),'sampling_log_probs':torch.full((len(raw),len(raw[0])),-.125)},
            non_tensors={'fresh200_sampling':objects},meta_info={'preserve_sampling_tokens':True})


def fixture(early=True):
    tokenizer=Tokenizer();worker=Worker(tokenizer)
    config=GenerationConfig(2,5,100,10,10,2,preserve_sampling_tokens=True,force_final_answer=False)
    manager=LLMGenerationManager(tokenizer,worker,config)
    def execute(predictions,pad,active,do_search=True):
        searches=['<search>' in p for p in predictions]
        return ['retrieval' if s else '' for s in searches],[not s for s in searches],[1]*len(predictions),searches
    manager.execute_predictions=execute
    controller=Controller.__new__(Controller)
    controller.config=NS(gigpo=NS(anchor_mode='observation'),seed=42,context_alignment='error',bapo=NS(resample_attempts=2))
    controller.method='bapo';controller.ranker=None;controller.stage=NS(early=early);controller.trainer=NS(tokenizer=tokenizer)
    gen=DataProto.from_dict({'input_ids':torch.tensor([[0,10],[0,11]]),'attention_mask':torch.tensor([[0,1],[0,1]]),'position_ids':torch.tensor([[0,0],[0,0]])},
        non_tensors={'fresh200_row_id':['row0','row1']},meta_info={'preserve_sampling_tokens':True,'policy_snapshot_id':'snapshot'})
    qa=DataProto.from_dict({'dummy':torch.zeros(2,1)},non_tensors={'index':['q0','q1'],'reward_model':[{'ground_truth':{'target':'y'}}]*2})
    return controller,manager,worker,gen,qa


def test_actual_bapo_early_path_keeps_raw_tokens_masks_and_cost():
    controller,manager,worker,gen,qa=fixture();metrics={}
    result,sources=controller.generate(manager,gen,qa,metrics)
    assert sources.tolist()==[0,1]
    assert result.batch['responses'].tolist()==[[2,7,8,4,0,3]]*2
    assert result.batch['sampling_mask'].tolist()==[[True,False,False,True,True,True]]*2
    assert result.batch['attention_mask'][:,-6:].tolist()==[[1]*6]*2
    assert worker.calls[1]==[[10,2,7,8],[11,2,7,8]]
    assert metrics['rollout/generated_tokens']==10 and metrics['rollout/retrieval_calls']==2
    assert audit_batch(result)['context_verified']
    from search_r1.baselines.rollout import select_rows
    reordered=select_rows(result,[1,0])
    assert reordered.non_tensor_batch['fresh200_row_id'].tolist()==['row1','row0']
    assert audit_batch(reordered)['context_verified']


def test_bapo_late_resampling_retains_source_journal_and_stage():
    controller,manager,worker,gen,qa=fixture(early=False);metrics={}
    with patch('search_r1.baselines.controller.decode_answers',return_value=['y','y']),patch('search_r1.baselines.controller.resample_rows',side_effect=[[1],[]]):
        result,sources=controller.generate(manager,gen,qa,metrics)
    assert sources.tolist()==[0,1] and controller.stage.early is False
    assert result.non_tensor_batch['fresh200_row_id'].tolist()==['row0','row1']
    assert result.non_tensor_batch['fresh200_turns'][1][0]['actual_request_token_ids']==[11]
    assert worker.calls[2]==[[11],[11]] # actual synthetic padding included.
    assert metrics['rollout/trajectories']==3 and metrics['rollout/padded_rows']==2
    assert metrics['rollout/generated_tokens']==20 # retained plus padded resample generations.
    assert audit_batch(result)['context_verified']


def test_effective_group_transform_mismatch_rejects_and_restores():
    engine=Engine(Tokenizer());caller=params()
    caller.top_p=.9
    with pytest.raises(ValueError,match='sampling transformation'):
        with capture_groups(engine,caller,[[10]]):pass
    assert 'add_request' not in vars(engine)
    caller=params()
    native=engine.add_request
    def bad(*args,**kwargs):
        native(*args,**kwargs);engine.scheduler[0].waiting[-1].sampling_params.top_k=5
    engine.add_request=bad
    with pytest.raises(ValueError,match='sampling transformation'):
        with capture_groups(engine,caller,[[10]]):engine.add_request('id',[10],caller)
    assert engine.add_request is bad


def test_effective_stop_set_mismatch_rejects():
    engine=Engine(Tokenizer());native=engine.add_request
    def bad(*args,**kwargs):
        native(*args,**kwargs);engine.scheduler[0].waiting[-1].sampling_params.all_stop_token_ids={0}
    engine.add_request=bad
    with pytest.raises(ValueError,match='effective stop set'):
        with capture_groups(engine,params(),[[10]]):engine.add_request('id',[10],params())


def test_real_chain_cropped_context_is_rejected_before_second_sample():
    controller,manager,worker,gen,qa=fixture();manager.config.max_prompt_length=3
    with pytest.raises(ValueError,match='cropped'):
        controller.generate(manager,gen,qa,{})
    assert len(worker.calls)==1


def test_native_stop_proof_required_before_actor_context_admission():
    controller,manager,worker,gen,qa=fixture();result,_=controller.generate(manager,gen,qa,{})
    turn=result.non_tensor_batch['fresh200_turns'][0][0]
    del turn['effective_stop_provenance']
    with pytest.raises(ValueError,match='effective group'):
        audit_batch(result)


def test_actual_validation_raw_greedy_preserves_eos_equal_pad_cost_and_context():
    from search_r1.baselines.bapo_validation import BapoValidationManager
    _,original,worker,gen,_=fixture()
    gen.meta_info.update(fresh200_evaluation=True,do_sample=False)
    manager=BapoValidationManager(original.tokenizer,worker,original.config,True)
    manager.execute_predictions=original.execute_predictions
    with patch('search_r1.baselines.bapo_validation.validation_answer_counts',return_value=dict(idk_count=0,answered_count=2,invalid_count=0)):
        output=manager.run_llm_loop(gen,gen.batch['input_ids'])
    assert output.batch['responses'].tolist()==[[2,7,8,4,0,3]]*2
    assert manager.cost['generated_tokens']==10 # all raw native tokens, including EOS==pad and trimmed EOS.
    assert worker.calls[1]==[[10,2,7,8],[11,2,7,8]]
    assert audit_batch(output)['probability_role']=='greedy_model_logprob_not_sampling_policy'
    output.meta_info.pop('fresh200_evaluation')
    with pytest.raises(ValueError,match='probability role'):
        audit_batch(output)


def test_greedy_capture_cannot_be_admitted_as_stochastic_policy():
    caller=params();caller.temperature=0.
    with pytest.raises(ValueError,match='sampling transformation'):
        with capture_groups(Engine(Tokenizer()),caller,[[10]]): pass
    caller.temperature=1.
    with pytest.raises(ValueError,match='sampling transformation'):
        with capture_groups(Engine(Tokenizer()),caller,[[10]],evaluation=True): pass


@pytest.mark.parametrize('rows',[2,64,512])
def test_actual_validate_pop_with4096_padding_preserves_full_initial_tensor(rows):
    # Actual3300 caller: dataset width4096 vs max_start2048. No native CUDA claim.
    from omegaconf import open_dict
    from test_models_and_driver import config
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from search_r1.baselines.bapo_validation import BapoValidationManager
    _,original,worker,_,_=fixture()
    c=config('bapo')
    with open_dict(c.trainer):c.trainer.fresh200_enabled=True
    c.trainer.n_gpus_per_node=2;c.trainer.nnodes=1
    c.data.max_start_length=2048;c.data.max_prompt_length=4096
    c.max_turns=2;c.force_final_answer=False
    t=RayPPOTrainer.__new__(RayPPOTrainer);t.config=c;t.tokenizer=original.tokenizer
    t.actor_rollout_wg=worker;t.baseline_controller=NS(method='bapo',ranker=None)
    ids=torch.zeros((rows,4096),dtype=torch.long);mask=torch.zeros_like(ids)
    # Include a genuine token equal to EOS/pad in attended context; cannot infer mask from tokens.
    ids[:,-3:]=torch.tensor([10,0,11]);mask[:,-3:]=1
    t.val_dataloader=[dict(input_ids=ids,attention_mask=mask,position_ids=(mask.cumsum(-1)-1)*mask,
        reward_model=np.array([{'ground_truth':{'target':'y'}}]*rows,dtype=object),data_source=np.array(['nq']*rows,dtype=object))]
    seen=[]
    def reward(data):
        seen.append(data);audit_batch(data)
        return torch.zeros_like(data.batch['responses'],dtype=torch.float32)
    t.val_reward_fn=reward
    with patch.object(BapoValidationManager,'execute_predictions',side_effect=original.execute_predictions),patch('search_r1.baselines.evaluation.batch_answer_metrics',return_value=[]):
        metrics=t._validate()
    assert metrics['val/evaluated_count/nq']==rows and metrics['val/cost/generated_tokens']==5*rows
    assert worker.calls[0]==[[10,0,11]]*rows
    assert worker.calls[1]==[[10,0,11,2,7,8]]*rows
    assert seen[0].batch['prompts'].tolist()==[[10,0,11]]*rows
    assert seen[0].batch['attention_mask'][0].tolist()==[1]*9
    assert seen[0].batch['attention_mask'][-1].tolist()==[1]*9
    assert audit_batch(seen[0])['probability_role']=='greedy_model_logprob_not_sampling_policy'


def test_full_padded_initial_raw_loop_rejects_real_overlong_context_before_worker():
    _,manager,worker,gen,_=fixture()
    ids=torch.full((2,20),10,dtype=torch.long)
    gen.batch['input_ids']=ids;gen.batch['attention_mask']=torch.ones_like(ids);gen.batch['position_ids']=torch.arange(20).repeat(2,1)
    with pytest.raises(ValueError,match='initial context would be cropped'):
        manager.run_llm_loop(gen,ids)
    assert not worker.calls


def test_initial_shape_guard_remains_strict_for_sliced_view():
    _,manager,worker,gen,_=fixture()
    with pytest.raises(ValueError,match='initial prompt mismatch'):
        manager.run_llm_loop(gen,gen.batch['input_ids'][:,-1:])
    assert not worker.calls
