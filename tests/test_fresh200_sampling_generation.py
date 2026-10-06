"""CPU controller fixture using a small DataProto stand-in, never native admission."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS, ModuleType
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


class Proto:
    def __init__(self,batch,non_tensor_batch=None,meta_info=None):
        self.batch=batch; self.non_tensor_batch=non_tensor_batch or {}; self.meta_info=meta_info or {}
    @classmethod
    def from_dict(cls,batch,non_tensors=None,meta_info=None):
        return cls(batch,{k:np.asarray(v,dtype=object) for k,v in (non_tensors or {}).items()},meta_info)
    def __len__(self): return next(iter(self.batch.values())).shape[0]


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep production imports untouched; load only under a private fixture namespace.
        package=ModuleType('_fresh200_controller_fixture'); package.__path__=[]
        sys.modules[package.__name__]=package
        spec=importlib.util.spec_from_file_location(package.__name__+'.sampling_trace',ROOT/'search_r1/llm_agent/sampling_trace.py')
        trace=importlib.util.module_from_spec(spec); sys.modules[spec.name]=trace; spec.loader.exec_module(trace)
        spec=importlib.util.spec_from_file_location(package.__name__+'.sampling_capture',ROOT/'search_r1/llm_agent/sampling_capture.py')
        capture=importlib.util.module_from_spec(spec); sys.modules[spec.name]=capture; spec.loader.exec_module(capture)
        previous=sys.modules.get('verl'); sys.modules['verl']=NS(DataProto=Proto)
        try:
            spec=importlib.util.spec_from_file_location(package.__name__+'.sampling_tokens',ROOT/'search_r1/llm_agent/sampling_tokens.py')
            module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
        finally:
            if previous is None: del sys.modules['verl']
            else: sys.modules['verl']=previous
        cls.controller=module; cls.trace=trace; cls.capture=capture

    def manager(self,crop=False):
        trace=self.trace; capture=self.capture
        class Tokenizer:
            pad_token_id=0; pad_token='<pad>'; eos_token_id=0
            def decode(self,ids,**kwargs):
                mapping={2:'<search>x</search>',3:'<answer>y</answer>',0:'<eos>'}
                return ''.join(mapping[t] for t in ids)
            def encode(self,text,**kwargs): return [7,8] if text else []
        class Worker:
            def generate_sequences(self,call):
                rows=[]; raw=[]
                for i,rid in enumerate(call.non_tensor_batch['fresh200_row_id']):
                    request=call.batch['input_ids'][i][call.batch['attention_mask'][i].bool()].tolist()
                    tokens=[2] if call.meta_info['fresh200_turn_index']==0 else [3,0]
                    complete=NS(token_ids=tokens,logprobs=[{t:NS(logprob=-.1)} for t in tokens],finish_reason='stop',stop_reason=0)
                    output=NS(prompt_token_ids=request,request_id=str(i),outputs=[complete])
                    rows.extend(trace.capture_requests([output],[request],[rid],call.meta_info['policy_snapshot_id'],call.meta_info['fresh200_call_id'],call.meta_info['fresh200_turn_index'],{},0,0))
                    record=rows[-1]
                    sampling=dict(n=1,best_of=1,temperature=1.,top_p=1.,top_k=-1,min_p=0.,presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.,ignore_eos=False,min_tokens=0,max_tokens=10,stop=[],stop_token_ids=[],logprobs=1,truncate_prompt_tokens=None,logits_processors=[],logit_bias=None,allowed_token_ids=None,guided_decoding=None)
                    record['sampling']=sampling
                    proof=dict(request_id=record['request_id'],sequence_id=i,actual_request_sha256=trace.sha(request),caller_sampling_sha256=trace.sha(sampling),effective_sampling=sampling,effective_all_stop_token_ids=[0],sequence_eos_token_id=0,tokenizer_eos_token_id=0,tokenizer_pad_token_id=0,actual_loaded_generation_config_fields={'eos_token_id':0})
                    proof['sha256']=trace.sha(proof); record['effective_stop_provenance']=proof
                    record['native_sha256']=trace.sha({k:v for k,v in record.items() if k!='native_sha256'})
                    raw.append(tokens)
                return Proto.from_dict({'responses':torch.tensor(raw),'sampling_log_probs':torch.full((len(raw),len(raw[0])),-.1)},non_tensors={'fresh200_sampling':self_objects(rows)})
        def execute(texts,pad,active,do_search):
            searching=['<search>' in text for text in texts]
            return ['retrieval' if s else '' for s in searching],[not s for s in searching],[1]*len(texts),searching
        return NS(config=NS(no_think_rl=False,num_gpus=2,max_start_length=5,max_prompt_length=4 if crop else 100,max_obs_length=10,max_turns=2,force_final_answer=False),tokenizer=Tokenizer(),actor_rollout_wg=Worker(),execute_predictions=execute)

    def input(self):
        return Proto.from_dict({'input_ids':torch.tensor([[0,10]]),'attention_mask':torch.tensor([[0,1]]),'position_ids':torch.tensor([[0,0]])},non_tensors={'fresh200_row_id':['row']},meta_info={'preserve_sampling_tokens':True,'policy_snapshot_id':'snapshot'})

    def test_raw_multi_turn_retrieval_and_padding_identity(self):
        data=self.input(); result=self.controller.run(self.manager(),data,data.batch['input_ids'])
        self.assertEqual(result.batch['responses'].tolist(),[[2,7,8,3]])
        self.assertEqual(result.batch['sampling_mask'].tolist(),[[True,False,False,True]])
        turns=result.non_tensor_batch['fresh200_turns'][0]
        self.assertEqual(turns[1]['raw_generated_token_ids'],[3,0]) # EOS retained in raw journal.
        self.assertEqual(turns[1]['actual_request_token_ids'],[10,2,7,8])
        self.assertTrue(self.controller.audit_batch(result)['context_verified'])
        self.assertFalse(result.meta_info['sampling_context_audit']['native_probability_verified'])
        result.batch['input_ids'][0,-1]=99
        with self.assertRaises(ValueError): self.controller.audit_batch(result)

    def test_context_crop_rejected(self):
        data=self.input(); manager=self.manager(crop=True); manager.config.max_prompt_length=3
        with self.assertRaisesRegex(ValueError,'truncated|cropped'):
            self.controller.run(manager,data,data.batch['input_ids'])


def self_objects(values):
    result=np.empty(len(values),dtype=object); result[:]=values
    return result

if __name__=='__main__': unittest.main()
