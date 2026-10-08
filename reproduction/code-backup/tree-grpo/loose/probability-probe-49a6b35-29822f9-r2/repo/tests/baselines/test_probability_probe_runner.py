"""CPU/source fixtures; these do not exercise actual HF110 or CUDA/FSDP."""
import ast
import copy
import importlib.util
import json
import os
from datetime import timedelta
from pathlib import Path
import socket
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from search_r1.baselines import probability_probe_loader as loader
from search_r1.baselines import probability_probe_runner as runner
from search_r1.baselines.probability_probe_capture import file_sha,tensor_digest


class SmallTiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=torch.nn.Module();self.model.embed_tokens=torch.nn.Embedding(7,3)
        self.lm_head=torch.nn.Linear(3,7,bias=False);self.lm_head.weight=self.model.embed_tokens.weight
    def forward(self,value):return self.lm_head(self.model.embed_tokens(value))


def fixture_data():
    ids=torch.tensor([[1,2,3,4],[2,3,4,5]])
    return dict(input_ids=ids,position_ids=torch.arange(4).expand(2,4),attention_mask=torch.ones_like(ids),
        responses=ids[:,-2:],old_log_probs=torch.tensor([[-.2,-.3],[-.4,-.5]]),
        sampled_log_probs=torch.tensor([[-.1,-.2],[-.3,-.4]]))


class LoaderTests(unittest.TestCase):
    def test_original_source_ast_boundary_is_exact_and_optimizer_tail_omitted(self):
        path=ROOT/'verl/workers/fsdp_workers.py'
        before,after,receipt=loader.split_original_constructor(path,file_sha(path))
        original=next(n for n in next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef)
            and n.name=='ActorRolloutRefWorker').body if isinstance(n,ast.FunctionDef) and n.name=='_build_model_optimizer')
        i=next(i for i,n in enumerate(original.body) if n.lineno==receipt['original_barrier_line'])
        j=next(i for i,n in enumerate(original.body) if n.lineno==receipt['omitted_optimizer_conditional_line'])
        self.assertEqual([ast.dump(n,include_attributes=False) for n in before.body[0].body[:-1]],
            [ast.dump(n,include_attributes=False) for n in original.body[:i]])
        self.assertEqual([ast.dump(n,include_attributes=False) for n in after.body],
            [ast.dump(n,include_attributes=False) for n in original.body[i:j]])
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='AdamW' for n in ast.walk(after)))

    def test_changed_source_sha_refused(self):
        with self.assertRaisesRegex(ValueError,'SHA changed'):
            loader.split_original_constructor(ROOT/'verl/workers/fsdp_workers.py','0'*64)

    def test_ambiguous_constructor_boundary_refused(self):
        value='class ActorRolloutRefWorker:\n def _build_model_optimizer(self):\n  torch.distributed.barrier()\n  torch.distributed.barrier()\n  if self._is_actor:\n   optim.AdamW(x)\n'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.py';path.write_text(value)
            with self.assertRaisesRegex(ValueError,'boundary'):loader.split_original_constructor(path,file_sha(path))

    def test_kept_training_call_refused(self):
        value='class ActorRolloutRefWorker:\n def _build_model_optimizer(self):\n  x.backward()\n  torch.distributed.barrier()\n  if self._is_actor:\n   optim.AdamW(x)\n'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.py';path.write_text(value)
            with self.assertRaisesRegex(ValueError,'forbidden'):loader.split_original_constructor(path,file_sha(path))

    def test_same_window_alias_real_live_cpu_tensors(self):
        model=SmallTiedModel()
        with loader.SameWindowTiedAliasTrace(model,model.model.embed_tokens,model.lm_head) as trace:
            model(torch.tensor([1,2]))
        self.assertTrue(trace.receipt()['tied_alias_identity_proven'])
        self.assertFalse(trace.receipt()['historical_3268_identity_proven'])

    def test_same_window_two_equal_copies_are_not_aliases(self):
        model=SmallTiedModel();model.lm_head.weight=torch.nn.Parameter(model.model.embed_tokens.weight.detach().clone())
        calls=[];handle=model.lm_head.register_forward_hook(lambda *args:calls.append(True))
        trace=loader.SameWindowTiedAliasTrace(model,model.model.embed_tokens,model.lm_head)
        with self.assertRaisesRegex(ValueError,'alias evidence'):
            with trace:model(torch.tensor([1,2]))
        handle.remove();self.assertEqual(calls,[True]) # refusal deferred past native model completion
        with self.assertRaisesRegex(ValueError,'failed/incomplete'):trace.receipt()

    def test_alias_trace_native_exception_cannot_publish(self):
        model=SmallTiedModel();trace=loader.SameWindowTiedAliasTrace(model,model.model.embed_tokens,model.lm_head)
        with self.assertRaisesRegex(RuntimeError,'native'):
            with trace:
                model(torch.tensor([1]));raise RuntimeError('native failure')
        with self.assertRaisesRegex(ValueError,'failed/incomplete'):trace.receipt()
        self.assertEqual(len(model._forward_pre_hooks),0)

    def test_alias_trace_missing_original_call_refuses(self):
        model=SmallTiedModel();trace=loader.SameWindowTiedAliasTrace(model,model.model.embed_tokens,model.lm_head)
        with self.assertRaisesRegex(ValueError,'absent/failed'):
            with trace:pass
        with self.assertRaisesRegex(ValueError,'failed/incomplete'):trace.receipt()

    def synthetic_hf(self,change=None,extra=False):
        model=SmallTiedModel();saved={name:value.detach().clone() for name,value in model.named_parameters(remove_duplicate=False)}
        if change:saved[change][0,0]+=1
        if extra:saved['extra']=torch.ones(1)
        class Shard:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def keys(self):return list(saved)
            def get_tensor(self,name):return saved[name]
        fake=ModuleType('safetensors');fake.safe_open=lambda *a,**k:Shard()
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'safetensors':fake}):
            (Path(tmp)/'fake.safetensors').touch()
            index={'weight_map':{name:'fake.safetensors' for name,_ in model.named_parameters(remove_duplicate=False)}}
            return loader.exhaustive_hf_cast_inventory(model,tmp,index,{'fake.safetensors':file_sha(Path(tmp)/'fake.safetensors')},2,1)

    def test_every_serialized_tied_key_is_read_unique_count_separate(self):
        targets,receipt=self.synthetic_hf()
        self.assertEqual((receipt['serialized_count'],receipt['unique_parameter_count']),(2,1))
        self.assertEqual({v['logical_name'] for v in targets},{'model.embed_tokens.weight','lm_head.weight'})
        self.assertEqual(len(set(receipt['serialized_to_unique'].values())),1)
        self.assertTrue(all(v['expected_cast_digest']['dtype']=='torch.bfloat16' for v in targets))

    def test_inconsistent_serialized_tied_value_refuses(self):
        with self.assertRaisesRegex(ValueError,'master differs'):self.synthetic_hf(change='lm_head.weight')

    def test_extra_serialized_shard_key_refuses(self):
        with self.assertRaisesRegex(ValueError,'missing/unexpected'):self.synthetic_hf(extra=True)


class RunnerTests(unittest.TestCase):
    def test_config_consumer_uses_exact_original_text_not_path(self):
        import hashlib
        text='actor_rollout_ref:\n  actor:\n    ppo_mini_batch_size: 384\n'
        bundle=dict(config_path='/no/path/open/allowed',config_text=text,
            identity=dict(config_sha256=hashlib.sha256(text.encode()).hexdigest()))
        fake=ModuleType('omegaconf');seen=[]
        def create(value):seen.append(value);return value
        fake.OmegaConf=SimpleNamespace(create=create,load=lambda *args:(_ for _ in ()).throw(AssertionError('path reopened')))
        with patch.dict(sys.modules,{'omegaconf':fake}):
            config=runner.verified_bundle_config(bundle)
        self.assertEqual(config,text);self.assertEqual(seen,[text])
        bundle['config_text']=text.replace('384','192')
        with patch.dict(sys.modules,{'omegaconf':fake}),self.assertRaisesRegex(ValueError,'frozen original bytes'):runner.verified_bundle_config(bundle)
        bundle['config_text']=None
        with patch.dict(sys.modules,{'omegaconf':fake}),self.assertRaisesRegex(ValueError,'text is required'):runner.verified_bundle_config(bundle)

    def test_raw_mask_is_new_explicit_allones_and_original_bytes_preserved(self):
        data=fixture_data();before=runner.input_identity(data)
        normalized=runner.canonical_response_data(data,'raw')
        self.assertTrue(normalized['loss_mask'].all());self.assertNotIn('loss_mask',data)
        self.assertEqual(before,runner.input_identity(data))

    def test_monitor_fullwidth_masks_use_response_suffix_without_changing_input(self):
        data=fixture_data();data['loss_mask']=torch.tensor([[1,1,0,1],[1,1,1,0]])
        data['policy_weights']=torch.full((2,4),.25)
        normalized=runner.canonical_response_data(data,'monitor')
        self.assertTrue(torch.equal(normalized['loss_mask'],torch.tensor([[0,1],[1,0]])))
        self.assertEqual(data['loss_mask'].shape,(2,4))

    def test_nonfinite_comparison_refused(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite'):runner.probability_statistics(torch.tensor([float('nan')]))

    def test_tree_weighted_numerator_denominator_are_not_unweighted_mean(self):
        result=runner.probability_statistics(torch.tensor([.1,.3]),torch.tensor([1.,3.]))
        self.assertAlmostEqual(result['tree_weight_denominator'],4.)
        self.assertAlmostEqual(result['tree_weighted_mean_abs'],.25,places=6)
        self.assertAlmostEqual(result['mean_abs'],.2,places=6)

    def test_selected_full_micro_row_buffers_and_order_preserved(self):
        data=runner.canonical_response_data(fixture_data(),'raw')
        selected=runner.prepare_native_inputs(data,[1,0],'cpu')
        self.assertTrue(torch.equal(selected['input_ids'],data['input_ids'][[1,0]]))
        self.assertEqual(selected['input_ids'].shape[1],data['input_ids'].shape[1])

    def test_invalid_masked_predecessor_rejected_before_native_call(self):
        data=runner.canonical_response_data(fixture_data(),'raw');data['attention_mask'][0,1]=0
        with self.assertRaisesRegex(ValueError,'prediction position'):runner.prepare_native_inputs(data,[0,1],'cpu')

    def test_complete_state_records_modes_buffers_and_rejects_new_gradient(self):
        model=SmallTiedModel();model.register_buffer('rotary_fixture',torch.arange(3.))
        state=runner.module_state(model);model.rotary_fixture[1]+=1
        self.assertNotEqual(state['buffers'],runner.module_state(model)['buffers'])
        model.model.embed_tokens.weight.grad=torch.ones_like(model.model.embed_tokens.weight)
        with self.assertRaisesRegex(ValueError,'Gradient created'):runner.module_state(model)

    def test_dropout_refused_even_eval_no_grad(self):
        model=torch.nn.Sequential(torch.nn.Dropout(.1));model.eval()
        with torch.no_grad(),self.assertRaisesRegex(ValueError,'Nonzero dropout'):runner.module_state(model)

    def test_actual_style_lazy_private_creation_only_same_master_is_allowed(self):
        model=SmallTiedModel();before=runner.module_state(model)['parameters']
        model.model.embed_tokens.weight._local_shard=model.model.embed_tokens.weight.data
        after=runner.module_state(model)['parameters']
        lazy=runner.verify_lazy_private_shards(before,after)
        self.assertEqual(len(lazy),2) # Both serialized tied names, one unique parameter.
        self.assertTrue(all(record['before'] is None for record in lazy.values()))
        self.assertTrue(all(record['after']['full_shape_dtype_bytes_sha256']==after[name]['full_shape_dtype_bytes_sha256'] for name,record in lazy.items()))

    def test_lazy_private_creation_with_wrong_bytes_refused(self):
        model=SmallTiedModel();before=runner.module_state(model)['parameters']
        model.model.embed_tokens.weight._local_shard=model.model.embed_tokens.weight.data+1
        with self.assertRaisesRegex(ValueError,'private shard differs'):
            runner.verify_lazy_private_shards(before,runner.module_state(model)['parameters'])

    def test_lazy_initialization_cannot_mask_public_master_update(self):
        model=SmallTiedModel();before=runner.module_state(model)['parameters']
        with torch.no_grad():model.model.embed_tokens.weight.add_(1)
        model.model.embed_tokens.weight._local_shard=model.model.embed_tokens.weight.data
        with self.assertRaisesRegex(ValueError,'public master bytes'):
            runner.verify_lazy_private_shards(before,runner.module_state(model)['parameters'])

    def test_existing_private_shard_change_or_deletion_refused(self):
        model=SmallTiedModel();param=model.model.embed_tokens.weight;param._local_shard=param.data.clone()
        before=runner.module_state(model)['parameters'];param._local_shard+=1
        with self.assertRaisesRegex(ValueError,'already-existing'):
            runner.verify_lazy_private_shards(before,runner.module_state(model)['parameters'])
        del param._local_shard
        with self.assertRaisesRegex(ValueError,'already-existing'):
            runner.verify_lazy_private_shards(before,runner.module_state(model)['parameters'])

    def test_global_refusal_deferred_to_matched_max(self):
        events=[]
        def reduce(flag,op):events.append(('max',int(flag)));flag.fill_(1)
        def operation():events.append(('operation',True));raise ValueError('rank-local evidence')
        with patch.object(runner.dist,'all_reduce',reduce):
            with self.assertRaisesRegex(runner.GlobalDiagnosticRefusal,'rank-local evidence'):
                runner.RankCoordinator('cpu').local_stage('stage',operation)
        self.assertEqual(events,[('operation',True),('max',1)])

    def test_peer_local_success_also_refused_on_global_flag(self):
        with patch.object(runner.dist,'all_reduce',lambda flag,op:flag.fill_(1)):
            with self.assertRaisesRegex(runner.GlobalDiagnosticRefusal,'local evidence: None'):
                runner.RankCoordinator('cpu').local_stage('stage',lambda:torch.tensor(1.))

    def test_same_logits_original_values_saved_and_duplicate_metadata_explicit(self):
        data=runner.canonical_response_data(fixture_data(),'raw')
        mapping=dict(rows=torch.tensor([0,0,1,1]),response_positions=torch.tensor([0,1,0,1]),
            logit_indices=torch.tensor([1,2,5,6]),labels=torch.tensor([3,4,4,5]))
        values=torch.tensor([-.2,-.3,-.4,-.5],dtype=torch.float64)
        tensors,evidence=runner.build_comparison_evidence(dict(token_values={'original_actor':values},metadata={}),
            dict(mapping=mapping),data,dict(rows=[0,1],exclude_duplicate=True),
            dict(kind='raw',asset='raw',source_group=0),{},4,0,0)
        self.assertTrue(torch.equal(tensors['original_old_log_probs'],data['old_log_probs'].flatten().double()))
        self.assertTrue(evidence['duplicate_excluded_from_occurrence_totals'])
        self.assertEqual(evidence['metrics']['original_actor_vs_actual_sampled']['physical_effective_token_occurrences'],4)

    def test_monitor_missing_companion_refuses_instead_of_position_alignment(self):
        data=fixture_data();data['loss_mask']=torch.ones(2,2);data['policy_weights']=torch.ones(2,2)
        data=runner.canonical_response_data(data,'monitor')
        mapping=dict(rows=torch.tensor([0]),response_positions=torch.tensor([0]),logit_indices=torch.tensor([1]),labels=torch.tensor([3]))
        with self.assertRaisesRegex(ValueError,'UNIMPLEMENTED'):
            runner.build_comparison_evidence(dict(token_values={'original_actor':torch.tensor([-.2])},metadata={}),
                dict(mapping=mapping),data,dict(rows=[0,1],exclude_duplicate=False),
                dict(kind='monitor',asset='monitor',source_group=0),{},4,0,0)

    def test_main_default_unreleased_bundle_cannot_initialize_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bundle.json';path.write_text(json.dumps({'status':'SOURCE_ONLY'}))
            with patch.object(sys,'argv',['probe','--bundle',str(path)]),patch.object(runner.dist,'init_process_group') as pg:
                with self.assertRaisesRegex(ValueError,'UNIMPLEMENTED'):runner.main()
                pg.assert_not_called()

    def test_actual_two_rank_loopback_gloo_global_refusal_no_rank_continues(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            torch.multiprocessing.spawn(gloo_refusal_worker,args=(port,tmp),nprocs=2,join=True)
            results=[json.loads((Path(tmp)/('rank%d.json'%rank)).read_text()) for rank in range(2)]
            self.assertEqual([r['status'] for r in results],['globally_refused','globally_refused'])
            self.assertTrue(all(r['native_completed'] and not r['next_native_started'] for r in results))
            self.assertEqual([r['sync_sequence'] for r in results],[1,1])


def gloo_refusal_worker(rank,port,out):
    # Actual CPU collectives, explicitly not an eight-rank CUDA/FSDP receipt.
    runner.dist.init_process_group('gloo',init_method='tcp://127.0.0.1:%d'%port,rank=rank,world_size=2,
        timeout=timedelta(seconds=15))
    state=dict(native_completed=False,next_native_started=False,status='unknown')
    coordinator=runner.RankCoordinator('cpu')
    def completed_native_then_local_evidence():
        state['native_completed']=True
        if rank==1:raise ValueError('synthetic rank1 post-forward evidence refusal')
        return 1
    try:
        coordinator.local_stage('matched post-native stage',completed_native_then_local_evidence)
        state['next_native_started']=True
    except runner.GlobalDiagnosticRefusal:state['status']='globally_refused'
    finally:
        state['sync_sequence']=coordinator.sequence
        (Path(out)/('rank%d.json'%rank)).write_text(json.dumps(state))
        runner.dist.destroy_process_group()


if __name__=='__main__':unittest.main()
