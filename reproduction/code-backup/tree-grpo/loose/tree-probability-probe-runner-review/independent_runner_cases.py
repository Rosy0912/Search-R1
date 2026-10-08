"""Tiny CPU runner counterexamples; original candidate stays unchanged."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'work/tree-probability-probe-runner'
sys.path.insert(0,str(REPO))
from search_r1.baselines import probability_probe_runner as R
from search_r1.baselines import probability_probe_capture as C
from search_r1.baselines import probability_probe_monitor as M


def raw(value=-1.):
    ids=torch.tensor([[1,2,3,4]])
    return dict(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.tensor([[0,1,2,3]]),
        responses=ids[:,-2:],old_log_probs=torch.full((1,2),value,dtype=torch.float64),
        sampled_log_probs=torch.full((1,2),-1.,dtype=torch.float64))


def captured():
    return dict(forward_kwargs=('input_ids','position_ids','attention_mask','use_cache'),
        input_shapes={'input_ids':[1,4]},input_dtypes={'input_ids':'torch.int64'},model_training=False)


def reproduce_load_assets_path_swap():
    path=REPO/'search_r1/baselines/probability_probe_runner.py'
    tree=ast.parse(path.read_text())
    parent=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_probe')
    function=next(n for n in parent.body if isinstance(n,ast.FunctionDef) and n.name=='load_assets')
    original_ast_sha=hashlib.sha256(ast.dump(function,include_attributes=False).encode()).hexdigest()
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'toy.pt';torch.save(raw(-1.),p);good=p.read_bytes();expected=hashlib.sha256(good).hexdigest()
        q=Path(t)/'other.pt';torch.save(raw(-99.),q);bad=q.read_bytes();calls=[]
        def racing_sha(asset):
            calls.append(str(asset))
            if len(calls)==1:
                answer=hashlib.sha256(p.read_bytes()).hexdigest();p.write_bytes(bad);return answer
            p.write_bytes(good);return hashlib.sha256(p.read_bytes()).hexdigest()
        data={}
        ns=dict(selection=dict(raw_groups=[dict(asset='raw')],monitor=dict(asset='raw')),
            local_data=data,bundle={'inventory':{'raw':{'path':str(p),'sha256':expected,'kind':'raw'}}},
            file_sha=racing_sha,require=R.require,torch=torch,canonical_response_data=R.canonical_response_data,
            bind_monitor_companion=R.bind_monitor_companion,rank=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(path),'exec'),ns)
        ns['load_assets']()
        return dict(both_asset_hash_checks_passed=True,final_asset_hash_is_expected=hashlib.sha256(p.read_bytes()).hexdigest()==expected,
            expected_loaded_old_log_probs=[[-1.,-1.]],actual_loaded_old_log_probs=data['raw']['old_log_probs'].tolist(),
            original_nested_function_ast_sha256=original_ast_sha,source_body_changed=False)


class IndependentRunnerCases(unittest.TestCase):
    def test_path_swap_between_hashes_accepts_wrong_tensor_bytes(self):
        result=reproduce_load_assets_path_swap()
        self.assertTrue(result['both_asset_hash_checks_passed'] and result['final_asset_hash_is_expected'])
        self.assertEqual(result['actual_loaded_old_log_probs'],[[-99.,-99.]])
        self.assertNotEqual(result['actual_loaded_old_log_probs'],result['expected_loaded_old_log_probs'])

    def test_input_mutation_refused_after_call(self):
        data=raw();events=[]
        def native(actor,original,temp):original['input_ids'][0,0]=10;events.append('native complete');return captured()
        with patch.object(R,'capture_original_call',side_effect=native):
            with self.assertRaisesRegex(ValueError,'changed input'):R.immutable_native_call(None,data,1.,None)
        self.assertEqual(events,['native complete'])

    def test_cpu_rng_restored_after_fake_native_consumption(self):
        data=raw();before=torch.get_rng_state().clone()
        def native(*args):torch.rand(3);return captured()
        with patch.object(R,'capture_original_call',side_effect=native):_,receipt=R.immutable_native_call(None,data,1.,None)
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(receipt['rng_unchanged'])

    def test_native_failure_cannot_return_receipt(self):
        with patch.object(R,'capture_original_call',side_effect=RuntimeError('native failure')):
            with self.assertRaisesRegex(RuntimeError,'native failure'):R.immutable_native_call(None,raw(),1.,None)

    def test_postnative_metadata_refusal_propagates_to_global_max(self):
        data=raw();events=[]
        def native(*args):events.append('native complete');return captured()
        def evidence(*args):events.append('metadata');raise ValueError('bad metadata')
        def reduce(flag,op):events.append(('MAX',int(flag)))
        with patch.object(R,'capture_original_call',side_effect=native),patch.object(R,'input_identity',side_effect=[{},ValueError('bad metadata')]),patch.object(R.dist,'all_reduce',side_effect=reduce):
            with self.assertRaisesRegex(R.GlobalDiagnosticRefusal,'bad metadata'):
                R.RankCoordinator('cpu').local_stage('fake forward',lambda:R.immutable_native_call(None,data,1.,None))
        self.assertEqual(events,['native complete',('MAX',1)])

    def test_aligned_plans_preserve_originals_and_mark_extra_calls(self):
        original=[C.micro_plan([[0,1,2,3]] if rank==0 else [[0,1,2,3],[4]]) for rank in range(8)]
        before=copy.deepcopy(original);aligned=C.synchronize_plans(original)
        self.assertEqual(original,before)
        for micro in [4,2,1]:
            self.assertEqual(len({len(plan[micro]) for plan in aligned}),1)
            for rank in range(8):
                valid=[c for c in aligned[rank][micro] if not c['exclude_duplicate']]
                self.assertEqual(valid,before[rank][micro])
                self.assertEqual(sum(len(c['rows']) for c in valid),4 if rank==0 else 5)

    def test_duplicate_micro4_cannot_replace_original_reference(self):
        data=R.canonical_response_data(raw(),'raw')
        mapping=dict(rows=torch.tensor([0,0]),response_positions=torch.tensor([0,1]),logit_indices=torch.tensor([1,2]),labels=torch.tensor([3,4]))
        baseline={};group=dict(kind='raw',asset='a',source_group=0)
        def call(micro,duplicate,values):
            return R.build_comparison_evidence(dict(token_values={'original_actor':torch.tensor(values,dtype=torch.float64)},metadata={}),
                dict(mapping=mapping),data,dict(rows=[0],exclude_duplicate=duplicate),group,baseline,micro,0,0)
        call(4,False,[-1.,-1.]);call(4,True,[-99.,-99.]);_,evidence=call(2,False,[-1.,-1.])
        self.assertEqual(evidence['metrics']['row-0-micro2-vs-original-micro4']['max_abs'],0.)

    def test_native_padback_mapping_matches_manual_packed_predecessor(self):
        data=raw();data['input_ids']=torch.tensor([[0,2,3,4]])
        data['attention_mask']=torch.tensor([[0,1,1,1]])
        data['position_ids']=torch.tensor([[0,0,1,2]])
        data['loss_mask']=torch.tensor([[0,1]])
        packed=C.response_prediction_map(data,True);dense=C.response_prediction_map(data,False)
        self.assertEqual(packed['logit_indices'].tolist(),[1]);self.assertEqual(dense['logit_indices'].tolist(),[2])
        self.assertEqual(packed['labels'].tolist(),[4]);self.assertEqual(packed['response_positions'].tolist(),[1])

    def test_nonfinite_parameter_buffer_and_private_shard_refused(self):
        for where in ['parameter','buffer','private']:
            model=torch.nn.Linear(2,2,bias=False);model.register_buffer('b',torch.zeros(1));model.weight._local_shard=model.weight.detach().clone()
            with torch.no_grad():
                if where=='parameter':model.weight[0,0]=float('nan')
                elif where=='buffer':model.b[0]=float('nan')
                else:model.weight._local_shard[0,0]=float('nan')
            with self.assertRaisesRegex(ValueError,'Nonfinite'):R.module_state(model)

    def test_same_master_new_private_shard_metadata_changes_state(self):
        model=torch.nn.Linear(2,2,bias=False);before=R.module_state(model)
        model.weight._local_shard=model.weight.detach().clone();after=R.module_state(model)
        self.assertNotEqual(before['parameters'],after['parameters'])
        self.assertEqual(before['parameters']['weight']['full_shape_dtype_bytes_sha256'],after['parameters']['weight']['full_shape_dtype_bytes_sha256'])
        # This demonstrates guard semantics only. It does not prove when actual Torch2.4 FSDP creates this field.

    def test_unreleased_main_refuses_before_any_cuda_or_process_group(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bundle.json';p.write_text('{"status":"SOURCE_ONLY"}')
            with patch.object(sys,'argv',['probe','--bundle',str(p)]),patch.object(R.torch.cuda,'set_device') as gpu,patch.object(R.dist,'init_process_group') as pg:
                with self.assertRaisesRegex(ValueError,'UNIMPLEMENTED'):R.main()
                gpu.assert_not_called();pg.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
