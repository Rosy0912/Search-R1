"""Independent tiny CPU counterexamples; no actual assets or remote execution."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'work'))
import freeze_tree_probability_selection as F


def raw():
    return dict(input_ids=torch.tensor([[9,8,7,6]]), responses=torch.tensor([[7,6]]),
        attention_mask=torch.ones((1,4),dtype=torch.int64), position_ids=torch.tensor([[0,1,2,3]]),
        old_log_probs=torch.tensor([[-1.,-2.]],dtype=torch.float64),
        sampled_log_probs=torch.tensor([[-1.2,-2.2]],dtype=torch.float64))


def monitor():
    r=raw(); r.pop('sampled_log_probs'); r['loss_mask']=torch.tensor([[0,1]])
    r['policy_weights']=torch.tensor([[0.,1.]])
    return r


def row(asset,i,delta,frac=.2,length=9,anomaly=False,sha='e'):
    return dict(asset=asset,row=i,source_group=i//4,max_abs=delta,above_dot01=1,
        effective_tokens=2,above_dot01_fraction=frac,full_attention_tokens=length,
        position_anomaly=anomaly,row_tensor_sha256=sha*64)


class IndependentCases(unittest.TestCase):
    def test_full_group_tail_and_cross_asset_ties(self):
        inv={x:{'original_micro4_groups':[[0,1,2,3],[4]]} for x in ['a','z']}
        rows=[row('z',0,1),row('a',0,1),row('a',4,.01,frac=0.,sha='0')]
        g,c=F.choose_groups(rows,inv)
        self.assertEqual(c[0]['asset'],'a'); self.assertEqual(c[1]['asset'],'z')
        self.assertEqual([x['row_indices'] for x in g],[[0,1,2,3],[4],[0,1,2,3]])

    def test_tail_is_another_group_even_when_max_also_max_fraction(self):
        inv={'a':{'original_micro4_groups':[[0,1,2,3],[4,5,6,7]]}}
        g,c=F.choose_groups([row('a',0,9,1.),row('a',1,8,.99),row('a',4,1,.1)],inv)
        self.assertEqual(c[1]['row'],4); self.assertEqual(len(g),2)

    def test_no_alternate_tail_explicitly_absent(self):
        inv={'a':{'original_micro4_groups':[[0,1,2,3]]}}
        _,c=F.choose_groups([row('a',0,1),row('a',1,.001)],inv)
        self.assertEqual(c[1]['status'],'ABSENT')

    def test_same_criteria_deduplicate_physical_group(self):
        inv={'a':{'original_micro4_groups':[[0,1,2,3]]}}
        g,c=F.choose_groups([row('a',0,.001,anomaly=True,sha='0')],inv)
        self.assertEqual(len(g),1);self.assertEqual(len(g[0]['criteria']),4)

    def test_empty_response_not_selected(self):
        r=raw();r['attention_mask'][:,-2:]=0
        rows=F.describe_rows(r,'a');self.assertEqual(rows[0]['effective_tokens'],0)
        self.assertIsNone(rows[0]['max_abs'])
        with self.assertRaises(ValueError):F.choose_groups(rows,{'a':{'original_micro4_groups':[[0]]}})

    def test_finite_operands_overflow_refused(self):
        r=raw();r['old_log_probs'][0,0]=1.7e308;r['sampled_log_probs'][0,0]=-1.7e308
        with self.assertRaises(ValueError):F.describe_rows(r,'a')

    def test_masked_nan_is_outside_probability_selection(self):
        r=raw();r['attention_mask'][0,-1]=0;r['sampled_log_probs'][0,1]=float('nan')
        rows=F.describe_rows(r,'a');self.assertEqual(rows[0]['effective_tokens'],1)

    def test_losszero_observation_is_in_later_context(self):
        idx=F.A.RawIndex();idx.add_snapshot(raw(),'a');m=monitor()
        v,c=F.monitor_companion(idx,m,{},'m','b'*64)
        self.assertEqual(c['matched_tokens'],1);self.assertEqual(v['matched_count'][0].tolist(),[0,1])
        m['input_ids'][0,2]=100;m['responses'][0,0]=100
        v,c=F.monitor_companion(idx,m,{},'m','b'*64)
        self.assertEqual(c['unknown_tokens'],1);self.assertFalse(bool(v['matched_mask'].any()))

    def test_position_current_token_and_prefix_bound(self):
        idx=F.A.RawIndex();idx.add_snapshot(raw(),'a')
        for which in ['prefix-position','current-position','current-token']:
            m=monitor()
            if which=='prefix-position':m['position_ids'][0,0]=3
            elif which=='current-position':m['position_ids'][0,-1]=8
            else:m['input_ids'][0,-1]=11;m['responses'][0,-1]=11
            v,c=F.monitor_companion(idx,m,{},'m','b'*64)
            self.assertEqual(c['unknown_tokens'],1,which)
            self.assertEqual(v['exact_context_key'].shape,(1,2,48))

    def test_unknown_zero_excluded_not_claimed_probability(self):
        idx=F.A.RawIndex();m=monitor()
        v,c=F.monitor_companion(idx,m,{},'m','b'*64)
        self.assertEqual(c,dict(effective_tokens=1,matched_tokens=0,unknown_tokens=1,matches_with_duplicate_raw_occurrences=0))
        self.assertEqual(v['sampled_min'][0,-1].item(),0);self.assertFalse(v['matched_mask'][0,-1].item())

    def test_exact_duplicate_count_extrema_and_nonfinite_refusal(self):
        idx=F.A.RawIndex();a=raw();b=raw();b['sampled_log_probs']-=1
        idx.add_snapshot(a,'a');idx.add_snapshot(b,'b')
        v,c=F.monitor_companion(idx,monitor(),{},'m','b'*64)
        self.assertEqual(v['matched_count'][0,-1].item(),2)
        self.assertAlmostEqual(v['sampled_min'][0,-1].item(),-3.2)
        self.assertAlmostEqual(v['sampled_max'][0,-1].item(),-2.2)
        b['sampled_log_probs'][0,-1]=float('nan');idx.add_snapshot(b,'bad')
        with self.assertRaises(ValueError):F.monitor_companion(idx,monitor(),{},'m','b'*64)

    def test_tensor_hash_binds_dtype_shape_and_value(self):
        a=torch.tensor([1,2]);self.assertNotEqual(F.tensor_sha(a),F.tensor_sha(a.reshape(1,2)))
        self.assertNotEqual(F.tensor_sha(a),F.tensor_sha(a.int()))
        self.assertNotEqual(F.tensor_sha(a),F.tensor_sha(torch.tensor([1,3])))

    def test_companion_shapes_and_binary_mask_refused(self):
        m=monitor();m['loss_mask'][0,0]=2
        with self.assertRaises(ValueError):F.monitor_companion(F.A.RawIndex(),m,{},'m','b'*64)
        m=monitor();m['policy_weights']=torch.ones((1,3))
        with self.assertRaises(ValueError):F.monitor_companion(F.A.RawIndex(),m,{},'m','b'*64)

    def test_new_json_refuses_nan_and_existing_target(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'new.json';F.save_new_json(p,{'a':1})
            with self.assertRaises(FileExistsError):F.save_new_json(p,{'a':2})
            with self.assertRaises(ValueError):F.save_new_json(Path(t)/'nan.json',{'a':float('nan')})

    def test_actual_main_wrong_host_and_visible_gpu_refused_before_assets(self):
        for host,visible in [('Mac-review',''),('h100-9','0')]:
            argv=['freeze','--run','/nonexistent','--package','/nonexistent','--cpu-reports','/nonexistent','--output','/nonexistent']
            with patch.object(sys,'argv',argv),patch.object(F.socket,'gethostname',return_value=host),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':visible}),patch.object(F.A,'verify_bundle',side_effect=AssertionError('assets must not be touched')):
                with self.assertRaisesRegex(ValueError,'Proper same-node'):F.main()

    def test_main_exact_source_ast_bindings(self):
        text=(ROOT/'work/freeze_tree_probability_selection.py').read_text();tree=ast.parse(text)
        self.assertEqual(F.CONTEXT_SHA,'28e665eb822651ff44559b2e9c6110d01542c5e6433db8dd4d07c659af7131ac')
        for exact in ["calls == list(range(1, 9))","index.rows == 11528 and index.valid_tokens == 3186179", "value['input_ids'].shape[0] == 32", "raw['saved_rows'] == raw['local_rows']", "value['distribution'] == raw['distribution']", "snap == raw_path.with_suffix('.pt')", "name.startswith('source/repo/')", "output.mkdir()", "actual_model_forwards=0", "actual_optimizer_writes=0", "probability_consistency_accepted=False", "long_training_ready=False"]:
            self.assertIn(exact,text)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in ['backward','step','cuda','generate','generate_sequences'] for n in calls))

    def test_main_changed_context_helper_refused_before_assets(self):
        argv=['freeze','--run','/nonexistent','--package','/nonexistent','--cpu-reports','/nonexistent','--output','/nonexistent']
        with patch.object(sys,'argv',argv),patch.object(F.socket,'gethostname',return_value='h100-9'),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.object(F.A,'file_sha',return_value='0'*64),patch.object(F.A,'verify_bundle',side_effect=AssertionError('assets must not be touched')):
            with self.assertRaisesRegex(ValueError,'Pinned context helper changed'):F.main()

    def test_main_output_outside_or_existing_refused_before_assets(self):
        with tempfile.TemporaryDirectory() as t:
            task=Path(t)/'task';(task/'instruct300'/'exists').mkdir(parents=True)
            for dest in [Path(t)/'outside',task/'instruct300'/'exists']:
                argv=['freeze','--run','/nonexistent','--package','/nonexistent','--cpu-reports','/nonexistent','--output',str(dest)]
                with patch.object(sys,'argv',argv),patch.object(F.socket,'gethostname',return_value='h100-9'),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.object(F.A,'TASK_ROOT',task),patch.object(F.A,'file_sha',return_value=F.CONTEXT_SHA),patch.object(F.A,'verify_bundle',side_effect=AssertionError('assets must not be touched')):
                    with self.assertRaisesRegex(ValueError,'New task-local output required'):F.main()


if __name__=='__main__':
    unittest.main(verbosity=2)
