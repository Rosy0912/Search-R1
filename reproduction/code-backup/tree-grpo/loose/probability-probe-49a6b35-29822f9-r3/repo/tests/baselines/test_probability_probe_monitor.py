"""Synthetic CPU exact-context companion fixtures, no actual assets/GPU."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import struct
import sys
import unittest

import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from search_r1.baselines.probability_probe_monitor import PREFIX_VERSION,bind_monitor_companion,exact_monitor_keys,matched_monitor_ranges
from search_r1.baselines.probability_probe_runner import probability_statistics


def sample():
    ids=torch.tensor([[1,2,3,4],[2,3,4,5]])
    data=dict(input_ids=ids,responses=ids[:,-2:],position_ids=torch.arange(4).expand(2,4),
        attention_mask=torch.ones_like(ids),loss_mask=torch.tensor([[1,1],[0,1]]))
    effective=data['attention_mask'][:,-2:].bool()&data['loss_mask'].bool()
    counts=torch.tensor([[2,0],[0,1]],dtype=torch.int64)
    identity={'frozen':'synthetic'}
    payload=dict(identity=identity,monitor_asset='fixture',monitor_asset_sha256='0'*64,
        prefix_version=PREFIX_VERSION.decode().replace('\x00','\\0'),matched_count=counts,
        sampled_min=torch.tensor([[-2.,0.],[0.,-.5]],dtype=torch.float64),
        sampled_max=torch.tensor([[0.,0.],[0.,-.5]],dtype=torch.float64),
        effective_mask=effective,matched_mask=counts>0,exact_context_key=exact_monitor_keys(data))
    return data,payload,identity


class MonitorTests(unittest.TestCase):
    def bind(self,data,payload,identity):return bind_monitor_companion(payload,data,identity,'fixture','0'*64)

    def test_same_context_verified_and_unknown_preserved(self):
        data,payload,identity=sample();before={k:v.clone() for k,v in payload.items() if isinstance(v,torch.Tensor)}
        attached=self.bind(data,payload,identity)
        self.assertEqual(attached['companion_matched_count'][0,1],0)
        self.assertTrue(all(torch.equal(payload[k],v) for k,v in before.items()))

    def test_loss_masked_observation_stays_in_later_causal_prefix(self):
        data,_,_=sample();key=exact_monitor_keys(data)[1,1]
        prefix=hashlib.sha256(PREFIX_VERSION)
        for token,position in ((2,0),(3,1),(4,2)):prefix.update(struct.pack('<qq',token,position))
        self.assertEqual(bytes(key.tolist()),prefix.digest()+struct.pack('<qq',5,3))
        self.assertEqual(int(exact_monitor_keys(data)[1,0].sum()),0)

    def test_padding_excluded_without_trimming_full_buffers(self):
        data,_,_=sample();data['attention_mask'][0,0]=0
        key=exact_monitor_keys(data)[0,0]
        prefix=hashlib.sha256(PREFIX_VERSION);prefix.update(struct.pack('<qq',2,1))
        self.assertEqual(bytes(key.tolist()),prefix.digest()+struct.pack('<qq',3,2))
        self.assertEqual(data['input_ids'].shape,(2,4))

    def test_position_change_invalidates_companion_despite_same_tokens(self):
        data,payload,identity=sample();data['position_ids']=data['position_ids'].clone();data['position_ids'][0,1]=9
        with self.assertRaisesRegex(ValueError,'causal prefixes'):self.bind(data,payload,identity)

    def test_retokenized_response_invalidates_context_key(self):
        data,payload,identity=sample();data['input_ids']=data['input_ids'].clone();data['input_ids'][0,-1]=6
        data['responses']=data['input_ids'][:,-2:]
        with self.assertRaisesRegex(ValueError,'causal prefixes'):self.bind(data,payload,identity)

    def test_wrong_monitor_asset_refused(self):
        data,payload,identity=sample();payload['monitor_asset_sha256']='1'*64
        with self.assertRaisesRegex(ValueError,'another monitor'):self.bind(data,payload,identity)

    def test_wrong_parent_identity_refused(self):
        data,payload,identity=sample();payload['identity']={'other':True}
        with self.assertRaisesRegex(ValueError,'identity differs'):self.bind(data,payload,identity)

    def test_reversed_duplicate_probability_range_refused(self):
        data,payload,identity=sample();payload['sampled_min'][0,0]=1
        with self.assertRaisesRegex(ValueError,'reversed'):self.bind(data,payload,identity)

    def test_float_count_cannot_be_used_as_occurrences(self):
        data,payload,identity=sample();payload['matched_count']=payload['matched_count'].double()
        with self.assertRaisesRegex(ValueError,'dtype'):self.bind(data,payload,identity)

    def test_matches_for_ineffective_observation_refused(self):
        data,payload,identity=sample();payload['matched_count'][1,0]=1
        payload['matched_mask'][1,0]=True
        with self.assertRaisesRegex(ValueError,'matched mask'):self.bind(data,payload,identity)

    def test_unknown_placeholders_never_subtracted_or_averaged(self):
        data,payload,identity=sample();data.update(self.bind(data,payload,identity))
        rows=torch.tensor([0,0,1]);positions=torch.tensor([0,1,1])
        # Unknown token100 contributes no erroneous comparison with placeholder0.
        current={'actual':torch.tensor([-1.,100.,-.2],dtype=torch.float64)}
        report,values=matched_monitor_ranges(data,rows,positions,current,torch.ones(3),probability_statistics)
        self.assertEqual((report['effective_occurrences'],report['matched_occurrences'],report['unknown_occurrences']),(3,2,1))
        self.assertEqual(report['matched_raw_occurrences_denominator'],3)
        self.assertEqual(report['matched_contexts_with_duplicate_raw_occurrences'],1)
        self.assertFalse(report['duplicate_probabilities_averaged'])
        self.assertAlmostEqual(report['variants']['actual']['closest_abs_distance_to_saved_range']['mean_abs'],.15)
        self.assertAlmostEqual(report['variants']['actual']['worst_abs_distance_to_saved_range']['max_abs'],1.)
        self.assertEqual(values['actual_minus_sampled_max'].numel(),2)

    def test_all_unknown_ranges_report_no_fake_probability_sample(self):
        data,payload,identity=sample();payload['matched_count'].zero_();payload['matched_mask'].zero_()
        data.update(self.bind(data,payload,identity))
        report,_=matched_monitor_ranges(data,torch.tensor([0]),torch.tensor([0]),{'actual':torch.tensor([-1.])},
            torch.ones(1),probability_statistics)
        self.assertEqual(report['matched_occurrences'],0)
        self.assertIsNone(report['variants']['actual']['worst_abs_distance_to_saved_range']['mean_abs'])

    def test_raw_key_definition_matches_independent_cpu_context_helper(self):
        # Existing accepted helper bytes are read locally, synthetic tensors only.
        path=ROOT.parents[0]/'analyze_tree_sampling_calibration.py'
        spec=importlib.util.spec_from_file_location('fixture_accepted_context',path)
        helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
        data,payload,_=sample();data['old_log_probs']=torch.zeros(2,2);data['policy_weights']=torch.ones(2,2)
        layout=helper.normalize(data,raw=False);actual=exact_monitor_keys(data)
        for row in range(2):
            for key,offset,*_ in helper.causal_keys(layout,row,supervised=True):
                self.assertEqual(key,bytes(actual[row,offset].tolist()))


if __name__=='__main__':unittest.main()
