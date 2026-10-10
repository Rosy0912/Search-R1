"""Toy-only CPU tests of identities, group signal counts and distributions."""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parent))
import selected_reward_group_distributions as dist
import selected_pair_statistics as bridge
import pairwise_pearson as pearson
import reconstruct_train_raw_c_preflight_v2 as c_gate
from test_selected_pair_statistics import SyntheticFixture,put,h


def toy_rows():
    """Five recurring conceptual categories, all synthetic and in-memory."""
    run='a'*32;step=20;attempt='b'*32;rows=[]
    for parent in range(512):
        pattern=parent%5
        C=([1]*5 if pattern==0 else ([0]*5 if pattern in (1,2,3) else [0,1,0,1,0]))
        valid=([True]*5 if pattern in (0,1,4) else ([False]*5 if pattern==2 else [True,False,True,False,True]))
        R=([1.,1.1,1.2,1.3,1.4] if pattern==0 else ([.05]*5 if pattern==1 else ([-.1]*5 if pattern==2
            else ([.05,-.1,.05,-.1,.05] if pattern==3 else [.05,1.,.05,1.,.05]))))
        upstream=h('toy-parent'+str(parent))
        for sibling in range(5):
            rows.append(dict(run_id=run,step=step,attempt_id=attempt,
                trajectory_id=f'{run}-{step}-attempt-{attempt}-{parent}-{upstream}-sample-{sibling}',
                upstream_group_sha256=upstream,C_raw=C[sibling],final_action_valid=valid[sibling],
                model_authored_search_count=sibling%3,R_reconstructed=R[sibling],
                b_reconstructed=sibling/4,d_reconstructed=(4-sibling)/4))
    return rows,run,step,attempt


class DistributionTests(unittest.TestCase):
    def setUp(self):self.rows,self.run,self.step,self.attempt=toy_rows()
    def analyze(self):return dist.analyze_rows(self.rows,self.run,self.step,self.attempt)
    def hold(self):
        with self.assertRaises(dist.Hold):self.analyze()

    def test_complete_counts_signal_partition_and_population_denominators(self):
        summary,records=self.analyze()
        self.assertEqual(summary['trajectory_count'],2560);self.assertEqual(len(records),512)
        self.assertEqual(summary['group_outcomes'],dict(all_C1=103,all_C0_including_invalid=307,mixed_C=102))
        self.assertEqual(summary['all_C0_subclasses'],dict(all_valid_wrong=103,all_invalid=102,valid_wrong_and_invalid=102))
        self.assertEqual(summary['variance_signals'],dict(C_zero_R_nonzero=205,C_nonzero_R_nonzero=102,both_zero=205,C_nonzero_R_zero=0))
        self.assertEqual(summary['variance_signals_by_outcome']['all_C0_including_invalid']['C_zero_R_nonzero'],102)
        self.assertEqual(sum(summary['R_histogram']['counts']),2560)
        self.assertEqual(summary['conditional']['all']['N'],2560)
        self.assertEqual(summary['conditional']['strict_correct']['N'],103*5+102*2)

    def test_original_parent_and_sibling_order_not_R_request_order(self):
        expected,_=self.analyze();random.Random(42).shuffle(self.rows);actual,records=self.analyze()
        self.assertEqual(expected,actual)
        self.assertEqual([r['original_parent_occurrence'] for r in records],list(range(512)))
        self.assertTrue(all(r['original_sibling_indices']==list(range(5)) for r in records))
        self.assertTrue(all(r['member_trajectory_ids'][4].endswith('-sample-4') for r in records))

    def test_std_population_and_sample_correction_match_numpy(self):
        import numpy as np
        summary,records=self.analyze();first=records[0]
        self.assertAlmostEqual(first['R']['std_population'],float(np.std(first['R_reconstructed'],ddof=0)),places=14)
        self.assertAlmostEqual(first['R']['std_sample_correction1'],float(np.std(first['R_reconstructed'],ddof=1)),places=14)
        self.assertEqual(first['C']['std_sample_correction1'],0.)
        values=[r['R_reconstructed'] for r in self.rows]
        self.assertAlmostEqual(summary['R_reconstructed']['std_population'],float(np.std(values)),places=14)

    def test_probe_quantiles_match_numpy_linear(self):
        import numpy as np
        summary,_=self.analyze()
        for cohort in summary['conditional'].values():
            if cohort['N']:self.assertTrue(0<=cohort['b']['p10']<=cohort['b']['p50']<=cohort['b']['p90']<=1)
        values=[r['b_reconstructed'] for r in self.rows];stats=summary['conditional']['all']['b']
        for name,q in [('p10',.1),('p50',.5),('p90',.9)]:self.assertEqual(stats[name],float(np.quantile(values,q)))

    def test_probe_conditioning_denominators_are_rows_not_group_average(self):
        summary,_=self.analyze();c=summary['conditional']
        self.assertEqual(c['searched']['N'],512*3);self.assertEqual(c['no_search']['N'],512*2)
        self.assertEqual(c['searched']['b']['N'],c['searched']['N'])
        self.assertEqual(c['strict_correct']['N']+c['valid_wrong']['N']+c['invalid_final_action']['N'],2560)
        expected=sum(row['b_reconstructed'] for row in self.rows if row['C_raw'])/c['strict_correct']['N']
        self.assertEqual(c['strict_correct']['b']['mean'],expected)

    def test_grounding_joint_product_not_product_of_marginal_means(self):
        summary,_=self.analyze();all_rows=summary['conditional']['all']
        self.assertEqual(all_rows['grounding_product_d_times_1_minus_b']['mean'],.375)
        marginal=all_rows['d']['mean']*(1-all_rows['b']['mean'])
        self.assertEqual(marginal,.25)
        self.assertNotEqual(all_rows['grounding_product_d_times_1_minus_b']['mean'],marginal)

    def test_zero_variance_exact_not_epsilon(self):
        rows=[dict(row) for row in self.rows[:5]]
        for row in rows:row['R_reconstructed']=1.
        self.assertTrue(dist.group_statistics(rows)['both_zero'])
        rows[-1]['R_reconstructed']=math.nextafter(1.,2.)
        stats=dist.group_statistics(rows);self.assertTrue(stats['C_zero_R_nonzero']);self.assertGreater(stats['R']['range'],0.)

    def test_no_correct_negative_inference_or_gradient_field(self):
        _,records=self.analyze()
        self.assertNotIn('advantage',records[0]);self.assertNotIn('gradient',records[0])
        self.assertEqual(records[1]['all_C0_subclass'],'all_valid_wrong')
        self.assertEqual(records[2]['all_C0_subclass'],'all_invalid')
        self.assertEqual(records[3]['all_C0_subclass'],'valid_wrong_and_invalid')

    def test_empty_conditional_defined_as_null(self):
        for row in self.rows:row.update(C_raw=0,final_action_valid=True,R_reconstructed=.05)
        summary,_=self.analyze();empty=summary['conditional']['strict_correct']
        self.assertEqual(empty['N'],0);self.assertIsNone(empty['b']['mean']);self.assertIsNone(empty['d']['p50'])

    def test_all_group_metrics_have512_denominator(self):
        summary,_=self.analyze()
        for stats in summary['group_metric_distributions'].values():
            for value in stats.values():self.assertEqual(value['N'],512)

    def test_missing_row(self):self.rows.pop();self.hold()
    def test_duplicate_row(self):self.rows[-1]=dict(self.rows[-2]);self.hold()
    def test_wrong_native_run(self):self.rows[0]['run_id']='c'*32;self.hold()
    def test_wrong_step(self):self.rows[0]['step']=40;self.hold()
    def test_bool_step(self):self.rows[0]['step']=True;self.hold()
    def test_wrong_attempt(self):self.rows[0]['attempt_id']='c'*32;self.hold()
    def test_wrong_ID_run(self):self.rows[0]['trajectory_id']=self.rows[0]['trajectory_id'].replace('a'*32,'c'*32,1);self.hold()
    def test_wrong_upstream_hash(self):self.rows[0]['upstream_group_sha256']='c'*64;self.hold()
    def test_nonbinary_C(self):self.rows[0]['C_raw']=2;self.hold()
    def test_correct_invalid_conflict(self):self.rows[0]['final_action_valid']=False;self.hold()
    def test_nan_R(self):self.rows[0]['R_reconstructed']=math.nan;self.hold()
    def test_nan_b(self):self.rows[0]['b_reconstructed']=math.nan;self.hold()
    def test_outside_d(self):self.rows[0]['d_reconstructed']=1.01;self.hold()
    def test_negative_search_count(self):self.rows[0]['model_authored_search_count']=-1;self.hold()
    def test_R_outside_native_support(self):self.rows[0]['R_reconstructed']=2.;self.hold()


class MathFixtures(unittest.TestCase):
    def test_histogram_boundaries_include1_and_nativeFP32maximum(self):
        values=[0.,.1,.5,1.]
        result=dist.histogram(values,dist.PROBE_EDGES);self.assertEqual(sum(result['counts']),4)
        self.assertEqual(result['counts'][0],1);self.assertEqual(result['counts'][1],1);self.assertEqual(result['counts'][-1],1)
        result=dist.histogram([1.600000023841858,-.10000000149011612,.05000000074505806],dist.R_EDGES)
        self.assertEqual(result['N'],3);self.assertEqual(result['counts'][-1],1)
    def test_empty_moments(self):self.assertEqual(dist.moments([])['N'],0);self.assertIsNone(dist.moments([])['mean'])
    def test_quantiles_singleton(self):self.assertEqual(dist.moments([3.])['p10'],3.);self.assertEqual(dist.moments([3.])['std_population'],0.)
    def test_quantiles_not_nearest_neighbor(self):self.assertEqual(dist.quantile([0.,1.],.1),.1)
    def test_histogram_rejects_outside(self):
        with self.assertRaises(dist.Hold):dist.histogram([-1.],dist.PROBE_EDGES)
    def test_nonfinite_moments(self):
        with self.assertRaises(dist.Hold):dist.moments([math.inf])


class AcceptedBridgeFixtures(unittest.TestCase):
    def test_full8_frozen_bridge_acceptance_then4096_original_groups(self):
        with tempfile.TemporaryDirectory() as root:
            fixture=SyntheticFixture(root,bridge.SELECTED_STEPS);accepted=fixture.analyze()
            fixture.config.update(pairwise_source_path=pearson.__file__,pairwise_source_sha256=h('synthetic dependency'),
                C_helper_path=c_gate.__file__)
            cfgpath=fixture.D/'bridge-config.json'; cfgsha=put(cfgpath,fixture.config)
            acceptedpath=fixture.D/'accepted-selected-pairs.json';acceptedsha=put(acceptedpath,accepted)
            config=dict(bridge_config_path=str(cfgpath),bridge_config_sha256=cfgsha,
                accepted_pair_statistics_path=str(acceptedpath),accepted_pair_statistics_sha256=acceptedsha)
            config.update({name:fixture.config[name] for name in ['diagnostic_base','pairwise_source_path',
                'pairwise_source_sha256','C_helper_path','C_helper_sha256']})
            report,records=dist.analyze(config,bridge,pearson,c_gate)
            self.assertEqual(report['trajectories'],20480);self.assertEqual(report['original_five_sibling_groups'],4096)
            self.assertEqual(len(records),4096);self.assertTrue(report['selected_only_not_TRAIN_population_1_to_150'])
            self.assertTrue(report['reconstructed_R_not_online_bitwise']);self.assertFalse(report['original_online_R_available'])
            self.assertTrue(report['probe_version_changes_not_policy_only_drift'])
            self.assertEqual([x['step'] for x in report['steps']],bridge.SELECTED_STEPS)
            for summary in report['steps']:self.assertEqual(sum(summary['R_histogram']['counts']),2560)
            # A genuine synthetic committed-ledger append changes only the all-ledger hash.
            ledger=fixture.native/'logs/actual-updates.jsonl';before=hashlib.sha256(ledger.read_bytes()).hexdigest()
            appended=dict(fixture.updates[-1],step=141,attempt_id=f'{141:032x}')
            with ledger.open('ab') as stream:stream.write(bridge.canonical(appended)+b'\n')
            fresh_ledger_sha=dist.accepted_bridge_report(config,bridge,pearson,c_gate)[3]
            self.assertNotEqual(fresh_ledger_sha,before)
            # Only harmless original-ledger append may differ from accepted bridge.
            fresh=dict(accepted);fresh['original_ledger_read_sha256']='0'*64
            with patch.object(bridge,'analyze_scope',return_value=fresh):
                self.assertEqual(dist.accepted_bridge_report(config,bridge,pearson,c_gate)[3],'0'*64)
            # Changed moments or any selected identity/source proof is never accepted.
            bad=copy.deepcopy(fresh);bad['pooled']['sumR']+=1.
            with patch.object(bridge,'analyze_scope',return_value=bad):
                with self.assertRaises(dist.Hold):dist.accepted_bridge_report(config,bridge,pearson,c_gate)
            bad=copy.deepcopy(fresh);bad['step_proofs'][0]['attempt_id']='b'*32
            with patch.object(bridge,'analyze_scope',return_value=bad):
                with self.assertRaises(dist.Hold):dist.accepted_bridge_report(config,bridge,pearson,c_gate)


if __name__=='__main__':unittest.main()
