"""Toy-only CPU order fixtures; no original inputs or node calls."""
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parent))
import pairwise_order_concordance as helper
from test_group_reward_dispersion import FullArtifactTests


class OrderMathTests(unittest.TestCase):
    def test_three_order_types_and_C_ties_are_not_discordance(self):
        counts,n,_=helper.group_counts([0,0,1,1,1],[0.,.2,.2,.1,.3])
        self.assertEqual(counts,dict(concordant=4,discordant=1,C_tie_only=4,R_tie_only=1,both_tied=0))
        result=helper.rates(counts,n)
        self.assertEqual(n,6);self.assertEqual(result['concordance_rate'],4/6)
        self.assertEqual(result['non_tied_concordance_rate'],4/5)
        self.assertEqual(result['half_R_tie_AUC_same_group'],.75)
        self.assertAlmostEqual(result['optional_Kendall_tau_b'],3/math.sqrt(54))
    def test_R_ties_count_as_strict_order_failure(self):
        counts,n,_=helper.group_counts([0,0,0,0,1],[.5]*5);result=helper.rates(counts,n)
        self.assertEqual(n,4);self.assertEqual(result['R_tie_only'],4)
        self.assertEqual(result['concordance_rate'],0.);self.assertEqual(result['half_R_tie_AUC_same_group'],.5)
        self.assertIsNone(result['optional_Kendall_tau_b'])
    def test_C_constant_but_R_varying_primary_undefined(self):
        counts,n,_=helper.group_counts([1]*5,[0,1,2,3,4]);result=helper.rates(counts,n)
        self.assertEqual(counts['C_tie_only'],10);self.assertEqual(n,0)
        self.assertIsNone(result['concordance_rate']);self.assertIsNone(result['optional_Kendall_tau_b'])
    def test_both_constant_all_tied(self):
        counts,n,_=helper.group_counts([0]*5,[.05]*5)
        self.assertEqual(counts['both_tied'],10);self.assertEqual(n,0)
    def test_exact_ties_no_epsilon(self):
        counts,n,_=helper.group_counts([0,0,0,0,1],[1.,1.,1.,1.,math.nextafter(1.,2.)])
        self.assertEqual(counts['concordant'],4);self.assertEqual(helper.rates(counts,n)['concordance_rate'],1.)
    def test_pair_order_invariant_under_permutation(self):
        C=[1,1,0,1,0];R=[.8,.9,.2,.7,.3];expected=helper.group_counts(C,R)
        p=list(range(5));random.Random(42).shuffle(p)
        self.assertEqual(helper.group_counts([C[i] for i in p],[R[i] for i in p]),expected)
    def test_k_times5_minus_k_all_k(self):
        for k in range(6):
            counts,n,_=helper.group_counts([1]*k+[0]*(5-k),[1.]*k+[0.]*(5-k))
            self.assertEqual(n,k*(5-k));self.assertEqual(counts['concordant'],n)
    def test_perfect_crossC_order_can_still_correct_below_group_mean(self):
        counts,n,s=helper.group_counts([1,1,1,1,0],[1.6,1.6,1.6,.98,.05])
        self.assertEqual(helper.rates(counts,n)['concordance_rate'],1.)
        self.assertEqual(s['C1_count_in_mixed'],4);self.assertEqual(s['C1_R_below_group_mean'],1)
        self.assertEqual(s['C1_R_above_group_mean'],3);self.assertEqual(s['C1_R_below_group_mean_margin_gt_1e_4'],1)
    def test_mean_equality_and_strict_margin(self):
        _,_,s=helper.group_counts([1,1,0,0,0],[1.,1.,1.,1.,1.])
        self.assertEqual(s['C1_R_equal_group_mean'],2);self.assertEqual(s['C1_R_below_group_mean'],0)
        _,_,s=helper.group_counts([1,1,0,0,0],[0.,5e-5,5e-5,5e-5,5e-5])
        self.assertEqual(s['C1_R_below_group_mean'],1);self.assertEqual(s['C1_R_below_group_mean_margin_gt_1e_4'],0)
    def test_nonbinary_or_nonfinite(self):
        for C,R in [([0,0,0,0,2],[0.]*5),([0]*5,[0.,0.,0.,0.,math.inf])]:
            with self.assertRaises(helper.Hold):helper.group_counts(C,R)


class GlobalRankTests(unittest.TestCase):
    def test_bucket_AUC_matches_all_strict_C_pairs_bruteforce(self):
        C=[0,0,1,1,1];R=[0.,.2,.2,.1,.3];buckets={}
        for c,r in zip(C,R):buckets.setdefault(r,[0,0])[c]+=1
        result=helper.global_correct_wrong_order(buckets)
        P=Q=U=0
        for c,r in zip(C,R):
            if c:
                for cc,rr in zip(C,R):
                    if not cc:P+=int(r>rr);Q+=int(r<rr);U+=int(r==rr)
        self.assertEqual((result['concordant'],result['discordant'],result['R_ties']),(P,Q,U))
        self.assertEqual(result['AUROC'],(P+.5*U)/(P+Q+U))
    def test_global_across_questions_not_primary_within(self):
        Cs=[[0,0,0,0,1]]*2;Rs=[[100.,100.,100.,100.,101.],[0.,0.,0.,0.,1.]];buckets={}
        for C,R in zip(Cs,Rs):
            counts,n,_=helper.group_counts(C,R);self.assertEqual(helper.rates(counts,n)['concordance_rate'],1.)
            for c,r in zip(C,R):buckets.setdefault(r,[0,0])[c]+=1
        self.assertEqual(helper.global_correct_wrong_order(buckets)['AUROC'],.75)
    def test_large_crosspairs_counted_without_enumeration(self):
        result=helper.global_correct_wrong_order({0.:[10**6,0],1.:[0,10**6]})
        self.assertEqual(result['C_strict_pairs'],10**12);self.assertEqual(result['AUROC'],1.)
    def test_one_class_AUC_undefined(self):
        result=helper.global_correct_wrong_order({0.:[8,0]});self.assertFalse(result['AUROC_defined']);self.assertIsNone(result['AUROC'])


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.fixture=FullArtifactTests();self.fixture.setUp();self.addCleanup(self.fixture.temp.cleanup)
    def analyze(self):
        f=self.fixture;return helper.analyze(str(f.summary),f.summary_sha,f.group_sha)
    def test_actual4096_toy_group_pairs_and_privacy(self):
        result=self.analyze();self.assertEqual(result['pooled_within_original_question']['total_pairs'],40960)
        self.assertEqual(result['pooled_within_original_question']['C_strict_comparable'],612*8)
        self.assertEqual(result['pooled_within_original_question']['concordance_rate'],1.)
        self.assertEqual(result['pooled_within_original_question']['C_tie_only'],1642*8)
        self.assertEqual(result['pooled_supplementary_correct_vs_group_R_mean']['C1_count_in_mixed'],204*8)
        self.assertEqual(result['supplementary_global_across_questions']['AUROC'],1.)
        encoded=json.dumps(result,separators=(',',':')).encode();self.assertLessEqual(len(encoded),20*1024)
        self.assertNotIn(b'member_trajectory_ids',encoded);self.assertNotIn(b'upstream_group_sha256',encoded)
    def test_missing_group_HOLD(self):
        f=self.fixture;f.records.pop();f.rewrite()
        with self.assertRaises(helper.Hold):self.analyze()
    def test_duplicate_group_HOLD(self):
        f=self.fixture;f.records[-1]=copy.deepcopy(f.records[-2]);f.rewrite()
        with self.assertRaises(helper.Hold):self.analyze()
    def test_source_artifact_SHA_HOLD(self):
        f=self.fixture;f.group_sha='0'*64
        with self.assertRaises(helper.Hold):self.analyze()
    def test_stored_sample_std_disagrees_HOLD(self):
        f=self.fixture;f.records[0]['R']['std_sample_correction1']+=.01;f.rewrite()
        with self.assertRaises(helper.Hold):self.analyze()


class StdinTests(unittest.TestCase):
    def test_stdin_codehash_no_helper_file_needed(self):
        source=Path(helper.__file__).read_bytes();digest=hashlib.sha256(source).hexdigest();output=io.StringIO()
        argv=['order.py','--summary','/not-read','--summary-sha256','0'*64,'--groups-sha256','0'*64,
            '--expected-node','impossible-toy-node','--helper-sha256',digest]
        with patch.dict(helper.__dict__,{'__published_helper_bytes__':source}),patch.object(sys,'argv',argv),\
                patch.object(helper,'file_sha',side_effect=AssertionError('no file read')),contextlib.redirect_stdout(output):
            self.assertEqual(helper.main(),2)
        self.assertEqual(json.loads(output.getvalue())['reason'],'not_bound_remote_CPU_process')


if __name__=='__main__':unittest.main()
