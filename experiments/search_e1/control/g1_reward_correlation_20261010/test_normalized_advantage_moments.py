"""Small pure synthetic CPU fixtures; no remote calls or original row inputs."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parent))
import normalized_advantage_moments as helper
import pairwise_order_concordance as bound_group
import test_group_reward_dispersion as artifact_fixtures

NATIVE='''def compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index, epsilon=1e-6):
    id2std = {}
    id2mean = {}
    id2score = {}
    for idx in id2score:
        id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
    for i in range(len(scores)):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return scores, scores
'''


def sha(raw):return hashlib.sha256(raw).hexdigest()
def fn_sha(source):return sha(ast.dump(ast.parse(source).body[0],include_attributes=False).encode())


class MathTests(unittest.TestCase):
    def test_constant_exact_zero_not_float_residual(self):
        for x in [0.,.05,1.,1.6,-.1]:
            A,sigma,var,shrink=helper.normalize_five([x]*5)
            self.assertEqual((A,sigma,var,shrink),([0.]*5,0.,0.,0.))
    def test_sample_correction1_and_population_four_fifths(self):
        A,sigma,var,shrink=helper.normalize_five([0,0,0,0,1])
        self.assertAlmostEqual(sigma,math.sqrt(.2),places=15);self.assertAlmostEqual(math.fsum(A),0.,places=14)
        self.assertAlmostEqual(var,shrink**2,places=14)
        self.assertAlmostEqual(math.fsum(a*a for a in A)/5,.8*shrink**2,places=14)
        self.assertLess(max(abs(a) for a in A),4/math.sqrt(5))
    def test_scale_offset_epsilon_only_breaks_scale_invariance(self):
        values=[0,0,1,1,1];A,sigma,_,shrink=helper.normalize_five(values)
        B,sigma2,_,shrink2=helper.normalize_five([.05+1.5*x for x in values])
        for a,b in zip(A,B):self.assertAlmostEqual(a/shrink,b/shrink2,places=14)
        self.assertAlmostEqual(sigma2,1.5*sigma,places=14)
    def test_small_sigma_shrinks_not_forced_to_zero(self):
        A,sigma,var,shrink=helper.normalize_five([0.,0.,0.,0.,1e-8])
        self.assertGreater(sigma,0);self.assertGreater(var,0);self.assertLess(shrink,.005)
    def test_all_correct_adds_R_active_group_and_squares(self):
        m=helper.PairedMoments();m.add([1]*5,[1.,1.1,1.2,1.3,1.4]);result=m.report()
        self.assertEqual(result['C']['sum_A_squared'],0);self.assertGreater(result['R']['sum_A_squared'],0)
        self.assertEqual(result['C_zero_R_active_groups'],1);self.assertIsNone(result['R_to_C_population_variance_ratio'])
    def test_allwrong_mixed_invalid_can_add_signal(self):
        m=helper.PairedMoments();m.add([0]*5,[-.1,.05,.05,.05,.05]);r=m.report()
        self.assertEqual(r['C']['sigma_exact_zero_groups'],1);self.assertEqual(r['R']['sigma_nonzero_groups'],1)
    def test_same_reward_cross_and_ratio(self):
        m=helper.PairedMoments();m.add([0,0,1,1,1],[0,0,1,1,1]);r=m.report()
        self.assertEqual(r['R_to_C_RMS_ratio'],1.);self.assertEqual(r['R_to_C_population_variance_ratio'],1.)
        self.assertAlmostEqual(r['C_R_mean_product_A'],r['C']['mean_A_squared'],places=15)
    def test_pooled_all_rows_not_average_step_ratios(self):
        a=helper.PairedMoments();a.add([0,0,0,0,1],[0,0,0,0,1])
        b=helper.PairedMoments();b.add([1]*5,[1.,1.1,1.2,1.3,1.4]);a.merge(b);r=a.report()
        self.assertEqual(r['C']['N_trajectories'],10);self.assertEqual(r['R']['sigma_nonzero_groups'],2)
        self.assertGreater(r['R_to_C_sum_squared_ratio'],1.99);self.assertLess(r['R_to_C_sum_squared_ratio'],2.01)
        self.assertAlmostEqual(r['R_to_C_RMS_ratio']**2,r['R_to_C_sum_squared_ratio'],places=14)
    def test_empty_short_tail_nonfinite_bool_rejected(self):
        for values in [[],[1.]*4,[1.]*6,[1.,1.,1.,1.,math.nan],[True]*5]:
            with self.assertRaises(helper.Hold):helper.normalize_five(values)
    def test_group_distribution_linear_quantiles(self):
        r=helper.distribution([0.,1.,2.,3.,4.]);self.assertEqual((r['p10'],r['p50'],r['p90']),(.4,2.,3.6))


class SourceTests(unittest.TestCase):
    def test_bound_published_dependency_hash(self):
        source=Path(bound_group.__file__).read_bytes();namespace=helper.load_bound_group(source,sha(source))
        self.assertEqual(namespace['STEPS'],helper.STEPS)
        with self.assertRaises(helper.Hold):helper.load_bound_group(source,'0'*64)
    def test_native_source_AST_semantics_and_no_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()/'core.py';path.write_text(NATIVE)
            proof=helper.native_formula_proof(str(path),fn_sha(NATIVE))
            self.assertEqual(proof['epsilon_default'],1e-6);self.assertTrue(proof['source_checked_by_AST_only_not_imported'])
            self.assertNotIn('torch',sys.modules)
            with self.assertRaises(helper.Hold):helper.native_formula_proof(str(path),'0'*64)
    def test_native_changed_semantics_HOLD_even_selfmatching_expected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()/'core.py'
            for source in [NATIVE.replace('epsilon=1e-6','epsilon=1e-5'),NATIVE.replace('torch.tensor([id2score[idx]]))','torch.tensor([id2score[idx]]), correction=0)'),
                NATIVE.replace('+ epsilon','+ 1e-6'),NATIVE.replace('* eos_mask','* other_mask')]:
                path.write_text(source)
                with self.assertRaises(helper.Hold):helper.native_formula_proof(str(path),fn_sha(source))
    def test_stdin_helper_hash_without_local_file_lookup(self):
        source=Path(helper.__file__).read_bytes();argv=['helper.py']
        for name,value in [('summary','/not-read'),('summary-sha256','0'*64),('groups-sha256','0'*64),('expected-node','impossible-node'),
            ('helper-sha256',sha(source)),('group-helper-sha256','0'*64),('native-core-algos','/not-read'),('native-grpo-ast-sha256','0'*64)]:argv+=['--'+name,value]
        output=io.StringIO()
        with patch.dict(helper.__dict__,{'__published_helper_bytes__':source}),patch.object(sys,'argv',argv),\
            patch.object(helper,'file_sha',side_effect=AssertionError('helper file must not be read')),contextlib.redirect_stdout(output):
            self.assertEqual(helper.main(),2)
        self.assertEqual(json.loads(output.getvalue())['reason'],'not_bound_remote_CPU_process')


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.fixture=artifact_fixtures.FullArtifactTests();self.fixture.setUp();self.addCleanup(self.fixture.temp.cleanup)
    def analyze(self):
        f=self.fixture;return helper.analyze(str(f.summary),f.summary_sha,f.group_sha,bound_group)
    def test_full4096_complete_identity_moments_and_privacy(self):
        result=self.analyze();self.assertEqual(result['original_groups'],4096);self.assertEqual(len(result['steps']),8)
        p=result['pooled_selected_steps'];self.assertEqual(p['C']['N_trajectories'],20480)
        self.assertEqual(p['C']['sigma_nonzero_groups'],102*8);self.assertEqual(p['R']['sigma_nonzero_groups'],307*8)
        self.assertAlmostEqual(p['C']['mean_A'],0.,places=14);self.assertAlmostEqual(p['R']['mean_A'],0.,places=14)
        self.assertEqual(p['C_zero_R_active_groups'],205*8)
        for key in ['sum_A_squared','variance_population_A','mean_A_squared']:
            self.assertAlmostEqual(p['C'][key],result['steps'][0]['C'][key]*(8 if key=='sum_A_squared' else 1),places=10)
        encoded=json.dumps(result,separators=(',',':')).encode();self.assertLessEqual(len(encoded),19*1024)
        self.assertNotIn(b'member_trajectory_ids',encoded);self.assertNotIn(b'upstream_group_sha256',encoded);self.assertNotIn(b'R_reconstructed":[',encoded)
    def test_hash_or_incomplete_rejected(self):
        f=self.fixture
        with self.assertRaises(bound_group.Hold):helper.analyze(str(f.summary),f.summary_sha,'0'*64,bound_group)
        f.records.pop();f.rewrite()
        with self.assertRaises(bound_group.Hold):self.analyze()
    def test_second_pass_changed_identity_or_SHA_HOLD(self):
        f=self.fixture
        class MutatingBound:
            row_identity=staticmethod(bound_group.row_identity)
            @staticmethod
            def analyze(*args):
                proof=bound_group.analyze(*args);f.groups.write_bytes(f.groups.read_bytes()+b'\n');return proof
        with self.assertRaises(helper.Hold):helper.analyze(str(f.summary),f.summary_sha,f.group_sha,MutatingBound)


if __name__=='__main__':unittest.main()
