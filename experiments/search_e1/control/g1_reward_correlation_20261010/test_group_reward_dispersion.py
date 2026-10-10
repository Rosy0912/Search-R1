"""CPU-only toy fixtures; no real group/member data or node calls."""
import copy
import contextlib
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
import group_reward_dispersion as helper
import selected_reward_group_distributions as grouped
from test_selected_reward_group_distributions import toy_rows


def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


class NumericalTests(unittest.TestCase):
    def stdin_main(self,source,expected):
        output=io.StringIO()
        argv=['group_reward_dispersion.py','--summary','/not-read','--summary-sha256','0'*64,
            '--groups-sha256','0'*64,'--expected-node','impossible-toy-node','--helper-sha256',expected]
        with patch.dict(helper.__dict__,{'__published_helper_bytes__':source}),patch.object(sys,'argv',argv),\
                patch.object(helper,'file_sha',side_effect=AssertionError('no helper file read')),\
                contextlib.redirect_stdout(output):
            self.assertEqual(helper.main(),2)
        return json.loads(output.getvalue())
    def test_stdin_source_bytes_hash_no_helper_file_required(self):
        source=Path(helper.__file__).read_bytes();result=self.stdin_main(source,sha(source))
        self.assertEqual(result['reason'],'not_bound_remote_CPU_process')
    def test_stdin_wrong_source_bytes_rejected(self):
        result=self.stdin_main(b'not the published source','0'*64)
        self.assertEqual(result['reason'],'unpublished_dispersion_helper')
    def test_sample_std_matches_numpy_correction1(self):
        import numpy as np
        values=[.05,-.1,.05,-.1,.05]
        self.assertAlmostEqual(helper.sample_std(values),float(np.std(values,ddof=1)),places=15)
    def test_sample_std_tiny_is_not_zero(self):
        self.assertGreater(helper.sample_std([1.,1.,1.,1.,math.nextafter(1.,2.)]),0.)
        self.assertEqual(helper.sample_std([.05]*5),0.)
    def test_quantiles_numpy_linear(self):
        import numpy as np
        values=[0.,1e-8,1e-3,.004,.007,.015,.10,.26]
        result=helper.distribution(values)
        for name,q in [('p10',.1),('p25',.25),('p50',.5),('p75',.75),('p90',.9)]:
            self.assertAlmostEqual(result[name],float(np.quantile(values,q)),places=16)
    def test_thresholds_strict_greater_not_ge(self):
        result=helper.distribution(helper.THRESHOLDS)
        self.assertEqual([x['count'] for x in result['counts_strictly_greater']],[5,4,3,2,1,0])
    def test_empty_class_null(self):
        result=helper.distribution([]);self.assertEqual(result['N'],0);self.assertIsNone(result['p50'])
        self.assertTrue(all(x['count']==0 for x in result['counts_strictly_greater']))
    def test_invalid_reward_length_or_nonfinite(self):
        for values in [[0.]*4,[0.,0.,0.,0.,math.nan]]:
            with self.assertRaises(helper.Hold):helper.sample_std(values)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        rows,self.run,self.step,self.attempt=toy_rows();_,records=grouped.analyze_rows(rows,self.run,self.step,self.attempt)
        self.row=records[0]
    def reject(self):
        with self.assertRaises(helper.Hold):helper.row_identity(self.row,self.run)
    def test_native_sibling_identity_and_allC1(self):
        step,parent,attempt,outcome,std=helper.row_identity(self.row,self.run)
        self.assertEqual((step,parent,attempt,outcome),(self.step,0,self.attempt,'all_C1'))
        self.assertAlmostEqual(std,math.sqrt(.025),places=14)
    def test_wrong_run(self):self.row['run_id']='b'*32;self.reject()
    def test_wrong_step(self):self.row['step']=21;self.reject()
    def test_parent_outside(self):self.row['original_parent_occurrence']=512;self.reject()
    def test_sibling_coverage(self):self.row['original_sibling_indices']=[0,1,2,3,3];self.reject()
    def test_sibling_ID(self):self.row['member_trajectory_ids'][-1]=self.row['member_trajectory_ids'][0];self.reject()
    def test_nonbinary_C(self):self.row['C_raw'][0]=2;self.reject()
    def test_wrong_stored_outcome(self):self.row['outcome']='mixed_C';self.reject()
    def test_fake_stored_std(self):self.row['R']['std_sample_correction1']+=.001;self.reject()
    def test_fake_exactzero(self):self.row['R']['zero_variance_exact_values']=True;self.reject()
    def test_nonfinite_stored_std(self):self.row['R']['std_sample_correction1']=math.inf;self.reject()


class FullArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        rows,run,step,attempt=toy_rows();summary,records=grouped.analyze_rows(rows,run,step,attempt)
        self.records=[];summaries=[]
        for step in helper.STEPS:
            s=copy.deepcopy(summary);s['step']=step;summaries.append(s)
            for original in records:
                row=copy.deepcopy(original);row['step']=step;row['attempt_id']=f'{step:032x}'
                parent=row['original_parent_occurrence'];upstream=row['upstream_group_sha256']
                row['member_trajectory_ids']=[f'{run}-{step}-attempt-{row["attempt_id"]}-{parent}-{upstream}-sample-{i}' for i in range(5)]
                self.records.append(row)
        self.groups=self.root/'groups.jsonl';self.summary=self.root/'summary.json'
        self.document=dict(status='ACCEPTED_SELECTED8_RECONSTRUCTED_REWARD_GROUP_DISTRIBUTIONS',selected_steps=helper.STEPS,
            trajectories=20480,original_five_sibling_groups=4096,original_online_R_available=False,steps=summaries,
            remote_group_artifact_path=str(self.groups))
        self.rewrite()
    def rewrite(self):
        self.groups.write_bytes(b''.join(canonical(row)+b'\n' for row in self.records));self.group_sha=sha(self.groups.read_bytes())
        self.document['remote_group_artifact_sha256']=self.group_sha;self.summary.write_bytes(canonical(self.document))
        self.summary_sha=sha(self.summary.read_bytes())
    def analyze(self):return helper.analyze(str(self.summary),self.summary_sha,self.group_sha)
    def reject(self):
        with self.assertRaises(helper.Hold):self.analyze()
    def test_full4096_perclass_magnitude_and_compact_privacy(self):
        result=self.analyze();self.assertEqual(result['original_groups'],4096);self.assertEqual(len(result['steps']),8)
        encoded=canonical(result);self.assertLessEqual(len(encoded),20*1024)
        self.assertNotIn(b'member_trajectory_ids',encoded);self.assertNotIn(b'upstream_group_sha256',encoded)
        self.assertTrue(result['raw_IDs_members_not_exported'])
        first=result['steps'][0]['classes']
        self.assertEqual({k:v['N'] for k,v in first.items()},dict(all_C1=103,mixed_C=102,all_C0=307))
        self.assertEqual(first['all_C0']['counts_strictly_greater'][0]['count'],102)
        self.assertEqual(first['all_C0']['counts_strictly_greater'][-1]['count'],0)
        self.assertEqual(first['all_C1']['counts_strictly_greater'][-1]['count'],103)
    def test_missing_group(self):self.records.pop();self.rewrite();self.reject()
    def test_duplicate_group(self):self.records[-1]=copy.deepcopy(self.records[-2]);self.rewrite();self.reject()
    def test_groups_SHA(self):self.group_sha='0'*64;self.reject()
    def test_summary_SHA(self):self.summary_sha='0'*64;self.reject()
    def test_partial_last_row(self):
        self.groups.write_bytes(self.groups.read_bytes()[:-1]);self.group_sha=sha(self.groups.read_bytes())
        self.document['remote_group_artifact_sha256']=self.group_sha;self.summary.write_bytes(canonical(self.document));self.summary_sha=sha(self.summary.read_bytes());self.reject()
    def test_class_count_mismatch(self):self.document['steps'][0]['group_outcomes']['all_C1']+=1;self.rewrite();self.reject()


if __name__=='__main__':unittest.main()
