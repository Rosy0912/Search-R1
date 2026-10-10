"""Small local-only group-controller fixtures; no node, raw data or GPU."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('group_control',HERE/'group_statistics_control_v1.py')
ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
from test_selected_reward_group_distributions import toy_rows
import selected_reward_group_distributions as dist


class GroupControlTests(unittest.TestCase):
    def ctx(self):return ctl.context(HERE/'historical_replay/parallel_selected/cohort-config.json')
    def bridge(self):
        c=self.ctx();S=Path(c['stats_source']);D=Path(c['diagnostic_base'])
        return dict(selected_steps=ctl.SELECTED,diagnostic_base=str(D),expected_node=c['cohort']['expected_node'],
            native_run_id=c['cohort']['original_run_id'],original_base=c['cohort']['base'],
            pairwise_source_path=str(S/'pairwise_pearson.py'),pairwise_source_sha256=c['dependency_sha256']['pairwise_pearson.py'],
            C_helper_path=c['C_helper'],C_helper_sha256=ctl.C_SHA,output_new=str(D/'statistics-selected-ff98085b6918-v2'))
    def build(self,b):
        return ctl.make_config(self.ctx(),ctl.canonical(b),b['output_new']+'/selected-pair-statistics.json','a'*64)
    def test_explicit_accepted_bridge_identity_and_new_output(self):
        b=self.bridge();c=self.build(b)
        self.assertEqual(c['bridge_config_sha256'],ctl.sha(ctl.canonical(b)))
        self.assertEqual(c['accepted_pair_statistics_sha256'],'a'*64)
        self.assertTrue(c['output_new'].endswith('group-statistics-selected-ff98085b6918-v1'))
        self.assertNotEqual(c['output_new'],b['output_new']);self.assertTrue(c['group_artifacts_remote_only'])
    def test_wrong_bridge_dependency_or_scope_HOLD(self):
        for key,value in [('selected_steps',[1]),('C_helper_sha256','a'*64),('pairwise_source_sha256','b'*64),('native_run_id','c'*32)]:
            b=self.bridge();b[key]=value
            with self.assertRaises(AssertionError):self.build(b)
    def test_wrong_remote_report_directory_or_hash_HOLD(self):
        b=self.bridge()
        with self.assertRaises(AssertionError):ctl.make_config(self.ctx(),ctl.canonical(b),'/data/foreign/selected-pair-statistics.json','a'*64)
        with self.assertRaises(AssertionError):ctl.make_config(self.ctx(),ctl.canonical(b),b['output_new']+'/selected-pair-statistics.json','notSHA')
    def test_full8_compact_summary_fits_no_group_members(self):
        rows,run,step,attempt=toy_rows();s,_=dist.analyze_rows(rows,run,step,attempt)
        summaries=[]
        for n in ctl.SELECTED:
            q=copy.deepcopy(s);q.update(step=n,probe_version='round-100');summaries.append(q)
        r=dict(status='ACCEPTED_SELECTED8_RECONSTRUCTED_REWARD_GROUP_DISTRIBUTIONS',selected_steps=ctl.SELECTED,
               trajectories=20480,original_five_sibling_groups=4096,steps=summaries)
        compact=ctl.compact_report(r);raw=ctl.canonical(compact)
        self.assertLess(len(raw),31*1024)
        self.assertNotIn(b'member_trajectory_ids',raw);self.assertNotIn(b'upstream_group_sha256',raw)
        self.assertTrue(compact['raw_groups_not_exported']);self.assertEqual(len(compact['steps']),8)
        for out in compact['steps']:
            self.assertEqual(sum(out['group_outcomes'].values()),512)
            self.assertEqual(out['conditional']['all']['N'],2560)
    def test_embedded_remote_code_compiles_CPU_only(self):
        for text in [ctl.REMOTE_DEPLOY,ctl.REMOTE_EXECUTE]:ast.parse(text)
        self.assertIn('Ran 32 tests',ctl.REMOTE_DEPLOY);self.assertNotIn('nvidia-smi',ctl.REMOTE_COMMON)
        self.assertIn("CUDA_VISIBLE_DEVICES=''",ctl.REMOTE_COMMON)
        self.assertLess(ctl.REMOTE_EXECUTE.index("accepted['total_pairs']==20480"),ctl.REMOTE_EXECUTE.index('r=subprocess.run'))
        self.assertIn('len(encoded)<=32*1024',ctl.REMOTE_EXECUTE)
    def test_publication_rejects_changed_bytes_before_contact(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d).resolve();(base/'toy.py').write_bytes(b'toy');p=base/'p.json'
            p.write_text(json.dumps(dict(source_commit='fake',files=[dict(path=ctl.PREFIX+'toy.py',sha256='0'*64)])))
            with patch.object(ctl,'HERE',base):
                with self.assertRaises(AssertionError):ctl.publication(p,['toy.py'])


if __name__=='__main__':unittest.main()
