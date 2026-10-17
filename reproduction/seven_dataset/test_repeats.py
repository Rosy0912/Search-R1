import copy
import json
from pathlib import Path
import tempfile
import unittest

from .contracts import sha
from .repeats import validate_repeat, inspect_repeat


class RepeatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.parent = dict(protocol='tree_seven_milestone_v1', group='original', outer=200,
            backend='slurm_lumia', base=str(root), checkpoint_identity='a'*64,
            evaluation_code_commit='b'*40, evaluation_run_id='c'*32,
            job='zhangjunming-ttt-e-orig-200-aaaa',
            output=str(root/'seven-evaluations'/('a'*64)/'output'),
            model_files={'weights':'d'*64}, training_config_sha256='e'*64,
            repeat_context=dict(requested_repeats=3,candidate_repeat_index=1))
        self.pp=root/'parent.json';self.pp.write_text(json.dumps(self.parent))
        first=Path(self.parent['output']).parent;first.mkdir(parents=True)
        self.sp=first/'submission.json'
        self.sub=dict(checkpoint_identity='a'*64,group='original',outer=200,
            job=self.parent['job'],output=self.parent['output'],plan=str(self.pp),
            plan_sha256=sha(self.pp),job_id='138984')
        self.sp.write_text(json.dumps(self.sub))
        self.plan=copy.deepcopy(self.parent)
        self.plan.update(evaluation_code_commit='f'*40,evaluation_run_id='1'*32,
            job='zhangjunming-ttt-e-orig-200-r2-aaaa',
            output=str(first/'repeats'/('c'*32)/'r2'/'output'),
            repeat=dict(index=2,count=3,parent_plan=str(self.pp),parent_plan_sha256=sha(self.pp),
                        parent_submission=str(self.sp),parent_submission_sha256=sha(self.sp)))

    def test_two_unique_outputs_three_total(self):
        two=inspect_repeat(self.plan)
        p=copy.deepcopy(self.plan);p['repeat']['index']=3
        p['job']=p['job'].replace('-r2-','-r3-');p['output']=p['output'].replace('/r2/','/r3/')
        self.assertNotEqual(two,inspect_repeat(p))

    def test_rejects_frozen_field_changes(self):
        for field,value in [('model_files',{}),('training_config_sha256','0'*64),
                            ('outer',150),('base','/elsewhere'),('repeat_context',{})]:
            with self.subTest(field=field):
                p=copy.deepcopy(self.plan);p[field]=value
                with self.assertRaises(Exception):inspect_repeat(p)

    def test_rejects_extra_or_replayed_repeat(self):
        for index in (1,4,True,'2'):
            p=copy.deepcopy(self.plan);p['repeat']['index']=index
            with self.assertRaises(Exception):inspect_repeat(p)
        p=copy.deepcopy(self.plan);p['evaluation_run_id']=self.parent['evaluation_run_id']
        with self.assertRaises(Exception):inspect_repeat(p)

    def test_rejects_first_output_and_wrong_name(self):
        for field,value in [('output',self.parent['output']),('job',self.parent['job'])]:
            p=copy.deepcopy(self.plan);p[field]=value
            with self.assertRaises(Exception):inspect_repeat(p)

    def test_rejects_parent_or_submission_tamper(self):
        self.pp.write_text(self.pp.read_text()+' ')
        with self.assertRaises(Exception):inspect_repeat(self.plan)
        self.pp.write_text(json.dumps(self.parent));self.sp.write_text('{}')
        with self.assertRaises(Exception):inspect_repeat(self.plan)

    def test_rejects_unrelated_receipt_and_scope(self):
        s=dict(self.sub,checkpoint_identity='0'*64)
        with self.assertRaises(Exception):validate_repeat(self.plan,self.parent,s)
        parent=dict(self.parent,group='budget-20')
        with self.assertRaises(Exception):validate_repeat(self.plan,parent,self.sub)

    def test_rejects_unknown_fields(self):
        p=copy.deepcopy(self.plan);p['science_override']=True
        with self.assertRaises(Exception):inspect_repeat(p)
        p=copy.deepcopy(self.plan);p['repeat']['unbounded_count']=True
        with self.assertRaises(Exception):inspect_repeat(p)


if __name__ == '__main__':unittest.main()
