import copy
import json
from pathlib import Path
import tempfile
import unittest

from .contracts import sha, milestone_job_name
from .repeats import validate_repeat, inspect_repeat, repeat_job_name


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

    def ablation(self, group, outer, index=2):
        parent=copy.deepcopy(self.parent)
        parent.update(group=group,outer=outer,
            job=milestone_job_name(group,outer,parent['checkpoint_identity'],'slurm_lumia'))
        sub=dict(self.sub,group=group,outer=outer,job=parent['job'])
        plan=copy.deepcopy(self.plan)
        plan.update(group=group,outer=outer,job=repeat_job_name(parent,index))
        plan['repeat']['index']=index
        plan['output']=plan['output'].replace('/r2/','/r'+str(index)+'/')
        return plan,parent,sub

    def test_authorized_remaining_ablations_both_repeats(self):
        for group,outer in [('budget-5',100),('budget-5',150),
                            ('budget-20',150),('generation-2',150)]:
            for index in (2,3):
                with self.subTest(group=group,outer=outer,index=index):
                    plan,parent,sub=self.ablation(group,outer,index)
                    self.assertEqual(validate_repeat(plan,parent,sub).name,'r'+str(index))

    def test_prior_completed_checkpoints_remain_out_of_repeat_scope(self):
        for group,outer in [('original',150),('C1-root-only',150),('C2-random-final',150),
                            ('budget-5',50),('budget-20',100),('generation-2',100),
                            ('position-early',50),('position-late',50)]:
            with self.subTest(group=group,outer=outer):
                with self.assertRaises(ValueError):validate_repeat(*self.ablation(group,outer))

    def test_ablation_repeat_cannot_change_science_or_parent(self):
        plan,parent,sub=self.ablation('generation-2',150)
        for field,value in [('training_config_sha256','0'*64),('model_files',{}),
                            ('group','budget-5'),('backend','pj_pt')]:
            changed=copy.deepcopy(plan);changed[field]=value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):validate_repeat(changed,parent,sub)
        with self.assertRaises(ValueError):validate_repeat(plan,parent,dict(sub,group='budget-5'))
        for field in ('replaces_job_id','pj_retired_submission'):
            changed=copy.deepcopy(parent);changed[field]='old'
            with self.assertRaises(ValueError):validate_repeat(plan,changed,sub)
        for index in (1,4,5,True):
            with self.assertRaises(ValueError):repeat_job_name(parent,index)

    def test_ablation_on_disk_parent_and_submission_binding(self):
        plan,parent,sub=self.ablation('budget-5',100)
        self.pp.write_text(json.dumps(parent));sub['plan_sha256']=sha(self.pp)
        self.sp.write_text(json.dumps(sub))
        plan['repeat'].update(parent_plan_sha256=sha(self.pp),parent_submission_sha256=sha(self.sp))
        self.assertEqual(inspect_repeat(plan).name,'r2')
        sub['outer']=150;self.sp.write_text(json.dumps(sub))
        plan['repeat']['parent_submission_sha256']=sha(self.sp)
        with self.assertRaises(ValueError):inspect_repeat(plan)


    def test_replacement_parent_repeats_bind_recovery_namespace_and_ancestor(self):
        old=dict(self.sub);old['job_id']='139019';old['plan_sha256']='9'*64
        self.sp.write_text(json.dumps(old))
        parent=copy.deepcopy(self.parent)
        parent.update(replaces_job_id='139019',previous_submission=str(self.sp),
            previous_submission_sha256=sha(self.sp),replaces_plan_sha256='9'*64)
        root=Path(parent['output']).parent/('recovery-'+parent['evaluation_code_commit'][:12])
        root.mkdir();parent['output']=str(root/'output');parent['job']+='-h1'
        self.pp.write_text(json.dumps(parent))
        sub=dict(self.sub,job_id='139020',job=parent['job'],output=parent['output'],plan_sha256=sha(self.pp))
        sp=root/'submission.json';sp.write_text(json.dumps(sub))
        plan=copy.deepcopy(parent)
        plan.update(evaluation_run_id='2'*32,job=repeat_job_name(parent,2),
                    output=str(root/'repeats'/parent['evaluation_run_id']/'r2'/'output'),
                    repeat=dict(index=2,count=3,parent_plan=str(self.pp),parent_plan_sha256=sha(self.pp),
                                parent_submission=str(sp),parent_submission_sha256=sha(sp)))
        self.assertEqual(inspect_repeat(plan),Path(plan['output']).parent)
        # Entry must retain the repeat namespace, not redirect it to the first recovery.
        from unittest.mock import patch
        from types import SimpleNamespace
        from . import entry
        planfile=Path(self.tmp.name)/'repeat.json';planfile.write_text(json.dumps(plan))
        modules={'omegaconf':SimpleNamespace(OmegaConf=None),
                 'search_r1.baselines.preflight':SimpleNamespace(validate_gpu_lease=None)}
        with patch.dict('sys.modules',modules),patch('sys.argv',['entry','--plan',str(planfile)]),patch.object(entry,'inspect_inputs',return_value=(None,[])),patch.object(entry,'run_frozen') as run:
            entry.main()
        self.assertEqual(run.call_args.args[-1],Path(plan['output']).parent)
        bad=copy.deepcopy(plan);bad['previous_submission_sha256']='0'*64
        with self.assertRaises(ValueError):inspect_repeat(bad)
        bad=copy.deepcopy(plan);bad['output']=self.plan['output']
        with self.assertRaises(ValueError):inspect_repeat(bad)
        old['checkpoint_identity']='0'*64;self.sp.write_text(json.dumps(old))
        with self.assertRaises(ValueError):inspect_repeat(plan)


if __name__ == '__main__':unittest.main()
