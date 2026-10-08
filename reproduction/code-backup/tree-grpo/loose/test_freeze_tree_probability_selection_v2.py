"""Tiny CPU counterexamples for immutable group selection and sampled ranges."""
import hashlib
import json
import unittest

import torch
import freeze_tree_probability_selection_v2 as F


def raw():
    return dict(input_ids=torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]),
        responses=torch.tensor([[3, 4], [3, 4]]), attention_mask=torch.ones((2,4),dtype=torch.int64),
        position_ids=torch.tensor([[0,1,2,3],[0,1,2,3]]),
        old_log_probs=torch.tensor([[-1.,-2.],[-1.,-2.]]),
        sampled_log_probs=torch.tensor([[-1.1,-2.1],[-1.2,-2.2]]))


def monitor():
    v=raw();v.pop('sampled_log_probs');v.update(loss_mask=torch.ones((2,2),dtype=torch.int64),
        policy_weights=torch.full((2,2),.5)); return v


def row(i, max_abs, fraction=.5, length=7, anomaly=False, sha='f'):
    return dict(asset='raw',row=i,source_group=i//4,max_abs=max_abs,
        above_dot01=1,effective_tokens=2,above_dot01_fraction=fraction,full_attention_tokens=length,
        position_anomaly=anomaly,row_tensor_sha256=sha*64)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.inventory={'raw':{'original_micro4_groups':[list(range(i,min(i+4,18))) for i in range(0,18,4)]}}

    def test_max_whole_group_and_no_substitution(self):
        groups,cats=F.choose_groups([row(5,2.)],self.inventory)
        self.assertEqual(groups[0]['row_indices'],[4,5,6,7]);self.assertEqual(len(groups),1)
        self.assertEqual([c['status'] for c in cats[-2:]],['ABSENT','ABSENT'])

    def test_five_criteria_partial_last_group(self):
        groups,cats=F.choose_groups([row(0,4.,.1),row(4,3.,1.),row(8,2.,.1,99),
            row(12,1.,.1,anomaly=True),row(17,.001,.0,sha='0')],self.inventory)
        self.assertEqual([g['row_indices'] for g in groups][-1],[16,17]);self.assertEqual(sum(len(g['row_indices']) for g in groups),18)
        self.assertTrue(all(c['status']=='SELECTED' for c in cats))

    def test_stable_ties_and_hash_control(self):
        rows=[row(5,.001,sha='0'),row(1,.001,sha='1')]
        g,c=F.choose_groups(rows,self.inventory); g2,c2=F.choose_groups(rows[::-1],self.inventory)
        self.assertEqual((g,c),(g2,c2));self.assertEqual(c[0]['row'],1);self.assertEqual(c[-1]['row'],5)

    def test_no_valid_refused(self):
        r=row(0,0);r['effective_tokens']=0
        with self.assertRaises(ValueError):F.choose_groups([r],self.inventory)

    def test_row_full_hash_and_invalid_delta(self):
        value=raw();r=F.describe_rows(value,'raw');self.assertFalse(r[0]['position_anomaly'])
        self.assertNotEqual(r[0]['row_tensor_sha256'],r[1]['row_tensor_sha256'])
        value['sampled_log_probs'][0,0]=float('nan')
        with self.assertRaises(ValueError):F.describe_rows(value,'raw')

    def test_anomaly_includes_position_offset_and_right_padding(self):
        value=raw();value['position_ids'][0]+=1;value['attention_mask'][1,1]=0
        r=F.describe_rows(value,'raw');self.assertTrue(r[0]['position_anomaly']);self.assertTrue(r[1]['position_anomaly'])

    def test_duplicate_range_is_not_mean(self):
        idx=F.A.RawIndex();idx.add_snapshot(raw(),'fixture');v,c=F.monitor_companion(idx,monitor(),{},'monitor','a'*64)
        self.assertEqual(c['matched_tokens'],4);self.assertTrue(bool((v['matched_count']==2).all()))
        self.assertLess(float(v['sampled_min'][0,0]),float(v['sampled_max'][0,0]))
        self.assertEqual(v['exact_context_key'].shape,(2,2,48))

    def test_unknown_context_and_loss_mask_exclusion(self):
        idx=F.A.RawIndex();idx.add_snapshot(raw(),'fixture');m=monitor();m['input_ids'][0,0]=8;m['loss_mask'][1,0]=0
        v,c=F.monitor_companion(idx,m,{},'monitor','a'*64)
        self.assertEqual(c['effective_tokens'],3);self.assertEqual(c['unknown_tokens'],2)
        self.assertFalse(bool(v['matched_mask'][0].any()));self.assertTrue(bool((v['sampled_min'][0]==0).all()))
        self.assertEqual(int(v['matched_count'][1,0]),0)

    def test_padding_never_creates_matching_prefix(self):
        idx=F.A.RawIndex();idx.add_snapshot(raw(),'fixture');m=monitor();m['attention_mask'][0,0]=0
        _,c=F.monitor_companion(idx,m,{},'monitor','a'*64);self.assertEqual(c['unknown_tokens'],2)

    def test_actual_cpu_terminal_schema_and_report_binding(self):
        d=dict(status='ACTUAL_H1009_CPU3273_REPORTS_AND_TERMINAL_INDEPENDENTLY_VERIFIED',
            job_id=3273,parent_calibration_job_id=3268,slurm_state='COMPLETED',slurm_exit_code='0:0',
            host='h100-9',gpu_count=0,cpu_count=4,ready_sha256=F.READY_SHA,
            package_manifest_sha256=F.PACKAGE_SHA,actual_optimizer_writes=0,actual_model_forwards=0,
            position_report_sha256=F.CPU_REPORTS['position-analysis-3273.json'],
            action_report_sha256=F.CPU_REPORTS['retained-actions-3273.json'])
        F.verify_cpu_terminal(d)
        for key,val in [('actual_optimizer_writes',1),('actual_optimizer_writes',False),
            ('actual_model_forwards',1),('slurm_state','FAILED'),('job_id',3271),
            ('action_report_sha256','0'*64)]:
            with self.assertRaises(ValueError):F.verify_cpu_terminal(dict(d,**{key:val}))

    def test_invalid_sampling_distribution_refused(self):
        d=dict(temperature=1.,top_p=1.,top_k=-1,min_p=0.,presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.,n=1)
        F.plain_distribution(d)
        for key,val in [('top_p',.9),('temperature',.6),('logits_processors',['x'])]:
            altered=dict(d,**{key:val})
            with self.assertRaises(ValueError): F.plain_distribution(altered)


if __name__=='__main__':
    unittest.main(verbosity=2)
