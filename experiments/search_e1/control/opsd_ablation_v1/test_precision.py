import unittest
import torch
from precision import PrecisionContract,SyncedGradScaler

class PrecisionTests(unittest.TestCase):
    def test_fp32_masters_fp16_embedding_and_linear(self):
        m=torch.nn.Sequential(torch.nn.Embedding(8,4),torch.nn.Linear(4,3));c=PrecisionContract('fp32_master_fp16','cpu').install(m)
        with c.context('teacher'):y=m(torch.tensor([1,2]))
        self.assertEqual(y.dtype,torch.float16);self.assertTrue(all(p.dtype==torch.float32 for p in m.parameters()))
        self.assertTrue(all(e['role']=='teacher' for e in c.events));c.close()
    def test_reject_bf16_upcast_claim(self):
        with self.assertRaises(ValueError):PrecisionContract('fp32_master_fp16','cpu').install(torch.nn.Linear(2,2).bfloat16())
    def test_scaled_sum_no_world_divide_then_unscale_clip_step(self):
        p=torch.nn.Parameter(torch.tensor([1.]));opt=torch.optim.AdamW([p],lr=.01);sc=SyncedGradScaler(scale=8.)
        sc.scale_loss(p.sum()*.1).backward();events=[]
        result=sc.finish_step([p],opt,lambda g:(events.append('sum'),g.mul_(2)),lambda v:(events.append('max'),v)[1],lambda state:events.append('agree'))
        self.assertTrue(result['accepted']);self.assertAlmostEqual(p.grad.item(),.2,places=6)
        self.assertEqual(events[0],'agree');self.assertEqual(opt.state[p]['step'].item(),1)
    def test_remote_overflow_skips_adam_restores_capsule_scale(self):
        p=torch.nn.Parameter(torch.tensor([1.]));opt=torch.optim.AdamW([p],lr=.01);sc=SyncedGradScaler(scale=8.)
        p.grad=torch.tensor([.8]);calls=[]
        def maxflag(flag):
            calls.append(flag);return 1 if len(calls)==3 else flag
        r=sc.finish_step([p],opt,lambda g:None,maxflag,lambda state:None)
        self.assertFalse(r['adam_advanced']);self.assertFalse(r['cursor_advanced']);self.assertEqual(p.item(),1);self.assertEqual(len(opt.state),0)
        resumed=SyncedGradScaler(scale=8.);resumed.load_state_dict(r['scaler_after']);self.assertEqual(resumed.scale,4.)
    def test_forward_failure_is_not_claimed_scaler_repair(self):
        p=torch.nn.Parameter(torch.tensor([1.]));o=torch.optim.AdamW([p]);s=SyncedGradScaler(scale=8.)
        r=s.finish_step([p],o,lambda g:None,lambda v:v,lambda state:None,local_failed=True)
        self.assertEqual(r['failure_kind'],'forward_or_logging_failure');self.assertFalse(r['overflow'])
        self.assertEqual(s.scale,8.);self.assertFalse(r['requires_durable_rng_retry_capsule']);self.assertTrue(r['requires_durable_failure_evidence'])
    def test_frozen_old_in_optimizer_group_is_rejected_before_adam(self):
        current=torch.nn.Parameter(torch.tensor([1.]));old=torch.nn.Parameter(torch.tensor([2.]),requires_grad=False)
        o=torch.optim.AdamW([current,old]);s=SyncedGradScaler()
        with self.assertRaises(ValueError):s.finish_step([current],o,lambda g:None,lambda v:v,lambda state:None)
        self.assertEqual(old.item(),2.);self.assertFalse(o.state)
    def test_local_inf_detected_after_sum(self):
        p=torch.nn.Parameter(torch.tensor([1.]));p.grad=torch.tensor([float('inf')]);o=torch.optim.AdamW([p]);s=SyncedGradScaler(scale=8.)
        self.assertFalse(s.finish_step([p],o,lambda g:None,lambda v:v,lambda state:None)['accepted'])

if __name__=='__main__':unittest.main()
