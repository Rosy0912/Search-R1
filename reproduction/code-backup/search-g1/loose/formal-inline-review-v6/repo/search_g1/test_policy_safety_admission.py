"""CPU fixture schema/config tests; fixtures are not actual GPU admission evidence."""
import tempfile,unittest,json,hashlib,os
from pathlib import Path
from unittest.mock import patch
import torch,torch.multiprocessing as mp
from omegaconf import OmegaConf
from search_g1.policy_safety import bind_thresholds,bind_actual_config


def config_worker(rank,init,out):
    torch.distributed.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=2)
    c=OmegaConf.create(dict(clip_ratio=.2,use_dynamic_bsz=False,policy_safety=dict(enabled=rank==1,dual_clip=3.,target_old_policy_kl=.01,kl_log_bound=30.,monitor_trajectories_per_rank=4)))
    try:bind_actual_config(c);result='wrongly_allowed'
    except RuntimeError:result='collectively_rejected'
    Path(out+str(rank)).write_text(result);torch.distributed.destroy_process_group()


def empty_selection_worker(rank,init,out):
    from search_g1.numerics import require_finite
    torch.distributed.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=2)
    lengths=torch.tensor([1.,2.,3.,0. if rank==1 else 4.])
    try:
        require_finite({'monitor_actor_lengths':lengths},stage='frozen_monitor_selection_before_any_minibatch',positive=('monitor_actor_lengths',))
        result='wrongly_allowed'
    except RuntimeError:result='collectively_rejected_before_write'
    Path(out+str(rank)).write_text(result);torch.distributed.destroy_process_group()


class AdmissionTests(unittest.TestCase):
    def fixture(self,root):
        b=Path(root);(b/'logs').mkdir()
        run=dict(run_id='fixture',initial_model_sha256={'weights':'fixture'},source_sha256={'source':'fixture'},input_sha256={'data':'fixture'})
        (b/'logs/run-manifest.json').write_text(json.dumps(run))
        probability=dict(accepted=True,run_id='fixture',source_sha256=run['source_sha256'],phase='initial',actual_optimizer_writes=0,ranks=list(range(8)))
        segment=dict(accepted=True,run_id='fixture',source_sha256=run['source_sha256'],initial_model_sha256=run['initial_model_sha256'],actual_optimizer_writes=1,all_parameters_and_adam_finite=True,monitor_definition='four_length_stratified_trajectories_per_rank_two_row_micro_weighting',monitor_identity_before={'fixture':1},monitor_identity_after={'fixture':1},old_logprob_sha256_before='fixture',old_logprob_sha256_after='fixture',negative_tail_limit=.02,target_old_policy_kl=.01)
        artifact=dict(accepted=True,role='healthy_pretraining_policy_calibration',**run,dual_clip=3.,target_old_policy_kl=.01,kl_log_bound=30.,negative_tail_limit=.02,actual_probability_gate_accepted=True,actual_native_short_segment_accepted=True)
        for name,data in [('probability',probability),('short_segment',segment)]:
            rel='logs/'+name+'.json';content=json.dumps(data).encode();(b/rel).write_bytes(content)
            artifact[name+'_proof']=rel;artifact[name+'_proof_sha256']=hashlib.sha256(content).hexdigest()
        target=b/'logs/policy-safety-calibration.json';target.write_text(json.dumps(artifact))
        config=OmegaConf.create(dict(calibration_manifest=str(target),dual_clip=3.,target_old_policy_kl=.01,kl_log_bound=30.,negative_tail_limit=None))
        return config,target,artifact

    def test_fixture_binding_and_corrupt_proof_rejection(self):
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,SEARCH_G1_BASE=root):
            c,p,a=self.fixture(root);self.assertEqual(bind_thresholds(c)[0],.02)
            (Path(root)/a['short_segment_proof']).write_text('{}')
            with self.assertRaises(RuntimeError):bind_thresholds(c)

    def test_string_false_is_rejected(self):
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,SEARCH_G1_BASE=root):
            c,p,a=self.fixture(root);a['accepted']='false';p.write_text(json.dumps(a))
            with self.assertRaises(RuntimeError):bind_thresholds(c)

    def test_actual_core_loss_keeps_masked_mean_and_extreme_gradients(self):
        from verl.trainer.ppo.core_algos import compute_policy_loss
        new=torch.tensor([800.,-800.,800.,float('nan')],requires_grad=True)
        old=torch.zeros(4);a=torch.tensor([-1.,1.,0.,float('nan')]);mask=torch.tensor([1.,1.,1.,0.])
        loss,clip,kl=compute_policy_loss(old,new,a,mask,.2,dual_clip=3.)
        self.assertAlmostEqual(float(loss.detach()),1.);loss.backward()
        self.assertTrue(torch.isfinite(new.grad).all());self.assertTrue(torch.isfinite(kl))

    def test_actual_two_rank_enabled_mismatch_rejects_before_branch(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(config_worker,args=(root+'/init',root+'/rank'),nprocs=2,join=True)
            self.assertEqual([Path(root+'/rank'+str(i)).read_text() for i in range(2)],['collectively_rejected']*2)

    def test_required_run_rejects_disabled_or_nonboolean_switch(self):
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,SEARCH_G1_BASE=root,SEARCH_G1_PROFILE='common8'):
            (Path(root)/'logs').mkdir()
            (Path(root)/'logs/run-manifest.json').write_text(json.dumps({'policy_safety_required':True}))
            for enabled in (False,0,'false'):
                with self.subTest(enabled=enabled),self.assertRaises(RuntimeError):
                    bind_actual_config(OmegaConf.create({'policy_safety':{'enabled':enabled}}))

    def test_empty_monitor_selection_rejects_all_ranks_before_write(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(empty_selection_worker,args=(root+'/init',root+'/rank'),nprocs=2,join=True)
            self.assertEqual([Path(root+'/rank'+str(i)).read_text() for i in range(2)],['collectively_rejected_before_write']*2)


if __name__=='__main__':unittest.main()
