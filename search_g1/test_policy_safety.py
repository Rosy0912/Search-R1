import math,unittest,tempfile,json
from pathlib import Path
import torch
import torch.multiprocessing as mp
from search_g1.policy_safety import dual_clip_terms,monitor


def admission_collective_worker(rank,path,out,mode):
    """Actual Gloo protocol test with an explicit verifier fixture, never admission."""
    import os,sys
    from types import ModuleType,SimpleNamespace
    from unittest.mock import patch
    from reproduction.common8.policy_calibration_plan import document_sha256
    from search_g1.policy_safety import bind_thresholds
    torch.distributed.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    try:
        base=Path(out+'-base-'+str(rank));(base/'logs').mkdir(parents=True)
        artifact=dict(schema_version=2,dual_clip=3.,target_old_policy_kl=.01,kl_log_bound=30.,
                      negative_tail_limit=.02,fixture_marker=rank if mode=='different' else 0)
        if rank==1 and mode=='schema': artifact['schema_version']=True
        if rank==1 and mode=='threshold': artifact['dual_clip']=4.
        (base/'logs/run-manifest.json').write_text(json.dumps(dict(run_id='fixture-parent')))
        target=base/'logs/policy-safety-calibration.json'
        if rank!=1 or mode!='missing': target.write_text(json.dumps(artifact))
        verifier=ModuleType('reproduction.common8.production_safety_admission')
        def verify_admission(_base,document,_run):
            if rank==1 and mode=='one_local_error': raise AssertionError('One local proof failure')
            return .02, '0'*64 if rank==1 and mode=='verifier_hash' else document_sha256(document)
        verifier.verify_admission=verify_admission
        config=SimpleNamespace(calibration_manifest=str(target),dual_clip=3.,target_old_policy_kl=.01,
                               kl_log_bound=30.,negative_tail_limit=None)
        result=dict(rejected=False,simulated_writes_after_gate=0)
        with patch.dict(os.environ,SEARCH_G1_BASE=str(base)),patch.dict(sys.modules,{verifier.__name__:verifier}):
            try:
                tail,sha=bind_thresholds(config)
                result.update(tail=tail,sha=sha,simulated_writes_after_gate=1)
            except RuntimeError as failure:
                result.update(rejected=True,error=str(failure))
        Path(out+str(rank)).write_text(json.dumps(result))
    finally:
        torch.distributed.destroy_process_group()


def phase_collective_worker(rank,path,out,mode):
    from search_g1.policy_safety import _phase_agreement,DISCOVERY_ROLE
    torch.distributed.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    try:
        if mode=='observe':
            result=monitor(torch.tensor([2.,0.]),torch.tensor([-1.,1.]),torch.ones(2),
                           torch.tensor([.001,.999]),c=3.,kl_log_bound=30.,target_kl=.01,
                           negative_tail_limit=None,execution_role=DISCOVERY_ROLE)
        else:
            rejected=False
            try:
                role=DISCOVERY_ROLE if rank==0 or mode!='role' else 'production'
                _phase_agreement(dict(role=role),['one local admission failure'] if rank==1 and mode=='failure' else [],None)
            except RuntimeError: rejected=True
            result=dict(rejected=rejected)
        Path(out+str(rank)).write_text(json.dumps(result))
    finally:
        torch.distributed.destroy_process_group()


def collective_worker(rank,path,out,weighted=False):
    torch.distributed.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    d=torch.tensor([[.1,.2]] if rank==0 else [[.3,2.]],dtype=torch.float64)
    a=torch.tensor([[-1.,1.]] if rank==0 else [[-1.,-1.]],dtype=torch.float64) if weighted else -torch.ones_like(d)
    w=torch.tensor([[.1,.2]] if rank==0 else [[.3,.4]],dtype=torch.float64) if weighted else torch.ones_like(d)
    r=monitor(d,a,torch.ones_like(d),w,c=3.,kl_log_bound=30.,target_kl=10.,negative_tail_limit=.1)
    Path(out+str(rank)).write_text(json.dumps(r))
    torch.distributed.destroy_process_group()


class SafetyTests(unittest.TestCase):
    def test_schema2_admission_local_proof_errors_and_hash_drift_collectively_reject(self):
        for mode in ('valid','one_local_error','different','schema','threshold','missing','verifier_hash'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                mp.spawn(admission_collective_worker,args=(root+'/init',root+'/rank',mode),nprocs=2,join=True)
                rows=[json.loads(Path(root+'/rank'+str(i)).read_text()) for i in range(2)]
            if mode=='valid':
                self.assertEqual(rows[0],rows[1]);self.assertEqual(rows[0]['tail'],.02)
                self.assertFalse(rows[0]['rejected'])
            else:
                self.assertTrue(all(row['rejected'] and row['simulated_writes_after_gate']==0 for row in rows))
                if mode=='one_local_error':
                    self.assertTrue(rows[0]['error'].endswith('[]'))
                    self.assertIn('One local proof failure',rows[1]['error'])

    def test_discovery_observes_only_tail_and_retains_every_other_brake(self):
        from search_g1.policy_safety import DISCOVERY_ROLE,CONFIRMATION_ROLE
        kwargs=dict(c=3.,kl_log_bound=30.,target_kl=.01,negative_tail_limit=None,execution_role=DISCOVERY_ROLE)
        d=torch.tensor([2.,0.]);a=torch.tensor([-1.,1.]);m=torch.ones(2)
        observed=monitor(d,a,m,torch.tensor([.001,.999]),**kwargs)
        self.assertTrue(observed['allowed']);self.assertEqual(observed['negative_tail_weight_fraction'],1.)
        self.assertEqual(observed['enforcement']['negative_tail'],'observe_only')
        kl=monitor(d,a,m,torch.tensor([.1,.9]),**kwargs)
        self.assertFalse(kl['allowed']);self.assertIn('old_policy_kl',kl['reasons'])
        for value in (float('nan'),31.):
            report=monitor(torch.tensor([value,0.]),a,m,m,**kwargs)
            self.assertFalse(report['allowed']);self.assertFalse(report['kl_estimate_complete'])
        for role in ('production',CONFIRMATION_ROLE):
            with self.assertRaises(ValueError): monitor(d,a,m,m,**dict(kwargs,execution_role=role))
            report=monitor(d,a,m,torch.tensor([.001,.999]),**dict(kwargs,execution_role=role,negative_tail_limit=.1))
            self.assertFalse(report['allowed']);self.assertIn('negative_advantage_tail',report['reasons'])

    def test_fresh_phase_uses_real_adam_state_and_actual_counters(self):
        from search_g1.policy_safety import verify_fresh_phase_state
        parameter=torch.nn.Parameter(torch.tensor([1.]));adam=torch.optim.AdamW([parameter],lr=1e-6)
        verify_fresh_phase_state(adam,1,0)
        for step,writes in ((True,0),(1,False),(2,0),(1,1)):
            with self.assertRaises(ValueError): verify_fresh_phase_state(adam,step,writes)
        parameter.sum().backward();adam.step()
        self.assertTrue(adam.state)
        with self.assertRaises(ValueError): verify_fresh_phase_state(adam,1,0)

    def test_actual_twenty_write_adam_snapshot_does_not_create_or_replace_state(self):
        from search_g1.policy_safety import calibration_state_snapshot
        module=torch.nn.Linear(2,1);adam=torch.optim.AdamW(module.parameters(),lr=1e-6)
        fields,before=calibration_state_snapshot(module,adam)
        self.assertEqual(before['adam_state_entries'],0);self.assertEqual(before['adam_steps'],[])
        for _ in range(20):
            adam.zero_grad();module(torch.ones(1,2)).sum().backward();adam.step()
        original={p:{key:id(value) for key,value in state.items()} for p,state in adam.state.items()}
        fields,after=calibration_state_snapshot(module,adam)
        self.assertEqual(after['adam_steps'],[20.,20.]);self.assertEqual(after['adam_state_tensor_count'],6)
        self.assertEqual(after['parameter_numel'],3)
        self.assertTrue(all(torch.isfinite(value).all() for value in fields.values()))
        self.assertEqual(original,{p:{key:id(value) for key,value in state.items()} for p,state in adam.state.items()})

    def test_cuda_uuid_count_is_strict_and_single_visible_queries_only_ordinal_zero(self):
        from unittest.mock import patch
        from reproduction.common8.native_allocation import cuda_uuids
        for count in (True,0,-1,1.,None):
            with self.assertRaises(ValueError): cuda_uuids(count=count)
        calls=[]
        class FakeCall:
            def __call__(self,buffer,ordinal):
                calls.append(ordinal)
                buffer[256:272]=bytes([ordinal+1])*16
                return 0
        class FakeLibrary:
            cudaGetDeviceProperties_v2=FakeCall()
        header='struct cudaDeviceProp { char name[256]; cudaUUID_t uuid;'
        with patch('ctypes.CDLL',return_value=FakeLibrary()),patch.object(Path,'read_text',return_value=header):
            one=cuda_uuids(count=1)
            self.assertEqual(calls,[0]);self.assertEqual(len(one),1)
            calls.clear();eight=cuda_uuids()
            self.assertEqual(calls,list(range(8)));self.assertEqual(len(set(eight)),8)

    def test_phase_role_and_local_failure_collective_and_observed_tail(self):
        for mode in ('role','failure','observe'):
            with tempfile.TemporaryDirectory() as root:
                mp.spawn(phase_collective_worker,args=(root+'/init',root+'/rank',mode),nprocs=2,join=True)
                results=[json.loads(Path(root+'/rank'+str(i)).read_text()) for i in range(2)]
            self.assertEqual(results[0],results[1])
            self.assertTrue(results[0]['allowed'] if mode=='observe' else results[0]['rejected'])
    def test_direct_formula_values_and_gradients(self):
        d=torch.tensor([-2.,-.1,.1,.3,1.,2.]*3,dtype=torch.float64,requires_grad=True)
        a=torch.tensor([1.]*6+[-1.]*6+[0.]*6,dtype=torch.float64)
        ratio=d.exp();ordinary=torch.maximum(-a*ratio,-a*ratio.clamp(.8,1.2))
        direct=torch.where(a<0,torch.minimum(ordinary,-a*3.),ordinary)
        stable=dual_clip_terms(d,a,torch.ones_like(a),.2,3.)
        self.assertTrue(torch.allclose(stable,direct))
        g1=torch.autograd.grad(stable.sum(),d,retain_graph=True)[0]
        g2=torch.autograd.grad(direct.sum(),d)[0]
        self.assertTrue(torch.allclose(g1,g2))

    def test_extreme_zero_padding_and_gradients(self):
        d=torch.tensor([800.,-800.,800.,float('nan')],requires_grad=True)
        a=torch.tensor([-1.,1.,0.,float('nan')]);m=torch.tensor([1.,1.,1.,0.])
        terms=dual_clip_terms(d,a,m,.2,3.);terms.sum().backward()
        self.assertTrue(torch.isfinite(terms).all() and torch.isfinite(d.grad).all())
        self.assertAlmostEqual(float(terms[0].detach()),3.);self.assertEqual(float(d.grad[0]),0.)
        with self.assertRaises(ValueError):dual_clip_terms(d,a,torch.ones_like(m),.2,3.)

    def test_statistics_reject_invalid_and_over_bound_without_fake_kl(self):
        for value in [800.,float('nan'),float('inf')]:
            d=torch.tensor([value]);r=monitor(d,-torch.ones(1),torch.ones(1),torch.ones(1),c=3.,kl_log_bound=30.,target_kl=.01,negative_tail_limit=.1)
            self.assertFalse(r['allowed']);self.assertIsNone(r['sampled_old_policy_kl'])
            self.assertFalse(r['kl_estimate_complete']);self.assertEqual(r['weighted_k3_numerator'],0.)
        r=monitor(torch.tensor([float('nan'),0.]),torch.tensor([float('nan'),1.]),torch.tensor([0.,1.]),torch.tensor([float('inf'),1.]),c=3.,kl_log_bound=30.,target_kl=.01,negative_tail_limit=.1)
        self.assertTrue(r['allowed']);self.assertEqual(r['sampled_old_policy_kl'],0.)

    def test_empty_and_negative_tail(self):
        d=torch.tensor([2.,0.]);a=-torch.ones(2);m=torch.ones(2)
        r=monitor(d,a,m,m,c=3.,kl_log_bound=30.,target_kl=10.,negative_tail_limit=.1)
        self.assertEqual(r['negative_tail_count_fraction'],.5);self.assertFalse(r['allowed'])
        r=monitor(d,a,torch.zeros(2),torch.zeros(2),c=3.,kl_log_bound=30.,target_kl=.01,negative_tail_limit=.1)
        self.assertIn('empty_effective_samples',r['reasons'])

    def test_existing_objective_weights_survive_partition(self):
        d=torch.tensor([[-2.,.1],[.2,1.],[2.,.3],[-.3,.4]],requires_grad=True)
        a=torch.tensor([[1.,1.],[-1.,-1.],[1.,-1.],[-1.,1.]])
        m=torch.tensor([[1.,1.],[1.,0.],[1.,1.],[0.,1.]])
        weights=m/torch.tensor([[3.],[3.],[3.],[3.]])
        full=(dual_clip_terms(d,a,m,.2,3.)*weights).sum()
        split=sum((dual_clip_terms(dd,aa,mm,.2,3.)*ww).sum() for dd,aa,mm,ww in zip(d.split(1),a.split(1),m.split(1),weights.split(1)))
        self.assertTrue(torch.equal(full,split))

    def test_actual_two_rank_global_tail_and_decision(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(collective_worker,args=(root+'/init',root+'/rank'),nprocs=2,join=True)
            a,b=[json.loads(Path(root+'/rank'+str(i)).read_text()) for i in range(2)]
        self.assertEqual(a,b);self.assertFalse(a['allowed'])
        self.assertEqual(a['negative_tail_count_fraction'],.25)
        self.assertEqual(a['negative_log_ratio_p99'],2.)
        expected=sum(math.expm1(x)-x for x in [.1,.2,.3,2.])/4
        self.assertAlmostEqual(a['sampled_old_policy_kl'],expected)

    def test_global_accumulators_independently_reconstruct_weighted_ratios(self):
        deltas=[.1,2.,-.2,.3]
        advantages=[-1.,-1.,1.,-1.]
        weights=[.1,.2,.3,.4]
        r=monitor(torch.tensor(deltas+[float('nan')],dtype=torch.float64),
                  torch.tensor(advantages+[float('nan')],dtype=torch.float64),
                  torch.tensor([1.,1.,1.,1.,0.],dtype=torch.float64),
                  torch.tensor(weights+[float('inf')],dtype=torch.float64),
                  c=3.,kl_log_bound=30.,target_kl=10.,negative_tail_limit=.5)
        numerator=sum(w*(math.expm1(d)-d) for d,w in zip(deltas,weights))
        mass=sum(weights)
        negative_mass=sum(w for a,w in zip(advantages,weights) if a<0)
        tail_mass=sum(w for d,a,w in zip(deltas,advantages,weights) if a<0 and d>math.log(3.))
        self.assertTrue(r['allowed'] and r['kl_estimate_complete'])
        for key,expected in [('weighted_k3_numerator',numerator),('effective_weight_mass',mass),
                             ('negative_effective_weight_mass',negative_mass),('tail_effective_weight_mass',tail_mass)]:
            self.assertAlmostEqual(r[key],expected)
        self.assertAlmostEqual(r['sampled_old_policy_kl'],r['weighted_k3_numerator']/r['effective_weight_mass'])
        self.assertAlmostEqual(r['negative_tail_weight_fraction'],r['tail_effective_weight_mass']/r['negative_effective_weight_mass'])
        self.assertAlmostEqual(r['dual_clip_saturated_objective_weight_fraction'],r['tail_effective_weight_mass']/r['effective_weight_mass'])
        self.assertNotAlmostEqual(r['negative_tail_weight_fraction'],r['negative_tail_count_fraction'])

    def test_nonfinite_global_totals_do_not_emit_finite_accumulators(self):
        r=monitor(torch.zeros(2,dtype=torch.float64),-torch.ones(2,dtype=torch.float64),
                  torch.ones(2,dtype=torch.float64),torch.full((2,),1e308,dtype=torch.float64),
                  c=3.,kl_log_bound=30.,target_kl=10.,negative_tail_limit=.5)
        self.assertFalse(r['allowed'] or r['kl_estimate_complete'])
        self.assertIn('nonfinite_global_totals',r['reasons'])
        for key in ('weighted_k3_numerator','effective_weight_mass',
                    'negative_effective_weight_mass','tail_effective_weight_mass'):
            self.assertIsNone(r[key])

    def test_two_rank_replicated_reports_are_one_global_observation(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(collective_worker,args=(root+'/init',root+'/rank',True),nprocs=2,join=True)
            reports=[json.loads(Path(root+'/rank'+str(i)).read_text()) for i in range(2)]
        self.assertEqual(reports[0],reports[1])
        expected_numerator=sum(w*(math.expm1(d)-d) for d,w in zip([.1,.2,.3,2.],[.1,.2,.3,.4]))
        for report in reports:
            self.assertFalse(report['allowed']);self.assertEqual(report['reasons'],['negative_advantage_tail'])
            self.assertAlmostEqual(report['weighted_k3_numerator'],expected_numerator)
            self.assertAlmostEqual(report['effective_weight_mass'],1.)
            self.assertAlmostEqual(report['negative_effective_weight_mass'],.8)
            self.assertAlmostEqual(report['tail_effective_weight_mass'],.4)
            self.assertAlmostEqual(report['sampled_old_policy_kl'],expected_numerator)
            self.assertAlmostEqual(report['negative_tail_weight_fraction'],.5)
            self.assertAlmostEqual(report['dual_clip_saturated_objective_weight_fraction'],.4)
            self.assertEqual(report['effective_token_count'],4)
            self.assertEqual(report['negative_token_count'],3)
            self.assertEqual(report['negative_tail_count'],1)
        # Select one consensus report for this observation; eight copies in rank
        # logs must not multiply its already-reduced numerator, masses or counts.
        replicated_rank_logs=reports*4
        self.assertTrue(all(row==replicated_rank_logs[0] for row in replicated_rank_logs))
        observation=replicated_rank_logs[0]
        self.assertAlmostEqual(observation['effective_weight_mass'],1.)
        self.assertEqual(observation['effective_token_count'],4)
        self.assertNotAlmostEqual(sum(row['effective_weight_mass'] for row in replicated_rank_logs),1.)


if __name__=='__main__':unittest.main()
