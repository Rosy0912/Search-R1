"""Independent CPU negative controls; no model, CUDA, network or real processes."""
import copy
import unittest
from unittest.mock import patch

import torch

try:
    from . import owned_process_audit as audit
    from .verify_five_updates import verify_rank_states
except ImportError:
    import owned_process_audit as audit
    from verify_five_updates import verify_rank_states


class IndependentCheckpointTests(unittest.TestCase):
    def fixture(self):
        checkpoint = dict(run_id='test-run', optimizer_updates=5,
                          cumulative_pair_exposure=160, cursor={'next_pair_cursor':160},
                          training_input={'pairs':'fixture'}, source_model_identity='source',
                          runtime_identity='runtime', all_rank_adapter_sha256=['actual']*8,
                          ranks=[dict(rank=i,file={'path':str(i)}) for i in range(8)])
        state = dict(schema='opsd-online-rank-state-v1', config_sha256='config',
                     adapter_tensor_sha256='actual',
                     rng=dict(python=(),numpy=(),torch_cpu=torch.zeros(4,dtype=torch.uint8),
                              cuda_all=[torch.zeros(4,dtype=torch.uint8)]),
                     optimizer=dict(param_groups=[dict(params=[0,1],lr=5e-6)],
                                    state={i:dict(step=torch.tensor(5.),exp_avg=torch.zeros(2),
                                                  exp_avg_sq=torch.zeros(2)) for i in range(2)}))
        for key in ('run_id','optimizer_updates','cumulative_pair_exposure','cursor',
                    'training_input','source_model_identity','runtime_identity'):
            state[key]=checkpoint[key]
        states={str(i):dict(copy.deepcopy(state),rank=i) for i in range(8)}
        return checkpoint,dict(_config_sha256='config',scientific={'learning_rate':5e-6}),states

    def check(self,checkpoint,config,states,digest='actual'):
        with patch.object(torch,'load',side_effect=lambda name,**kw: states[name]):
            return verify_rank_states(checkpoint,config,torch,digest,2)

    def test_complete_eight_rank_fixture_accepts(self):
        self.assertEqual(len(self.check(*self.fixture())),8)
        self.assertFalse(torch.cuda.is_initialized())

    def test_identically_missing_adam_state_all_ranks_rejected(self):
        checkpoint,config,states=self.fixture()
        for state in states.values(): del state['optimizer']['state'][1]
        with self.assertRaisesRegex(ValueError,'state is missing'):
            self.check(checkpoint,config,states)

    def test_actual_adapter_digest_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'adapter bytes differ'):
            self.check(*self.fixture(),digest='different-actual-bytes')

    def test_one_rank_moment_difference_rejected(self):
        checkpoint,config,states=self.fixture()
        states['7']['optimizer']['state'][1]['exp_avg'][0]=1
        with self.assertRaisesRegex(ValueError,'not identical'):
            self.check(checkpoint,config,states)


class IndependentOwnershipTests(unittest.TestCase):
    def fixture(self):
        def node(pid,uid,tick,ppid,state='S'):
            return dict(pid=pid,uid=uid,starttick=tick,ppid=ppid,pgid=pid,state=state,
                        comm='fixture',kernel_uids=[uid]*4,proc_directory_owner_uid=uid,
                        argv=['fixture'],argv_sha256=str(pid))
        self.nodes={1:node(1,0,1,0),10:node(10,1005,100,1)}
        self.env={10:{}}
        self.owner=self.nodes[10]

    def scan(self):
        def environment(pid):
            value=self.env.get(pid,{})
            if isinstance(value,Exception): raise value
            return value
        return audit.audit_same_uid('ours',1005,100,[],['/fixture/trainer.py'],
                    ignored_exact=[self.owner],ancestry_anchors=[self.owner],
                    snapshot_reader=self.nodes.__getitem__,environment_reader=environment,
                    pid_reader=lambda:list(self.nodes))

    def child(self,ppid=10,state='S'):
        self.nodes[20]=dict(self.owner,pid=20,starttick=110,ppid=ppid,state=state,
                            argv_sha256='child')

    def test_exact_live_controller_is_exempt(self):
        self.fixture()
        self.assertTrue(self.scan()['accepted_no_owned_or_opaque'])

    def test_unregistered_child_cleared_run_still_rejected(self):
        self.fixture();self.child();self.env[20]={}
        result=self.scan()
        self.assertFalse(result['accepted_no_owned_or_opaque'])
        self.assertTrue(result['blockers'][0]['owned_parent_chain_matches'])

    def test_reparented_same_run_orphan_rejected(self):
        self.fixture();self.child(ppid=1);self.env[20]={'E1_RUN_ID':'ours'}
        self.assertFalse(self.scan()['accepted_no_owned_or_opaque'])

    def test_unknown_opaque_live_rejected(self):
        self.fixture();self.child(ppid=1);self.env[20]=PermissionError('fixture')
        self.assertEqual(self.scan()['blockers'][0]['reason'],
                         'opaque_live_process_not_proven_infrastructure')

    def test_unknown_zombie_rejected_even_readable_environment(self):
        self.fixture();self.child(ppid=1,state='Z');self.env[20]={}
        self.assertEqual(self.scan()['blockers'][0]['reason'],
                         'inert_zombie_origin_not_proven_non_owned')


if __name__=='__main__': unittest.main()
