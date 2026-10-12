import copy
import unittest
from reproduction.a800_common.predecessor_admission import validate_binding, validate_documents, EXPECTED_PREDECESSOR
from reproduction.common8.verify_data import MANIFEST_SHA


class PredecessorAdmissionTests(unittest.TestCase):
    def fixture(self):
        binding = dict(EXPECTED_PREDECESSOR, full_acceptance_sha256='e'*64, cleanup_sha256='f'*64)
        child = dict(run_id='1'*32, execution_id='2'*32, model_revision='3'*40,
            initial_model_sha256={'weight': '4'*64}, input_sha256={'manifest.json': MANIFEST_SHA})
        parent = dict(child, **{k: binding[k] for k in ('run_id','execution_id','source_commit')},
            execution_backend='ssh_local', training_world_size=4)
        proof = dict(run_id=parent['run_id'], structural_accepted=True, actual_outer_updates=200,
            per_rank_optimizer_opportunities=4000, final_saved_tensor_count=435,
            final_counts=dict(nq=3610, a=8000, b=8000, c=8000, d=8000, e=8000, f=8103),
            per_rank_actual_optimizer_writes={str(rank):3993 for rank in range(4)})
        cleanup = dict(run_id=parent['run_id'], execution_id=parent['execution_id'], cleaned=True,
            execution_backend='ssh_local', scope='only recorded project PID births and private local corpus cache')
        updates = [dict(step=step, run_id=parent['run_id'], opportunities_processed=20,
            actual_optimizer_writes=13 if step==103 else 20,
            optimizer_writes_after=20*step-(7 if step>=103 else 0)) for step in range(1,201)]
        return binding, child, parent, proof, cleanup, updates

    def test_accepts_actual_complete_outers_with_skips_and_rejects_wrong_parent_or_missing_completion(self):
        args = self.fixture(); validate_binding(args[0])
        self.assertEqual(validate_documents(*args), 3993)
        for index, key, value in ((3,'actual_outer_updates',199), (4,'cleaned',False),
                (2,'run_id','9'*32), (3,'final_saved_tensor_count',434)):
            modified = copy.deepcopy(args); modified[index][key] = value
            with self.assertRaises(AssertionError): validate_documents(*modified)
        modified = copy.deepcopy(args); modified[5].pop(103)
        with self.assertRaises(AssertionError): validate_documents(*modified)
        modified = copy.deepcopy(args); modified[2]['input_sha256']['manifest.json'] = '0'*64
        with self.assertRaises(AssertionError): validate_documents(*modified)
        with self.assertRaises(AssertionError): validate_binding(args[0] | {'base':'/tmp/other'})
        with self.assertRaises(AssertionError): validate_binding(args[0] | {'run_id':'a'*32})
