import tempfile,unittest,json
from pathlib import Path
from reproduction.common8.worker_evidence import read_console_form


class EvidenceTests(unittest.TestCase):
    def make(self,root,omit=False,mutate=False):
        b=Path(root);(b/'logs').mkdir();(b/'logs/run-manifest.json').write_text(json.dumps(dict(policy_safety_required=True)))
        for rank in range(8):
            rows=[]
            for mini in range(20):
                for post in range(2):
                    if omit and rank==7 and mini==19 and post:continue
                    identity=dict(selection_indices=[0,106,213,319],actor_token_lengths=[2,3,4,5],tensor_sha256={'old':'frozen'})
                    if mutate and rank==7 and mini==19:identity['tensor_sha256']['old']='changed'
                    rows.append(dict(kind='POLICY_MONITOR',run_id='run',rank=rank,allowed=True,kl_estimate_complete=True,context=dict(logical_update=1,minibatch_index=mini),stage='after_minibatch' if post else 'before_minibatch',optimizer_writes=mini+post,calibration_sha256='bound',monitor_identity=identity,objective_weight_definition='equal two-row micro masked_mean; weighted global numerator/denominator'))
            rows.append(dict(kind='OUTER_BUDGET',run_id='run',rank=rank,logical_update=1,cumulative_writes=20))
            (b/'logs'/f'optimizer-rank-{rank}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        return b

    def test_complete_monitor_schedule(self):
        with tempfile.TemporaryDirectory() as root:
            text=read_console_form(self.make(root),'run')
            self.assertEqual(text.count('SEARCH_G1_OUTER_BUDGET='),8)

    def test_missing_post_check_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(AssertionError):read_console_form(self.make(root,omit=True),'run')

    def test_mutated_old_identity_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(AssertionError):read_console_form(self.make(root,mutate=True),'run')


if __name__=='__main__':unittest.main()
