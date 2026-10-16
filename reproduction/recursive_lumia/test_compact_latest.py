import unittest
from .compact_latest import loss_status


class MilestoneLossTest(unittest.TestCase):
    def ranks(self):
        return [dict(rank=i,observed_loss_scalars=80,all_observed_loss_finite=True) for i in range(8)]

    def test_evaluation_pending_uses_real_sparse_losses_without_claiming_summary(self):
        result=loss_status({},50,self.ranks(),True)
        self.assertTrue(result['loss_finite'])
        self.assertTrue(result['summary_pending'])
        self.assertIsNone(result['reported_pg_loss_finite'])
        self.assertIsNone(result['pg_loss'])

    def test_wrong_or_nonfinite_summary_never_hidden_by_sparse_evidence(self):
        for step,value in [(49,0.1),(50,float('nan')),(50,float('inf'))]:
            with self.assertRaises(ValueError):
                loss_status({'training/actual_outer_updates':step,'actor/pg_loss':value},50,self.ranks(),True)

    def test_missing_rank_loss_or_boundary_rejected(self):
        for ranks,boundary in [(self.ranks()[:7],True),(self.ranks(),False),
                                (self.ranks()[:7]+[dict(rank=7,observed_loss_scalars=0,all_observed_loss_finite=None)],True)]:
            with self.assertRaises(ValueError):loss_status({},50,ranks,boundary)
        with self.assertRaises(ValueError):loss_status({},49,self.ranks(),True)

    def test_completed_summary_keeps_exact_value(self):
        result=loss_status({'training/actual_outer_updates':50,'actor/pg_loss':0.003},50,self.ranks(),True)
        self.assertEqual(result['pg_loss'],0.003)
        self.assertFalse(result['summary_pending'])


if __name__=='__main__':unittest.main()
