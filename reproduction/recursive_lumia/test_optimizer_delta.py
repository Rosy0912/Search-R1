import unittest
from .compact_latest import optimizer_delta


def ranks(writes,total):
    return [dict(rank=i,actual_adam_writes=writes,total_adam_writes=total,
                 all_observed_loss_finite=None) for i in range(8)]


class ActualOptimizerDelta(unittest.TestCase):
    def test_partial_round_and_following_complete_round(self):
        self.assertEqual(optimizer_delta(ranks(10,3570),ranks(20,3560)),10)
        self.assertEqual(optimizer_delta(ranks(20,3590),ranks(10,3570)),20)

    def test_initial_and_zero_write_rounds(self):
        self.assertEqual(optimizer_delta(ranks(20,20),None),20)
        self.assertEqual(optimizer_delta(ranks(0,20),ranks(20,20)),0)

    def test_reject_reset_bad_increment_nonfinite_and_missing_rank(self):
        for current in [ranks(20,20),ranks(20,3591),ranks(21,3591),ranks(20,3590)[:-1]]:
            with self.assertRaises(ValueError):optimizer_delta(current,ranks(10,3570))
        current=ranks(20,3590);current[7]['all_observed_loss_finite']=False
        with self.assertRaises(ValueError):optimizer_delta(current,ranks(10,3570))

    def test_reject_rank_disagreement_even_if_individual_deltas_match(self):
        current=ranks(20,3590);current[7].update(actual_adam_writes=19,total_adam_writes=3589)
        with self.assertRaises(ValueError):optimizer_delta(current,ranks(10,3570))


if __name__=='__main__':unittest.main()
