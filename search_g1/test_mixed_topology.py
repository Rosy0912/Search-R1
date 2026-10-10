import unittest
from reproduction.common8.topology import actor_local_micro,actor_global_micro,actor_local_mini

class MicroTests(unittest.TestCase):
    def run_manifest(self,micro):
        return {'profile':'common8','training_world_size':8,'budget':{'world_size':8},
                'actor_micro_batch_per_rank':micro}

    def test_eight_rank_accumulation_choices_and_same_adam_budget(self):
        for micro,accum in [(2,8),(4,4)]:
            run=self.run_manifest(micro)
            self.assertEqual(actor_global_micro(run),8*micro)
            self.assertEqual(actor_local_mini(run)//actor_local_micro(run),accum)
            self.assertEqual((2560//8)//actor_local_mini(run),20)

    def test_invalid_micro_rejected(self):
        for micro in [0,3,8,True]:
            with self.assertRaises(AssertionError):actor_local_micro(self.run_manifest(micro))

if __name__=='__main__':unittest.main()
