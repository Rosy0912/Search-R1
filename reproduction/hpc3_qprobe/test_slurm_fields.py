"""Synthetic Slurm parser fixtures never establish native GPU admission."""
import unittest
from reproduction.common8.native_allocation import fields,allocated_indices
from reproduction.common8.stage_retrieval import memory_bytes


class SlurmFieldTests(unittest.TestCase):
    def test_actual_allocation_used_without_borrowing_requested_resources(self):
        pending=fields('JobState=PENDING ReqTRES=cpu=96,mem=512G,gres/gpu=8 AllocTRES=(null)')
        self.assertNotIn('TRES',pending)
        actual=fields('JobState=RUNNING ReqTRES=mem=512G AllocTRES=cpu=96,mem=512G,node=1,gres/gpu=8')
        self.assertEqual(memory_bytes(actual['TRES']),512*2**30)
        self.assertEqual(fields('TRES=cpu=96,mem=512G')['TRES'],'cpu=96,mem=512G')

    def test_physical_indices_require_actual_detail_and_eight_unique_cards(self):
        self.assertEqual(allocated_indices(' Nodes=ACD1-1 CPU_IDs=0-95 Mem=524288 GRES=gpu:8(IDX:0-7)','ACD1-1'),list(range(8)))
        for value in ['ReqTRES=gres/gpu=8',' Nodes=ACD1-1 GRES=gpu:8(IDX:0-6)',
                      ' Nodes=ACD1-1 GRES=gpu:8(IDX:0-6,6)']:
            with self.assertRaises(AssertionError):allocated_indices(value,'ACD1-1')


if __name__=='__main__':unittest.main()
