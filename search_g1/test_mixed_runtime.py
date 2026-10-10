"""Verify that selection preserves real sampling values and trajectory metadata."""
import unittest
import numpy as np
import torch
from verl import DataProto
from search_g1.mixed_runtime import subset, padded_concat, ROW_META


def chunk(offset, width):
    rows = 10
    tensors = {'prompts': torch.full((rows, 3), 9, dtype=torch.long),
               'responses': torch.arange(rows*width).reshape(rows,width)+offset,
               'input_ids': torch.full((rows,3+width),9,dtype=torch.long),
               'sampling_log_probs': torch.arange(rows*width).reshape(rows,width).float()/100,
               'sampling_mask': torch.ones(rows,width,dtype=torch.bool)}
    return DataProto.from_dict(tensors=tensors,
        non_tensors={'trajectory_id': np.array([str(offset+i) for i in range(rows)],dtype=object)},
        meta_info={k:list(range(offset,offset+rows)) for k in ROW_META})


class RuntimeSelectionTests(unittest.TestCase):
    def test_selection_from_multiple_candidate_chunks(self):
        first, second = chunk(0,2),chunk(100,4)
        picked = [subset(first,[5,6,7,8,9]),subset(second,[0,1,2,3,4])]
        original_probs = picked[0].batch['sampling_log_probs'].clone()
        merged = padded_concat(picked,0)
        self.assertEqual(len(merged),10)
        for key in ROW_META:self.assertEqual(merged.meta_info[key],[5,6,7,8,9,100,101,102,103,104])
        self.assertEqual(merged.non_tensor_batch['trajectory_id'].tolist(),['5','6','7','8','9','100','101','102','103','104'])
        self.assertTrue(torch.equal(merged.batch['sampling_log_probs'][:5,:2],original_probs))
        self.assertFalse(bool(merged.batch['sampling_mask'][:5,2:].any()))
        self.assertEqual(int(merged.batch['sampling_mask'].sum()),30)

    def test_bad_row_metadata_rejected(self):
        data=chunk(0,2);data.meta_info['turns_stats']=[1]
        with self.assertRaises(ValueError):subset(data,[0])

if __name__=='__main__':unittest.main()
