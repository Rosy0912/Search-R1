import unittest
from reproduction.common8.submit_mixed_native import validate_native_queue
class QueueTests(unittest.TestCase):
    def test_unrelated_allocation_can_queue_without_cancellation(self):
        self.assertEqual(validate_native_queue('3371|zhy072-fqg-vb-Di1500sf\n'),[['3371','zhy072-fqg-vb-Di1500sf']])
        self.assertEqual(validate_native_queue(''),[])
    def test_existing_search_g1_blocks_duplicate_even_if_unrelated_also_exists(self):
        with self.assertRaises(AssertionError):validate_native_queue('3371|other\n3372|search-g1-mixed200\n')
    def test_malformed_queue_rejected(self):
        for text in ('bad|other','3371','3371|'):
            with self.assertRaises(AssertionError):validate_native_queue(text)
