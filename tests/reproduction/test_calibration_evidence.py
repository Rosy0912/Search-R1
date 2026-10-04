import unittest
from search_g1.calibration_evidence import evidence_deleted_prefix, validate_provenance


class CalibrationEvidence(unittest.TestCase):
    def test_delete_only_actual_retrieval_preserves_correction_and_reasoning(self):
        model1 = 'reasoning <information>authored</information>'
        correction = 'retry <information>instruction</information>'
        model2 = '<search>query</search>'
        evidence = '<information>actual evidence</information>'
        model3 = '<answer>result</answer>'
        pieces = [model1, correction, model2, evidence, model3]
        bounds = []; cursor = 0
        for part in pieces:
            bounds.append([cursor, cursor + len(part)]); cursor += len(part)
        text = ''.join(pieces); end = text.index('result')
        result = evidence_deleted_prefix(text, end, [bounds[0], bounds[2], bounds[4]],
                                         [bounds[3]], [bounds[1]])
        self.assertEqual(result, model1 + correction + model2 +
                         '\n<information></information>\n<answer>')

    def test_invalid_overlap_gap_or_out_of_bounds_refused(self):
        for models, evidence, corrections in [([[0, 3]], [[2, 4]], []),
                                               ([[0, 2]], [[3, 4]], []),
                                               ([[0, 5]], [], [])]:
            with self.assertRaises(ValueError):
                validate_provenance('abcd', models, evidence, corrections)

    def test_no_retrieval_keeps_prefix_and_empty_model_segment(self):
        self.assertEqual(evidence_deleted_prefix('abcd', 3, [[0, 0], [0, 4]], [], []), 'abc')

    def test_bad_answer_boundary_refused(self):
        with self.assertRaises(ValueError):
            evidence_deleted_prefix('abcd', 5, [[0, 4]], [], [])


if __name__ == '__main__':
    unittest.main()
