import unittest

from search_g1.reward_schedule import SCHEDULE, binary_phase, scheduled_score


class RewardScheduleTests(unittest.TestCase):
    def test_boundary_and_binary_values(self):
        self.assertEqual(scheduled_score(1.4, True, SCHEDULE, 50), 1.4)
        self.assertEqual(scheduled_score(0.2, False, SCHEDULE, 50), 0.2)
        for step in (51, 100, 200):
            self.assertEqual(scheduled_score(1.4, True, SCHEDULE, step), 1.0)
            self.assertEqual(scheduled_score(0.2, False, SCHEDULE, step), 0.0)

    def test_disabled_and_invalid_binding(self):
        self.assertEqual(scheduled_score(1.4, True, None, None), 1.4)
        for step in (None, 0, 201, True):
            with self.assertRaises(ValueError):
                binary_phase(SCHEDULE, step)
        with self.assertRaises(ValueError):
            binary_phase('unknown', 51)
