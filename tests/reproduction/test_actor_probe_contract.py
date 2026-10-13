import unittest

from search_g1.actor_probe_contract import (MODE, actor_probes, experiment_fields, head_round, probe_rounds,
    validate_version, checkpoint_head_round, validate_fitted_version, validate_checkpoint_version)
from search_g1.reward_schedule import ACTOR_PROBE_SCHEDULE, SCHEDULE, binary_phase, reward_boundary, scheduled_score


def run_binding():
    return dict(run_id='new', reward_schedule=ACTOR_PROBE_SCHEDULE,
                input_sha256={'calibration-300.parquet': 'training-only'},
                **experiment_fields(ACTOR_PROBE_SCHEDULE))


class ActorProbeContractTests(unittest.TestCase):
    def test_after100_boundary_preserves_legacy_after50(self):
        self.assertEqual(reward_boundary(SCHEDULE), 50)
        self.assertFalse(binary_phase(ACTOR_PROBE_SCHEDULE, 100))
        self.assertTrue(binary_phase(ACTOR_PROBE_SCHEDULE, 101))
        self.assertEqual(scheduled_score(-.1, False, ACTOR_PROBE_SCHEDULE, 101), 0.)
        self.assertEqual(scheduled_score(1.7, True, ACTOR_PROBE_SCHEDULE, 200), 1.)
        self.assertEqual(scheduled_score(.3, False, ACTOR_PROBE_SCHEDULE, 100), .3)

    def test_iteration_version_boundary_includes_terminal200(self):
        run = run_binding()
        self.assertTrue(actor_probes(run))
        self.assertEqual([head_round(i) for i in (10,11,100,101,200)], [0,10,90,100,190])
        self.assertEqual(probe_rounds(run), tuple(range(0, 201, 10)))
        self.assertEqual(probe_rounds({}), (0, 50, 100, 150))

    def test_post_complete200_head_is_saved_but_never_predicts_a_formal201(self):
        run = run_binding()
        def version(step):
            return dict(run_id='new', version=f'v{step}', round=step, actor_step=step,
                representation_source=MODE, calibration_data_sha256='training-only')
        self.assertEqual(validate_version(run,version(190),200),version(190))
        self.assertEqual(validate_fitted_version(run,version(200)),version(200))
        self.assertEqual(validate_checkpoint_version(run,version(200),200),version(200))
        self.assertEqual([checkpoint_head_round(i) for i in (0,5,50,100,150,200)], [0,0,50,100,150,200])
        for head,logical in ((200,200),(200,201),(190,201)):
            with self.subTest(head=head,logical=logical),self.assertRaises(ValueError):
                validate_version(run,version(head),logical)
        with self.assertRaises(ValueError):validate_checkpoint_version(run,version(190),200)
        for step in (True,-1,201):
            with self.assertRaises(ValueError):checkpoint_head_round(step)

    def test_partial_and_shared_head_contracts_fail_closed(self):
        for key in ('probe_representation_source', 'probe_refresh_interval', 'probe_parameter_ownership'):
            run = run_binding(); del run[key]
            with self.assertRaises(ValueError): actor_probes(run)
        for changes in ({'probe_refresh_interval': True}, {'probe_refresh_interval': 50},
                        {'probe_parameter_ownership': 'actor_lm_head'},
                        {'initial_probe_adoption_source_base': '/old'}, {'reward_schedule': SCHEDULE}):
            with self.assertRaises(ValueError): actor_probes(run_binding() | changes)

    def test_wrong_actor_data_or_stale_version_is_rejected(self):
        run = run_binding()
        valid = dict(run_id='new', version='v10', round=10, actor_step=10,
                     representation_source=MODE, calibration_data_sha256='training-only')
        self.assertEqual(validate_version(run, valid, 11), valid)
        for change in ({'run_id': 'old'}, {'round': 0}, {'actor_step': 0},
                       {'representation_source': 'frozen_teacher'}, {'calibration_data_sha256': 'dev512'}):
            with self.assertRaises(ValueError): validate_version(run, valid | change, 11)
        for logical in (0, True, 201):
            with self.assertRaises(ValueError): head_round(logical)
