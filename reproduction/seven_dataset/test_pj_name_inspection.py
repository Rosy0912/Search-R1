from types import SimpleNamespace
import unittest
from .pj_name_inspection import CONTROL, CONTROL_UID, prove_absent
from .budget5_name_recovery import validate_prior


NAMES = ['short-r' + str(i) for i in (1, 2, 3)] + ['legacy-' + 'x' * 33 + str(i) for i in (1, 2, 3)]


class Fake:
    def __init__(self):
        self.gets = []
        self.existing = set()
        self.paginated = False

    def list_training_jobs(self, filter_str, page_size):
        import json
        name = json.loads(filter_str.split(' = ', 1)[1])
        rows = [{'name': CONTROL, 'uid': CONTROL_UID}] if name == CONTROL else []
        if name in self.existing:
            rows.append({'name': name, 'uid': 'existing'})
        return dict(total_size=len(rows), training_jobs=rows,
                    next_page_token='more' if self.paginated else '')

    def get_training_job(self, name):
        self.gets.append(name)
        exc = RuntimeError('not found')
        exc.response = SimpleNamespace(status_code=404)
        raise exc


class NameProofTest(unittest.TestCase):
    def test_long_name_not_sent_to_invalid_get_endpoint(self):
        client = Fake()
        proof = prove_absent(client, NAMES)
        self.assertEqual(len(proof['exact_jobs']), 6)
        self.assertEqual(client.gets, NAMES[:3])

    def test_existing_legacy_name_blocks(self):
        client = Fake()
        client.existing.add(NAMES[-1])
        with self.assertRaises(ValueError):
            prove_absent(client, NAMES)

    def test_pagination_does_not_prove_absence(self):
        client = Fake()
        client.paginated = True
        with self.assertRaises(ValueError):
            prove_absent(client, NAMES)

    def test_400_is_not_absence(self):
        class Invalid(Fake):
            def get_training_job(self, name):
                exc = RuntimeError('bad request')
                exc.response = SimpleNamespace(status_code=400)
                raise exc
        with self.assertRaises(RuntimeError):
            prove_absent(Invalid(), NAMES)

    def test_unknown_prior_phase_cannot_continue(self):
        with self.assertRaises(ValueError):
            validate_prior({'phase': 'WAITING_FINAL150'}, {})

    def test_pretransfer_only_and_export_identity_bound(self):
        prior = dict(pid=36239, group='budget-5', phase='FAILED_PRESERVE_EVIDENCE_NO_AUTO_RETRY',
            error='value length must be between 1 and 32', checkpoint_identity='c',
            archive_bytes=10, archive_sha256='s')
        proof = {k: prior[k] for k in ['checkpoint_identity', 'archive_bytes', 'archive_sha256']}
        validate_prior(prior, proof)
        for field in ['preflight', 'submissions', 'transfer_started', 'transfer']:
            with self.assertRaises(ValueError):
                validate_prior({**prior, field: None}, proof)
        with self.assertRaises(ValueError):
            validate_prior(prior, {**proof, 'archive_sha256': 'changed'})


if __name__ == '__main__':
    unittest.main()
