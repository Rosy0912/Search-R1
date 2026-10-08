"""Reject uncalibrated or relabeled evidence before guarded fresh200 starts."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

from reproduction.common8 import fresh200_bootstrap as bootstrap
from reproduction.common8.fresh200_bootstrap import candidate_thresholds
from reproduction.common8.policy_calibration_plan import canonical_bytes, document_sha256
from reproduction.common8.verify_data import digest


def diagnostic():
    rows = [dict(allowed=True, kl_estimate_complete=True,
                 sampled_old_policy_kl=.001, negative_tail_weight_fraction=i / 1000)
            for i in range(40)]
    before = dict(question_ids=['question' + str(i) for i in range(512)], question_count=512,
                  weighted_em=.2, weighted_f1=.3, valid_action_rate=.9)
    return dict(plan=dict(candidate_thresholds=dict(dual_clip=3., epsilon=.2,
                       target_old_policy_kl=.01, kl_log_bound=30.)),
                segment=dict(structural=dict(global_monitors=rows),
                             evaluation_before=before, evaluation_after=copy.deepcopy(before)))


class CandidateTests(unittest.TestCase):
    def test_exact_actual_maximum_without_added_margin(self):
        result = candidate_thresholds(diagnostic())
        self.assertEqual(result['negative_tail_limit'], .039)
        self.assertEqual(result['target_old_policy_kl'], .01)

    def test_zero_observed_tail_stays_zero(self):
        item = diagnostic()
        for row in item['segment']['structural']['global_monitors']:
            row['negative_tail_weight_fraction'] = 0.
        self.assertEqual(candidate_thresholds(item)['negative_tail_limit'], 0.)

    def test_incomplete_or_rejected_monitor_is_not_a_calibration(self):
        for mutation in ('missing', 'rejected', 'incomplete', 'above_kl', 'nan', 'bad_tail'):
            with self.subTest(mutation=mutation):
                item = diagnostic(); rows = item['segment']['structural']['global_monitors']
                if mutation == 'missing': rows.pop()
                if mutation == 'rejected': rows[-1]['allowed'] = False
                if mutation == 'incomplete': rows[-1]['kl_estimate_complete'] = False
                if mutation == 'above_kl': rows[-1]['sampled_old_policy_kl'] = .010001
                if mutation == 'nan': rows[-1]['negative_tail_weight_fraction'] = float('nan')
                if mutation == 'bad_tail': rows[-1]['negative_tail_weight_fraction'] = 1.01
                with self.assertRaises(ValueError): candidate_thresholds(item)

    def test_original_prospective_health_cannot_be_widened(self):
        for field in ('weighted_em', 'weighted_f1', 'valid_action_rate'):
            item = diagnostic()
            item['segment']['evaluation_after'][field] -= .00001
            with self.assertRaises(ValueError): candidate_thresholds(item)

    def test_unpaired_questions_rejected(self):
        item = diagnostic(); item['segment']['evaluation_after']['question_ids'].reverse()
        with self.assertRaises(ValueError): candidate_thresholds(item)


class OriginalAuthenticationCacheTests(unittest.TestCase):
    """Mocked original CPU verifier boundary; no actual native/tensor evidence."""
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name).resolve();self.base=root/'new';self.source=root/'old'
        (self.source/'repo').mkdir(parents=True);(self.source/'repo/source.py').write_text('immutable fixture\n')
        source_map={'source.py':digest(self.source/'repo/source.py')}
        self.original=dict(source_sha256=source_map,source_commit=bootstrap.ORIGINAL_COMMIT,run_id='old')
        self.write(self.source/'logs/run-manifest.json',self.original)
        item=diagnostic();item['segment'].update(completed_outer_updates=1,production_admission=False)
        self.write(self.source/'logs/policy-safety-calibration-plan.json',item['plan'])
        self.write(self.source/'logs/policy-calibration-segment.json',item['segment'])
        files={name:digest(self.source/name) for name in ('logs/run-manifest.json',
            'logs/policy-safety-calibration-plan.json','logs/policy-calibration-segment.json')}
        self.audit=dict(run=self.original,plan=item['plan'],segment=item['segment'],source_file_sha256=files)
        self.proof=self.base/'preparation/original-launch.json'
        self.terminal=self.base/'preparation/original-terminal.json'
        self.write(self.proof,dict(base=str(self.source),source_commit=bootstrap.ORIGINAL_COMMIT,
                                   source_sha256=source_map,job_id='3298'))
        self.write(self.terminal,dict(base=str(self.source),job_id='3298',job_returncode=0,
            source_hashes_match=True,job_stdout='JobId=3298 JobState=COMPLETED ExitCode=0:0 Restarts=0 NodeList=h100-8',
            **{'run-manifest.json':dict(run_id='old')}))
        self.run=dict(run_id='new',diagnostic_proof=str(self.proof),diagnostic_terminal_proof=str(self.terminal),
            policy_calibration_discovery_base=str(self.source),diagnostic_proof_sha256=digest(self.proof),
            diagnostic_terminal_proof_sha256=digest(self.terminal))
        self.write(self.base/'logs/run-manifest.json',self.run)
        self.cache=self.base/'logs/diagnostic-authentication.json'
        self.source_map_sha=document_sha256(source_map)
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(canonical_bytes(value))
    def authenticate(self,process):
        with mock.patch('reproduction.common8.formal_contract.assert_formal_run'), \
             mock.patch.object(bootstrap,'ORIGINAL_LAUNCH_SHA',digest(self.proof)), \
             mock.patch.object(bootstrap,'ORIGINAL_SOURCE_MAP_SHA',self.source_map_sha), \
             mock.patch.object(bootstrap.subprocess,'run',process):
            return bootstrap.authenticate_original_diagnostic(self.base,self.source,self.proof,self.terminal)
    def verifier(self):
        return mock.Mock(return_value=SimpleNamespace(stdout=json.dumps(self.audit)))
    def test_existing_cache_still_executes_complete_original_audit_and_preserves_archive_bytes(self):
        process=self.verifier();first=self.authenticate(process);before=self.cache.read_bytes()
        second=self.authenticate(process)
        self.assertEqual(process.call_count,2)
        self.assertEqual(first,second);self.assertEqual(self.cache.read_bytes(),before)
        self.assertEqual(process.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
        self.assertEqual(process.call_args.kwargs['env']['PYTHONPATH'],str(self.source/'repo'))
    def test_cached_measurements_identity_or_required_graph_cannot_replace_original_reconstruction(self):
        process=self.verifier();self.authenticate(process);original=self.cache.read_bytes()
        for field in ('segment','plan','run','source_file_sha256'):
            cached=json.loads(original)
            if field=='segment':cached[field]['structural']['global_monitors'][0]['negative_tail_weight_fraction']=1.
            if field=='plan':cached[field]['candidate_thresholds']['target_old_policy_kl']=1.
            if field=='run':cached[field]['run_id']='forged-old'
            if field=='source_file_sha256':cached[field]={}
            self.write(self.cache,cached);before=self.cache.read_bytes()
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'complete original audit'):
                self.authenticate(process)
            self.assertEqual(self.cache.read_bytes(),before)
        self.assertEqual(process.call_count,5)

    def test_frozen_kl_candidate_not_inflated_to_cover_instability(self):
        item = diagnostic(); item['plan']['candidate_thresholds']['target_old_policy_kl'] = 1.
        with self.assertRaises(ValueError): candidate_thresholds(item)


class InitialArchiveTests(unittest.TestCase):
    """File graph/lineage fixtures; native probability and formal identity are mocked."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.round = self.base / 'artifacts/round_0-adopted-fixture'
        self.run = dict(run_id='new-run', initial_probe_version='new-run:0:fixture',
                        initial_model='/official/base', source_sha256={'source.py': 'a' * 64},
                        initial_model_sha256={'model': 'b' * 64}, input_sha256={'input': 'c' * 64}, budget={'recipe': 200})
        for name, raw in (
                ('heads/commit_probe.pkl', b'actual copied initial commit'),
                ('heads/boundary_probe.pkl', b'actual copied initial boundary'),
                ('heads/calibration-report.json', b'{}'),
                ('dumps/calibration-label-proof.json', b'{}'),
                ('provenance/original-version-manifest.json', b'{}'),
                ('provenance/original-diagnostic-proof.json', b'{"old":true}'),
                ('provenance/original-native-terminal-proof.json', b'{"complete":true}')):
            path = self.round / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        self.diagnostic = diagnostic()
        self.diagnostic.update(run=dict(run_id='old-run'), new_run_id='new-run')
        fingerprint = dict(encoder_sha256={'model': 'b' * 64},
            head_sha256={name: digest(self.round / 'heads' / name) for name in ('commit_probe.pkl', 'boundary_probe.pkl')})
        self.diagnostic['plan']['initial_reward_fingerprint'] = fingerprint
        origin = dict(run_id='old-run', archive_sha256=digest(self.round / 'provenance/original-version-manifest.json'),
            diagnostic_proof_sha256=digest(self.round / 'provenance/original-diagnostic-proof.json'),
            terminal_proof_sha256=digest(self.round / 'provenance/original-native-terminal-proof.json'))
        self.initial = dict(run_id='new-run', round=0, version=self.run['initial_probe_version'],
            encoder=self.run['initial_model'], commit=str(self.round / 'heads/commit_probe.pkl'),
            boundary=str(self.round / 'heads/boundary_probe.pkl'), adoption_origin=origin, **fingerprint,
            calibration_report_sha256=digest(self.round / 'heads/calibration-report.json'),
            dump_sha256={'calibration-label-proof.json': digest(self.round / 'dumps/calibration-label-proof.json')})
        self.write(self.round / 'version-manifest.json', self.initial)
        self.write(self.base / 'artifacts/live/manifest.json', self.initial)
        self.adoption = dict(schema_version=3, role='fresh200_initial_probe_adoption', run_id='new-run', adopted=True,
            version=self.run['initial_probe_version'], target_source_sha256=self.run['source_sha256'],
            target_budget_sha256=document_sha256(self.run['budget']), origin=origin,
            initial_reward_fingerprint=fingerprint, initial_archive=str((self.round / 'version-manifest.json').relative_to(self.base)),
            initial_archive_sha256=digest(self.round / 'version-manifest.json'),
            live_manifest_sha256=digest(self.base / 'artifacts/live/manifest.json'))
        self.write(self.base / 'logs/initial-probe-adoption.json', self.adoption)
        self.write(self.base / 'logs/diagnostic-authentication.json', self.diagnostic)
        receipts = bootstrap.initial_probe_files(self.base, self.run, self.diagnostic)
        receipts['logs/diagnostic-authentication.json'] = digest(self.base / 'logs/diagnostic-authentication.json')
        self.artifact = dict(schema_version=3, role=bootstrap.ROLE, accepted=True, run_id='new-run',
            healthy_checkpoint_accepted=False, same_recipe_health_accepted=False,
            actual_production_short_train_accepted=False, old_updates_counted_as_production=0,
            initial_probe_version=self.run['initial_probe_version'], receipts_sha256=receipts,
            diagnostic_authentication='logs/diagnostic-authentication.json', diagnostic_run_id='old-run',
            probability=dict(run_id='new-run', actual_optimizer_writes=0, source_sha256=self.run['source_sha256']),
            **{key: self.run[key] for key in ('source_sha256', 'initial_model_sha256', 'input_sha256', 'budget')},
            **candidate_thresholds(self.diagnostic))

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(canonical_bytes(value))

    def verify(self):
        with mock.patch('reproduction.common8.formal_contract.assert_formal_run'), \
             mock.patch('reproduction.common8.production_safety_admission._verify_parent_probability'):
            return bootstrap.verify_bootstrap(self.base, self.artifact, self.run)

    def test_valid_refit50_live_pointer_does_not_invalidate_frozen_initial0(self):
        original = self.verify()
        updated = dict(self.initial, round=50, version='new-run:50:refit', encoder='/checkpoint50',
                       encoder_sha256={'model': 'd' * 64}, head_sha256={'new-head': 'e' * 64})
        self.write(self.base / 'artifacts/live/manifest.json', updated)
        self.assertEqual(self.verify(), original)
        self.assertNotIn('artifacts/live/manifest.json', self.artifact['receipts_sha256'])

    def test_initial_archive_heads_dumps_and_provenance_remain_frozen_after_refit(self):
        for name in ('version-manifest.json', 'heads/commit_probe.pkl', 'dumps/calibration-label-proof.json',
                     'provenance/original-native-terminal-proof.json'):
            path = self.round / name; raw = path.read_bytes(); path.write_bytes(raw + b' ')
            with self.subTest(name=name), self.assertRaises(ValueError): self.verify()
            path.write_bytes(raw)

    def test_missing_frozen_file_or_live_pointer_hash_rejected(self):
        name = self.adoption['initial_archive']; value = self.artifact['receipts_sha256'].pop(name)
        with self.assertRaises(ValueError): self.verify()
        self.artifact['receipts_sha256'][name] = value
        self.artifact['receipts_sha256']['artifacts/live/manifest.json'] = digest(self.base / 'artifacts/live/manifest.json')
        with self.assertRaises(ValueError): self.verify()

    def test_relabeling_initial_archive_does_not_bypass_origin_version_or_path_binding(self):
        for key, value in (('version', 'new-run:50:forged'), ('encoder', '/checkpoint50'),
                           ('adoption_origin', dict(self.initial['adoption_origin'], run_id='other'))):
            original = copy.deepcopy(self.initial); modified = dict(original, **{key: value})
            self.write(self.round / 'version-manifest.json', modified)
            forged = digest(self.round / 'version-manifest.json')
            adoption = dict(self.adoption, initial_archive_sha256=forged, live_manifest_sha256=forged)
            self.write(self.base / 'logs/initial-probe-adoption.json', adoption)
            with self.subTest(key=key), self.assertRaises(ValueError):
                bootstrap.initial_probe_files(self.base, self.run, self.diagnostic)
            self.write(self.round / 'version-manifest.json', original)
        self.write(self.base / 'logs/initial-probe-adoption.json', dict(self.adoption, initial_archive='../outside.json'))
        with self.assertRaises(ValueError): bootstrap.initial_probe_files(self.base, self.run, self.diagnostic)


if __name__ == '__main__':
    unittest.main()
