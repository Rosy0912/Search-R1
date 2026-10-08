"""Reject uncalibrated or relabeled evidence before guarded fresh200 starts."""
import copy
import json
import os
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
    return dict(role='original_f044_diagnostic_authentication',
                plan=dict(candidate_thresholds=dict(dual_clip=3., epsilon=.2,
                       target_old_policy_kl=.01, kl_log_bound=30.)),
                segment=dict(structural=dict(global_monitors=rows),
                             evaluation_before=before, evaluation_after=copy.deepcopy(before)))


def warmup_diagnostic():
    from reproduction.common8.warmup_measurement_plan import (
        ROLE, PHASE_BUDGET, THRESHOLDS, HEALTH, METHOD_ROLE, TAIL_METHOD,
    )
    from reproduction.common8.formal_contract import FORMAL_MATHEMATICS
    value = diagnostic()
    value.update(role=bootstrap.WARMUP_AUTHENTICATION_ROLE, run=dict(started_unix=2.0))
    value['plan'].update(role=ROLE, phase_budget=copy.deepcopy(PHASE_BUDGET),
        candidate_thresholds=copy.deepcopy(THRESHOLDS),
        method=dict(schema_version=2, role=METHOD_ROLE, created_unix=1.0,
                    phase_budget=copy.deepcopy(PHASE_BUDGET), health_rule=copy.deepcopy(HEALTH),
                    candidate_thresholds=copy.deepcopy(THRESHOLDS), negative_tail_method=TAIL_METHOD,
                    mathematics_sha256=document_sha256(FORMAL_MATHEMATICS)))
    value['segment'].update(completed_outer_updates=2, optimizer_writes_per_rank=40)
    value['segment']['structural']['global_monitors'] = [dict(
        allowed=True, kl_estimate_complete=True, sampled_old_policy_kl=.001,
        negative_tail_weight_fraction=i / 1000) for i in range(80)]
    value['segment']['evaluation_before']['step'] = 0
    value['segment']['evaluation_after']['step'] = 2
    return value


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

    def test_failed3298_native_terminal_is_still_rejected_before_original_cpu_verifier(self):
        terminal=json.loads(self.terminal.read_text())
        terminal['job_stdout']='JobId=3298 JobState=FAILED ExitCode=1:0 Restarts=0 NodeList=h100-8'
        self.write(self.terminal,terminal)
        self.run['diagnostic_terminal_proof_sha256']=digest(self.terminal)
        self.write(self.base/'logs/run-manifest.json',self.run)
        process=self.verifier()
        with self.assertRaises(ValueError):self.authenticate(process)
        process.assert_not_called()


class WarmupCandidateTests(unittest.TestCase):
    def test_exact_actual_all80_maximum_and_no_floor(self):
        item=warmup_diagnostic()
        self.assertEqual(candidate_thresholds(item)['negative_tail_limit'],.079)
        for row in item['segment']['structural']['global_monitors']:
            row['negative_tail_weight_fraction']=0.
        self.assertEqual(candidate_thresholds(item)['negative_tail_limit'],0.)

    def test_wrong_budget_rejected_or_incomplete_or_failed80_is_not_success(self):
        for mutation in ('old40','79','81','rejected','above_kl','nan','one_outer','20_writes'):
            with self.subTest(mutation=mutation):
                item=warmup_diagnostic();rows=item['segment']['structural']['global_monitors']
                if mutation=='old40':del rows[40:]
                if mutation=='79':rows.pop()
                if mutation=='81':rows.append(copy.deepcopy(rows[-1]))
                if mutation=='rejected':rows[-1]['allowed']=False
                if mutation=='above_kl':rows[-1]['sampled_old_policy_kl']=.010001
                if mutation=='nan':rows[-1]['negative_tail_weight_fraction']=float('nan')
                if mutation=='one_outer':item['segment']['completed_outer_updates']=1
                if mutation=='20_writes':item['segment']['optimizer_writes_per_rank']=20
                with self.assertRaises(ValueError):candidate_thresholds(item)

    def test_pre_registered_method_and_zero_health_drop_cannot_be_changed(self):
        for mutation in ('method','future_method','kl','health','paired_ids','em_drop','f1_drop','action_drop'):
            with self.subTest(mutation=mutation):
                item=warmup_diagnostic();method=item['plan']['method']
                if mutation=='method':method['negative_tail_method']='max_with_floor'
                if mutation=='future_method':method['created_unix']=2.001
                if mutation=='kl':item['plan']['candidate_thresholds']['target_old_policy_kl']=.2
                if mutation=='health':method['health_rule']['allowed_drop']['weighted_em']=1.
                if mutation=='paired_ids':item['segment']['evaluation_after']['question_ids'].reverse()
                if mutation=='em_drop':item['segment']['evaluation_after']['weighted_em']-=.00001
                if mutation=='f1_drop':item['segment']['evaluation_after']['weighted_f1']-=.00001
                if mutation=='action_drop':item['segment']['evaluation_after']['valid_action_rate']-=.00001
                with self.assertRaises(ValueError):candidate_thresholds(item)

    def test_unknown_measurement_role_has_no_implicit_old_branch(self):
        item=diagnostic();item.pop('role')
        with self.assertRaises(ValueError):candidate_thresholds(item)


class WarmupAuthenticationTests(unittest.TestCase):
    """Native/source/tensor subprocess boundaries are mocked CPU fixtures."""
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name).resolve();self.base=root/'production';self.source=root/'measurement'
        (self.source/'repo').mkdir(parents=True)
        self.source_file=self.source/'repo/source.py';self.source_file.write_bytes(b'CPU frozen typed source fixture')
        source_map={'source.py':digest(self.source_file)}
        item=warmup_diagnostic()
        science=dict(budget={'CPU fixture science':200},initial_model_sha256={'model':'b'*64},
                     input_sha256={'data':'c'*64},calibration_question_ids=['q'+str(i) for i in range(300)],
                     model_revision='CPU fixture only')
        self.original=dict(science,run_id='measured-run',source_commit='d'*40,source_sha256=source_map,
                           policy_calibration_role=bootstrap.WARMUP_MEASUREMENT_ROLE,
                           measurement_parent_run_id='actual-prepared-parent',started_unix=2.0)
        item['plan'].update(slurm_job_id='4000',parent_production_run_id='actual-prepared-parent')
        item['segment'].update(run_id=self.original['run_id'],production_admission=False,healthy_checkpoint_accepted=False)
        for name,value in (
            ('logs/run-manifest.json',self.original),
            ('logs/warmup57-measurement-plan.json',item['plan']),
            ('logs/warmup57-measurement-segment.json',item['segment']),
            ('preparation/measurement-parent-run-manifest.json',{'run_id':'actual-prepared-parent'}),
            ('preparation/warmup57-measurement-method.json',item['plan']['method']),
            ('logs/pipeline-status.json',{'run_id':'measured-run','stage':bootstrap.WARMUP_FINISHED_STAGE}),
            ('logs/cleanup-status.json',{'run_id':'measured-run','cleaned':True}),
            ('logs/process-ledger.json',{'run_id':'measured-run','jobs':{}})):
            self.write(self.source/name,value)
        self.audit=dict(run=copy.deepcopy(self.original),plan=item['plan'],segment=item['segment'],
                        source_file_sha256={str(path.relative_to(self.source)):digest(path)
                                            for path in self.source.rglob('*') if path.is_file() and path.suffix=='.json'})
        self.proof=self.base/'preparation/measurement-launch.json';self.terminal=self.base/'preparation/measurement-terminal.json'
        self.write(self.proof,dict(execution_role=bootstrap.WARMUP_MEASUREMENT_ROLE,base=str(self.source),
                                  job_id='4000',source_commit='d'*40,source_sha256=source_map))
        self.write(self.terminal,dict(base=str(self.source),job_id='4000',job_returncode=0,source_hashes_match=True,
            job_stdout='JobId=4000 JobState=COMPLETED ExitCode=0:0 Restarts=0 NodeList=h100-8',
            **{'run-manifest.json':copy.deepcopy(self.original)}))
        self.run=dict(science,run_id='new-production',diagnostic_execution_role=bootstrap.WARMUP_MEASUREMENT_ROLE,
                      diagnostic_proof=str(self.proof),diagnostic_terminal_proof=str(self.terminal),
                      diagnostic_proof_sha256=digest(self.proof),diagnostic_terminal_proof_sha256=digest(self.terminal),
                      policy_calibration_discovery_base=str(self.source))
        self.write(self.base/'logs/run-manifest.json',self.run)

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(canonical_bytes(value))

    def authenticate(self,process=None):
        process=process or mock.Mock(return_value=SimpleNamespace(stdout=json.dumps(self.audit)))
        with mock.patch('reproduction.common8.formal_contract.assert_formal_run'), \
             mock.patch('reproduction.common8.formal_contract.assert_scientific_run'), \
             mock.patch('reproduction.common8.source_files.runtime_files',return_value=[self.source_file]), \
             mock.patch.object(bootstrap,'_warmup_cleanup') as cleanup, \
             mock.patch.object(bootstrap.subprocess,'run',process):
            result=bootstrap.authenticate_warmup57_measurement(self.base,self.source,self.proof,self.terminal)
        cleanup.assert_called_once_with(self.source,self.original,self.audit['plan'])
        return result,process

    def test_completed_measurement_is_cpu_original_source_and_not_production_updates(self):
        result,process=self.authenticate()
        self.assertEqual(result['role'],bootstrap.WARMUP_AUTHENTICATION_ROLE)
        self.assertFalse(result['production_admission']);self.assertFalse(result['same_recipe_health_accepted'])
        self.assertTrue(result['measurement_recipe_health_accepted'])
        self.assertEqual(result['measurements_counted_as_production'],0)
        self.assertEqual(process.call_args.kwargs['cwd'],self.source/'repo')
        self.assertEqual(process.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
        self.assertEqual(process.call_args.kwargs['env']['PYTHONPATH'],str(self.source/'repo'))
        script=process.call_args.args[0][2]
        self.assertIn('require_live_workers=False',script)
        self.assertIn('global_step_2',script)
        self.assertNotIn('verify_child',script)

    def test_cache_never_replaces_repeated_full_source_reconstruction(self):
        result,process=self.authenticate();cache=self.base/'logs/diagnostic-authentication.json';raw=cache.read_bytes()
        second,_=self.authenticate(process)
        self.assertEqual(process.call_count,2);self.assertEqual(result,second);self.assertEqual(raw,cache.read_bytes())
        cached=json.loads(raw);cached['segment']['optimizer_writes_per_rank']=20;self.write(cache,cached)
        forged=cache.read_bytes()
        with self.assertRaisesRegex(ValueError,'complete original audit'):self.authenticate(process)
        self.assertEqual(forged,cache.read_bytes())

    def test_failed_incomplete_purged_or_restarted_native_never_calls_verifier(self):
        actual=json.loads(self.terminal.read_text())
        for value in ('JobId=4000 JobState=FAILED ExitCode=1:0 Restarts=0 NodeList=h100-8',
                      'JobId=4000 JobState=RUNNING ExitCode=0:0 Restarts=0 NodeList=h100-8',
                      'JobId=4000 JobState=COMPLETED ExitCode=0:0 Restarts=1 NodeList=h100-8',
                      'slurm_load_jobs error: Invalid job id specified'):
            terminal=dict(actual,job_stdout=value);self.write(self.terminal,terminal)
            self.run['diagnostic_terminal_proof_sha256']=digest(self.terminal)
            self.write(self.base/'logs/run-manifest.json',self.run)
            process=mock.Mock()
            with self.assertRaises(ValueError):self.authenticate(process)
            process.assert_not_called()

    def test_original_or_incomplete_phase_cannot_be_relabelled_warmup(self):
        for field,value in (('completed_outer_updates',1),('optimizer_writes_per_rank',20)):
            old=self.audit['segment'][field];self.audit['segment'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.authenticate()
            self.audit['segment'][field]=old
        self.audit['source_file_sha256'].pop('preparation/warmup57-measurement-method.json')
        with self.assertRaises(ValueError):self.authenticate()

    def test_explicit_dispatch_has_no_failure_or_missing_role_fallback(self):
        with mock.patch.object(bootstrap,'authenticate_warmup57_measurement',side_effect=ValueError('measurement failed')) as new, \
             mock.patch.object(bootstrap,'authenticate_original_diagnostic') as old:
            with self.assertRaises(ValueError):
                bootstrap.authenticate_diagnostic(self.base,self.source,self.proof,self.terminal)
            new.assert_called_once();old.assert_not_called()
        self.run.pop('diagnostic_execution_role');self.write(self.base/'logs/run-manifest.json',self.run)
        with mock.patch.object(bootstrap,'authenticate_warmup57_measurement') as new, \
             mock.patch.object(bootstrap,'authenticate_original_diagnostic') as old:
            with self.assertRaises(ValueError):bootstrap.authenticate_diagnostic(self.base,self.source,self.proof,self.terminal)
            new.assert_not_called();old.assert_not_called()

    def test_failed_own_cleanup_blocks_authentication_cache(self):
        with mock.patch('reproduction.common8.formal_contract.assert_formal_run'), \
             mock.patch('reproduction.common8.formal_contract.assert_scientific_run'), \
             mock.patch('reproduction.common8.source_files.runtime_files',return_value=[self.source_file]), \
             mock.patch.object(bootstrap.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps(self.audit))), \
             mock.patch.object(bootstrap,'_warmup_cleanup',side_effect=ValueError('owned cleanup incomplete')):
            with self.assertRaises(ValueError):
                bootstrap.authenticate_warmup57_measurement(self.base,self.source,self.proof,self.terminal)
        self.assertFalse((self.base/'logs/diagnostic-authentication.json').exists())


class HistoricalCleanupTests(unittest.TestCase):
    """Real owned historical JSON reads; no worker is required live."""
    def test_complete_stopped_records_accepted_and_partial_cleanup_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();run={'run_id':'cpu-measurement'};plan={'slurm_job_id':'4000'}
            logs=base/'logs';logs.mkdir()
            def write(name,value):(logs/name).write_bytes(canonical_bytes(value))
            write('pipeline-status.json',dict(run_id=run['run_id'],stage=bootstrap.WARMUP_FINISHED_STAGE))
            write('cleanup-status.json',dict(run_id=run['run_id'],cleaned=True))
            process=dict(pid=123456789,start_ticks=987654321,uid=os.getuid())
            jobs={name:dict(state='STOPPED',ledger={'original':process}) for name in ('trainer','retriever','probe-server')}
            write('process-ledger.json',dict(run_id=run['run_id'],jobs=jobs))
            root=Path('/dev/shm')/'search-g1-common8-cpu-measurement-4000'
            for name,path in (('retrieval-staging-claim.json',root),('corpus-cache-claim.json',root/'datasets-cache')):
                write(name,dict(run_id=run['run_id'],slurm_job_id='4000',uid=os.getuid(),phase='CLEANED',root=str(path)))
            with mock.patch('reproduction.a800.owned_processes.same',return_value=False):
                bootstrap._warmup_cleanup(base,run,plan)
            jobs['trainer']['state']='FAILED';write('process-ledger.json',dict(run_id=run['run_id'],jobs=jobs))
            with self.assertRaises(ValueError):bootstrap._warmup_cleanup(base,run,plan)


class InitialArchiveTests(unittest.TestCase):
    """File graph/lineage fixtures; native probability and formal identity are mocked."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.round = self.base / 'artifacts/round_0-adopted-fixture'
        self.run = dict(run_id='new-run', initial_probe_version='new-run:0:fixture',
                        diagnostic_execution_role=bootstrap.ORIGINAL_DIAGNOSTIC_ROLE,
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
            diagnostic_execution_role=bootstrap.ORIGINAL_DIAGNOSTIC_ROLE,
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
