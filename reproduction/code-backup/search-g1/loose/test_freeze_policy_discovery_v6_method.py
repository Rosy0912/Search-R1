"""CPU protocol fixtures, never native/GPU/healthy method proof.

Future run/CPU/Slurm/criteria documents are explicitly synthetic. Real local
source readers and strict _unstarted/_identity helpers are exercised with tiny
temporary files. Only scontrol/remote orchestration boundaries are mocked; no
remote or actual preregistration/method file is created in project preparation.
"""
import base64
import copy
import inspect
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch

import freeze_policy_discovery_v6_method as v
import test_submit_policy_discovery_v6_native as discovery_tests


def prereg_fixture():
    # A CPU schema fixture, not root's prospective document or threshold choice.
    return dict(role=v.ROLE,source_commit='a'*40,created_unix=200.,
        health_rule_document=dict(role='policy_safety_health_rule',
            provenance=dict(origin='CPU-only fixture',not_a_population_statistical_guarantee=True),
            criteria=copy.deepcopy(v.HEALTH_CRITERIA)),negative_tail_selection=copy.deepcopy(v.TAIL_SELECTION),
        fixed_candidates=copy.deepcopy(v.CANDIDATES),negative_tail_limit=None,sampling_distribution=copy.deepcopy(v.SAMPLING),
        target_cpu_proof_sha256='b'*64,actual_target_cpu_tests=999,actual_discovery_observations_for_this_source=0,
        production_admission=False,healthy_checkpoint_accepted=False,prospective_identity_binding_complete=False,
        scope='CPU fixture only, not actual native criteria or health')


def validate_prereg(doc):
    raw=v.common.canonical(doc)
    return v.validate_preregistration(raw,expected_sha256=v.common.sha(raw),source_commit='a'*40,
        target_cpu_proof_sha256='b'*64,expected_cpu_tests=999)


def metadata_fixture(base='/mnt/data/shaoshitong/g1d-11223344',parent='/mnt/data/shaoshitong/g1p-11223344'):
    prereg=prereg_fixture();b=Path(base)
    mapping={f'f{i:03}.py':'c'*64 for i in range(351)}
    mapping.update({'verl/version/version':'c'*64,'verl/single_controller/version/version':'c'*64,
                    'reproduction/common8/sampling_capture.py':'c'*64})
    launch=dict(returncode=0,job_id='9999',stdout='9999\n',submitted_unix=210.,base=base,parent_base=parent,
        source_commit='a'*40,source_sha256=mapping,target_cpu_proof_sha256='b'*64,actual_target_cpu_tests=999)
    command=str(b/'repo/reproduction/common8/launch_calibration_native.sh')
    output=str(b/'logs/native-discovery-v6-%j.log')
    launch['argv']=['sbatch','--exclusive','--nodes=1','--gres=gpu:h100:8','--mem=512G',
        '--job-name=search-g1-policy-discovery-v6','--chdir='+str(b/'repo'),'--output='+output,command]
    fields=dict(JobId='9999',JobState='RUNNING',ExitCode='0:0',NodeList='h100-8',BatchHost='h100-8',
        Command=command,WorkDir=str(b/'repo'),Restarts='0',UserId='cpu-fixture(0)',NumNodes='1',
        MinMemoryNode='512G',TresPerNode='gres:gpu:h100:8',SubmitTime='2026-10-06T10:00:00',
        StartTime='2026-10-06T10:00:01',StdOut=output.replace('%j','9999'),StdErr=output.replace('%j','9999'))
    observed=dict(base=base,parent_base=parent,job_id='9999',checked_unix=220.,observation_finished_unix=225.,
        job_returncode=0,job_stderr='',job_stdout=' '.join(key+'='+value for key,value in fields.items()),source_hashes_match=True,
        child_actual_unstarted_verified=True,parent_actual_unstarted_verified=True,
        child_actual_unstarted=copy.deepcopy(v.UNSTARTED),parent_actual_unstarted=copy.deepcopy(v.UNSTARTED),
        actual_updates=[],actual_behavior_gates=[],child_manifest_sha256='d'*64,child_manifest_slurm_job_id='9999',
        parent_run_manifest=dict(run_id='2'*32,source_commit='a'*40,completed_updates=0,manifest_sha256='e'*64),
        **{'run-manifest.json':dict(run_id='1'*32,parent_production_run_id='2'*32,source_commit='a'*40,completed_updates=0,
            policy_calibration_role='policy_safety_calibration_discovery'),
           'pipeline-status.json':dict(run_id='1'*32,stage='INITIAL_ACTOR_REFERENCE_ROLLOUT_SMOKE')})
    return prereg,launch,observed


def validate_zero(prereg,launch,observed):
    return v.validate_zero_observation(launch,observed,prereg=prereg,source_commit='a'*40,
        source_sha256=launch['source_sha256'],target_cpu_proof_sha256='b'*64,expected_cpu_tests=999)


def payload_fixture(base,parent):
    # Actual CPU files and real pure runtime helpers; scheduler is a fixture.
    sys.path.insert(0,str(v.REPO))
    from reproduction.common8 import test_policy_calibration_plan as plan_tests
    fixture=plan_tests.CalibrationPlanTests();fixture.setUp()
    mapping=discovery_tests.new_mapping(base);mapping2=discovery_tests.new_mapping(parent)
    assert mapping==mapping2
    for namespace in (base,parent):
        (namespace/'logs').mkdir();(namespace/'preparation').mkdir()
    production=copy.deepcopy(fixture.parent)
    production.update(run_id='2'*32,source_commit='a'*40,source_sha256=mapping,
        train_log=str(parent/'logs/search-g1-common8-training.log'),checkpoint_dir=str(parent/'checkpoints/search-g1-common8'))
    child=copy.deepcopy(production)
    child.update(run_id='1'*32,parent_production_run_id='2'*32,policy_calibration_slurm_job_id='9999',
        policy_calibration_role='policy_safety_calibration_discovery',
        train_log=str(base/'logs/search-g1-common8-training.log'),checkpoint_dir=str(base/'checkpoints/search-g1-common8'))
    child_raw=v.common.canonical(child);parent_raw=v.common.canonical(production)
    (base/'logs/run-manifest.json').write_bytes(child_raw);(parent/'logs/run-manifest.json').write_bytes(parent_raw)
    status=dict(run_id='1'*32,stage='INITIAL_ACTOR_REFERENCE_ROLLOUT_SMOKE')
    (base/'logs/pipeline-status.json').write_bytes(v.common.canonical(status))
    prereg,launch,observed=metadata_fixture(str(base),str(parent));launch['source_sha256']=mapping
    observed['child_manifest_sha256']=v.common.sha(child_raw)
    observed['parent_run_manifest']['manifest_sha256']=v.common.sha(parent_raw)
    fields=v.discovery._job_fields(observed['job_stdout'])
    return dict(prereg=prereg,launch=launch,launch_sha256='f'*64,observed=observed,observation_sha256='9'*64,
        actual_zero_identity=dict(child_run_id='1'*32,parent_run_id='2'*32,native_job_id='9999',scheduler_fields=fields),
        source_commit='a'*40,source_sha256=mapping,source_map_sha256=v.common.sha(v.common.canonical(mapping)),
        prereg_sha256='8'*64,target_cpu_proof_sha256='b'*64,actual_target_cpu_tests=999)


class PreregistrationTests(unittest.TestCase):
    def test_explicit_conservative_schema_does_not_choose_tail(self):
        result=validate_prereg(prereg_fixture())
        self.assertIsNone(result['negative_tail_limit']);self.assertFalse(result['production_admission'])
        self.assertFalse(result['healthy_checkpoint_accepted']);self.assertFalse(result['prospective_identity_binding_complete'])
        self.assertEqual(result['negative_tail_selection']['required_global_observations'],40)
        self.assertIn('monitored32',result['negative_tail_selection']['scope'])

    def test_old_runtime_sha_implicit_source_count_and_raw_change_reject(self):
        raw=v.common.canonical(prereg_fixture())
        for kw in [dict(expected_sha256='0'*64),dict(expected_cpu_tests=None),dict(expected_cpu_tests=True),
                   dict(source_commit=v.discovery.OLD_COMMIT),dict(source_commit='HEAD')]:
            default=dict(expected_sha256=v.common.sha(raw),source_commit='a'*40,target_cpu_proof_sha256='b'*64,expected_cpu_tests=999)
            with self.assertRaises(ValueError):v.validate_preregistration(raw,**(default|kw))
        with self.assertRaises(ValueError):
            v.validate_preregistration(raw+b'\n',expected_sha256=v.common.sha(raw),source_commit='a'*40,
                target_cpu_proof_sha256='b'*64,expected_cpu_tests=999)

    def test_rule_tolerance_tail_margin_candidate_sampler_and_health_drift_reject(self):
        changes=[lambda d:d.update(negative_tail_limit=1),lambda d:d.update(production_admission=True),
                 lambda d:d.update(healthy_checkpoint_accepted=True),lambda d:d.update(prospective_identity_binding_complete=True),
                 lambda d:d.update(actual_discovery_observations_for_this_source=True),lambda d:d.update(previous_method_preserved_sha256='0'*64),
                 lambda d:d['health_rule_document']['criteria']['allowed_drop'].update(weighted_em=.01),
                 lambda d:d['health_rule_document']['criteria']['allowed_drop'].update(weighted_f1=.01),
                 lambda d:d['health_rule_document']['criteria']['allowed_drop'].update(valid_action_rate=.01),
                 lambda d:d['health_rule_document']['provenance'].update(not_a_population_statistical_guarantee=False),
                 lambda d:d['negative_tail_selection'].update(arbitrary_margin_or_floor=True),
                 lambda d:d['negative_tail_selection'].update(required_global_observations=32),
                 lambda d:d['negative_tail_selection'].update(scope='Full2560 objective guarantee'),
                 lambda d:d['fixed_candidates'].update(dual_clip=2.),lambda d:d['fixed_candidates'].update(epsilon=.1),
                 lambda d:d['fixed_candidates'].update(target_old_policy_kl=.02),lambda d:d['fixed_candidates'].update(kl_log_bound=40.),
                 lambda d:d['sampling_distribution'].update(top_p=.95)]
        for change in changes:
            doc=prereg_fixture();change(doc)
            with self.assertRaises(ValueError):validate_prereg(doc)

    def test_missing_any_actual_prereg_cpu_input_reject_before_reads_or_remote(self):
        args=dict(source_commit='a'*40,target_cpu_proof_path='fixture',target_cpu_proof_sha256='b'*64,
            expected_cpu_tests=999,preregistration_path='fixture',preregistration_sha256='c'*64)
        for key in args:
            with patch.object(v,'_read_sealed') as read,patch.object(v.common,'remote') as remote:
                with self.assertRaises(ValueError):v.preregistration_inputs(**(args|{key:None}))
                read.assert_not_called();remote.assert_not_called()

    def test_pure_prereg_input_uses_real_temp_git_and_rootauthored_bytes(self):
        old=discovery_tests.actual_old_cpu();proof=discovery_tests.future_cpu_fixture(old)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();mapping=discovery_tests.new_mapping(root);repo=root/'repo'
            recipe=v.common.strict_json((v.REPO/'reproduction/common8/recipe.json').read_bytes())
            # Use real committed353 objects, then exercise recipe at a CPU boundary.
            proof['source_sha256']=mapping
            def git(*args):return subprocess.check_output(['git',*args],cwd=repo,text=True,stderr=subprocess.DEVNULL).strip()
            git('init');git('add','.');git('-c','user.name=CPU fixture','-c','user.email=fixture@example.invalid','commit','-m','CPU only')
            commit=git('rev-parse','HEAD');cpu_raw=v.common.canonical(proof);cpu=root/'cpu.json';cpu.write_bytes(cpu_raw)
            doc=prereg_fixture();doc.update(source_commit=commit,target_cpu_proof_sha256=v.common.sha(cpu_raw),created_unix=proof['checked_unix']+1)
            raw=v.common.canonical(doc);prereg=root/'prereg.json';prereg.write_bytes(raw)
            with patch.object(v.discovery,'new_recipe') as recipe_boundary,patch.object(v.common,'remote') as remote:
                result=v.preregistration_inputs(source_commit=commit,target_cpu_proof_path=cpu,target_cpu_proof_sha256=v.common.sha(cpu_raw),
                    expected_cpu_tests=999,preregistration_path=prereg,preregistration_sha256=v.common.sha(raw),repo=repo)
                recipe_boundary.assert_called_once();remote.assert_not_called()
            self.assertEqual(result['source_sha256'],mapping);self.assertEqual(result['prereg'],doc)
            self.assertEqual(prereg.read_bytes(),raw)


class ZeroObservationTests(unittest.TestCase):
    def test_actual_identity_schema_fixture_is_prospective_not_health(self):
        prereg,launch,observed=metadata_fixture();out=validate_zero(prereg,launch,observed)
        self.assertEqual(out['native_job_id'],'9999');self.assertFalse(out['production_admission'])
        self.assertFalse(out['healthy_checkpoint_accepted'])

    def test_running_missing_resource_recycled_birth_terminal_or_late_prereg_reject(self):
        prereg,launch,observed=metadata_fixture()
        for key,value in [('JobState','FAILED'),('ExitCode','1:0'),('JobId','9998'),('Command','/wrong'),('WorkDir','/wrong'),
                          ('MinMemoryNode','256G'),('TresPerNode','gres:gpu:h100:4'),('Restarts','1'),('NodeList','h100-9')]:
            bad=copy.deepcopy(observed);fields=v.discovery._job_fields(bad['job_stdout']);fields[key]=value
            bad['job_stdout']=' '.join(key+'='+value for key,value in fields.items())
            with self.assertRaises(ValueError):validate_zero(prereg,launch,bad)
        for change in [lambda l:l.update(submitted_unix=199.),lambda l:l.update(returncode=True),
                       lambda l:l.update(source_commit=v.discovery.OLD_COMMIT),lambda l:l.update(actual_target_cpu_tests=True),
                       lambda l:l['argv'].remove('--exclusive'),lambda l:l.update(base=v.discovery.OLD_BASE)]:
            bad=copy.deepcopy(launch);change(bad)
            with self.assertRaises(ValueError):validate_zero(prereg,bad,observed)

    def test_both_strict_zero_metadata_required_not_only_completed_counter(self):
        prereg,launch,observed=metadata_fixture()
        for change in [lambda o:o.update(child_actual_unstarted_verified=False),lambda o:o.update(parent_actual_unstarted_verified=False),
                       lambda o:o['child_actual_unstarted'].update(training_claim_exists=True),
                       lambda o:o['parent_actual_unstarted'].update(train_log_exists=True),
                       lambda o:o['child_actual_unstarted'].update(optimizer_evidence_paths=['rank0']),
                       lambda o:o['child_actual_unstarted'].update(checkpoint_entries=['model']),
                       lambda o:o.update(actual_updates=[{}]),lambda o:o.update(actual_behavior_gates=[{}]),
                       lambda o:o['run-manifest.json'].update(completed_updates=True),
                       lambda o:o['parent_run_manifest'].update(completed_updates=1),
                       lambda o:o['pipeline-status.json'].update(stage='BOUNDED_POLICY_CALIBRATION_ONE_OUTER')]:
            bad=copy.deepcopy(observed);change(bad)
            with self.assertRaises(ValueError):validate_zero(prereg,launch,bad)

    def test_source_capture_run_crossparent_manifest_and_timeline_drift_reject(self):
        prereg,launch,observed=metadata_fixture()
        for change in [lambda o:o.update(source_hashes_match=False),lambda o:o.update(child_manifest_sha256='unknown'),
                       lambda o:o.update(checked_unix=209.),lambda o:o.update(observation_finished_unix=219.),
                       lambda o:o['run-manifest.json'].update(parent_production_run_id='3'*32),
                       lambda o:o['parent_run_manifest'].update(run_id='1'*32),
                       lambda o:o['parent_run_manifest'].update(source_commit='b'*40),
                       lambda o:o.update(child_manifest_slurm_job_id='9998')]:
            bad=copy.deepcopy(observed);change(bad)
            with self.assertRaises(ValueError):validate_zero(prereg,launch,bad)

    def test_missing_launch_and_zero_args_reject_before_any_local_contract_reads(self):
        for kwargs in [dict(),dict(launch_path='fixture'),dict(launch_path='fixture',launch_sha256='c'*64)]:
            with patch.object(v,'preregistration_inputs') as check,patch.object(v.common,'remote') as remote:
                with self.assertRaises(ValueError):v.freeze_inputs(**kwargs)
                check.assert_not_called();remote.assert_not_called()


class NativeMetadataAndExclusiveTests(unittest.TestCase):
    def run_native(self,payload):
        result=S(returncode=0,stdout=payload['observed']['job_stdout'],stderr='')
        with patch.object(subprocess,'run',return_value=result):
            return v._native_freeze(payload)

    def test_actual_pure_runtime_unstarted_and_identity_then_exact_bytes_in_both_dirs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();payload=payload_fixture(root/'child',root/'parent')
            receipt=self.run_native(payload);content=v.verify_receipt(receipt,payload)
            self.assertFalse(receipt['production_admission']);self.assertFalse(receipt['physical_model_execution_authenticated'])
            doc=v.common.strict_json(content);self.assertIsNone(doc['negative_tail_limit'])
            for name in ('child','parent'):
                path=root/name/'preparation'/v.NAMES['method']
                self.assertEqual(path.read_bytes(),content);self.assertEqual(stat.S_IMODE(path.stat().st_mode),0o600)

    def test_existing_claim_or_single_rank_partial_ledger_prevents_all_method_mutations(self):
        for filename in ('training-launch-claim.json','optimizer-rank-3.jsonl','policy-calibration-segment.json'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve();payload=payload_fixture(root/'child',root/'parent')
                (root/'child/logs'/filename).write_bytes(b'CPU fixture prior attempt')
                with self.assertRaises(Exception):self.run_native(payload)
                for name in ('child','parent'):self.assertFalse((root/name/'preparation'/v.NAMES['method']).exists())

    def test_source_extra_changed_manifest_crossmath_or_query_unknown_no_mutation(self):
        for kind in ('extra','source','manifest','query'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve();payload=payload_fixture(root/'child',root/'parent')
                if kind=='extra':(root/'child/repo/extra.py').write_bytes(b'CPU extra')
                elif kind=='source':(root/'parent/repo/f001.py').write_bytes(b'CPU changed')
                elif kind=='manifest':
                    path=root/'parent/logs/run-manifest.json';doc=v.common.strict_json(path.read_bytes())
                    doc['input_sha256']['train.parquet']='9'*64;path.write_bytes(v.common.canonical(doc))
                if kind=='query':
                    with patch.object(subprocess,'run',return_value=S(returncode=1,stdout='',stderr='unknown')):
                        with self.assertRaises(ValueError):v._native_freeze(payload)
                else:
                    with self.assertRaises(ValueError):self.run_native(payload)
                for name in ('child','parent'):self.assertFalse((root/name/'preparation'/v.NAMES['method']).exists())

    def test_preexisting_native_method_blocks_both_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();payload=payload_fixture(root/'child',root/'parent')
            path=root/'child/preparation'/v.NAMES['method'];path.write_bytes(b'partial prior freeze')
            with self.assertRaises(ValueError):self.run_native(payload)
            self.assertEqual(path.read_bytes(),b'partial prior freeze')
            self.assertFalse((root/'parent/preparation'/v.NAMES['method']).exists())

    def test_unknown_mid_native_write_retains_first_byte_file_and_rejects_reentry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();payload=payload_fixture(root/'child',root/'parent')
            actual=v._exclusive_bytes;calls=[]
            def interrupted(path,content):
                calls.append(path)
                if len(calls)==2:raise RuntimeError('CPU simulated unknown write')
                actual(path,content)
            with patch.object(v,'_exclusive_bytes',side_effect=interrupted):
                with self.assertRaises(RuntimeError):self.run_native(payload)
            path=root/'parent/preparation'/v.NAMES['method'];raw=path.read_bytes()
            with self.assertRaises(ValueError):self.run_native(payload)
            self.assertEqual(path.read_bytes(),raw);self.assertFalse((root/'child/preparation'/v.NAMES['method']).exists())

    def test_receipt_wrong_source_tail_candidate_zero_cpu_or_binding_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();payload=payload_fixture(root/'child',root/'parent');receipt=self.run_native(payload)
            original=v.common.strict_json(base64.b64decode(receipt['method_bytes_base64']))
            for change in [lambda d:d.update(negative_tail_limit=1),lambda d:d.update(production_admission=True),
                           lambda d:d.update(source_commit='f'*40),lambda d:d['fixed_candidates'].update(kl_log_bound=31.),
                           lambda d:d['prospective_binding'].update(actual_target_cpu_tests=1),
                           lambda d:d['prospective_binding'].update(zero_update_observation_sha256='0'*64),
                           lambda d:d['prospective_binding'].update(preregistered_unix=0)]:
                doc=copy.deepcopy(original);change(doc);raw=v.common.canonical(doc)
                altered=receipt|dict(method_bytes_base64=base64.b64encode(raw).decode(),method_sha256=v.common.sha(raw))
                with self.assertRaises(ValueError):v.verify_receipt(altered,payload)

    def test_local_exclusive_mode_and_partial_file_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory).resolve()/'method';v._exclusive_bytes(path,b'CPU bytes')
            self.assertEqual(path.read_bytes(),b'CPU bytes');self.assertEqual(stat.S_IMODE(path.stat().st_mode),0o600)
            with self.assertRaises(FileExistsError):v._exclusive_bytes(path,b'replace')
            self.assertEqual(path.read_bytes(),b'CPU bytes')

    def test_missing_preconditions_or_reserved_intent_prevent_remote_and_contract_write(self):
        with tempfile.TemporaryDirectory() as directory:
            prep=Path(directory).resolve()
            with patch.object(v.common,'remote') as remote:
                with self.assertRaises(ValueError):v.freeze(prep=prep)
                remote.assert_not_called();self.assertEqual(list(prep.iterdir()),[])
            (prep/v.NAMES['intent']).write_bytes(b'{partial')
            with patch.object(v,'freeze_inputs') as inputs,patch.object(v.common,'remote') as remote:
                with self.assertRaises(ValueError):v.freeze(prep=prep)
                inputs.assert_not_called();remote.assert_not_called()

    def test_unknown_remote_freeze_retains_local_durable_intent_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();payload=payload_fixture(root/'child',root/'parent');prep=root/'control';prep.mkdir()
            with patch.object(v,'freeze_inputs',return_value=payload),patch.object(v.common,'remote',side_effect=RuntimeError('CPU unknown remote')) as remote:
                with self.assertRaises(RuntimeError):v.freeze(prep=prep)
                raw=(prep/v.NAMES['intent']).read_bytes()
                self.assertEqual(remote.call_count,1)
                with self.assertRaises(ValueError):v.freeze(prep=prep)
                self.assertEqual(remote.call_count,1);self.assertEqual((prep/v.NAMES['intent']).read_bytes(),raw)
                self.assertFalse((prep/v.NAMES['method']).exists());self.assertFalse((prep/v.NAMES['receipt']).exists())

    def test_embedded_compile_scope_no_gpu_explicit_identity_and_all_files_newv6(self):
        code=v.native_code();compile(code,'<CPU-only-native-metadata>','exec')
        self.assertNotIn('nvidia-smi',code);self.assertNotIn('torch.load',code);self.assertNotIn('model_smoke',code)
        self.assertNotIn('old_method',code);self.assertNotIn('495d3024',code);self.assertNotIn('61eeb304',code)
        self.assertIn('_unstarted(namespace,manifest)',code)
        self.assertLess(code.index('discovery._verify_new_mapping(namespace,mapping)'),code.index('from reproduction.common8.freeze_calibration_plan'))
        self.assertIn('os.O_EXCL|os.O_NOFOLLOW',code)
        self.assertTrue(all('v6' in name for name in v.NAMES.values()))
        for name in v.NAMES.values():self.assertFalse((v.PREP/name).exists())
        for old,sha in [('freeze_policy_discovery_v4_method.py','d2dbabbbc1a7c15ce6d9b64abf9b203adf5c29c64fef81b9b6cb62a7e4c5cd87')]:
            self.assertEqual(v.common.sha((v.PREP/old).read_bytes()),sha)


if __name__=='__main__':unittest.main()
