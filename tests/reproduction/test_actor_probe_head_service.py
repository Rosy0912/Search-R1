"""RAM-only classifiers and tiny scalar files; no native/GPU/model admission."""
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from reproduction.common8.actor_probe_head_service import HeadService, verify_manifest, build_app
from reproduction.common8.actor_probe_head_admission import verify_endpoint_documents, verify_inputs_and_source
from search_g1.actor_probe_contract import MODE, head_round
from tests.reproduction.test_actor_probe_recovery import documents


class ClassifierFixture:
    classes_ = np.array([0,1]); n_features_in_ = 2
    def predict_proba(self,values):
        assert values.shape == (2,2) and values.dtype == np.float32
        p = .25 + .25*(values[:,0] > 0)
        return np.stack((1-p,p),axis=1)


def fitted_fixture_loader(heads):
    # Target joblib is not installed on the Mac. Only its deserializer is
    # substituted; the real service validates/hashes the scalar fixture files.
    return patch.dict('sys.modules',{'joblib':SimpleNamespace(load=lambda path:heads[Path(path).name])})


def fixture(root,step=0):
    base=Path(root).resolve(); (base/'artifacts/live').mkdir(parents=True)
    parent,child,binding,_=documents()
    run=parent if step==0 else child
    if step:
        binding.update(completed_outer=step,actual_optimizer_writes=20*step,
            attempt_index=step+2,aborted_attempts=2)
    run['calibration_question_ids']=[f'fixture-id-{i}' for i in range(300)]
    run['initial_probe_version']=None if step==0 else 'saved-version'
    round_id=head_round(step+1)
    manifest=dict(run_id=run['run_id'],round=round_id,actor_step=round_id,
        version='saved-version' if step else run['run_id']+':0:initial-fixture',representation_source=MODE,
        parameter_ownership='independent_binary_heads_only',encoder='/data/unit/immutable-fit-export',
        calibration_data_sha256=run['input_sha256']['calibration-300.parquet'],head_sha256={},
        calibration_report=dict(unique_question_count=300,question_grouped_cv=True,
            require_all_cv_folds=True,configured_cv_folds=2,commit_class_counts=[120,180],
            boundary_class_counts=[150,150],commit_cv_auc=.6,boundary_cv_auc=.7),
        label_proof=dict(question_count=300,group_size=5,question_ids=run['calibration_question_ids'],
            closed_book_label='exact_match_no_retrieval',
            reliance_label='normalized_answer_change_after_evidence_deletion',
            model_authored_answer_spans=True,boundary_all_reserved_questions=True,
            searched_valid_trajectory_count=400,
            calibration_protocol=dict(rollout_max_new=500,correction_feedback=True),
            evidence_deletion='typed_retrieval_ranges_only_preserve_model_and_correction'))
    heads={}
    for key in ('commit','boundary'):
        path=base/'artifacts'/f'{key}_probe.pkl';path.write_text(json.dumps({'fixture':key}))
        manifest[key]=str(path);manifest['head_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        heads[path.name]=dict(clf=ClassifierFixture(),auc=.7,layer=21,
            position='Xa' if key=='commit' else 'Xp',label='flip' if key=='commit' else 'closedbook')
    live=base/'artifacts/live/manifest.json';live.write_text(json.dumps(manifest,indent=2))
    producer=dict(pid=7,start_ticks=11,sid=7,uid=os.getuid())
    service=HeadService(base,run,producer)
    return base,run,manifest,service,heads


class HeadServiceTests(unittest.TestCase):
    def test_actual_calibration_audit_and_complete_runtime_files_rechecked_at_gate(self):
        from reproduction.common8.source_files import runtime_files, VERSION_FILES
        with tempfile.TemporaryDirectory() as root:
            base=Path(root).resolve(); repo=base/'repo';repo.mkdir();(base/'data').mkdir()
            for name in VERSION_FILES:
                path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture version')
            source=repo/'scorer.py';source.write_text('# fixture source only\n')
            calibration=base/'data/calibration-300.parquet';calibration.write_text('{"fixture":"scalar only"}')
            audit=base/'data/audit.json';audit.write_text('{"fixture":"reserved ids"}')
            sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            run=dict(source_commit='a'*40,source_sha256={str(p.relative_to(repo)):sha(p) for p in runtime_files(repo)},
                input_sha256={p.name:sha(p) for p in (calibration,audit)})
            def git(args,**kwargs):return run['source_commit']+'\n' if args[1]=='rev-parse' else ''
            with patch('reproduction.common8.actor_probe_head_admission.subprocess.check_output',git):
                verify_inputs_and_source(base,run)
                for path in (calibration,audit,source):
                    original=path.read_bytes();path.write_text('{"fixture":"changed"}')
                    with self.assertRaises(AssertionError):verify_inputs_and_source(base,run)
                    path.write_bytes(original)
                (repo/'unbound-worker.py').write_text('# unexpected runtime file\n')
                with self.assertRaises(AssertionError):verify_inputs_and_source(base,run)

    def test_actual_http_routes_enforce_version_and_suspend_protocol_on_ram_fixture(self):
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as root:
            _,_,manifest,service,heads=fixture(root)
            with fitted_fixture_loader(heads), TestClient(build_app(service)) as client:
                health=client.get('/health');self.assertEqual(health.status_code,200)
                self.assertFalse(health.json()['frozen_encoder_loaded'])
                self.assertEqual(client.post('/readout_smoke',json={'version':'stale'}).status_code,409)
                self.assertEqual(client.post('/readout_smoke',json={'wrong_field':'stale'}).status_code,422)
                request={'version':manifest['version']}
                smoke=client.post('/readout_smoke',json=request);self.assertEqual(smoke.status_code,200)
                self.assertFalse(smoke.json()['actor_features_exercised'])
                self.assertEqual(client.post('/suspend',json=request).status_code,200)
                self.assertEqual(client.post('/readout_smoke',json=request).status_code,409)
                self.assertEqual(client.post('/resume',json=request).status_code,200)
                self.assertEqual(client.post('/readout_smoke',json=request).status_code,200)

    def test_fresh_and_saved_origins_use_head_plumbing_without_claiming_actor_accuracy(self):
        for step in (0,5,50,100,150):
            with self.subTest(step=step),tempfile.TemporaryDirectory() as root:
                base,run,manifest,service,heads=fixture(root,step)
                with fitted_fixture_loader(heads):
                    health=service.health();smoke=service.smoke(manifest['version'])
                    sha=hashlib.sha256((base/'artifacts/live/manifest.json').read_bytes()).hexdigest()
                    origin=verify_endpoint_documents(run,manifest,health,smoke,sha)
                    self.assertEqual(origin['completed_outer'],step)
                    self.assertEqual(health['round'],head_round(step+1))
                    self.assertFalse(health['frozen_encoder_loaded'])
                    self.assertFalse(smoke['actor_features_exercised'])
                    self.assertFalse(smoke['task_accuracy_claimed'])
                    self.assertEqual(smoke['readouts'],{'commit':[.25,.5],'boundary':[.25,.5]})

    def test_suspend_resume_and_new_pair_refresh_keep_fixed_version_until_complete_publication(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,manifest,service,heads=fixture(root)
            with fitted_fixture_loader(heads):
                service.health();self.assertTrue(service.suspend(manifest['version'])['suspended'])
                with self.assertRaises(AssertionError):service.smoke(manifest['version'])
                self.assertTrue(service.resume(manifest['version'])['resumed'])
                service.suspend(manifest['version'])
                self.assertEqual(service.health()['version'],manifest['version'])
                self.assertTrue(service.health()['suspended'])
                changed=dict(manifest,version='next-version',round=10,actor_step=10)
                (base/'artifacts/live/manifest.json').write_text(json.dumps(changed,indent=2))
                self.assertEqual(service.health()['version'],'next-version')
                self.assertFalse(service.health()['suspended'])
                with self.assertRaises(AssertionError):service.smoke(manifest['version'])
                self.assertEqual(service.smoke('next-version')['round'],10)

    def test_rebound_uuid_skipped_round_cached_corruption_and_wrong_classes_are_rejected(self):
        for problem in ('uuid','round','corrupt','classes'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as root:
                base,run,manifest,service,heads=fixture(root)
                with fitted_fixture_loader(heads):
                    if problem!='classes':service.health()
                    if problem=='uuid':manifest['encoder']='/data/unit/changed'
                    elif problem=='round':manifest.update(version='future',round=20,actor_step=20)
                    elif problem=='corrupt':Path(manifest['boundary']).write_text('{"fixture":"wrong"}')
                    else:heads['boundary_probe.pkl']['clf'].classes_=np.array([0,2])
                    if problem in ('uuid','round'):(base/'artifacts/live/manifest.json').write_text(json.dumps(manifest,indent=2))
                    with self.assertRaises(AssertionError):service.health()

    def test_actual_service_publishes_terminal_pair_without_prediction201_admission(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,manifest,service,heads=fixture(root)
            live=base/'artifacts/live/manifest.json'
            manifest.update(version='head190',round=190,actor_step=190)
            live.write_text(json.dumps(manifest))
            with fitted_fixture_loader(heads):
                self.assertEqual(service.health()['round'],190)
                service.suspend('head190')
                terminal=dict(manifest,version='head200',round=200,actor_step=200)
                live.write_text(json.dumps(terminal))
                self.assertEqual(service.health()['round'],200)
                self.assertFalse(service.health()['suspended'])
                self.assertEqual(service.smoke('head200')['round'],200)
                for logical in (200,201):
                    with self.assertRaises(ValueError):verify_manifest(run,terminal,logical_update=logical)
                live.write_text(json.dumps(dict(terminal,version='head210',round=210,actor_step=210)))
                with self.assertRaises(ValueError):service.health()

    def test_endpoint_identity_origin_hash_and_false_accuracy_claims_cannot_be_admitted(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,manifest,service,heads=fixture(root,100)
            with fitted_fixture_loader(heads):
                health=service.health();smoke=service.smoke(manifest['version'])
            sha=hashlib.sha256((base/'artifacts/live/manifest.json').read_bytes()).hexdigest()
            for problem in ('run','round','hash','teacher','accuracy','actor','nan','producer'):
                with self.subTest(problem=problem):
                    h=copy.deepcopy(health);s=copy.deepcopy(smoke)
                    if problem=='run':h['identity']['run_id']='0'*32
                    elif problem=='round':h['round']=0
                    elif problem=='hash':h['live_manifest_sha256']='0'*64
                    elif problem=='teacher':h['frozen_encoder_loaded']=True
                    elif problem=='accuracy':s['task_accuracy_claimed']=True
                    elif problem=='actor':s['actor_features_exercised']=True
                    elif problem=='nan':s['readouts']['commit'][0]=float('nan')
                    else:s['producer']['start_ticks']+=1
                    with self.assertRaises(AssertionError):verify_endpoint_documents(run,manifest,h,s,sha)

    def test_teacher_representation_wrong_target_and_calibration_leak_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            _,run,manifest,_,_=fixture(root)
            for problem in ('teacher','shared_head','question_ids','label','folds'):
                with self.subTest(problem=problem):
                    wrong=copy.deepcopy(manifest)
                    if problem=='teacher':wrong['representation_source']='frozen'
                    elif problem=='shared_head':wrong['parameter_ownership']='generation_head'
                    elif problem=='question_ids':wrong['label_proof']['question_ids'][0]='other-validation-id'
                    elif problem=='label':wrong['label_proof']['closed_book_label']='reward_guess'
                    else:wrong['calibration_report']['require_all_cv_folds']=False
                    with self.assertRaises((AssertionError,ValueError)):verify_manifest(run,wrong)


if __name__=='__main__':unittest.main()
