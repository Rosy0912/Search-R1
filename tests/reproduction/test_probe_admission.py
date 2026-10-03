import unittest
import ast,json,os,types
from pathlib import Path
from search_g1.signals import validate_probe_reply
from unittest.mock import patch
from reproduction.a800 import probe_admission as m

class AdmissionTests(unittest.TestCase):
    def fixture(self):
        run=dict(run_id='r',initial_model='/model',initial_model_sha256={'weights':'h'})
        proof=dict(question_count=300,searched_valid_trajectory_count=4,group_size=5,question_ids=list(range(300)),closed_book_label='exact_match_no_retrieval',reliance_label='normalized_answer_change_after_evidence_deletion',model_authored_answer_spans=True,boundary_all_reserved_questions=True)
        report=dict(unique_question_count=300,commit_class_counts=[2,2],boundary_class_counts=[150,150],commit_cv_auc=.6,boundary_cv_auc=.7,question_grouped_cv=True)
        live=dict(run_id='r',round=0,encoder='/model',version='r:0:v',calibration_report=report,label_proof=proof,encoder_sha256={'weights':'h'},head_sha256={'commit_probe.pkl':'h','boundary_probe.pkl':'h'},commit='/commit',boundary='/boundary')
        return run,live
    def test_both_real_calibration_contracts_required(self):
        run,live=self.fixture()
        with patch.object(m,'digest',return_value='h'):
            self.assertEqual(m.calibrated_version(run,live),'r:0:v')
            live['calibration_report']['boundary_class_counts']=[300,0]
            with self.assertRaises(AssertionError):m.calibrated_version(run,live)
    def test_wrong_run_or_modified_heads_rejected(self):
        run,live=self.fixture()
        with patch.object(m,'digest',return_value='changed'):
            with self.assertRaises(AssertionError):m.calibrated_version(run,live)
        live['run_id']='other'
        with self.assertRaises(AssertionError):m.calibrated_version(run,live)
    def test_proxy_labels_and_duplicate_questions_rejected(self):
        run,live=self.fixture();live['label_proof']['question_ids']=[0]*300
        with self.assertRaises(AssertionError):m.calibrated_version(run,live)

class RewardVersionTests(unittest.TestCase):
    def test_actual_reward_client_rejects_stale_before_update(self):
        source=Path(__file__).resolve().parents[2]/'verl/trainer/main_ppo_eco.py'
        tree=ast.parse(source.read_text());fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_probe_call')
        module=ast.Module(body=[fn],type_ignores=[]);ns=dict(os=os,json=json,validate_probe_reply=validate_probe_reply)
        exec(compile(ast.fix_missing_locations(module),str(source),'exec'),ns)
        items=[{'id':'q'}]
        reply=dict(version='old',results=[dict(id='q',version='old',flip=.5,boundary=.5)])
        response=types.SimpleNamespace(raise_for_status=lambda:None,json=lambda:reply)
        client=types.SimpleNamespace(expected_probe_version='correct',probe_url='unused',probe_timeout=1,probe_backoff_cap=4,probe_max_wait=4,_probe_session=types.SimpleNamespace(post=lambda *a,**kw:response))
        with patch.dict(os.environ,{'SEARCH_G1_BASE':'/unused'}),patch('time.sleep'):
            with self.assertRaises(RuntimeError):ns['_probe_call'](client,items)
            self.assertFalse(hasattr(client,'last_probe_version'))
            reply['version']='correct';reply['results'][0]['version']='correct'
            self.assertEqual(ns['_probe_call'](client,items),reply['results'])
            self.assertEqual(client.last_probe_version,'correct')

if __name__=='__main__':unittest.main()
