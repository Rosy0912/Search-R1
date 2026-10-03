import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import requests
from verl.trainer.main_ppo_eco import EcoRewardManager

class ProbeTransportAndVersion(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        root=Path(self.directory.name);(root/'logs').mkdir()
        (root/'logs/run-manifest.json').write_text(json.dumps({'initial_probe_version':'run:0:actual'}))
        self.env=patch.dict(os.environ,{'SEARCH_G1_BASE':str(root),'ECO_USE_CF':'0'});self.env.start()
        self.manager=EcoRewardManager(None,0)
        self.manager.probe_max_wait=8
        self.manager._probe_session=MagicMock()
        self.items=[{'id':'trajectory-1'}]
        self.addCleanup(self.directory.cleanup);self.addCleanup(self.env.stop)
    def response(self,version='run:0:actual',probability=.7):
        response=MagicMock()
        response.json.return_value={'version':version,'results':[{'id':'trajectory-1','version':version,'flip':probability,'boundary':.2}]}
        return response
    def test_actual_method_reads_manifest_version_and_returns_valid_reply(self):
        self.manager._probe_session.post.return_value=self.response()
        with patch('time.sleep') as sleep:
            result=self.manager._probe_call(self.items)
        self.assertEqual(result[0]['flip'],.7)
        self.assertEqual(self.manager.expected_probe_version,'run:0:actual')
        self.assertEqual(self.manager.last_probe_version,'run:0:actual');sleep.assert_not_called()
    def test_refitted_expected_version_is_used_instead_of_initial_manifest(self):
        self.manager.expected_probe_version='run:50:refitted'
        self.manager._probe_session.post.return_value=self.response('run:50:refitted')
        self.manager._probe_call(self.items)
        self.assertEqual(self.manager.last_probe_version,'run:50:refitted')
    def test_wrong_version_is_not_retried_or_reported_as_recovered(self):
        self.manager._probe_session.post.return_value=self.response('run:0:wrong')
        out=io.StringIO()
        with patch('time.sleep') as sleep,contextlib.redirect_stdout(out):
            with self.assertRaisesRegex(ValueError,'Evaluator version mismatch'):self.manager._probe_call(self.items)
        sleep.assert_not_called();self.assertNotIn('recovered',out.getvalue())
        self.assertFalse(hasattr(self.manager,'last_probe_version'))
    def test_nonfinite_probability_is_not_retried(self):
        self.manager._probe_session.post.return_value=self.response(probability=float('nan'))
        with patch('time.sleep') as sleep:
            with self.assertRaisesRegex(ValueError,'Invalid finite'):self.manager._probe_call(self.items)
        sleep.assert_not_called()
    def test_bad_json_is_not_retried_as_transport_failure(self):
        response=self.response();response.json.side_effect=requests.exceptions.JSONDecodeError('bad JSON','x',0)
        self.manager._probe_session.post.return_value=response
        with patch('time.sleep') as sleep:
            with self.assertRaises(requests.exceptions.JSONDecodeError):self.manager._probe_call(self.items)
        sleep.assert_not_called()
    def test_transport_recovers_only_after_validated_reply(self):
        self.manager._probe_session.post.side_effect=[requests.exceptions.ConnectionError('temporary'),self.response()]
        out=io.StringIO()
        with patch('time.sleep') as sleep,contextlib.redirect_stdout(out):
            rows=self.manager._probe_call(self.items)
        sleep.assert_called_once_with(4);self.assertEqual(rows[0]['version'],'run:0:actual')
        self.assertIn('recovered after 1 retries',out.getvalue())
    def test_programming_error_fails_immediately(self):
        self.manager._probe_session.post.side_effect=NameError('programming error')
        with patch('time.sleep') as sleep:
            with self.assertRaises(NameError):self.manager._probe_call(self.items)
        sleep.assert_not_called()
    def test_http_errors_fail_without_retry(self):
        for status in (400,422,500,503):
            response=self.response();response.raise_for_status.side_effect=requests.exceptions.HTTPError(str(status))
            self.manager._probe_session.post.return_value=response
            with patch('time.sleep') as sleep:
                with self.assertRaises(requests.exceptions.HTTPError):self.manager._probe_call(self.items)
            sleep.assert_not_called()
    def test_total_budget_counts_request_time_and_bounds_request_timeout(self):
        clock=[0.0];self.manager.probe_max_wait=5;self.manager.probe_timeout=600
        def failed_request(*args,**kwargs):
            self.assertEqual(kwargs['timeout'],5)
            clock[0]+=4
            raise requests.exceptions.Timeout('slow request')
        self.manager._probe_session.post.side_effect=failed_request
        with patch('time.monotonic',side_effect=lambda:clock[0]),patch('time.sleep',side_effect=lambda n:clock.__setitem__(0,clock[0]+n)) as sleep:
            with self.assertRaisesRegex(RuntimeError,'total wait exceeded'):self.manager._probe_call(self.items)
        self.assertEqual(clock[0],5);sleep.assert_called_once_with(1)
    def test_late_success_is_rejected_before_reward_version_publication(self):
        clock=[0.0];self.manager.probe_max_wait=5
        def late_success(*args,**kwargs):
            clock[0]=6;return self.response()
        self.manager._probe_session.post.side_effect=late_success
        with patch('time.monotonic',side_effect=lambda:clock[0]):
            with self.assertRaisesRegex(RuntimeError,'refusing late reward'):self.manager._probe_call(self.items)
        self.assertFalse(hasattr(self.manager,'last_probe_version'))
    def test_empty_batch_needs_no_transport(self):
        self.assertEqual(self.manager._probe_call([]),[])
        self.manager._probe_session.post.assert_not_called()
if __name__=='__main__':unittest.main()
