import io
import unittest
import urllib.error
from unittest.mock import Mock
from .health import wait_retriever_health

class HealthTest(unittest.TestCase):
    def test_transient_then_success(self):
        opener=Mock(side_effect=[TimeoutError(),io.StringIO('{"job":"j"}')]);sleep=Mock()
        self.assertEqual(wait_retriever_health('http://r','j',opener=opener,sleeper=sleep),{'job':'j'})
        self.assertEqual(opener.call_count,2);sleep.assert_called_once_with(5)
    def test_exhaustion_is_bounded(self):
        opener=Mock(side_effect=TimeoutError());sleep=Mock()
        with self.assertRaises(TimeoutError):wait_retriever_health('http://r','j',opener=opener,sleeper=sleep)
        self.assertEqual(opener.call_count,6);self.assertEqual(sleep.call_count,5)
    def test_identity_mismatch_does_not_retry(self):
        opener=Mock(return_value=io.StringIO('{"job":"other"}'))
        with self.assertRaises(ValueError):wait_retriever_health('http://r','j',opener=opener,sleeper=Mock())
        self.assertEqual(opener.call_count,1)
    def test_auth_error_does_not_retry(self):
        opener=Mock(side_effect=urllib.error.HTTPError('http://r',403,'no',None,None))
        with self.assertRaises(urllib.error.HTTPError):wait_retriever_health('http://r','j',opener=opener,sleeper=Mock())
        self.assertEqual(opener.call_count,1)
    def test_invalid_bound_rejected(self):
        with self.assertRaises(ValueError):wait_retriever_health('http://r','j',attempts=100)
