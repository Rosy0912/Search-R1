"""Actual HTTP body timeout and strict one-replay budget, CPU only."""
import contextlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from . import transport_recovery as module
from .protocol import digest


class TransportRecoveryTests(unittest.TestCase):
    def test_native_success_identical_arguments_result_and_rng(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'output').mkdir(); calls = []; before = random.getstate()
            def request(url, queries):
                calls.append((url, queries)); return ([[{'text': 'fixture'}]], 'same-response')
            result = module.perform_search('http://fixture/retrieve', ['a'], root, 0, 1, request)
            self.assertEqual(calls, [('http://fixture/retrieve', ['a'])])
            self.assertEqual(result, ([[{'text': 'fixture'}]], 'same-response'))
            self.assertEqual(random.getstate(), before)
            self.assertFalse((Path(root) / 'output/http-single-replay-claim.json').exists())

    def test_actual_http_body_timeout_then_identical_native_payload(self):
        received = []; response = json.dumps({'result': [[{'text': 'stable'}]]}).encode()
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length'])); received.append(body)
                number = len(received)
                self.send_response(200); self.send_header('Content-Length', str(len(response))); self.end_headers()
                if number == 1: time.sleep(.2)
                try: self.wfile.write(response)
                except BrokenPipeError: pass
            def log_message(self, *args): pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as root:
                (Path(root) / 'output').mkdir()
                def native(url, queries):
                    body = json.dumps(dict(queries=queries, topk=3, return_scores=True)).encode()
                    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req, timeout=.05) as reply: data = json.loads(reply.read())
                    return data['result'], digest(data)
                result = module.perform_search('http://127.0.0.1:'+str(server.server_port)+'/retrieve', ['query'], root, 0, 1, native)
                self.assertEqual(len(received), 2); self.assertEqual(received[0], received[1])
                self.assertEqual(result, ([[{'text': 'stable'}]], digest(json.loads(response))))
                self.assertTrue((Path(root) / 'output/http-single-replay-claim.json').exists())
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_allocation_budget_is_shared_across_ranks(self):
        with tempfile.TemporaryDirectory() as root, patch.object(module.time, 'sleep'):
            (Path(root) / 'output').mkdir(); calls = []
            def transient(url, queries):
                calls.append(queries); raise socket.timeout('fixture')
            with self.assertRaises(socket.timeout):
                module.perform_search('http://fixture', ['q'], root, 0, 1, transient)
            self.assertEqual(len(calls), 2)
            with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                module.perform_search('http://fixture', ['q'], root, 1, 1, transient)
            self.assertEqual(len(calls), 3)

    def test_http_error_and_invalid_content_are_not_retried(self):
        for error in [urllib.error.HTTPError('http://fixture', 500, 'error', {}, None), ValueError('bad JSON')]:
            with tempfile.TemporaryDirectory() as root:
                (Path(root) / 'output').mkdir(); calls = []
                def bad(url, queries):
                    calls.append(1); raise error
                with self.assertRaises(type(error)):
                    module.perform_search('http://fixture', ['q'], root, 0, 1, bad)
                self.assertEqual(calls, [1])
                self.assertFalse((Path(root) / 'output/http-single-replay-claim.json').exists())

    def test_request_identity_and_attempt_cost_journal(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / 'output'; out.mkdir()
            module.perform_search('http://fixture', ['a', 'b'], root, 2, 9, lambda u, q: ([], 'response'))
            start = json.loads((out/'http-rank-2-call-9-attempt-1.start.json').read_text())
            end = json.loads((out/'http-rank-2-call-9-attempt-1.complete.json').read_text())
            self.assertEqual(start['request_identity'], end['request_identity'])
            self.assertEqual(start['logical_query_count'], 2)
            self.assertEqual(start['timeout_seconds'], 180)

    def test_wrapped_transport_error_only(self):
        self.assertTrue(module.transport_exception(urllib.error.URLError(socket.timeout())))
        self.assertTrue(module.transport_exception(http.client.IncompleteRead(b'partial')))
        self.assertFalse(module.transport_exception(urllib.error.URLError('DNS configuration error')))


if __name__ == '__main__': unittest.main()
