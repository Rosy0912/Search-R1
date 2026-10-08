"""Call the unchanged snapshot reader and retain actual failed transport bytes."""
import datetime
import hashlib
import json
from pathlib import Path
import runpy
import sys

W = Path(__file__).resolve().parent
P = W / 'snapshot_actual.py'
PIN = '7b2e0b058b6616abd641f37950de2e4cf4864f5e61ae0d232661097d4d389083'
assert hashlib.sha256(P.read_bytes()).hexdigest() == PIN
D = W / ('snapshot-preserved-caller-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
D.mkdir()
(D / 'intent.json').write_text(json.dumps({'original_reader_sha256': PIN,
    'external_transport_timeout_seconds': 360,
    'scope': 'Read-only actual snapshot transport, not training control or acceptance'}))
sys.path.insert(0, str(W))
import transport
original = transport.remote


def remote(cmd, data=None, timeout=60):
    try:
        return original(cmd, data, timeout=max(360, timeout))
    except Exception as error:
        for field in ('stdout', 'stderr'):
            value = getattr(error, field, None)
            if value is not None:
                (D / ('failure-original.' + field)).write_bytes(value if isinstance(value, bytes) else value.encode())
        record = {'scope': 'Actual snapshot transport failure, not training failure',
            'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'error_type': type(error).__name__, 'returncode': getattr(error, 'returncode', None),
            'timeout': getattr(error, 'timeout', None)}
        (D / 'failure-original.json').write_text(json.dumps(record))
        print(json.dumps({'caller_directory': str(D), 'failed': record}), flush=True)
        raise SystemExit(1)


transport.remote = remote
sys.argv = [str(P), '--job-id', '3261']
runpy.run_path(str(P), run_name='__main__')
