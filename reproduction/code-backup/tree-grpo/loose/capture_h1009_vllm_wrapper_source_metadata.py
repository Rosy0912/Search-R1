"""Capture installed wrapper source bytes only; no imports or tensor access."""
import base64
import gzip
import hashlib
import json
import socket
from pathlib import Path

assert socket.gethostname().split('.')[0] == 'h100-9'
base = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/env/lib/python3.10/site-packages')
names = ['vllm/utils.py', 'torch/utils/_contextlib.py']
files = {}
for name in names:
    path = base / name
    assert path.is_file() and path.resolve().is_relative_to(base.resolve())
    payload = path.read_bytes()
    assert len(payload) < 1024 ** 2
    files[name] = dict(path=str(path), bytes=len(payload),
                       sha256=hashlib.sha256(payload).hexdigest(),
                       gzip_base64=base64.b64encode(gzip.compress(payload)).decode())
print(json.dumps(dict(status='INSTALLED_WRAPPER_SOURCE_CAPTURE_ONLY_NO_RUNTIME_ACCEPTANCE',
                      hostname=socket.gethostname(), files=files,
                      package_imports=[], tensor_files_loaded=False,
                      model_initialized=False, gpu_called=False,
                      optimizer_writes=0), sort_keys=True, allow_nan=False))
