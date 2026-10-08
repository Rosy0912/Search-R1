"""Hash installed implementation binaries, without import or copying binaries."""
import hashlib
import json
import socket
from pathlib import Path

assert socket.gethostname().split('.')[0] == 'h100-9'
base = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/env/lib/python3.10/site-packages')
paths = sorted((base/'vllm').glob('_C*.so')) + sorted(base.glob('flash_attn_2_cuda*.so'))
assert paths and any(p.name.startswith('_C.') for p in paths)
assert any(p.name.startswith('flash_attn_2_cuda') for p in paths)
records = []
for path in paths:
    assert path.is_file() and path.resolve().is_relative_to(base.resolve())
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for part in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(part)
    records.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest()))
print(json.dumps(dict(status='INSTALLED_KERNEL_BINARY_METADATA_ONLY_NO_RUNTIME_ACCEPTANCE',
                      hostname=socket.gethostname(), records=records,
                      binaries_copied=False, package_imports=[], gpu_called=False,
                      tensor_assets_loaded=False, optimizer_writes=0), sort_keys=True))
