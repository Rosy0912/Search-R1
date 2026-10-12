"""Install the already-reviewed first-five provenance bytes for job 3359 only."""
import base64
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009')
SOURCE = '0fe642bd0c71f0b5f9813cbbad7bdd94e0625244'
TARGET = ROOT / 'tools/deployment/first5-node11-3359/aligned-first5-math-provenance.json'
EXPECTED_SHA256 = 'a59a0503e2cbb66448e9ddf3b1ef3bc0e7cf7e0d716b68f273f3cfd664435e13'
# Construct from the exact reviewed JSON so a typo in the compact transport literal cannot pass.
PAYLOAD = json.dumps({
    'scope': 'generic functions only; old root/source/job/behavior policy excluded',
    'original_file_sha256': '3ea5370c5812a9460d580eb1f94e2f1a848b8423a790f23ea06470e8dbabd967',
    'functions': ['digest', 'finite', 'require', 'resolve_qwen2_tied_alias', 'safetensor_map', 'tokenizer_proof'],
    'extracted_sha256': '32c037c9e81bb9b0f3c30396f09623db57a3d763754a5d29b3c28e37964e3815',
}, indent=2).encode() + b'\n'

assert os.uname().nodename == 'h100-11'
assert ROOT.is_dir() and ROOT.resolve() == ROOT
assert (ROOT / 'scheduler-head/formal-submit-receipt.json').is_file()
receipt = json.loads((ROOT / 'scheduler-head/formal-submit-receipt.json').read_text())
assert receipt['job'] == '3359' and receipt['token'] == '16c570e111f947e2b0026983df2f7938'
assert receipt['source'] == SOURCE
assert TARGET.parent.resolve() == TARGET.parent and not TARGET.parent.is_symlink()
assert hashlib.sha256(PAYLOAD).hexdigest() == EXPECTED_SHA256
if TARGET.exists():
    assert TARGET.is_file() and not TARGET.is_symlink() and TARGET.read_bytes() == PAYLOAD
    action = 'EXACT_ALREADY_PRESENT'
else:
    with TARGET.open('xb') as fh:
        fh.write(PAYLOAD)
        fh.flush()
        os.fsync(fh.fileno())
    action = 'EXACT_CREATED'
print(json.dumps({'job': 3359, 'source_commit': SOURCE, 'path': str(TARGET),
                  'sha256': EXPECTED_SHA256, 'action': action,
                  'scope': 'missing reviewed provenance only; no training/runtime change'}))
