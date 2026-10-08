"""Via management SSH stdout: read 156 proofs and official tokenizer; no writes/GPU/weights."""
import datetime
import hashlib
import inspect
import json
import socket
import sys
from pathlib import Path

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
CANDIDATE = ROOT / 'logs/policy-safety-native-52befe9-candidate-v2.json'
EXPECTED = '7faa41598152201ca0a5e9cd1b41c06c6525ad1ffa2f8577360f29f929bd0a6a'
SOURCE = ROOT / 'repo-policy-safety-52befe9'
MODEL = ROOT / 'assets/qwen2.5-3b-instruct-aa8e725'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


assert socket.gethostname().split('.')[0] == 'h100-8'
assert digest(CANDIDATE) == EXPECTED
candidate = json.loads(CANDIDATE.read_text())
assert len(candidate['evidence']) == 156
records = []
for item in candidate['evidence']:
    p = Path(item['path'])
    before = p.stat()
    actual = digest(p)
    after = p.stat()
    assert (before.st_size, before.st_mtime_ns, before.st_dev, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino)
    assert actual == item['sha256'], str(p)
    records.append(dict(path=str(p), bytes=before.st_size, sha256=actual,
                        mtime_ns=before.st_mtime_ns, device=before.st_dev, inode=before.st_ino))
identity_files = {}
raw = {}
for name in ('config.json', 'generation_config.json', 'tokenizer_config.json', 'tokenizer.json',
             'VERIFIED-MODEL-MANIFEST.json', 'model.safetensors.index.json'):
    p = MODEL / name
    expected = next(x['sha256'] for x in candidate['files'] if x['path'] == str(p))
    actual = digest(p)
    assert actual == expected
    identity_files[name] = dict(path=str(p), bytes=p.stat().st_size, sha256=actual)
    raw[name] = json.loads(p.read_text())
sys.path.insert(0, str(SOURCE))
import torch
assert not torch.cuda.is_initialized()
from verl.utils import hf_tokenizer
import verl.utils.tokenizer as actual_tokenizer_module
assert inspect.getfile(actual_tokenizer_module) == str(SOURCE / 'verl/utils/tokenizer.py')
tokenizer = hf_tokenizer(str(MODEL), local_files_only=True)
assert not torch.cuda.is_initialized()
assert isinstance(tokenizer.pad_token_id, int)
print(json.dumps(dict(status='READ_ONLY_EVIDENCE_AND_OFFICIAL_TOKENIZER_IDENTITY_NOT_GPU_NOT_NATIVE_ACCEPTANCE',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), node=socket.gethostname(), executable=sys.executable,
    source_commit=candidate['code_commit'], native_candidate_sha256=EXPECTED,
    inherited_evidence_count=len(records), inherited_evidence_total_bytes=sum(r['bytes'] for r in records), evidence=records,
    official_model_path=str(MODEL), official_revision=candidate['fresh_model_revision'], identity_files=identity_files,
    model_pad_token_id=raw['config.json'].get('pad_token_id'),
    generation_pad_token_id=raw['generation_config.json'].get('pad_token_id'),
    configured_pad_token=raw['tokenizer_config.json'].get('pad_token'),
    configured_eos_token=raw['tokenizer_config.json'].get('eos_token'),
    actual_hf_pad_token_id=tokenizer.pad_token_id, actual_hf_eos_token_id=tokenizer.eos_token_id,
    actual_hf_pad_token=tokenizer.pad_token, actual_hf_eos_token=tokenizer.eos_token,
    tokenizer_class=type(tokenizer).__name__, tokenizer_import_path=inspect.getfile(actual_tokenizer_module),
    torch=torch.__version__, cuda_initialized=False, weights_loaded=False, remote_files_written=False,
    large_asset_hashes_performed=0, native_acceptance=None), allow_nan=False))
