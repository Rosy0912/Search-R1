"""One isolated publication process, with unchanged-input digest reuse across both phases."""
import importlib
import json
import sys
from pathlib import Path
from verified_digest_cache import VerifiedDigestCache

ROOT = Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009')
sys.path.insert(0, str(ROOT/'tools/deployment/first5-node11-3359-v2'))
import publish_first5_release as publisher
cache = VerifiedDigestCache()
gate = publisher.source_gate()
# Only verifier-local module references change; production files and training process stay intact.
for name in ('search_r1.baselines.fresh200_gate',):
    module = importlib.import_module(name)
    module.digest = cache.digest
for name in ('search_r1.baselines.aligned_partial_resume', 'search_r1.baselines.aligned_complete5_resume'):
    module = importlib.import_module(name)
    module.exact_file = cache.exact_file
publisher.cpu.digest = cache.digest
receipt = ROOT/'evidence/first5-independent-review3359-20261009-2210/FIRST5-3359-RELEASE-RECEIPT-CANDIDATE.json'
receipt_sha = 'f4b7ee5063576d9c486f1833eb47bd9d7b5c292be64d6d1f13e66131903a4f3c'
report_sha = '729dac5e0ec64be9f97df839cbaa644c19682c166b573923c7b2b2efe21b04cc'
cpu_sha = 'cee8e9e4d671fd998ad1474af98f7f05b353db0d7867dfe9c7c3e96e3fd21198'
output = ROOT/'evidence/first5-release3359-cached-20261009'
original_atomic_publish = publisher.atomic_publish
def checked_atomic_publish(target, raw):
    cache.verify_unchanged()
    return original_atomic_publish(target, raw)
publisher.atomic_publish = checked_atomic_publish
plan = publisher.prepare(receipt, receipt_sha, report_sha, cpu_sha, output, gate)
cache.verify_unchanged()
print(json.dumps({'phase':'prepare','bytes_read':cache.bytes_read,'cache_hits':cache.cache_hits}), flush=True)
plan_path = output/'release-plan.json'
result = publisher.publish(plan_path, cache.digest(plan_path), receipt_sha, gate)
cache.verify_unchanged()
print(json.dumps({'phase':'publish','result':result,'bytes_read':cache.bytes_read,'cache_hits':cache.cache_hits}), flush=True)
