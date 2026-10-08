"""Read-only target integration of the exact native submission binding view."""
import copy
import datetime
import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
sys.path.insert(0, str(ROOT / 'tools/action-prefix-native-2262942-control-v2'))
sys.path.insert(0, str(ROOT / 'tools/action-prefix-native-2262942-v2'))
from deployment_contract_v1 import (READY, BINDING, INTENT, RECEIPT, HELPERS,
    CONTROLS, read_bound, require, sha, validate_ready, write_once)
from action_prefix_native_control_v1 import validate_binding

require(socket.gethostname().split('.')[0] == 'h100-8', 'Wrong node')
out = ROOT / 'logs/ACTION-PREFIX-NATIVE-FINALBINDING-INTEGRATION-v2.json'
require(not out.exists(), 'Existing integration proof; inspect rather than repeat')
require(not any(p.exists() for p in (BINDING, INTENT, RECEIPT)), 'Submission claim already exists')
ready_sha = '04d04fe286a2575d4126cd13cadf0974e675bd66959f16e0230590e64e0d5d7c'
value = read_bound(dict(path=str(READY), sha256=ready_sha))
validate_ready(value)
binding = copy.deepcopy(value)
added = []
for record in (value['candidate_record'], value['physical_record'], value['deployment_review']):
    same = [r for r in binding['evidence'] if r['path'] == record['path']]
    require(not same or same == [record], 'Conflicting bound acceptance proof')
    if not same:
        binding['evidence'].append(record)
        added.append(record)
require(validate_binding(binding, HELPERS, sha) is True, 'Actual final binding rejected')
queue = subprocess.run(['squeue', '-h', '-w', 'h100-8', '-o', '%i %j %T'], text=True, capture_output=True)
compute = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                          '--format=csv,noheader'], text=True, capture_output=True)
require(queue.returncode == compute.returncode == 0, 'Unreadable occupancy')
result = dict(status='PASS_ACTUAL_TARGET_FINAL_BINDING_INTEGRATION_ONLY',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), node=socket.gethostname(),
    ready_sha256=ready_sha, code_commit=value['code_commit'], candidate_record=value['candidate_record'],
    physical_record=value['physical_record'], deployment_review=value['deployment_review'],
    files=len(binding['files']), evidence=len(binding['evidence']), added_evidence=added,
    prospective_binding_sha256=hashlib.sha256((json.dumps(binding, indent=2, allow_nan=False)+'\n').encode()).hexdigest(),
    validator=dict(path=str(HELPERS/'action_prefix_native_control_v1.py'),
                   sha256=sha(HELPERS/'action_prefix_native_control_v1.py')),
    submit_helper=dict(path=str(CONTROLS/'submit_once_v1.py'), sha256=sha(CONTROLS/'submit_once_v1.py')),
    queue=dict(returncode=queue.returncode, stdout=queue.stdout, stderr=queue.stderr),
    gpu_compute=dict(returncode=compute.returncode, stdout=compute.stdout, stderr=compute.stderr),
    scope='Real target raw paths, resolved aliases and SHA in exact future binding; no GPU native/production acceptance',
    all_evidence_sha_actually_checked=True, six_required_helpers_sha_actually_checked=True,
    config_driver_sha_actually_checked=True, fresh_full_large_asset_scan=False,
    binding_written=False, intent_written=False, submitted=False, gpu_used=False,
    production_thresholds=None, production_probability_tolerance=None)
write_once(out, result)
print(json.dumps(dict(path=str(out), sha256=sha(out), bytes=out.stat().st_size, result=result)), flush=True)
