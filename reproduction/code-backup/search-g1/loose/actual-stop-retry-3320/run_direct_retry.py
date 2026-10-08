"""Direct runtime retry explicitly requested by the latest human instruction.

Historical CPU/review artifacts are preserved and labelled historical. No new
review/test admission is asserted. Intrinsic sampling/finite/training safeguards
and the user's opportunity/write accounting remain unchanged.
"""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'numeric-repair-native-v6'
REPO = Path('/Users/shaoshitong/Documents/Search-G1/repo')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transport = load_module('saved_user_lan_transport', ORIGINAL / 'user_authorized_lan_transport.py')
native = load_module('saved_native_retry_engine', ORIGINAL / 'submit_formal_inline_native.py')
# Slurm reports a requested single-node range while the saved job is PENDING.
# Preserve the strict allocated-node check for RUNNING/terminal records.
native.OBSERVE_CODE = native.OBSERVE_CODE.replace(
    "row['NumNodes']=='1'",
    "(row['NumNodes']=='1' or (row['JobState']=='PENDING' and row['NumNodes']=='1-1'))")
sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control
control.route = transport.route

FAILED_READONLY = native.ARTIFACT_INPUT_CODE + r'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);b=Path('/mnt/data/shaoshitong/g1f-eaf41587')
run=json.loads(bounded_artifact_bytes(b/'logs/run-manifest.json',256*1024))
stage=json.loads(bounded_artifact_bytes(b/'logs/pipeline-status.json',256*1024))
cleanup=json.loads(bounded_artifact_bytes(b/'logs/cleanup-status.json',256*1024))
ledger=json.loads(bounded_artifact_bytes(b/'logs/process-ledger.json',4*1024*1024))
assert run['run_id']==stage['run_id']==cleanup['run_id']==ledger['run_id']=='88eb6ead1e244594ad4d2ebac36a7a59'
assert stage['stage']=='FAILED' and cleanup['cleaned'] is True
jobs,births=validated_ledger(ledger);assert all(x['state']=='STOPPED' for x in jobs.values())
for identity in births:
 proc=Path('/proc')/str(identity['pid'])
 try:
  row=bounded_proc_bytes(proc/'stat',4096).decode().rsplit(')',1)[1].split()
  assert (int(row[19]),int(row[3]),proc.stat().st_uid)!=(identity['start_ticks'],identity['sid'],identity['uid'])
 except (FileNotFoundError,ProcessLookupError):pass
active=[];marker=('SEARCH_G1_BASE='+str(b)).encode()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid==os.getuid() and marker in bounded_proc_bytes(proc/'environ',1024*1024).split(b'\0'):active.append(int(proc.name))
 except (FileNotFoundError,ProcessLookupError):pass
assert not active
queue=subprocess.run(['squeue','-h','-n','search-g1-formal-inline-eaf41587','-o','%i|%T|%j'],capture_output=True,text=True,timeout=15)
assert queue.returncode==0 and not queue.stdout.strip()
for name,expected in p['failed_source_sha256'].items():
 assert hashlib.sha256(bounded_artifact_bytes(b/'repo'/name,16*1024*1024)).hexdigest()==expected
print(json.dumps(dict(base=str(b),run_id=run['run_id'],stage=stage['stage'],cleaned=True,checked_unix=time.time(),recorded_birth_count=len(births),namespace_empty=True,exact_job_name_queue_empty=True,original_failed_source_unchanged=True,native_terminal_success=False)))
'''


def prepare_inputs():
    inputs = json.loads((HERE / 'direct-runtime-inputs.json').read_text())
    files = {name: (REPO / name).read_bytes() for name in inputs['source_sha256']}
    for name, content in files.items():
        assert hashlib.sha256(content).hexdigest() == inputs['source_sha256'][name]
    method = (HERE / 'formal-method.json').read_bytes()
    assert hashlib.sha256(method).hexdigest() == inputs['method_sha256']
    assert inputs['actual_target_cpu_tests'] == 0
    assert inputs['target_cpu_proof_applies_to_current_source'] is False
    assert inputs['pre_execution_review_applies_to_current_source'] is False
    return inputs, files, method


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('--execute', '--observe', '--cleanup-source'):
        raise ValueError('Use exactly one saved-attempt mode')
    if sys.argv[1] != '--execute':
        result = native.observe(prep=HERE, cleanup_source=sys.argv[1] == '--cleanup-source')
    else:
        assert not (HERE / native.NAMES['intent']).exists(), 'Saved intent exists: observe only'
        inputs, files, method = prepare_inputs()
        failed = json.loads((ORIGINAL / native.NAMES['intent']).read_text())
        def reconcile():
            return native.remote('h100-8', FAILED_READONLY,
                dict(failed_source_sha256=failed['source_sha256']), timeout=60,
                producer_sha256=inputs['cpu_producer_sha256'])
        snapshot = reconcile()
        native.save(HERE / 'failed3320-pre-retry-readonly.json', snapshot)
        inputs['failed3320_pre_retry'] = snapshot
        native.load_inputs = lambda args, repo=REPO: (inputs, files, method)
        original_remote = native.remote
        def remote(host, code, payload, **kwargs):
            if code == native.SUBMIT_CODE:
                snapshot = original_remote('h100-8', FAILED_READONLY,
                    dict(failed_source_sha256=failed['source_sha256']), timeout=60,
                    producer_sha256=inputs['cpu_producer_sha256'])
                native.save(HERE / 'failed3320-pre-sbatch-readonly.json', snapshot)
            return original_remote(host, code, payload, **kwargs)
        native.remote = remote
        result = native.submit(None, prep=HERE, repo=REPO)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
