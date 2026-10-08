"""Prepare and run the next fresh retry under the existing direct-run authority.

The old native3324 failure and missing cleanup-status remain historical facts.
New source has no new CPU/review admission. The only runtime repair changes GPU
occupancy from a fixed98% rejection to monitoring, with bounded context-release
handling owned by the committed training source. No old weights are resumed.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'numeric-repair-native-v6'
PREVIOUS = HERE.parent / 'actual-stop-retry-3320'
REPO = Path('/Users/shaoshitong/Documents/Search-G1/repo')
SOURCE_COMMIT = 'ddf71bf874e943d9ad1c535517f2a4342e4baecf'
FAILED_INTENT_SHA = 'd0bcd2b2b57bcbee3af43f4bf812bcfe3f6915fb820c3b2b64d0849d4d507c48'
FAILURE_CAPTURE = PREVIOUS / 'goal-native3324-live-progress-1791370004298358000.json'
FAILURE_CAPTURE_SHA = 'b20beb095bcbc06b9f08a1e771bfede2e7807deda47279cbf5d3aefb7433f922'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transport = load_module('saved_user_lan_transport', ORIGINAL / 'user_authorized_lan_transport.py')
native = load_module('saved_native_retry_engine', ORIGINAL / 'submit_formal_inline_native.py')
# Same prior launcher adaptation: PENDING can report the requested range1-1.
# RUNNING and terminal allocation identity continue to require exactly one node.
native.OBSERVE_CODE = native.OBSERVE_CODE.replace(
    "row['NumNodes']=='1'",
    "(row['NumNodes']=='1' or (row['JobState']=='PENDING' and row['NumNodes']=='1-1'))")
sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control
control.route = transport.route

FAILED_READONLY = native.ARTIFACT_INPUT_CODE + r'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);b=Path('/mnt/data/shaoshitong/g1f-ef7a5a0c')
assert b.resolve()==b and not b.is_symlink()
values={};artifact_sha={}
for name,limit in (('run-manifest.json',256*1024),('pipeline-status.json',256*1024),('process-ledger.json',16*1024*1024)):
 raw=bounded_artifact_bytes(b/'logs'/name,limit)
 artifact_sha[name]=hashlib.sha256(raw).hexdigest()
 assert artifact_sha[name]==p['failed_artifact_sha256'][name], 'Immutable failed3324 artifact changed'
 values[name]=json.loads(raw)
run=values['run-manifest.json'];stage=values['pipeline-status.json'];ledger=values['process-ledger.json']
assert run['run_id']==stage['run_id']==ledger['run_id']=='c0f329fb6282460a9fea8811aa8f47f2'
assert run['source_commit']=='19fdfc32cc1c62d9723a02bf658ad8e4c2b332aa'
assert run['source_sha256']==p['failed_source_sha256'] and len(run['source_sha256'])==373
assert stage['stage'] in ('FAILED','OWNED_CLEANUP_FAILED')
cleanup_path=b/'logs/cleanup-status.json'
assert not cleanup_path.exists() and not cleanup_path.is_symlink(), 'Missing original cleanup-status must stay missing'
jobs,occurrences=validated_ledger(ledger)
assert len(jobs)==9 and all(x['state']=='STOPPED' for x in jobs.values())
births={validated_birth(identity):identity for identity in occurrences}
assert len(births)==508
classification={'absent':0,'changed':0}
for birth,identity in births.items():
 proc=Path('/proc')/str(identity['pid'])
 try:
  row=bounded_proc_bytes(proc/'stat',4096).decode().rsplit(')',1)[1].split()
  actual=(identity['pid'],int(row[19]),int(row[3]),proc.stat().st_uid)
  assert actual!=birth, 'Recorded3324 process birth remains present'
  classification['changed']+=1
 except (FileNotFoundError,ProcessLookupError):classification['absent']+=1
active=[];marker=('SEARCH_G1_BASE='+str(b)).encode();prefix=str(b).encode()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=os.getuid():continue
  args=bounded_proc_bytes(proc/'cmdline',64*1024).split(b'\0')
  environment=bounded_proc_bytes(proc/'environ',1024*1024).split(b'\0')
  if marker in environment or any(arg==prefix or arg.startswith(prefix+b'/') for arg in args):active.append(int(proc.name))
 except (FileNotFoundError,ProcessLookupError):pass
assert not active, 'Failed3324 training namespace still present'
queue=subprocess.run(['squeue','-h','-n','search-g1-formal-inline-ef7a5a0c','-o','%i|%T|%j'],capture_output=True,text=True,timeout=15)
assert queue.returncode==0 and not queue.stdout.strip()
for name,expected in p['failed_source_sha256'].items():
 assert hashlib.sha256(bounded_artifact_bytes(b/'repo'/name,2*1024*1024)).hexdigest()==expected
historical=p['historical_native_failure']
expected={'JobId':'3324','JobName':'search-g1-formal-inline-ef7a5a0c','JobState':'FAILED','ExitCode':'1:0','Restarts':'0','NodeList':'h100-8','NumNodes':'1','StartTime':'2026-10-07T15:01:24','EndTime':'2026-10-07T18:44:07','TimeLimit':'8-00:00:00'}
assert historical['returncode']==0 and historical['identity_matches_saved_job'] is True and historical['state_known'] is True
assert all(historical['fields'][key]==value for key,value in expected.items())
record=subprocess.run(['scontrol','show','job','-o','3324'],capture_output=True,text=True,timeout=15)
actual_native=None
if record.returncode==0:
 assert len(record.stdout.strip().splitlines())==1
 actual_native={}
 for token in record.stdout.split():
  key,value=token.split('=',1);assert key and key not in actual_native;actual_native[key]=value
 assert all(actual_native[key]==value for key,value in expected.items())
 assert actual_native['UserId']=='root(0)' and actual_native['Command']==str(b/'repo/reproduction/common8/launch_native.sh')
 assert actual_native['WorkDir']==str(b/'repo') and actual_native['BatchHost']=='h100-8'
else:
 assert 'invalid job id' in (record.stdout+record.stderr).lower(), 'Unknown current scheduler read failure'
# The external v2 recorder has its own namespace. Observe its exact birth only;
# never stop, signal, relaunch, write to or exempt a training namespace process.
watcher=Path('/proc/3902673');watcher_current='absent'
try:
 row=bounded_proc_bytes(watcher/'stat',4096).decode().rsplit(')',1)[1].split()
 same=(int(row[19]),int(row[3]),watcher.stat().st_uid)==(2230041223,3902673,0)
 watcher_current='same_birth' if same else 'pid_reused'
 if same:
  assert marker not in bounded_proc_bytes(watcher/'environ',1024*1024).split(b'\0')
except (FileNotFoundError,ProcessLookupError):pass
print(json.dumps(dict(role='failed3324_separate_actual_nonduplication',base=str(b),run_id=run['run_id'],job_id='3324',stage=stage['stage'],cleanup_status_missing=True,original_cleanup_success=False,owned_job_count=len(jobs),all_owned_jobs_stopped=True,recorded_birth_occurrences=len(occurrences),distinct_recorded_birth_count=len(births),current_birth_classification_counts=classification,all_recorded_births_absent_or_changed=True,namespace_empty=True,exact_job_name_queue_empty=True,original_failed_source_unchanged=True,artifact_sha256=artifact_sha,historical_native_failure=historical,current_native_fields=actual_native,current_native_record_available=actual_native is not None,current_native_returncode=record.returncode,current_native_stdout_sha256=hashlib.sha256(record.stdout.encode()).hexdigest(),external_recorder_pid=3902673,external_recorder_current_birth=watcher_current,external_recorder_action='PRESERVED_NO_SIGNAL_NO_WRITE',checked_unix=time.time(),native_terminal_success=False,scope='Fresh read-only resource reconciliation of failed3324; missing cleanup remains missing; no old state or success admission; no GPU check')))
'''


def sealed_json(path, expected):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    return json.loads(raw)


def prepare_inputs():
    state = json.loads((HERE / 'preparation-state.json').read_text())
    assert state['inputs_ready'] is True and state['source_commit'] == SOURCE_COMMIT
    inputs = json.loads((HERE / 'direct-runtime-inputs.json').read_text())
    assert inputs['source_commit'] == SOURCE_COMMIT and inputs['source_count'] == 373
    files = {name: (REPO / name).read_bytes() for name in inputs['source_sha256']}
    assert len(files) == 373
    for name, content in files.items():
        assert hashlib.sha256(content).hexdigest() == inputs['source_sha256'][name]
    method = (HERE / 'formal-method.json').read_bytes()
    assert hashlib.sha256(method).hexdigest() == inputs['method_sha256']
    assert inputs['actual_target_cpu_tests'] == 0 and inputs['actual_cpu_modules'] == []
    assert inputs['target_cpu_proof_applies_to_current_source'] is False
    assert inputs['pre_execution_review_applies_to_current_source'] is False
    assert inputs['actual_production_updates'] == 0 and inputs['fresh_states'] is True
    return inputs, files, method


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('--execute', '--observe', '--cleanup-source'):
        raise ValueError('Use exactly one saved-attempt mode')
    if sys.argv[1] != '--execute':
        result = native.observe(prep=HERE, cleanup_source=sys.argv[1] == '--cleanup-source')
    else:
        assert not (HERE / native.NAMES['intent']).exists(), 'Saved intent exists: observe only'
        inputs, files, method = prepare_inputs()
        failed = sealed_json(PREVIOUS / native.NAMES['intent'], FAILED_INTENT_SHA)
        baseline = sealed_json(FAILURE_CAPTURE, FAILURE_CAPTURE_SHA)
        assert failed['base'] == baseline['base'] == '/mnt/data/shaoshitong/g1f-ef7a5a0c'
        assert baseline['native_job_id'] == '3324' and baseline['run_id'] == 'c0f329fb6282460a9fea8811aa8f47f2'
        payload = dict(failed_source_sha256=failed['source_sha256'],
            failed_artifact_sha256={name: baseline['files']['logs/' + name]['captured_sha256']
                for name in ('run-manifest.json', 'pipeline-status.json', 'process-ledger.json')},
            historical_native_failure=baseline['native'])
        def reconcile():
            return native.remote('h100-8', FAILED_READONLY, payload, timeout=60,
                producer_sha256=inputs['cpu_producer_sha256'])
        snapshot = reconcile()
        native.save(HERE / 'failed3324-pre-retry-readonly.json', snapshot)
        inputs['failed3324_pre_retry'] = snapshot
        native.load_inputs = lambda args, repo=REPO: (inputs, files, method)
        original_remote = native.remote
        def remote(host, code, supplied, **kwargs):
            if code == native.SUBMIT_CODE:
                snapshot = original_remote('h100-8', FAILED_READONLY, payload, timeout=60,
                    producer_sha256=inputs['cpu_producer_sha256'])
                native.save(HERE / 'failed3324-pre-sbatch-readonly.json', snapshot)
            return original_remote(host, code, supplied, **kwargs)
        native.remote = remote
        result = native.submit(None, prep=HERE, repo=REPO)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
