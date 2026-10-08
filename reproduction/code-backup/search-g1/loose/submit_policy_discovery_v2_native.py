"""Independent discovery retry after an authenticated zero-update failed job.

This external draft is outside runtime. It never changes the original 3274
files. Explicit new commit, fresh actual target CPU proof and prior scoped
cleanup proof are required. Missing/unknown evidence rejects before mutation.
Submission, source handle and observations have separate v2 claim paths.
"""
import argparse
import base64
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

PREP = Path('/Users/shaoshitong/Documents/Search-G1/preparation')
REPO = PREP.parent / 'repo'
sys.path.insert(0, str(PREP))
sys.path.insert(0, str(REPO))
import submit_policy_confirmation_native as common

NAMES = dict(intent='policy-discovery-v2-native-submit-intent.json',
             package='policy-discovery-v2-native-package-proof.json',
             handle='policy-discovery-v2-retrieval-export-handle.json',
             launch='policy-discovery-v2-native-launch.json')
# Keep the original failed identity independent of a later confirmation draft's
# explicitly updated contract. The original 3274 source must never move.
OLD_COMMIT = '61f6e1da532e08553410666b430d1933a0e4a49a'
REQUIRED_CACHE_TEST = 'reproduction.common8.test_temporary_cache'


def check_commit(commit):
    common.require(isinstance(commit, str) and len(commit) == 40 and
                   all(c in '0123456789abcdef' for c in commit) and commit != OLD_COMMIT,
                   'Explicit new full runtime commit is required; old132 proof cannot admit retry')


def original_inputs(prep=PREP):
    old = common.strict_json((prep / 'policy-discovery-native-launch.json').read_bytes())
    common.require(old['job_id'] == '3274' and old['source_commit'] == OLD_COMMIT,
                   'Wrong original immutable failed handle')
    proof = common.strict_json((prep / 'calibration-phase-target-cpu-v3-result.json').read_bytes())
    handle = common.strict_json((prep / 'policy-discovery-retrieval-export-handle.json').read_bytes())
    source_cleanup = common.strict_json((prep / 'policy-discovery-retrieval-source-cleanup.json').read_bytes())
    common.require(source_cleanup['cleaned'] is True and source_cleanup['handle'] == handle,
                   'Original tiny source cleanup has not been verified')
    return old, proof


def cleanup_schema(cleanup, old):
    common.require(cleanup['base'] == old['base'] and str(cleanup['job_id']) == old['job_id'] and
                   isinstance(cleanup['run_id'], str) and len(cleanup['run_id']) == 32 and
                   all(c in '0123456789abcdef' for c in cleanup['run_id']) and
                   type(cleanup['actual_completed_updates']) is int and cleanup['actual_completed_updates'] == 0 and
                   all(cleanup[name] is True for name in ('owned_namespace_empty', 'owned_shm_root_absent', 'cleaned',
                       'owned_temporary_cleanup_complete', 'failed_training_run_remains_failed')) and
                   cleanup['production_admission'] is False, 'Not the actual scoped zero-update failed cleanup')
    return cleanup


def validate_cpu(proof, old_proof):
    """Require new actual bytes and suite, never treating the old132 as new."""
    summary = common.strict_json(proof['stdout'])
    common.require(type(proof['returncode']) is int and proof['returncode'] == 0 and
                   type(proof['shell_syntax_returncode']) is int and proof['shell_syntax_returncode'] == 0 and
                   summary['successful'] is True and summary['cuda_available'] is False and
                   summary['skipped'] == [] and type(summary['tests_run']) is int and summary['tests_run'] > 0 and
                   type(summary['failures']) is int and summary['failures'] == 0 and
                   type(summary['errors']) is int and summary['errors'] == 0, 'Fresh target CPU regressions did not pass')
    common.require(proof['source_sha256'] != old_proof['source_sha256'] and
                   proof['checked_unix'] > old_proof['checked_unix'] and
                   isinstance(proof['modules'], list) and len(set(proof['modules'])) == len(proof['modules']) and
                   set(old_proof['modules']) <= set(proof['modules']) and REQUIRED_CACHE_TEST in proof['modules'],
                   'New cache implementation and full actual target suite are required')
    common.require(str(proof['base']).startswith('/dataset/shaoshitong/') or
                   str(proof['base']).startswith('/mnt/data/shaoshitong/'), 'Not a target-cluster CPU proof')
    return proof['source_sha256']


def committed_sources(commit, mapping, repo=REPO):
    """Exact complete runtime tree from explicit commit, not mutable HEAD."""
    check_commit(commit)
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], cwd=repo, text=True).splitlines()
    expected = {name for name in names if
        (Path(name).suffix in ('.py', '.sh', '.yaml', '.yml', '.json') or
         name in ('verl/version/version', 'verl/single_controller/version/version')) and
        not {'.git', '__pycache__', 'evidence'}.intersection(Path(name).parts)}
    common.require(expected == set(mapping), 'Actual target proof is not the full committed runtime file map')
    files = {}
    for name in sorted(expected):
        content = subprocess.check_output(['git', 'cat-file', 'blob', commit + ':' + name], cwd=repo)
        common.require(common.sha(content) == mapping[name], 'Source changed after actual target CPU proof: ' + name)
        files[name] = content
    return files


def load_inputs(commit, cpu_path, cleanup_path, prep=PREP, repo=REPO):
    check_commit(commit)
    old, old_proof = original_inputs(prep)
    cpu_bytes = Path(cpu_path).read_bytes(); proof = common.strict_json(cpu_bytes)
    cleanup_bytes = Path(cleanup_path).read_bytes(); cleanup = common.strict_json(cleanup_bytes)
    cleanup_schema(cleanup, old)
    history_bytes = (prep / 'policy-discovery-native-observation.json').read_bytes()
    observed = common.strict_json(history_bytes)
    history = {key: observed[key] for key in ('base', 'parent_base', 'job_id', 'job_stdout',
               'job_returncode', 'job_stderr', 'source_hashes_match', 'actual_updates',
               'active_owned_pids', 'actual_namespace_processes', 'run-manifest.json')}
    previous_terminal_schema(history, old, cleanup['run_id'])
    files = committed_sources(commit, validate_cpu(proof, old_proof), repo)
    return dict(old_launch=old, previous_cleanup=cleanup,
                previous_cleanup_proof_sha256=common.sha(cleanup_bytes),
                previous_terminal_observation=history, previous_terminal_observation_sha256=common.sha(history_bytes),
                target_cpu_proof_sha256=common.sha(cpu_bytes), source_commit=commit,
                source_sha256=proof['source_sha256'], actual_target_cpu_tests=common.strict_json(proof['stdout'])['tests_run']), files


def previous_terminal_schema(history, old, run_id):
    """Bind actual collected terminal observation before Slurm purged the job."""
    from reproduction.common8.native_allocation import fields
    job = fields(history['job_stdout'])
    common.require(history['base'] == old['base'] and history['parent_base'] == old['parent_base'] and
                   history['job_id'] == old['job_id'] and type(history['job_returncode']) is int and history['job_returncode'] == 0 and
                   history['source_hashes_match'] is True and history['actual_updates'] == [] and
                   history['active_owned_pids'] == [] and
                   not [row for row in history['actual_namespace_processes'] if row['state'] != 'Z'] and
                   history['run-manifest.json']['run_id'] == run_id and
                   history['run-manifest.json']['source_commit'] == OLD_COMMIT and
                   job['JobId'] == old['job_id'] and job['JobState'] == 'FAILED' and
                   job['NodeList'] == 'h100-8' and
                   job['Command'] == old['base'] + '/repo/reproduction/common8/launch_calibration_native.sh' and
                   job['WorkDir'] == old['base'] + '/repo' and
                   job['StdOut'] == old['base'] + '/logs/native-discovery-' + old['job_id'] + '.log',
                   'Missing exact original collected FAILED native/run/command observation')
    return job


def native_terminal_route(old, history, run_id, command):
    """Distinguish a true current terminal observation from an explicit purge."""
    from reproduction.common8.native_allocation import fields
    previous_terminal_schema(history, old, run_id)
    if command.returncode == 0:
        job = fields(command.stdout)
        common.require(job['JobId'] == old['job_id'] and job['NodeList'] == 'h100-8' and
                       job['JobState'] in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL') and
                       job['Command'] == old['base'] + '/repo/reproduction/common8/launch_calibration_native.sh',
                       'Original native failed/terminal state not independently observed')
        return 'actual_current_terminal_native'
    common.require('Invalid job id' in command.stderr + command.stdout, 'Native observation failed; not a proven purge')
    queue = subprocess.run(['squeue', '-h', '-j', old['job_id'], '-o', '%i|%T|%j|%Z|%o'],
                           capture_output=True, text=True, timeout=15)
    common.require((queue.returncode == 0 and not queue.stdout.strip()) or
                   (queue.returncode != 0 and 'Invalid job id' in queue.stderr + queue.stdout and not queue.stdout.strip()),
                   'Purged job absence not independently confirmed')
    return 'exact_historical_failed_command_and_actual_slurm_purged'


def previous_readonly(payload):
    """Actual original job/source/PID/namespace/temporary-root reread only."""
    old, cleanup = payload['old_launch'], payload['previous_cleanup']
    cleanup_schema(cleanup, old)
    b = Path(old['base']); sys.path.insert(0, str(b / 'repo'))
    from reproduction.common8.native_allocation import fields
    from reproduction.a800.owned_processes import same, identity
    previous_terminal_schema(payload['previous_terminal_observation'], old, cleanup['run_id'])
    run = common.strict_json((b / 'logs/run-manifest.json').read_bytes())
    common.require(run['run_id'] == cleanup['run_id'] and run['policy_calibration_slurm_job_id'] == old['job_id'] and
                   run['source_commit'] == OLD_COMMIT and type(run['completed_updates']) is int and
                   run['completed_updates'] == 0, 'Original actual zero-update run identity differs')
    for name, digest in old['source_sha256'].items():
        common.require(common.sha((b / 'repo' / name).read_bytes()) == digest, 'Original failed source changed')
    command = subprocess.run(['scontrol', 'show', 'job', '-o', old['job_id']], capture_output=True, text=True, timeout=15)
    route = native_terminal_route(old, payload['previous_terminal_observation'], cleanup['run_id'], command)
    status = common.strict_json((b / 'logs/pipeline-status.json').read_bytes())
    common.require(status['run_id'] == run['run_id'] and status['stage'] in ('FAILED', 'OWNED_CLEANUP_FAILED'),
                   'Do not convert failed run to success')
    updates = b / 'logs/actual-updates.jsonl'
    common.require(not updates.exists() or not updates.read_bytes().strip(), 'Original completed updates are not zero')
    common.require(not list((b / 'logs').glob('optimizer-rank-*.jsonl')), 'Original optimizer evidence exists; do not pretend fresh0')
    checkpoints = Path(run['checkpoint_dir'])
    common.require(not checkpoints.exists() or not any(checkpoints.iterdir()), 'Original checkpoint exists')
    ledger = common.strict_json((b / 'logs/process-ledger.json').read_bytes())
    common.require(ledger['run_id'] == run['run_id'], 'Original owned ledger run mismatch')
    for child in ledger['jobs'].values():
        common.require(child['state'] == 'STOPPED', 'Original owned child still active')
        for process in child['ledger'].values():
            common.require(not same(process) or identity(process['pid'])['state'] == 'Z', 'Original actual owned process remains live')
    marker = ('SEARCH_G1_BASE=' + str(b)).encode()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            if proc.stat().st_uid != os.getuid() or marker not in (proc / 'environ').read_bytes().split(b'\0'): continue
            common.require((proc / 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z', 'Original actual namespace remains active')
        except (FileNotFoundError, ProcessLookupError): pass
    root = Path('/dev/shm') / ('search-g1-common8-' + run['run_id'] + '-' + old['job_id'])
    for name, target in (('retrieval-staging-claim.json', root), ('corpus-cache-claim.json', root / 'datasets-cache')):
        claim = common.strict_json((b / 'logs' / name).read_bytes())
        common.require(claim['run_id'] == run['run_id'] and claim['slurm_job_id'] == old['job_id'] and
                       claim['uid'] == os.getuid() and claim['phase'] == 'CLEANED' and Path(claim['root']) == target,
                       'Original actual owned cleanup claim differs')
    common.require(not root.exists() and not root.is_symlink(), 'Original temporary root still exists')
    return dict(role='previous_failed_discovery_readonly_verification', run_id=run['run_id'], job_id=old['job_id'],
                original_failed_run_preserved=True, actual_completed_updates=0,
                terminal_observation_route=route,
                owned_namespace_empty=True, owned_shm_root_absent=True,
                production_admission=False, healthy_checkpoint_accepted=False)


def previous_code():
    # A tiny namespace object allows the same reviewed stdlib validators to run
    # remotely without copying this external orchestrator into the old source.
    helpers = '\n'.join(inspect.getsource(fn) for fn in (common.canonical, common.sha, common.require, common.strict_json))
    return ('import hashlib,json,os,subprocess,sys\nfrom pathlib import Path\nfrom types import SimpleNamespace\n' +
            helpers + '\ncommon=SimpleNamespace(canonical=canonical,sha=sha,require=require,strict_json=strict_json)\n' +
            'OLD_COMMIT=' + repr(OLD_COMMIT) + '\n' + inspect.getsource(cleanup_schema) + '\n' +
            inspect.getsource(previous_terminal_schema) + '\n' +
            inspect.getsource(native_terminal_route) + '\n' +
            inspect.getsource(previous_readonly) + '\np=json.load(sys.stdin);print(json.dumps(previous_readonly(p)))\n')


PREPARE_CODE = r'''
import base64,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);old=Path(p['old_launch']['parent_base'])
for directory in (p['parent_base'],p['base']):
 b=Path(directory);assert not b.exists();b.mkdir(parents=True,exist_ok=False)
 for name in ('repo','data','preparation','logs','tmp','ray','artifacts','cache/triton','cache/inductor','cache/hf'):(b/name).mkdir(parents=True,exist_ok=True)
 for name in ('env','bootstrap-assets','bootstrap-tools','assets','retriever-gpu-lib'):
  source=old/name;assert source.is_dir();(b/name).symlink_to(source.resolve(),target_is_directory=True)
 for name,data in p['files'].items():
  relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
  target=b/'repo'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(data))
  assert hashlib.sha256(target.read_bytes()).hexdigest()==p['source_sha256'][name]
 for name in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):
  source=old/'data'/name;target=b/'data'/name;assert not source.is_symlink();shutil.copy2(source,target)
  assert hashlib.sha256(source.read_bytes()).hexdigest()==hashlib.sha256(target.read_bytes()).hexdigest()
 (b/'preparation/official-model-manifest-original.json').write_bytes((old/'preparation/official-model-manifest-original.json').read_bytes())
 with (b/'preparation/source-package-intent.json').open('x') as f:json.dump({k:v for k,v in p.items() if k!='files'},f,indent=2);f.flush();os.fsync(f.fileno())
b=Path(p['base']);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(b/'repo'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
code='import verl,verl.single_controller;from verl.workers.fsdp_workers import ActorRolloutRefWorker;from reproduction.common8 import prepare_calibration_run,calibration_runtime,temporary_cache,adopt_initial_probes,production_safety_admission;from reproduction.common8.verify_data import verify;import pathlib,sys,json,inspect,hashlib;from types import SimpleNamespace;from ray._private.node import Node;s=inspect.getsource(Node._prepare_socket_file);good="/"+"a"*106;dummy=SimpleNamespace(_sockets_dir="/unused",_make_inc_temp=lambda **kwargs:good);assert sys.platform=="linux" and Node._prepare_socket_file(dummy,None,"plasma_store")==good\ntry:\n dummy._make_inc_temp=lambda **kwargs:"/"+"a"*107;Node._prepare_socket_file(dummy,None,"plasma_store")\nexcept OSError:pass\nelse:raise AssertionError("Actual installed Ray accepted108-byteAF_UNIX")\nprint(json.dumps(dict(data=verify(pathlib.Path(sys.argv[1])),installed_ray_socket_check_sha256=hashlib.sha256(s.encode()).hexdigest(),installed_limit_tested_bytes=[107,108],scope="Actual installed AF_UNIX method CPU boundary check; no Ray/GPU startup")))'
r=subprocess.run([str(b/'env/bin/python'),'-c',code,str(b/'data')],cwd=b/'repo',env=env,capture_output=True,text=True,timeout=75)
proof=dict(base=str(b),parent_base=p['parent_base'],checked_unix=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,source_commit=p['source_commit'],source_sha256=p['source_sha256'],ray_socket_budgets=p['ray_socket_budgets'],actual_production_updates=0,production_admission=False,healthy_checkpoint_accepted=False,scope='CPU imports+originalsixbytes+actualinstalledRayboundary; not GPU admission or new training')
with (b/'preparation/package-cpu-proof.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
assert r.returncode==0,r.stderr
syntax=subprocess.run(['bash','-n',str(b/'repo/reproduction/common8/launch_calibration_native.sh'),str(b/'repo/reproduction/common8/train.sh')],capture_output=True,text=True);assert syntax.returncode==0,syntax.stderr
print(json.dumps(proof))
'''

SUBMIT_CODE = r'''
import hashlib,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['base']);parent=Path(p['parent_base'])
for namespace in (b,parent):
 assert not (namespace/'logs/run-manifest.json').exists()
 for name,digest in p['source_sha256'].items():assert hashlib.sha256((namespace/'repo'/name).read_bytes()).hexdigest()==digest
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for name,size in p['handle']['files'].items():
 with opener.open(urllib.request.Request('http://10.33.0.13:19874/'+name,method='HEAD'),timeout=10) as response:assert response.status==200 and int(response.headers['Content-Length'])==size
args=['sbatch','--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h100:8','--mem=512G','--exclusive','--time=08:00:00','--job-name=search-g1-policy-discovery-v2','--chdir='+str(b/'repo'),'--output='+str(b/'logs/native-discovery-v2-%j.log'),'--export=ALL,SEARCH_G1_BASE='+str(b)+',SEARCH_G1_PRODUCTION_BASE='+str(parent)+',SEARCH_G1_CALIBRATION_ROLE=policy_safety_calibration_discovery,SEARCH_G1_SOURCE_COMMIT='+p['source_commit'],str(b/'repo/reproduction/common8/launch_calibration_native.sh')]
with (b/'preparation/native-submit-claim.json').open('x') as f:json.dump(dict(argv=args,created_unix=time.time(),base=str(b),parent_base=str(parent),target_cpu_proof_sha256=p['target_cpu_proof_sha256'],scope='Independent retry after stoppedfailed3274; production0'),f);f.flush();os.fsync(f.fileno())
with (b/'preparation/retrieval-source-handle.json').open('x') as f:json.dump(p['handle'],f,indent=2);f.flush();os.fsync(f.fileno())
r=subprocess.run(args,capture_output=True,text=True)
proof=dict(base=str(b),parent_base=str(parent),source_commit=p['source_commit'],source_sha256=p['source_sha256'],target_cpu_proof_sha256=p['target_cpu_proof_sha256'],argv=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,submitted_unix=time.time(),actual_production_updates=0)
if r.returncode==0:proof['job_id']=r.stdout.strip().split(';')[0];assert proof['job_id'].isdigit()
with (b/'preparation/native-launch.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
assert r.returncode==0,r.stderr;print(json.dumps(proof))
'''


def submit(commit, cpu_path, cleanup_path, prep=PREP, repo=REPO):
    common.require(not (prep / NAMES['intent']).exists(), 'v2 already reserved; observe same handle, never repeat')
    inputs, files = load_inputs(commit, cpu_path, cleanup_path, prep, repo)
    token = uuid.uuid4().hex[:8]
    parent = '/mnt/data/shaoshitong/g1p-' + token
    base = '/mnt/data/shaoshitong/g1d-' + token
    budgets = dict(parent=common.ray_socket_budget(parent), discovery=common.ray_socket_budget(base))
    # New source and prior cleanup are independently read before any mutation.
    previous = common.remote('h100-8', previous_code(), inputs, timeout=75,
                             interpreter=inputs['old_launch']['base'] + '/env/bin/python')
    export = '/dataset/shaoshitong/g1-original-export-discovery-v2-' + token
    source = files['reproduction/common8/assets_export.py'].decode()
    needle = "BASE=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export')"
    common.require(source.count(needle) == 1, 'Explicit new source asset exporter mismatch')
    source = source.replace(needle, 'BASE=Path(' + repr(export) + ')').encode()
    intent = dict(inputs, base=base, parent_base=parent, export_base=export,
                  export_source_sha256=common.sha(source), created_unix=time.time(),
                  external_orchestrator_sha256=common.sha(Path(__file__).read_bytes()),
                  external_common_helper_sha256=common.sha(Path(common.__file__).read_bytes()),
                  ray_socket_budgets=budgets, previous_readonly_proof=previous,
                  actual_production_updates=0, scope='New official0 parent and discovery; original failed3274 remains immutable')
    common.save(prep / NAMES['intent'], intent)
    payload = dict(intent, files={name: base64.b64encode(data).decode() for name, data in files.items()})
    package = common.remote('h100-8', PREPARE_CODE, payload)
    common.save(prep / NAMES['package'], package)
    # Package includes real installedRay boundary + original data byte proof;
    # source process starts only after this preparation has actually succeeded.
    handle = common.remote('h100-9', common.START_SOURCE_CODE, dict(export=export,
                            source=base64.b64encode(source).decode(), source_sha256=common.sha(source)))
    common.save(prep / NAMES['handle'], handle)
    common.remote('h100-8', previous_code(), inputs, timeout=75,
                  interpreter=inputs['old_launch']['base'] + '/env/bin/python')
    launch = common.remote('h100-8', SUBMIT_CODE, dict(intent, handle=handle))
    common.save(prep / NAMES['launch'], launch)
    return {key: value for key, value in launch.items() if key not in ('source_sha256', 'argv')}


def observe(prep=PREP, cleanup_source=False):
    intent = common.strict_json((prep / NAMES['intent']).read_bytes())
    observation = common.remote('h100-8', common.OBSERVE_CODE, intent)
    suffix = str(time.time_ns())
    common.save(prep / ('policy-discovery-v2-observation-' + suffix + '.json'), observation)
    if not cleanup_source:
        return observation
    common.require(common.cleanup_eligible(observation), 'Original assets still needed or native/namespace state unknown')
    path = prep / NAMES['handle']
    handle = common.strict_json(path.read_bytes()) if path.exists() else common.remote('h100-9', common.SOURCE_HANDLE_CODE, intent)
    common.require(handle['export'] == intent['export_base'] and
                   handle['source_sha256'] == intent['export_source_sha256'], 'Wrong v2 source handle')
    proof = common.remote('h100-9', common.CLEANUP_SOURCE_CODE, handle)
    common.save(prep / ('policy-discovery-v2-retrieval-source-cleanup-' + suffix + '.json'), proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--submit', action='store_true'); group.add_argument('--observe', action='store_true')
    group.add_argument('--cleanup-source', action='store_true'); group.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--source-commit'); parser.add_argument('--target-cpu-proof'); parser.add_argument('--previous-cleanup-proof')
    args = parser.parse_args()
    if args.observe or args.cleanup_source:
        common.require(not any((args.source_commit, args.target_cpu_proof, args.previous_cleanup_proof)), 'Observe only the original saved v2 handle')
        result = observe(cleanup_source=args.cleanup_source)
    else:
        common.require(all((args.source_commit, args.target_cpu_proof, args.previous_cleanup_proof)),
                       'Explicit new full commit + new actual target CPU proof + previous scoped cleanup proof required')
        if args.submit:
            result = submit(args.source_commit, args.target_cpu_proof, args.previous_cleanup_proof)
        else:
            inputs, files = load_inputs(args.source_commit, args.target_cpu_proof, args.previous_cleanup_proof)
            result = dict(role='v2_discovery_local_structure_check', source_commit=inputs['source_commit'],
                          runtime_file_count=len(files), actual_target_cpu_tests=inputs['actual_target_cpu_tests'],
                          actual_production_updates=0, production_admission=False, healthy_checkpoint_accepted=False,
                          physical_execution_authenticated=False, scope='No remote call or service/submission')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
