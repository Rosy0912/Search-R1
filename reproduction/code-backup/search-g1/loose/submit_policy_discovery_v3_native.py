"""Independent fresh discovery after an authenticated one-write partial failure.

This external draft is outside runtime. It never changes the original 3278
files. Explicit new commit and fresh full target CPU proof are required. The old
run has zero completed outers but one Adam write on each of eight ranks. Missing/unknown evidence rejects before mutation.
Submission, source handle and observations have separate v3 claim paths.
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

NAMES = dict(intent='policy-discovery-v3-native-submit-intent.json',
             package='policy-discovery-v3-native-package-proof.json',
             handle='policy-discovery-v3-retrieval-export-handle.json',
             launch='policy-discovery-v3-native-launch.json')
# Keep the original failed identity independent of a later confirmation draft's
# explicitly updated contract. The original 3278 source must never move.
OLD_COMMIT = '130d17a0d6f0a3ab4b4aee730928f461e25105f0'
OLD_RUN = '6ed2e25db05342479641cb158e4ee866'
OLD_PARENT_RUN = '03ba871da8604c768f38366b4f72f7de'
OLD_BASE = '/mnt/data/shaoshitong/g1d-6f2e0963'
FAILURE_NAME = 'policy-discovery-v2-failure-scalars-1791243502314265.json'
OBSERVATION_NAME = 'policy-discovery-v2-native-observation.json'
RUNNING_NAME = 'policy-discovery-v2-native-observation-1791242144130813.json'
CPU_OLD_NAME = 'archive-cache-fix-target-cpu-v2-result.json'
NEW_TEST_MODULES = frozenset(('reproduction.common8.test_sampling_capture',
 'reproduction.common8.test_behavior_trace', 'reproduction.common8.test_behavior_probability',
 'reproduction.common8.test_behavior_runtime', 'reproduction.common8.test_generation_behavior'))
REQUIRED_CACHE_TEST = 'reproduction.common8.test_temporary_cache'


def check_commit(commit):
    common.require(isinstance(commit, str) and len(commit) == 40 and
                   all(c in '0123456789abcdef' for c in commit) and commit != OLD_COMMIT,
                   'Explicit new full runtime commit is required; old147 proof cannot admit retry')


def original_inputs(prep=PREP):
    old = common.strict_json((prep / 'policy-discovery-v2-native-launch.json').read_bytes())
    common.require(type(old['returncode']) is int and old['returncode'] == 0 and
                   old['job_id'] == '3278' and old['source_commit'] == OLD_COMMIT and
                   old['base'] == OLD_BASE and old['parent_base'] == '/mnt/data/shaoshitong/g1p-6f2e0963',
                   'Wrong immutable partial-write discovery handle')
    proof = common.strict_json((prep / CPU_OLD_NAME).read_bytes())
    common.require(common.sha((prep / CPU_OLD_NAME).read_bytes()) == old['target_cpu_proof_sha256'] and
                   proof['source_sha256'] == old['source_sha256'] and
                   common.strict_json(proof['stdout'])['tests_run'] == 147, 'Wrong original147 CPU bytes')
    handle = common.strict_json((prep / 'policy-discovery-v2-retrieval-export-handle.json').read_bytes())
    cleanup = common.strict_json((prep / 'policy-discovery-v2-retrieval-source-cleanup-1791240547627033000.json').read_bytes())
    common.require(cleanup['cleaned'] is True and cleanup['handle'] == handle,
                   'Old source service has not been scoped-cleaned')
    return old, proof


def partial_schema(failure, old):
    """Zero completed outers is explicitly different from zero optimizer writes."""
    common.require(failure['base'] == old['base'] == OLD_BASE and failure['run_id'] == OLD_RUN and
                   failure['source_hashes_match'] is True and
                   failure['pipeline-status.json']['run_id'] == OLD_RUN and
                   failure['pipeline-status.json']['stage'] == 'FAILED' and
                   failure['cleanup-status.json']['run_id'] == OLD_RUN and
                   failure['cleanup-status.json']['cleaned'] is True and
                   failure['owned_staging_root_exists'] is False and
                   failure['owned_staging_root_is_symlink'] is False,
                   'Wrong failed partial/cleanup identity')
    common.require(set(failure['rank_evidence']) == {str(rank) for rank in range(8)}, 'Missing old rank evidence')
    for rank in range(8):
        evidence = failure['rank_evidence'][str(rank)]
        digest = evidence['sha256']
        common.require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
                       'Invalid original ledger SHA')
        rows = evidence['rows']
        common.require(len(rows) == 3 and [row['kind'] for row in rows] ==
                       ['POLICY_MONITOR', 'OPTIMIZER_WRITE', 'POLICY_MONITOR'], 'Not exactly one partial Adam write')
        for row in rows:
            common.require(row['run_id'] == OLD_RUN and type(row['rank']) is int and row['rank'] == rank and
                           row['context'] == dict(logical_update=1, minibatch_index=0, prior_optimizer_writes=0),
                           'Original rank/write identity changed')
        before, write, after = rows
        common.require(before['stage'] == 'before_minibatch' and before['allowed'] is True and
                       type(before['optimizer_writes']) is int and before['optimizer_writes'] == 0 and
                       after['stage'] == 'after_minibatch' and after['allowed'] is False and
                       type(after['optimizer_writes']) is int and after['optimizer_writes'] == 1 and
                       after['reasons'] == ['old_policy_kl'] and
                       type(write['write_count']) is int and write['write_count'] == 1 and
                       write['learning_rates'] == [1e-6], 'Old partial failure must retain its real write')
    return failure


def history_schema(history, observed, old):
    from reproduction.common8.native_allocation import fields
    actual = fields(history['job_stdout'])
    common.require(history['job_returncode'] == 0 and actual['JobId'] == '3278' and
                   actual['JobState'] == 'RUNNING' and actual['NodeList'] == 'h100-8' and
                   actual['NumNodes'] == '1' and actual['MinMemoryNode'] == '512G' and
                   actual['Command'] == OLD_BASE + '/repo/reproduction/common8/launch_calibration_native.sh' and
                   actual['WorkDir'] == OLD_BASE + '/repo' and
                   actual['StdOut'] == OLD_BASE + '/logs/native-discovery-v2-3278.log',
                   'Missing actual historical RUNNING allocation/command')
    for snapshot in (history, observed):
        common.require(snapshot['base'] == OLD_BASE and snapshot['parent_base'] == old['parent_base'] and
                       snapshot['job_id'] == '3278' and snapshot['source_hashes_match'] is True and
                       snapshot['run-manifest.json']['run_id'] == OLD_RUN and
                       snapshot['run-manifest.json']['source_commit'] == OLD_COMMIT and
                       snapshot['run-manifest.json']['parent_production_run_id'] == OLD_PARENT_RUN,
                       'Original observed native identity changed')
    common.require(observed['actual_updates'] == [] and observed['active_owned_pids'] == [] and
                   not [row for row in observed['actual_namespace_processes'] if row['state'] != 'Z'] and
                   observed['pipeline-status.json']['run_id'] == OLD_RUN and
                   observed['pipeline-status.json']['stage'] == 'FAILED' and
                   observed['cleanup-status.json']['run_id'] == OLD_RUN and
                   observed['cleanup-status.json']['cleaned'] is True and
                   observed['process-ledger.json']['run_id'] == OLD_RUN and
                   all(child['state'] == 'STOPPED' for child in observed['process-ledger.json']['jobs'].values()) and
                   type(observed['process-ledger.json']['jobs']['trainer']['exit_code']) is int and
                   observed['process-ledger.json']['jobs']['trainer']['exit_code'] == 1,
                   'Actual workflow failure/exit/cleanup was not observed')
    common.require(observed['parent_actual_unstarted_verified'] is True and
                   all(value is False or value == [] for value in observed['parent_actual_unstarted'].values()),
                   'Original production parent is not unstarted')
    return actual


def validate_cpu(proof, old_proof):
    """Require new actual bytes and suite, never treating the old147 as new."""
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
                   set(old_proof['modules']) <= set(proof['modules']) and REQUIRED_CACHE_TEST in proof['modules'] and
                   set(proof['modules']) == set(old_proof['modules']) | NEW_TEST_MODULES and
                   len(proof['modules']) == 16 and summary['tests_run'] > 147,
                   'All sixteen actual target suites including actual behavior protocol are required')
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


def new_recipe(files):
    """Check effective literal overrides, rejecting duplicate scientific keys."""
    recipe = common.strict_json(files['reproduction/common8/recipe.json'])
    common.require(isinstance(recipe, list) and all(isinstance(value, str) and '=' in value for value in recipe),
                   'Invalid committed common8 recipe')
    effective = {}
    for override in recipe:
        key, value = override.split('=', 1)
        common.require(key not in effective, 'Duplicate effective recipe override')
        effective[key] = value
    expected = {'actor_rollout_ref.rollout.top_p': '1.0', 'actor_rollout_ref.rollout.temperature': '1',
                'actor_rollout_ref.rollout.n_agent': '5', 'actor_rollout_ref.rollout.n': '1',
                'data.train_batch_size': '512', 'actor_rollout_ref.actor.ppo_mini_batch_size': '128',
                'actor_rollout_ref.actor.ppo_micro_batch_size': '16',
                'actor_rollout_ref.rollout.log_prob_micro_batch_size': '32',
                'actor_rollout_ref.ref.log_prob_micro_batch_size': '32',
                'actor_rollout_ref.actor.ppo_epochs': '1', 'actor_rollout_ref.actor.optim.lr': '1e-6'}
    common.require(all(effective.get(key) == value for key, value in expected.items()),
                   'New neutral sampling/scientific budget recipe differs')
    for name in ('behavior_trace', 'behavior_probability', 'behavior_runtime', 'generation_behavior'):
        common.require('reproduction/common8/' + name + '.py' in files, 'Missing actual behavior source')
    return effective


def load_inputs(commit, cpu_path, prep=PREP, repo=REPO):
    check_commit(commit)
    old, old_proof = original_inputs(prep)
    cpu_bytes = Path(cpu_path).read_bytes(); proof = common.strict_json(cpu_bytes)
    failure_bytes = (prep / FAILURE_NAME).read_bytes()
    observed_bytes = (prep / OBSERVATION_NAME).read_bytes()
    history_bytes = (prep / RUNNING_NAME).read_bytes()
    failure, observed, history = map(common.strict_json, (failure_bytes, observed_bytes, history_bytes))
    partial_schema(failure, old); history_schema(history, observed, old)
    files = committed_sources(commit, validate_cpu(proof, old_proof), repo)
    new_recipe(files)
    return dict(old_launch=old, previous_partial_failure=failure,
                previous_failure_sha256=common.sha(failure_bytes), previous_observation=observed,
                previous_observation_sha256=common.sha(observed_bytes), previous_running=history,
                previous_running_sha256=common.sha(history_bytes),
                target_cpu_proof_sha256=common.sha(cpu_bytes), source_commit=commit,
                source_sha256=proof['source_sha256'], actual_target_cpu_tests=common.strict_json(proof['stdout'])['tests_run']), files


def native_terminal_route(old, command):
    """Purge does not establish or invent the historical Slurm terminal state."""
    from reproduction.common8.native_allocation import fields
    if command.returncode == 0:
        job = fields(command.stdout)
        common.require(job['JobId'] == '3278' and job['NodeList'] == 'h100-8' and
                       job['JobState'] in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL') and
                       job['Command'] == OLD_BASE + '/repo/reproduction/common8/launch_calibration_native.sh',
                       'Original job remains active or current terminal identity changed')
        return dict(route='actual_current_terminal_native', slurm_terminal_state=job['JobState'])
    common.require(command.returncode != 0 and not command.stdout.strip() and
                   command.stderr.strip() == 'slurm_load_jobs error: Invalid job id specified',
                   'Native query error is not the exact scoped Slurm purge')
    queue = subprocess.run(['squeue', '-h', '-j', '3278', '-o', '%i|%T|%j|%Z|%o'],
                           capture_output=True, text=True, timeout=15)
    common.require((queue.returncode == 0 and not queue.stdout.strip() and not queue.stderr.strip()) or
                   (queue.returncode != 0 and not queue.stdout.strip() and
                    queue.stderr.strip() == 'slurm_load_jobs error: Invalid job id specified'),
                   'Scoped purge queue absence not authenticated')
    return dict(route='historical_running_actual_workflow_failed_current_purged', slurm_terminal_state=None)


def verify_old_manifest(raw, observed):
    """Bind complete original bytes to the poll's deliberately trimmed subset."""
    common.require(common.sha(raw) == observed['child_manifest_sha256'], 'Actual old manifest bytes changed')
    run = common.strict_json(raw)
    subset = observed['run-manifest.json']
    required = {'run_id', 'completed_updates', 'source_commit', 'train_log', 'checkpoint_dir',
                'policy_calibration_role', 'parent_production_run_id'}
    common.require(isinstance(subset, dict) and set(subset) == required and
                   all(key in run and run[key] == value for key, value in subset.items()),
                   'Actual old manifest saved subset changed')
    common.require(run['run_id'] == OLD_RUN and run['source_commit'] == OLD_COMMIT and
                   run['profile'] == 'common8' and
                   run['policy_calibration_role'] == 'policy_safety_calibration_discovery' and
                   run['parent_production_run_id'] == OLD_PARENT_RUN and
                   run['policy_calibration_slurm_job_id'] == '3278' and
                   type(run['completed_updates']) is int and run['completed_updates'] == 0,
                   'Actual old run/job/source/completed identity changed')
    return run


def previous_readonly(payload):
    """Read old immutable one-write evidence; never reset or resume its Adam."""
    old = payload['old_launch']; failure = partial_schema(payload['previous_partial_failure'], old)
    b = Path(old['base']); sys.path.insert(0, str(b / 'repo'))
    observed = payload['previous_observation']; history_schema(payload['previous_running'], observed, old)
    from reproduction.a800.owned_processes import same, identity
    run = verify_old_manifest((b / 'logs/run-manifest.json').read_bytes(), observed)
    for name, digest in old['source_sha256'].items():
        path = b / 'repo' / name
        common.require(path.resolve() == path and common.sha(path.read_bytes()) == digest, 'Original failed source changed')
    for rank in range(8):
        path = b / 'logs' / ('optimizer-rank-' + str(rank) + '.jsonl')
        common.require(path.resolve() == path, 'Old ledger symlink rejected')
        raw = path.read_bytes()
        common.require(common.sha(raw) == failure['rank_evidence'][str(rank)]['sha256'], 'Actual partial ledger changed')
        rows = [common.strict_json(line) for line in raw.splitlines()]
        # Revalidate its exact actual sequence, not only a saved summary.
        current = dict(failure, rank_evidence=dict(failure['rank_evidence']))
        current['rank_evidence'][str(rank)] = dict(sha256=common.sha(raw), rows=rows)
        partial_schema(current, old)
    status = common.strict_json((b / 'logs/pipeline-status.json').read_bytes())
    cleanup = common.strict_json((b / 'logs/cleanup-status.json').read_bytes())
    common.require(status == observed['pipeline-status.json'] and status['stage'] == 'FAILED' and
                   cleanup == observed['cleanup-status.json'] and cleanup['cleaned'] is True,
                   'Actual workflow failure/cleanup changed')
    updates = b / 'logs/actual-updates.jsonl'
    common.require(not updates.exists() or not updates.read_bytes().strip(), 'Old completed outers are not zero')
    for name in ('policy-calibration-segment.json', 'policy-calibration-segment-complete.json'):
        common.require(not (b / 'logs' / name).exists(), 'Old partial run acquired a completed segment')
    checkpoints = Path(run['checkpoint_dir'])
    common.require(checkpoints == b / 'checkpoints/search-g1-common8' and
                   (not checkpoints.exists() or not any(checkpoints.iterdir())), 'Old checkpoint exists or namespace changed')
    ledger = common.strict_json((b / 'logs/process-ledger.json').read_bytes())
    common.require(ledger == observed['process-ledger.json'] and ledger['jobs']['trainer']['exit_code'] == 1,
                   'Actual trainer exit/owned ledger changed')
    for child in ledger['jobs'].values():
        common.require(child['state'] == 'STOPPED', 'Old owned child remains active')
        for process in child['ledger'].values():
            common.require(not same(process) or identity(process['pid'])['state'] == 'Z', 'Actual old owned process still live')
    marker = ('SEARCH_G1_BASE=' + str(b)).encode()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            if proc.stat().st_uid != os.getuid() or marker not in (proc / 'environ').read_bytes().split(b'\0'): continue
            common.require((proc / 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z', 'Old namespace remains active')
        except (FileNotFoundError, ProcessLookupError): pass
    root = Path('/dev/shm') / ('search-g1-common8-' + OLD_RUN + '-3278')
    for name, target in (('retrieval-staging-claim.json', root), ('corpus-cache-claim.json', root / 'datasets-cache')):
        claim = common.strict_json((b / 'logs' / name).read_bytes())
        common.require(claim['run_id'] == OLD_RUN and claim['slurm_job_id'] == '3278' and
                       claim['uid'] == os.getuid() and claim['phase'] == 'CLEANED' and Path(claim['root']) == target,
                       'Actual old owned cleanup claim differs')
    common.require(not root.exists() and not root.is_symlink(), 'Old owned temporary root still exists')
    command = subprocess.run(['scontrol', 'show', 'job', '-o', '3278'], capture_output=True, text=True, timeout=15)
    route = native_terminal_route(old, command)
    return dict(role='previous_partial_discovery_readonly_verification', run_id=OLD_RUN, job_id='3278',
                source_commit=OLD_COMMIT, actual_completed_updates=0, actual_optimizer_writes_per_rank=1,
                original_failed_run_preserved=True, terminal_observation=route,
                owned_namespace_empty=True, owned_shm_root_absent=True,
                production_admission=False, healthy_checkpoint_accepted=False)


def previous_code():
    helpers = '\n'.join(inspect.getsource(fn) for fn in (common.canonical, common.sha, common.require, common.strict_json))
    functions = '\n'.join(inspect.getsource(fn) for fn in (partial_schema, history_schema, native_terminal_route,
                                                          verify_old_manifest, previous_readonly))
    constants = '\n'.join(name + '=' + repr(globals()[name]) for name in ('OLD_COMMIT','OLD_RUN','OLD_PARENT_RUN','OLD_BASE'))
    return ('import hashlib,json,os,subprocess,sys\nfrom pathlib import Path\nfrom types import SimpleNamespace\n' +
            helpers + '\ncommon=SimpleNamespace(canonical=canonical,sha=sha,require=require,strict_json=strict_json)\n' +
            constants + '\n' + functions + '\np=json.load(sys.stdin);print(json.dumps(previous_readonly(p)))\n')


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
code='import verl,verl.single_controller;from verl.workers.fsdp_workers import ActorRolloutRefWorker;from reproduction.common8 import prepare_calibration_run,calibration_runtime,temporary_cache,adopt_initial_probes,production_safety_admission,behavior_trace,behavior_probability,behavior_runtime,generation_behavior;from reproduction.common8.verify_data import verify;import pathlib,sys,json,inspect,hashlib;from types import SimpleNamespace;from ray._private.node import Node;s=inspect.getsource(Node._prepare_socket_file);good="/"+"a"*106;dummy=SimpleNamespace(_sockets_dir="/unused",_make_inc_temp=lambda **kwargs:good);assert sys.platform=="linux" and Node._prepare_socket_file(dummy,None,"plasma_store")==good\ntry:\n dummy._make_inc_temp=lambda **kwargs:"/"+"a"*107;Node._prepare_socket_file(dummy,None,"plasma_store")\nexcept OSError:pass\nelse:raise AssertionError("Actual installed Ray accepted108-byteAF_UNIX")\nprint(json.dumps(dict(data=verify(pathlib.Path(sys.argv[1])),installed_ray_socket_check_sha256=hashlib.sha256(s.encode()).hexdigest(),installed_limit_tested_bytes=[107,108],scope="Actual installed AF_UNIX method CPU boundary check; no Ray/GPU startup")))'
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
args=['sbatch','--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h100:8','--mem=512G','--exclusive','--time=08:00:00','--job-name=search-g1-policy-discovery-v3','--chdir='+str(b/'repo'),'--output='+str(b/'logs/native-discovery-v3-%j.log'),'--export=ALL,SEARCH_G1_BASE='+str(b)+',SEARCH_G1_PRODUCTION_BASE='+str(parent)+',SEARCH_G1_CALIBRATION_ROLE=policy_safety_calibration_discovery,SEARCH_G1_SOURCE_COMMIT='+p['source_commit'],str(b/'repo/reproduction/common8/launch_calibration_native.sh')]
with (b/'preparation/native-submit-claim.json').open('x') as f:json.dump(dict(argv=args,created_unix=time.time(),base=str(b),parent_base=str(parent),target_cpu_proof_sha256=p['target_cpu_proof_sha256'],scope='Independent retry after stoppedfailed3278; production0'),f);f.flush();os.fsync(f.fileno())
with (b/'preparation/retrieval-source-handle.json').open('x') as f:json.dump(p['handle'],f,indent=2);f.flush();os.fsync(f.fileno())
fd=os.open(str(b/'preparation'),os.O_DIRECTORY)
try:os.fsync(fd)
finally:os.close(fd)
r=subprocess.run(args,capture_output=True,text=True)
proof=dict(base=str(b),parent_base=str(parent),source_commit=p['source_commit'],source_sha256=p['source_sha256'],target_cpu_proof_sha256=p['target_cpu_proof_sha256'],argv=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,submitted_unix=time.time(),actual_production_updates=0)
if r.returncode==0:proof['job_id']=r.stdout.strip().split(';')[0];assert proof['job_id'].isdigit()
with (b/'preparation/native-launch.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
assert r.returncode==0,r.stderr;print(json.dumps(proof))
'''


def submit(commit, cpu_path, prep=PREP, repo=REPO):
    common.require(not (prep / NAMES['intent']).exists(), 'v3 already reserved; observe same handle, never repeat')
    inputs, files = load_inputs(commit, cpu_path, prep, repo)
    token = uuid.uuid4().hex[:8]
    parent = '/mnt/data/shaoshitong/g1p-' + token
    base = '/mnt/data/shaoshitong/g1d-' + token
    budgets = dict(parent=common.ray_socket_budget(parent), discovery=common.ray_socket_budget(base))
    # New source and prior cleanup are independently read before any mutation.
    previous = common.remote('h100-8', previous_code(), inputs, timeout=75,
                             interpreter=inputs['old_launch']['base'] + '/env/bin/python')
    export = '/dataset/shaoshitong/g1-original-export-discovery-v3-' + token
    source = files['reproduction/common8/assets_export.py'].decode()
    needle = "BASE=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export')"
    common.require(source.count(needle) == 1, 'Explicit new source asset exporter mismatch')
    source = source.replace(needle, 'BASE=Path(' + repr(export) + ')').encode()
    intent = dict(inputs, base=base, parent_base=parent, export_base=export,
                  export_source_sha256=common.sha(source), created_unix=time.time(),
                  external_orchestrator_sha256=common.sha(Path(__file__).read_bytes()),
                  external_common_helper_sha256=common.sha(Path(common.__file__).read_bytes()),
                  ray_socket_budgets=budgets, previous_readonly_proof=previous,
                  actual_production_updates=0, scope='New official0 parent and discovery; original partial-one-write3278 remains immutable')
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
    common.save(prep / ('policy-discovery-v3-observation-' + suffix + '.json'), observation)
    if not cleanup_source:
        return observation
    common.require(common.cleanup_eligible(observation), 'Original assets still needed or native/namespace state unknown')
    path = prep / NAMES['handle']
    handle = common.strict_json(path.read_bytes()) if path.exists() else common.remote('h100-9', common.SOURCE_HANDLE_CODE, intent)
    common.require(handle['export'] == intent['export_base'] and
                   handle['source_sha256'] == intent['export_source_sha256'], 'Wrong v3 source handle')
    proof = common.remote('h100-9', common.CLEANUP_SOURCE_CODE, handle)
    common.save(prep / ('policy-discovery-v3-retrieval-source-cleanup-' + suffix + '.json'), proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--submit', action='store_true'); group.add_argument('--observe', action='store_true')
    group.add_argument('--cleanup-source', action='store_true'); group.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--source-commit'); parser.add_argument('--target-cpu-proof')
    args = parser.parse_args()
    if args.observe or args.cleanup_source:
        common.require(not any((args.source_commit, args.target_cpu_proof)), 'Observe only the original saved v3 handle')
        result = observe(cleanup_source=args.cleanup_source)
    else:
        common.require(all((args.source_commit, args.target_cpu_proof)),
                       'Explicit new full commit + new actual sixteen-suite target CPU proof required')
        if args.submit:
            result = submit(args.source_commit, args.target_cpu_proof)
        else:
            inputs, files = load_inputs(args.source_commit, args.target_cpu_proof)
            result = dict(role='v3_discovery_local_structure_check', source_commit=inputs['source_commit'],
                          runtime_file_count=len(files), actual_target_cpu_tests=inputs['actual_target_cpu_tests'],
                          actual_production_updates=0, production_admission=False, healthy_checkpoint_accepted=False,
                          physical_execution_authenticated=False, scope='No remote call or service/submission')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
