"""Prepare one independent confirmation of an explicit root-frozen selection.

This external orchestration is deliberately outside the immutable runtime.
Default/check mode is local and cannot start a service or submit a job. Submit
requires complete independently reread discovery evidence and owned cleanup.
Any partial/unknown attempt keeps its claims: inspect it, never resubmit.
"""
import argparse
import base64
import hashlib
import inspect
import json
import math
import os
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
REPO, PREP = ROOT / 'repo', ROOT / 'preparation'
FROZEN_COMMIT = '61f6e1da532e08553410666b430d1933a0e4a49a'
FROZEN_METHOD_SHA256 = '1f41e41ace87161e9885c4d57873e388f9b77ae717ed9427fdfe5c71addce057'
DISCOVERY_ROLE = 'policy_safety_calibration_discovery'
CONFIRMATION_ROLE = 'policy_safety_calibration_confirmation'
CANONICAL_FILES = ('manifest.json', 'train.parquet', 'validation.parquet',
                   'stage.parquet', 'final.parquet', 'exclusions-audit.json')
SELECTION_NAME = 'policy-confirmation-selection.json'
METHOD_NAME = 'policy-calibration-selection-method.json'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(content):
    return hashlib.sha256(content).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(content):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(content, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def selection_schema(selection, launch):
    """Validate explicit frozen choices, without choosing thresholds or health."""
    names = {'schema_version', 'role', 'selected_by', 'selection_provenance',
             'parent_base', 'discovery_base', 'parent_production_run_id',
             'discovery_run_id', 'discovery_slurm_job_id', 'source_commit',
             'source_map_sha256', 'scientific_identity_sha256',
             'discovery_manifest_sha256', 'discovery_plan_file_sha256',
             'discovery_plan_sha256', 'discovery_evidence_sha256',
             'selection_method_sha256', 'negative_tail_limit', 'health_rule', 'threshold_selection'}
    require(isinstance(selection, dict) and set(selection) == names, 'Wrong frozen selection schema')
    require(type(selection['schema_version']) is int and selection['schema_version'] == 1 and
            selection['role'] == 'policy_safety_confirmation_orchestration_selection' and
            selection['selected_by'] == 'root_explicit_selection' and
            isinstance(selection['selection_provenance'], dict) and bool(selection['selection_provenance']),
            'Missing explicit root selection provenance')
    require(selection['source_commit'] == launch['source_commit'] == FROZEN_COMMIT,
            'Confirmation must use the frozen parent runtime, not current HEAD')
    require(selection['parent_base'] == launch['parent_base'] and
            selection['discovery_base'] == launch['base'] and
            selection['discovery_slurm_job_id'] == launch['job_id'], 'Wrong discovery handle')
    require(selection['parent_base'] != selection['discovery_base'], 'Parent and discovery are not separate')
    for name in ('parent_base', 'discovery_base'):
        path = Path(selection[name])
        require(path.is_absolute() and '..' not in path.parts and path.as_posix() == selection[name] and
                selection[name].startswith('/mnt/data/shaoshitong/'), 'Unbound native namespace')
    for name in ('parent_production_run_id', 'discovery_run_id'):
        value = selection[name]
        require(isinstance(value, str) and len(value) == 32 and all(c in '0123456789abcdef' for c in value),
                'Invalid actual run UUID')
    require(selection['parent_production_run_id'] != selection['discovery_run_id'], 'Child reused parent UUID')
    for name in ('source_map_sha256', 'scientific_identity_sha256', 'discovery_manifest_sha256',
                 'discovery_plan_file_sha256', 'discovery_plan_sha256', 'discovery_evidence_sha256'):
        value = selection[name]
        require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                'Invalid frozen hash: ' + name)
    require(selection['source_map_sha256'] == sha(canonical(launch['source_sha256'])), 'Frozen source map changed')
    require(selection['selection_method_sha256'] == FROZEN_METHOD_SHA256, 'Root pre-discovery selection method changed')
    tail = selection['negative_tail_limit']
    require(type(tail) in (int, float) and math.isfinite(tail) and 0 <= tail <= 1,
            'Explicit actual finite tail candidate is required')
    for name, role in (('health_rule', 'policy_safety_health_rule'),
                       ('threshold_selection', 'policy_safety_threshold_selection')):
        frozen = selection[name]
        require(isinstance(frozen, dict) and set(frozen) == {'document', 'sha256'} and
                sha(canonical(frozen['document'])) == frozen['sha256'] and
                frozen['document']['role'] == role and
                isinstance(frozen['document']['provenance'], dict) and bool(frozen['document']['provenance']),
                'Wrong frozen rule/provenance: ' + name)
    choice = selection['threshold_selection']['document']['criteria']
    require(choice['discovery_evidence_sha256'] == selection['discovery_evidence_sha256'] and
            choice['health_rule_sha256'] == selection['health_rule']['sha256'] and
            canonical(choice['candidate_thresholds']) == canonical(dict(
                dual_clip=3.0, epsilon=0.2, target_old_policy_kl=0.01,
                kl_log_bound=30.0, negative_tail_limit=tail)), 'Frozen candidates differ from selected evidence/rule')
    # Shape/method checks only. Actual paired measurements are checked on target.
    criteria = selection['health_rule']['document']['criteria']
    require(criteria['method'] == 'paired_fixed512_absolute_allowed_drop' and
            set(criteria) <= {'method', 'allowed_drop', 'minimum_before', 'minimum_after'}, 'Unknown health rule')
    supported = {'weighted_em', 'weighted_f1', 'valid_action_rate',
                 'executed_search_question_rate', 'executed_search_calls_per_question'}
    require({'weighted_em', 'weighted_f1', 'valid_action_rate'} <= set(criteria['allowed_drop']) <= supported,
            'Frozen EM/F1/valid-action health criteria are mandatory')
    for name in ('allowed_drop', 'minimum_before', 'minimum_after'):
        values = criteria.get(name, {})
        require(isinstance(values, dict) and set(values) <= supported, 'Unknown health criterion')
        for value in values.values():
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    'Invalid explicit health value')
    return selection


def method_binding(content, selection):
    require(sha(content) == selection['selection_method_sha256'] == FROZEN_METHOD_SHA256,
            'Original root selection-method bytes changed')
    method = strict_json(content)
    require(method['role'] == 'root_explicit_policy_calibration_selection_method' and
            method['source_commit'] == selection['source_commit'] and
            method['parent_production_run_id'] == selection['parent_production_run_id'] and
            method['discovery_run_id'] == selection['discovery_run_id'] and
            method['before_completed_discovery_update'] is True and
            method['negative_tail_limit'] is None and
            method['production_admission'] is False and method['healthy_checkpoint_accepted'] is False and
            canonical(method['health_rule_document']) == canonical(selection['health_rule']['document']),
            'Selection changed the root-frozen health rule or scientific identity')
    choice = selection['threshold_selection']['document']['criteria']['candidate_thresholds']
    require(canonical({key: choice[key] for key in method['fixed_candidates']}) ==
            canonical(method['fixed_candidates']), 'Selection changed the original candidate mathematics')
    return method


def measured_selection(selection, segment):
    """Check the frozen empirical-max method; never generate a candidate."""
    structural = segment['structural']
    require(structural['execution_role'] == DISCOVERY_ROLE and
            structural['run_id'] == selection['discovery_run_id'] and
            structural['calibration_sha256'] == selection['discovery_plan_sha256'] and
            type(structural['outer_count']) is int and structural['outer_count'] == 1 and
            type(structural['optimizer_writes_per_rank']) is int and structural['optimizer_writes_per_rank'] == 20 and
            structural['enforcement']['negative_tail'] == 'observe_only', 'Not the complete observe-only discovery')
    observations = structural['global_monitors']
    require(type(observations) is list and len(observations) == 40, 'Need exactly 40 coalesced global observations')
    tails = []
    for ordinal, observation in enumerate(observations):
        mini, stage = ordinal // 2, ('before_minibatch', 'after_minibatch')[ordinal % 2]
        expected = dict(logical_update=1, minibatch_index=mini, prior_optimizer_writes=mini)
        require(canonical(observation['context']) == canonical(expected) and
                observation['stage'] == stage and observation['run_id'] == selection['discovery_run_id'] and
                observation['calibration_sha256'] == selection['discovery_plan_sha256'] and
                type(observation['optimizer_writes']) is int and observation['optimizer_writes'] == mini + ordinal % 2 and
                observation['execution_role'] == DISCOVERY_ROLE and
                observation['enforcement']['negative_tail'] == 'observe_only' and
                observation['allowed'] is True and observation['reasons'] == [], 'Incomplete/reordered global observation')
        value = observation['negative_tail_weight_fraction']
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                'Actual weighted negative-tail fraction is invalid')
        tails.append(value)
    require(selection['negative_tail_limit'] == max(tails), 'Selected tail differs from exact actual maximum; no margin/floor')
    return max(tails)


def save(path, value):
    """Durable exclusive claim; a partial file deliberately blocks replay."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(str(Path(path).parent), os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def frozen_sources(launch, repo=REPO):
    """Read commit blobs and exact known map, never mutable HEAD files."""
    require(launch['source_commit'] == FROZEN_COMMIT, 'Wrong immutable source commit')
    files = {}
    for name, expected in launch['source_sha256'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unbound runtime source')
        content = subprocess.check_output(['git', 'cat-file', 'blob', FROZEN_COMMIT + ':' + name], cwd=repo)
        require(sha(content) == expected, 'Frozen commit/blob map mismatch: ' + name)
        files[name] = content
    require(len(files) == 344, 'Unexpected immutable runtime file set')
    return files


def target_evidence(selection, launch):
    """Read actual source archives/weights/512 rows on cluster; no GPU kernels.

    The real verifier reconstructs 40 global reports, 20 writes/rank, finite
    fresh1/Adam/delta, rank .05/configuration and exact fixed512 arithmetic.
    The selected health rule is then applied to before/after, twice to the
    same completed discovery. This does NOT compare a measurement to itself.
    """
    selection_schema(selection, launch)
    parent_base, discovery = Path(selection['parent_base']), Path(selection['discovery_base'])
    require(parent_base.resolve() == parent_base and discovery.resolve() == discovery, 'Namespace symlink rejected')
    sys.path.insert(0, str(parent_base / 'repo'))
    from reproduction.common8.policy_calibration_plan import _identity, DISCOVERY_ROLE
    from reproduction.common8.freeze_calibration_plan import _unstarted
    from reproduction.common8.production_safety_admission import verify_child, evaluate_health
    from reproduction.common8.source_files import runtime_files
    parent = strict_json((parent_base / 'logs/run-manifest.json').read_bytes())
    require(parent.get('policy_calibration_role', 'production') == 'production' and
            parent['run_id'] == selection['parent_production_run_id'] and
            parent['source_commit'] == FROZEN_COMMIT, 'Wrong original fresh parent/source')
    _unstarted(parent_base, parent)
    require(sha(canonical(_identity(parent))) == selection['scientific_identity_sha256'], 'Parent science identity changed')
    for namespace in (parent_base, discovery):
        mapping = {str(path.relative_to(namespace / 'repo')): sha(path.read_bytes())
                   for path in runtime_files(namespace / 'repo')}
        require(mapping == launch['source_sha256'], 'Actual frozen namespace source changed')
    receipts = {}
    run, plan, segment = verify_child(discovery, parent, DISCOVERY_ROLE, receipts, 'discovery/')
    require(run['run_id'] == selection['discovery_run_id'] and
            plan['slurm_job_id'] == selection['discovery_slurm_job_id'], 'Wrong actual complete child')
    for name, relative in (('discovery_manifest_sha256', 'logs/run-manifest.json'),
                           ('discovery_plan_file_sha256', 'logs/policy-safety-calibration-plan.json'),
                           ('discovery_evidence_sha256', 'logs/policy-calibration-segment.json')):
        require(sha((discovery / relative).read_bytes()) == selection[name], 'Frozen actual artifact changed: ' + name)
    require(sha(canonical(plan)) == selection['discovery_plan_sha256'], 'Canonical plan hash changed')
    measured_selection(selection, segment)
    health = evaluate_health(selection['health_rule'], segment, segment)
    require(health['accepted'] is True, 'Discovery failed the explicit frozen health rule')
    # Source verification has checked its owned ledger; independently census the
    # actual namespace too, so an unledgered owned child cannot remain active.
    marker = ('SEARCH_G1_BASE=' + str(discovery)).encode()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or proc.stat().st_uid != os.getuid(): continue
        try:
            if marker in (proc / 'environ').read_bytes().split(b'\0'):
                state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
                require(state == 'Z', 'Discovery namespace remains active: ' + proc.name)
        except (FileNotFoundError, ProcessLookupError): pass
    command = subprocess.run(['scontrol', 'show', 'job', '-o', selection['discovery_slurm_job_id']],
                             capture_output=True, text=True, timeout=15)
    from reproduction.common8.native_allocation import fields
    actual = fields(command.stdout)
    require(command.returncode == 0 and actual['JobId'] == selection['discovery_slurm_job_id'] and
            actual['JobState'] == 'COMPLETED', 'Actual discovery native job has not completed')
    return dict(role='confirmation_source_prerequisite_verification', run_id=run['run_id'],
                parent_production_run_id=parent['run_id'], source_commit=parent['source_commit'],
                discovery_evidence_sha256=selection['discovery_evidence_sha256'],
                frozen_health_rule_satisfied=True, health_rule_sha256=health['rule_sha256'],
                discovery_owned_cleanup_verified=True, actual_production_updates=0,
                production_admission=False, healthy_checkpoint_accepted=False)


def remote(host, code, payload, timeout=90, interpreter='python3'):
    """Unknown remote outcome is never converted into permission to retry."""
    compile(code, '<remote>', 'exec')
    sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
    import control
    result = subprocess.run(control.route(host, shlex.quote(interpreter) + ' -c ' + shlex.quote(code)),
                            input=json.dumps(payload, allow_nan=False), capture_output=True,
                            text=True, timeout=timeout)
    require(result.returncode == 0, 'Observe the same saved intent/remote claims; do not retry: ' + result.stderr[-2500:])
    return strict_json(result.stdout)


def verification_code():
    return ('import hashlib,json,math,os,subprocess,sys\nfrom pathlib import Path\n' +
            '\n'.join(inspect.getsource(function) for function in
                      (canonical, sha, require, strict_json, selection_schema, measured_selection, target_evidence)) +
            '\nFROZEN_COMMIT=' + repr(FROZEN_COMMIT) + '\nFROZEN_METHOD_SHA256=' + repr(FROZEN_METHOD_SHA256) +
            '\nDISCOVERY_ROLE=' + repr(DISCOVERY_ROLE) +
            '\np=json.load(sys.stdin);print(json.dumps(target_evidence(p["selection"],p["launch"])))\n')


def native_wrapper(files):
    """Preserve frozen native environment/horizon; insert only identity binding."""
    source = files['reproduction/common8/launch_calibration_native.sh'].decode()
    needle = 'exec python -m reproduction.common8.run_pipeline'
    require(source.count(needle) == 1, 'Frozen launcher changed')
    return source.replace(needle, 'python "$BASE/preparation/submit_policy_confirmation_native.py" --native-bind --base "$BASE"\n' + needle).encode()


def ray_socket_budget(base):
    """Conservative fresh-session path for actual Ray's plasma_store basename.

    The recorded native error proves the date/time/microsecond/session PID and
    basename form. Ten PID digits are conservatively used before any service.
    This calculation is not a claim that Ray has started successfully.
    """
    path = (str(base) + '/ray/session_9999-12-31_23-59-59_999999_9999999999/sockets/plasma_store')
    count = len(path.encode('utf-8'))
    require(count <= 107, 'Conservative Ray AF_UNIX path exceeds the installed 107-byte limit')
    return dict(worst_plasma_socket_path=path, worst_plasma_socket_bytes=count,
                af_unix_limit_bytes=107, pid_digits=10, actual_ray_started=False)


def native_bind(base):
    """Bind frozen choices to the newly prepared actual child UUID, before Adam."""
    base = Path(base).resolve()
    selection_bytes = (base / 'preparation' / SELECTION_NAME).read_bytes()
    package = strict_json((base / 'preparation/source-package-intent.json').read_bytes())
    require(sha(selection_bytes) == package['selection_sha256'], 'Selection bytes changed after reservation')
    selection = strict_json(selection_bytes); launch = package['discovery_launch']
    method_binding((base / 'preparation' / METHOD_NAME).read_bytes(), selection)
    target_evidence(selection, launch)
    from reproduction.common8.prepare_calibration_run import native_preparation
    from reproduction.common8.policy_calibration_plan import _identity
    from reproduction.common8.freeze_calibration_plan import _unstarted
    job_id = native_preparation()
    run = strict_json((base / 'logs/run-manifest.json').read_bytes())
    require(str(base) == package['base'] and run['policy_calibration_role'] == CONFIRMATION_ROLE and
            run['policy_calibration_slurm_job_id'] == job_id and job_id != selection['discovery_slurm_job_id'] and
            run['run_id'] not in (selection['parent_production_run_id'], selection['discovery_run_id']) and
            run['parent_production_run_id'] == selection['parent_production_run_id'] and
            run['policy_calibration_discovery_base'] == selection['discovery_base'] and
            run['initial_probe_version'] is None, 'Not a fresh independent actual confirmation')
    _unstarted(base, run)
    require(sha(canonical(_identity(run))) == selection['scientific_identity_sha256'], 'Child scientific identity differs')
    # Bound external code is not silently included in the immutable runtime map.
    require(sha(Path(__file__).read_bytes()) == package['external_orchestrator_sha256'], 'External binder changed')
    require(sha((base / 'preparation/launch-confirmation-native.sh').read_bytes()) ==
            package['external_wrapper_sha256'], 'External launcher changed')
    claim = base / 'preparation/confirmation-bind-claim.json'
    save(claim, dict(run_id=run['run_id'], slurm_job_id=job_id, selection_sha256=sha(selection_bytes),
                     started_unix=time.time(), scope='Frozen identity binding only; no health or model admission'))
    discovery = Path(selection['discovery_base'])
    for relative, name, expected in (
        ('logs/policy-safety-calibration-plan.json', 'policy-safety-discovery-plan.json', selection['discovery_plan_file_sha256']),
        ('logs/policy-calibration-segment.json', 'policy-safety-discovery-evidence.json', selection['discovery_evidence_sha256'])):
        content = (discovery / relative).read_bytes(); require(sha(content) == expected, 'Original source bytes changed')
        with (base / 'preparation' / name).open('xb') as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
    candidate = dict(schema_version=1, role='policy_safety_confirmation_candidates',
                     parent_production_run_id=selection['parent_production_run_id'],
                     confirmation_run_id=run['run_id'], discovery_run_id=selection['discovery_run_id'],
                     discovery_evidence_sha256=selection['discovery_evidence_sha256'],
                     negative_tail_limit=selection['negative_tail_limit'],
                     health_rule=selection['health_rule'], threshold_selection=selection['threshold_selection'])
    save(base / 'preparation/policy-safety-confirmation-candidates.json', candidate)
    result = dict(role='confirmation_frozen_candidate_binding', run_id=run['run_id'], slurm_job_id=job_id,
                  selection_sha256=sha(selection_bytes), actual_production_updates=0,
                  production_admission=False, healthy_checkpoint_accepted=False,
                  child_model_probe_http_admission_required=True)
    save(base / 'preparation/confirmation-bind-complete.json', result)
    return result


PREPARE_CODE = r'''
import base64,hashlib,inspect,json,os,shutil,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['base']);parent=Path(p['parent_base'])
assert not b.exists();b.mkdir(parents=True,exist_ok=False)
for name in ('repo','data','preparation','logs','tmp','ray','artifacts','cache/triton','cache/inductor','cache/hf'):(b/name).mkdir(parents=True,exist_ok=True)
for name in ('env','bootstrap-assets','bootstrap-tools','assets','retriever-gpu-lib'):
 source=parent/name;assert source.is_dir();(b/name).symlink_to(source.resolve(),target_is_directory=True)
for name,data in p['files'].items():
 relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
 target=b/'repo'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(data))
 assert hashlib.sha256(target.read_bytes()).hexdigest()==p['source_sha256'][name]
for name in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):
 source=parent/'data'/name;target=b/'data'/name;shutil.copy2(source,target)
 assert hashlib.sha256(source.read_bytes()).hexdigest()==hashlib.sha256(target.read_bytes()).hexdigest()
(b/'preparation/official-model-manifest-original.json').write_bytes((parent/'preparation/official-model-manifest-original.json').read_bytes())
for name,key in (('submit_policy_confirmation_native.py','external_orchestrator'),('launch-confirmation-native.sh','external_wrapper'),('policy-confirmation-selection.json','selection_bytes'),('policy-calibration-selection-method.json','method_bytes')):
 content=base64.b64decode(p[key]);expected=p[{'external_orchestrator':'external_orchestrator_sha256','external_wrapper':'external_wrapper_sha256','selection_bytes':'selection_sha256','method_bytes':'selection_method_sha256'}[key]]
 assert hashlib.sha256(content).hexdigest()==expected;(b/'preparation'/name).write_bytes(content)
(b/'preparation/launch-confirmation-native.sh').chmod(0o700)
(b/'preparation/source-package-intent.json').write_text(json.dumps({k:v for k,v in p.items() if k not in ('files','external_orchestrator','external_wrapper','selection_bytes','method_bytes')},indent=2))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(b/'repo'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
code='from reproduction.common8.verify_data import verify;import pathlib,sys,json,inspect,hashlib;from types import SimpleNamespace;from ray._private.node import Node;s=inspect.getsource(Node._prepare_socket_file);good="/"+"a"*106;dummy=SimpleNamespace(_sockets_dir="/unused",_make_inc_temp=lambda **kwargs:good);assert sys.platform=="linux" and Node._prepare_socket_file(dummy,None,"plasma_store")==good\ntry:\n dummy._make_inc_temp=lambda **kwargs:"/"+"a"*107;Node._prepare_socket_file(dummy,None,"plasma_store")\nexcept OSError:pass\nelse:raise AssertionError("Installed Ray accepted108-byteAF_UNIX")\nprint(json.dumps(dict(data=verify(pathlib.Path(sys.argv[1])),installed_ray_socket_check_sha256=hashlib.sha256(s.encode()).hexdigest(),installed_limit_tested_bytes=[107,108],scope="Actual installed AF_UNIX method CPU boundary check; no Ray startup")))'
r=subprocess.run([str(b/'env/bin/python'),'-c',code,str(b/'data')],cwd=b/'repo',env=env,capture_output=True,text=True,timeout=70)
assert r.returncode==0,r.stderr
syntax=subprocess.run(['bash','-n',str(b/'preparation/launch-confirmation-native.sh')],capture_output=True,text=True);assert syntax.returncode==0,syntax.stderr
proof=dict(base=str(b),parent_base=str(parent),returncode=r.returncode,stdout=r.stdout,source_commit=p['source_commit'],ray_socket_budget=p['ray_socket_budget'],actual_production_updates=0,production_admission=False,healthy_checkpoint_accepted=False,scope='Canonical six-file target-side copy and CPU structure only')
with (b/'preparation/package-cpu-proof.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(proof))
'''

START_SOURCE_CODE = r'''
import base64,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['export']);assert not b.exists()
with socket.socket() as sock:sock.bind(('10.33.0.13',19874))
b.mkdir(exist_ok=False);s=b/'serve_original_retrieval.py';s.write_bytes(base64.b64decode(p['source']))
assert hashlib.sha256(s.read_bytes()).hexdigest()==p['source_sha256']
with (b/'start-claim.json').open('x') as f:json.dump(dict(export=str(b),source_sha256=p['source_sha256'],created_unix=time.time()),f);f.flush();os.fsync(f.fileno())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='');env.pop('LD_LIBRARY_PATH',None)
with (b/'server.log').open('x') as f:process=subprocess.Popen(['nice','-n','19','python3',str(s)],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text().rsplit(')',1)[1].split()
with (b/'launcher-process.json').open('x') as f:json.dump(dict(pid=process.pid,start_ticks=int(stat[19]),sid=int(stat[3]),uid=os.getuid(),source_sha256=p['source_sha256']),f);f.flush();os.fsync(f.fileno())
h=None
for _ in range(100):
 assert process.poll() is None
 try:h=json.loads((b/'handle.json').read_text());break
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
assert h is not None and h['pid']==process.pid
h.update(export=str(b),source_sha256=p['source_sha256']);print(json.dumps(h))
'''

SUBMIT_CODE = r'''
import hashlib,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['base']);assert not (b/'logs/run-manifest.json').exists()
for name,expected in p['source_sha256'].items():assert hashlib.sha256((b/'repo'/name).read_bytes()).hexdigest()==expected
for name,key in (('submit_policy_confirmation_native.py','external_orchestrator_sha256'),('launch-confirmation-native.sh','external_wrapper_sha256'),('policy-confirmation-selection.json','selection_sha256'),('policy-calibration-selection-method.json','selection_method_sha256')):
 assert hashlib.sha256((b/'preparation'/name).read_bytes()).hexdigest()==p[key]
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for name,size in p['handle']['files'].items():
 with opener.open(urllib.request.Request('http://10.33.0.13:19874/'+name,method='HEAD'),timeout=10) as r:assert r.status==200 and int(r.headers['Content-Length'])==size
args=['sbatch','--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h100:8','--mem=512G','--exclusive','--time=08:00:00','--job-name=search-g1-policy-confirmation','--chdir='+str(b/'repo'),'--output='+str(b/'logs/native-confirmation-%j.log'),'--export=ALL,SEARCH_G1_BASE='+str(b)+',SEARCH_G1_PRODUCTION_BASE='+p['parent_base']+',SEARCH_G1_CALIBRATION_ROLE=policy_safety_calibration_confirmation,SEARCH_G1_SOURCE_COMMIT='+p['source_commit']+',SEARCH_G1_DISCOVERY_BASE='+p['discovery_base'],str(b/'preparation/launch-confirmation-native.sh')]
with (b/'preparation/native-submit-claim.json').open('x') as f:json.dump(dict(argv=args,created_unix=time.time(),base=str(b),selection_sha256=p['selection_sha256'],scope='Independent one-outer confirmation; production0 remains fresh'),f);f.flush();os.fsync(f.fileno())
with (b/'preparation/retrieval-source-handle.json').open('x') as f:json.dump(p['handle'],f);f.flush();os.fsync(f.fileno())
r=subprocess.run(args,capture_output=True,text=True)
proof=dict(base=str(b),parent_base=p['parent_base'],discovery_base=p['discovery_base'],source_commit=p['source_commit'],source_sha256=p['source_sha256'],selection_sha256=p['selection_sha256'],argv=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,submitted_unix=time.time(),actual_production_updates=0)
if r.returncode==0:proof['job_id']=r.stdout.strip().split(';')[0];assert proof['job_id'].isdigit()
with (b/'preparation/native-launch.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
assert r.returncode==0,r.stderr;print(json.dumps(proof))
'''

OBSERVE_CODE = r'''
import json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['base']);out=dict(base=str(b),checked_unix=time.time(),actual_production_updates=0)
launch=b/'preparation/native-launch.json';out['submission_outcome_known']=launch.exists()
job=None
if launch.exists():
 out['launch']=json.loads(launch.read_text());job=out['launch'].get('job_id')
if job:
 r=subprocess.run(['scontrol','show','job','-o',job],capture_output=True,text=True,timeout=15)
 out.update(job_stdout=r.stdout,job_returncode=r.returncode,job_stderr=r.stderr)
for name in ('pipeline-status.json','cleanup-status.json','retrieval-staging-status.json'):
 path=b/'logs'/name
 if path.exists():out[name]=json.loads(path.read_text())
active=[];marker=('SEARCH_G1_BASE='+str(b)).encode()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=os.getuid() or marker not in (proc/'environ').read_bytes().split(b'\0'):continue
  row=(proc/'stat').read_text().rsplit(')',1)[1].split()
  if row[0]!='Z':active.append(dict(pid=int(proc.name),start_ticks=int(row[19]),sid=int(row[3]),uid=proc.stat().st_uid))
 except (FileNotFoundError,ProcessLookupError):pass
out['actual_active_namespace_processes']=active
out['native_submit_claim_exists']=(b/'preparation/native-submit-claim.json').exists()
print(json.dumps(out))
'''

SOURCE_HANDLE_CODE = r'''
import hashlib,json,sys
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['export_base']);h=json.loads((b/'handle.json').read_text())
assert hashlib.sha256((b/'serve_original_retrieval.py').read_bytes()).hexdigest()==p['export_source_sha256']
h.update(export=str(b),source_sha256=p['export_source_sha256']);print(json.dumps(h))
'''

CLEANUP_SOURCE_CODE = r'''
import hashlib,json,os,signal,sys,time
from pathlib import Path
h=json.load(sys.stdin);b=Path(h['export']);source=b/'serve_original_retrieval.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()==h['source_sha256']
path=Path('/proc')/str(h['pid'])
if path.exists():
 descriptor=os.pidfd_open(h['pid'])
 try:
  row=(path/'stat').read_text().rsplit(')',1)[1].split()
  assert (int(row[19]),int(row[3]),path.stat().st_uid)==(h['start_ticks'],h['sid'],h['uid'])
  assert str(source).encode() in (path/'cmdline').read_bytes()
  if row[0]!='Z':
   signal.pidfd_send_signal(descriptor,signal.SIGTERM)
   for _ in range(50):
    if not path.exists() or (path/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
    time.sleep(.1)
  assert not path.exists() or (path/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
 finally:os.close(descriptor)
proof=dict(cleaned=True,handle=h,checked_unix=time.time(),scope='Only this source-SHA/PID-birth/SID/UID service; fixed pidfd signal')
with (b/'cleanup.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(proof))
'''


def load_inputs(prep=PREP):
    """Missing root selection rejects before intent, target copy or service."""
    selection_bytes = (prep / SELECTION_NAME).read_bytes()
    selection = strict_json(selection_bytes)
    launch = strict_json((prep / 'policy-discovery-native-launch.json').read_bytes())
    selection_schema(selection, launch)
    method_binding((prep / METHOD_NAME).read_bytes(), selection)
    cleanup = strict_json((prep / 'policy-discovery-retrieval-source-cleanup.json').read_bytes())
    handle = strict_json((prep / 'policy-discovery-retrieval-export-handle.json').read_bytes())
    require(cleanup['cleaned'] is True and cleanup['handle'] == handle, 'Original discovery source cleanup not verified')
    proof = strict_json((prep / 'calibration-phase-target-cpu-v3-result.json').read_bytes())
    summary = strict_json(proof['stdout'])
    require(proof['returncode'] == proof['shell_syntax_returncode'] == 0 and
            summary['successful'] is True and not summary['skipped'] and not summary['cuda_available'] and
            proof['source_sha256'] == launch['source_sha256'], 'Not the immutable actual target regression proof')
    return selection_bytes, selection, launch


def submit(prep=PREP, repo=REPO):
    intent_path = prep / 'policy-confirmation-native-submit-intent.json'
    require(not intent_path.exists(), 'Attempt already reserved; observe the same namespace, never repeat')
    selection_bytes, selection, launch = load_inputs(prep)
    files = frozen_sources(launch, repo)
    # Actual historical reread happens before any remote mutation/service.
    evidence = remote('h100-8', verification_code(), dict(selection=selection, launch=launch), timeout=180,
                      interpreter=selection['parent_base'] + '/env/bin/python')
    token = uuid.uuid4().hex[:8]
    base = '/mnt/data/shaoshitong/g1c-' + token
    socket_proof = ray_socket_budget(base)
    export = '/dataset/shaoshitong/g1-common8-original-retrieval-export-confirmation-' + token
    orchestrator, wrapper = Path(__file__).read_bytes(), native_wrapper(files)
    source = files['reproduction/common8/assets_export.py'].decode()
    needle = "BASE=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export')"
    require(source.count(needle) == 1, 'Frozen asset exporter changed')
    source = source.replace(needle, 'BASE=Path(' + repr(export) + ')').encode()
    intent = dict(base=base, parent_base=selection['parent_base'], discovery_base=selection['discovery_base'],
                  export_base=export, source_commit=FROZEN_COMMIT, source_sha256=launch['source_sha256'],
                  selection_sha256=sha(selection_bytes), external_orchestrator_sha256=sha(orchestrator),
                  selection_method_sha256=selection['selection_method_sha256'],
                  export_source_sha256=sha(source),
                  ray_socket_budget=socket_proof,
                  external_wrapper_sha256=sha(wrapper), discovery_launch=launch,
                  created_unix=time.time(), source_prerequisite_proof=evidence,
                  actual_production_updates=0, scope='Independent fresh official0 one-outer confirmation, never resume child weights/Adam')
    save(intent_path, intent)
    payload = dict(intent, files={name: base64.b64encode(value).decode() for name, value in files.items()},
                   external_orchestrator=base64.b64encode(orchestrator).decode(),
                   external_wrapper=base64.b64encode(wrapper).decode(), selection_bytes=base64.b64encode(selection_bytes).decode(),
                   method_bytes=base64.b64encode((prep / METHOD_NAME).read_bytes()).decode())
    prepared = remote('h100-8', PREPARE_CODE, payload)
    save(prep / 'policy-confirmation-native-package-proof.json', prepared)
    handle = remote('h100-9', START_SOURCE_CODE, dict(export=export,
                    source=base64.b64encode(source).decode(), source_sha256=sha(source)))
    save(prep / 'policy-confirmation-retrieval-export-handle.json', handle)
    # Recheck the same source, health and fresh parent before the one submission.
    remote('h100-8', verification_code(), dict(selection=selection, launch=launch), timeout=180,
           interpreter=selection['parent_base'] + '/env/bin/python')
    launched = remote('h100-8', SUBMIT_CODE, dict(intent, handle=handle))
    save(prep / 'policy-confirmation-native-launch.json', launched)
    return {key: value for key, value in launched.items() if key not in ('source_sha256', 'argv')}


def cleanup_eligible(observation):
    staging = observation.get('retrieval-staging-status.json', {})
    terminal = observation.get('job_returncode') == 0 and any(
        'JobState=' + value in observation.get('job_stdout', '') for value in
        ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL'))
    return ((staging.get('stage') == 'COMPLETE' and staging.get('accepted') is True) or
            (terminal and not observation['actual_active_namespace_processes']))


def observe(prep=PREP, cleanup=False):
    """Observe one saved namespace, including a lost submission response."""
    intent = strict_json((prep / 'policy-confirmation-native-submit-intent.json').read_bytes())
    observation = remote('h100-8', OBSERVE_CODE, intent)
    # Exclusive timestamped evidence preserves earlier observations/unknowns.
    suffix = str(time.time_ns())
    save(prep / ('policy-confirmation-observation-' + suffix + '.json'), observation)
    if not cleanup:
        return observation
    require(cleanup_eligible(observation), 'Source may still be needed or native/namespace cleanup is unknown')
    handle_path = prep / 'policy-confirmation-retrieval-export-handle.json'
    handle = strict_json(handle_path.read_bytes()) if handle_path.exists() else remote('h100-9', SOURCE_HANDLE_CODE, intent)
    require(handle['export'] == intent['export_base'] and handle['source_sha256'] == intent['export_source_sha256'],
            'Source handle differs from the one reserved namespace')
    proof = remote('h100-9', CLEANUP_SOURCE_CODE, handle)
    save(prep / ('policy-confirmation-retrieval-source-cleanup-' + suffix + '.json'), proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--submit', action='store_true')
    group.add_argument('--native-bind', action='store_true')
    group.add_argument('--check-selection', action='store_true')
    group.add_argument('--observe', action='store_true')
    group.add_argument('--cleanup-source', action='store_true')
    parser.add_argument('--base')
    args = parser.parse_args()
    if args.native_bind:
        require(args.base is not None, 'Explicit native child base required')
        result = native_bind(args.base)
    elif args.submit:
        require(args.base is None, 'Submission reserves a new namespace; do not reuse a base')
        result = submit()
    elif args.observe or args.cleanup_source:
        require(args.base is None, 'Observation uses the same reserved handle')
        result = observe(cleanup=args.cleanup_source)
    else:
        require(args.base is None, 'Local check does not operate a target')
        content, selection, launch = load_inputs()
        frozen_sources(launch)
        result = dict(role='local_confirmation_selection_structure_check', selection_sha256=sha(content),
                      source_commit=FROZEN_COMMIT, structural_verified=True,
                      actual_production_updates=0, production_admission=False,
                      healthy_checkpoint_accepted=False, physical_execution_authenticated=False,
                      scope='No remote call, service, job submission or actual evidence acceptance')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
