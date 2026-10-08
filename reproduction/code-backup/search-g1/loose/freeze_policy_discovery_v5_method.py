"""Prepare/freeze root-authored v5 prospective criteria, never select a tail.

Importing or checking this module performs no remote action and writes no
preregistration. Root must author and SHA-seal the actual preregistration before
the new native submission. Every future source/CPU/count/launch/zero identity is
an explicit argument; historical v4 identities cannot substitute for them.

A future --freeze reserves one durable local intent, then performs one native
metadata reread and O_EXCL write in each namespace. An unknown/partial outcome
blocks replay. No GPU/model/Adam tensor audit or healthy/production admission is
claimed. Runtime source, source methods, previous jobs and old files are inert.
"""
import argparse
import base64
import copy
import hashlib
import inspect
import json
import math
import os
import stat
import sys
import time
from pathlib import Path

PREP = Path('/Users/shaoshitong/Documents/Search-G1/preparation')
sys.path.insert(0, str(PREP))
import submit_policy_discovery_v5_native as discovery

common = discovery.common
REPO = discovery.REPO
ROLE = 'root_policy_calibration_method_preregistration_before_native_submission'
METHOD_ROLE = 'root_explicit_policy_calibration_selection_method'
NAMES = dict(intent='policy-calibration-v5-method-freeze-intent.json',
             method='policy-calibration-v5-selection-method.json',
             receipt='policy-calibration-v5-method-freeze-receipt.json')
UNSTARTED = dict(training_claim_exists=False, optimizer_evidence_paths=[], train_log_exists=False,
                checkpoint_entries=[], actual_updates_exists=False, segment_exists=False, segment_complete_exists=False)
CANDIDATES = dict(dual_clip=3.0, epsilon=0.2, target_old_policy_kl=0.01, kl_log_bound=30.0)
SAMPLING = dict(temperature=1.0, top_p=1.0,
    denominator='actual_selected_sampling_log_probs_after_all_token_context_and_0.05_gate')
TAIL_SELECTION = dict(
    method='Maximum actual global negative_tail_weight_fraction across all ordered pre/post mini checks of the complete discovery outer',
    required_global_observations=40,
    duplicate_rank_global_reports='Verify all eight copies agree and count each ordinal observation once',
    data_requirement='Complete actual 20 Adam writes and 40 allowed finite checks/rank, actual finite checkpoint1 parameter delta, paired ordered512 before/after, completed native and owned cleanup',
    arbitrary_margin_or_floor=False, fraction_with_zero_negative_mass=0.0,
    confirmation='Fresh official0 independent native job, same initial probe heads/encoder/reward math and original recipe; every actual check must remain within the frozen candidate envelope and other guards',
    scope='Empirical monitored32 trajectories/one discovery outer; no weighted quantile or population-risk claim')
HEALTH_CRITERIA = dict(method='paired_fixed512_absolute_allowed_drop',
                      allowed_drop=dict(weighted_em=0.0, weighted_f1=0.0, valid_action_rate=0.0))


def _sha_file(path):
    return common.sha(Path(path).read_bytes())


def _read_sealed(path, digest, label):
    common.require(path is not None and discovery._digest(digest), 'Explicit original ' + label + ' bytes/SHA are required')
    raw = Path(path).read_bytes()
    common.require(common.sha(raw) == digest, 'Original ' + label + ' SHA differs')
    return raw, common.strict_json(raw)


def validate_preregistration(raw, *, expected_sha256, source_commit, target_cpu_proof_sha256, expected_cpu_tests):
    """Validate explicit root-authored bytes; never invent/write its contract."""
    discovery.check_commit(source_commit)
    common.require(discovery._digest(expected_sha256) and isinstance(raw, bytes) and common.sha(raw) == expected_sha256 and
                   discovery._digest(target_cpu_proof_sha256) and type(expected_cpu_tests) is int and expected_cpu_tests > 0,
                   'Explicit new preregistration/CPU SHA/count are required')
    value = common.strict_json(raw)
    required = {'role','source_commit','created_unix','health_rule_document','negative_tail_selection','fixed_candidates',
                'negative_tail_limit','sampling_distribution','target_cpu_proof_sha256','actual_target_cpu_tests',
                'actual_discovery_observations_for_this_source','production_admission','healthy_checkpoint_accepted',
                'prospective_identity_binding_complete','scope'}
    common.require(set(value) == required and value['role'] == ROLE and value['source_commit'] == source_commit and
        value['target_cpu_proof_sha256'] == target_cpu_proof_sha256 and type(value['actual_target_cpu_tests']) is int and
        value['actual_target_cpu_tests'] == expected_cpu_tests and discovery._number(value['created_unix']) and
        value['created_unix'] > 0 and type(value['actual_discovery_observations_for_this_source']) is int and
        value['actual_discovery_observations_for_this_source'] == 0 and value['negative_tail_limit'] is None and
        value['production_admission'] is False and value['healthy_checkpoint_accepted'] is False and
        value['prospective_identity_binding_complete'] is False and isinstance(value['scope'],str) and bool(value['scope'].strip()),
        'Root preregistration role/source/zero state/closed schema differs')
    rule = value['health_rule_document']
    common.require(isinstance(rule,dict) and set(rule) == {'role','provenance','criteria'} and
        rule['role'] == 'policy_safety_health_rule' and isinstance(rule['provenance'],dict) and bool(rule['provenance']) and
        rule['provenance'].get('not_a_population_statistical_guarantee') is True and
        common.canonical(rule['criteria']) == common.canonical(HEALTH_CRITERIA),
        'Fixed paired512 EM/F1/valid-action drop0 criterion or provenance differs')
    common.require(common.canonical(value['negative_tail_selection']) == common.canonical(TAIL_SELECTION) and
        common.canonical(value['fixed_candidates']) == common.canonical(CANDIDATES) and
        common.canonical(value['sampling_distribution']) == common.canonical(SAMPLING),
        'Frozen40 empirical monitored32 max/no margin/floor or c3/eps.2/KL.01/positivebound30 mathematics differs')
    return value


def preregistration_inputs(*, source_commit=None, target_cpu_proof_path=None, target_cpu_proof_sha256=None,
                          expected_cpu_tests=None, preregistration_path=None, preregistration_sha256=None, repo=REPO):
    """Pure local new Git/blob + original CPU + explicit criteria verification."""
    discovery.check_commit(source_commit)
    common.require(target_cpu_proof_path is not None and discovery._digest(target_cpu_proof_sha256) and
        preregistration_path is not None and discovery._digest(preregistration_sha256) and
        type(expected_cpu_tests) is int and expected_cpu_tests > 0, 'Every new source/CPU/count/root criterion input is explicit')
    prereg_raw, _ = _read_sealed(preregistration_path, preregistration_sha256, 'root preregistration')
    prereg = validate_preregistration(prereg_raw, expected_sha256=preregistration_sha256, source_commit=source_commit,
        target_cpu_proof_sha256=target_cpu_proof_sha256, expected_cpu_tests=expected_cpu_tests)
    cpu_raw, proof = _read_sealed(target_cpu_proof_path, target_cpu_proof_sha256, 'target CPU proof')
    old_raw = (PREP / discovery.CPU_OLD_NAME).read_bytes(); old = common.strict_json(old_raw)
    common.require(common.sha(old_raw) == '5399795ab3c6b50400d378493aecede6307663672e32ebc38a13fc201ac64e35' and
        target_cpu_proof_sha256 != common.sha(old_raw), 'An original historical target proof is not a new runtime proof')
    mapping = discovery.validate_cpu(proof, old, expected_cpu_tests)
    files = discovery.committed_sources(source_commit, mapping, repo)
    discovery.new_recipe(files)
    common.require(prereg['created_unix'] >= proof['checked_unix'], 'Preregistration predates the actual admitted source proof')
    return dict(prereg=prereg, prereg_bytes_base64=base64.b64encode(prereg_raw).decode(), prereg_sha256=preregistration_sha256,
        source_commit=source_commit, source_sha256=mapping, source_map_sha256=common.sha(common.canonical(mapping)),
        target_cpu_proof_sha256=target_cpu_proof_sha256, actual_target_cpu_tests=expected_cpu_tests)


def validate_zero_observation(launch, observed, *, prereg, source_commit, source_sha256,
                              target_cpu_proof_sha256, expected_cpu_tests):
    """Authenticate a captured actual RUNNING birth and strict zero metadata."""
    common.require(type(launch['returncode']) is int and launch['returncode'] == 0 and
        isinstance(launch['job_id'],str) and launch['job_id'].isdigit() and launch['stdout'].strip().split(';')[0] == launch['job_id'] and
        discovery._number(launch['submitted_unix']) and launch['submitted_unix'] > prereg['created_unix'] and
        launch['source_commit'] == source_commit and launch['source_sha256'] == source_sha256 and
        launch['target_cpu_proof_sha256'] == target_cpu_proof_sha256 and
        type(launch['actual_target_cpu_tests']) is int and launch['actual_target_cpu_tests'] == expected_cpu_tests,
        'Actual new successful sbatch/source/CPU identity or prospective preregistration time differs')
    child, parent = Path(launch['base']), Path(launch['parent_base'])
    import re
    common.require(re.fullmatch(r'/mnt/data/shaoshitong/g1d-[0-9a-f]{8}',str(child)) is not None and
        str(parent) == str(child).replace('/g1d-','/g1p-') and str(child) == launch['base'] and
        str(parent) == launch['parent_base'] and str(child) != discovery.OLD_BASE,
        'Actual independent short v5 child/parent namespaces differ')
    for name in (str(child),str(parent)): common.ray_socket_budget(name)
    snapshot = observed['run-manifest.json']; production = observed['parent_run_manifest']
    common.require(discovery._digest(snapshot['run_id'],32) and discovery._digest(production['run_id'],32) and
        snapshot['run_id'] != production['run_id'] and snapshot['parent_production_run_id'] == production['run_id'] and
        snapshot['source_commit'] == production['source_commit'] == source_commit and
        snapshot['policy_calibration_role'] == 'policy_safety_calibration_discovery' and
        type(snapshot['completed_updates']) is int and snapshot['completed_updates'] == 0 and
        type(production['completed_updates']) is int and production['completed_updates'] == 0 and
        discovery._digest(observed['child_manifest_sha256']) and discovery._digest(production['manifest_sha256']) and
        observed['child_manifest_slurm_job_id'] == launch['job_id'], 'Actual independent fresh0 child/parent manifest identity differs')
    discovery._snapshot_binding(observed, launch, snapshot['run_id'], production['run_id'])
    common.require(type(observed['job_returncode']) is int and observed['job_returncode'] == 0 and observed['job_stderr'] == '' and
        observed['source_hashes_match'] is True and observed['child_actual_unstarted_verified'] is True and
        observed['parent_actual_unstarted_verified'] is True and observed['child_actual_unstarted'] == UNSTARTED and
        observed['parent_actual_unstarted'] == UNSTARTED and observed['actual_updates'] == [] and
        observed['actual_behavior_gates'] == [], 'Actual source/both strict unstarted/update evidence differs')
    status = observed['pipeline-status.json']
    common.require(status['run_id'] == snapshot['run_id'] and status['stage'] not in
        ('BOUNDED_POLICY_CALIBRATION_ONE_OUTER','FAILED','OWNED_CLEANUP_FAILED'), 'Captured zero stage is after admission to the training attempt')
    job = discovery._job_fields(observed['job_stdout'])
    command = str(child/'repo/reproduction/common8/launch_calibration_native.sh');workdir = str(child/'repo')
    common.require(job['JobId'] == launch['job_id'] and job['JobState'] == 'RUNNING' and job['ExitCode'] == '0:0' and
        job['NodeList'] == job['BatchHost'] == 'h100-8' and job['NumNodes'] == '1' and job['MinMemoryNode'] == '512G' and
        job['TresPerNode'] == 'gres:gpu:h100:8' and job['Restarts'] == '0' and job['Command'] == command and
        job['WorkDir'] == workdir, 'Captured actual native RUNNING eight-H100/512/command/job identity differs')
    argv = launch['argv']
    common.require(isinstance(argv,list) and all(isinstance(value,str) for value in argv) and argv[0] == 'sbatch' and
        argv[-1] == command and '--chdir='+workdir in argv and '--exclusive' in argv and
        '--gres=gpu:h100:8' in argv and '--mem=512G' in argv and '--nodes=1' in argv and
        '--job-name=search-g1-policy-discovery-v5' in argv, 'New native resource/once-launch arguments differ')
    output = [value.split('=',1)[1] for value in argv if value.startswith('--output=')]
    common.require(len(output) == 1 and job['StdOut'] == job['StdErr'] == output[0].replace('%j',launch['job_id']),
        'Actual new native output identity differs')
    common.require(discovery._scheduler_time(job['SubmitTime']) <= discovery._scheduler_time(job['StartTime']),
        'Actual job start predates submission')
    return dict(child_run_id=snapshot['run_id'],parent_run_id=production['run_id'],native_job_id=launch['job_id'],
                scheduler_fields=job,production_admission=False,healthy_checkpoint_accepted=False)


def freeze_inputs(*, launch_path=None, launch_sha256=None, observation_path=None, observation_sha256=None,
                  prep=PREP, **kwargs):
    common.require(launch_path is not None and discovery._digest(launch_sha256) and observation_path is not None and
        discovery._digest(observation_sha256), 'Explicit actual new launch and raw zero-observation SHA/path are required')
    inputs = preregistration_inputs(**kwargs)
    launch_raw, launch = _read_sealed(launch_path,launch_sha256,'new native launch')
    observed_raw, observed = _read_sealed(observation_path,observation_sha256,'actual zero observation')
    common.require(Path(launch_path).resolve() == Path(prep).resolve()/discovery.NAMES['launch'] and
        Path(observation_path).resolve().parent == Path(prep).resolve() and
        Path(observation_path).name.startswith('policy-discovery-v5-observation-'), 'New launch/zero files belong to another generation')
    identity = validate_zero_observation(launch,observed,prereg=inputs['prereg'],source_commit=inputs['source_commit'],
        source_sha256=inputs['source_sha256'],target_cpu_proof_sha256=inputs['target_cpu_proof_sha256'],
        expected_cpu_tests=inputs['actual_target_cpu_tests'])
    return dict(inputs,launch=launch,launch_sha256=launch_sha256,observed=observed,observation_sha256=observation_sha256,
                actual_zero_identity=identity)


def _exclusive_bytes(path, content):
    """Private durable O_EXCL; no overwrite/recovery of a partial prior write."""
    path = Path(path)
    common.require(path.parent.resolve() == path.parent and path.parent.is_dir() and path.parent.stat().st_uid == os.getuid(),
        'Freeze parent directory is not the canonical current-user directory')
    fd = os.open(str(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(content);stream.flush();os.fsync(stream.fileno())
    fd = os.open(str(path.parent),os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _native_freeze(payload):
    """Metadata-only actual reread before O_EXCL; dependencies are authenticated first."""
    import subprocess
    import re
    launch=payload['launch'];observed=payload['observed'];prereg=payload['prereg']
    base=Path(launch['base']);parent=Path(launch['parent_base']);mapping=launch['source_sha256']
    common.require(base.resolve()==base and parent.resolve()==parent and len(mapping)==353,
        'Actual namespaces/source map differs')
    # Authenticate source bytes and exact source set before importing its helpers.
    for namespace in (base,parent):
        discovery._verify_original_mapping(namespace,mapping)
        repo=namespace/'repo'
        actual={str(path.relative_to(repo)) for path in repo.rglob('*') if path.is_file() and
            (path.suffix in ('.py','.sh','.yaml','.yml','.json') or
             str(path.relative_to(repo)) in ('verl/version/version','verl/single_controller/version/version')) and
            not {'.git','__pycache__','evidence'}.intersection(path.relative_to(repo).parts)}
        common.require(actual==set(mapping),'Pretraining actual source contains missing/extra/unknown files')
    sys.path.insert(0,str(base/'repo'))
    from reproduction.common8.freeze_calibration_plan import _unstarted
    from reproduction.common8.policy_calibration_plan import _identity,canonical_bytes
    command=subprocess.run(['scontrol','show','job','-o',launch['job_id']],capture_output=True,text=True,timeout=15)
    common.require(type(command.returncode) is int and command.returncode==0 and command.stderr=='',
        'Actual current native query failed; unknown is not an unstarted job')
    current=discovery._job_fields(command.stdout);saved=payload['actual_zero_identity']['scheduler_fields']
    common.require(current['JobState']=='RUNNING' and current['ExitCode']=='0:0' and
        all(current[key]==saved[key] for key in discovery.BIRTH_FIELDS),
        'Actual current native birth/resource identity differs from original zero observation')
    child_raw=(base/'logs/run-manifest.json').read_bytes();run=common.strict_json(child_raw)
    parent_raw=(parent/'logs/run-manifest.json').read_bytes();production=common.strict_json(parent_raw)
    common.require(common.sha(child_raw)==observed['child_manifest_sha256'] and
        common.sha(parent_raw)==observed['parent_run_manifest']['manifest_sha256'] and
        run['run_id']==payload['actual_zero_identity']['child_run_id'] and production['run_id']==payload['actual_zero_identity']['parent_run_id'] and
        run['run_id']!=production['run_id'] and run['parent_production_run_id']==production['run_id'] and
        run['policy_calibration_slurm_job_id']==launch['job_id'] and run['policy_calibration_role']=='policy_safety_calibration_discovery' and
        run['source_commit']==production['source_commit']==launch['source_commit']==prereg['source_commit'] and
        run['source_sha256']==production['source_sha256']==mapping and
        canonical_bytes(_identity(run))==canonical_bytes(_identity(production)),
        'Actual current manifests/source/model/data/math identity differs from the captured fresh0 identities')
    for namespace,manifest in ((base,run),(parent,production)):
        common.require(Path(manifest['train_log'])==namespace/'logs/search-g1-common8-training.log' and
            Path(manifest['checkpoint_dir'])==namespace/'checkpoints/search-g1-common8',
            'Strict unstarted paths escape actual owned namespace')
        _unstarted(namespace,manifest)
        logs=namespace/'logs'
        common.require(not list(logs.glob('behavior-probability-outer-*.json')),
            'Actual behavior gate already exists before method freeze')
    status=common.strict_json((base/'logs/pipeline-status.json').read_bytes())
    common.require(status['run_id']==run['run_id'] and status['stage'] not in
        ('BOUNDED_POLICY_CALIBRATION_ONE_OUTER','FAILED','OWNED_CLEANUP_FAILED'),
        'Actual pipeline reached the training attempt/failure before prospective freeze')
    current_unix=time.time()
    common.require(prereg['created_unix']<launch['submitted_unix']<=observed['checked_unix']<=
        observed['observation_finished_unix']<=current_unix,'Actual prospective freeze/capture timeline differs')
    method=dict(role=METHOD_ROLE,source_commit=launch['source_commit'],parent_production_run_id=production['run_id'],
        discovery_run_id=run['run_id'],frozen_unix=current_unix,before_completed_discovery_update=True,
        before_any_optimizer_write=True,health_rule_document=prereg['health_rule_document'],
        negative_tail_selection=prereg['negative_tail_selection'],fixed_candidates=prereg['fixed_candidates'],
        negative_tail_limit=None,production_admission=False,healthy_checkpoint_accepted=False,
        sampling_distribution=prereg['sampling_distribution'],prospective_binding=dict(
            launch_sha256=payload['launch_sha256'],native_job_id=launch['job_id'],discovery_base=str(base),parent_base=str(parent),
            child_manifest_sha256=common.sha(child_raw),parent_manifest_sha256=common.sha(parent_raw),
            zero_update_observation_sha256=payload['observation_sha256'],preregistration_sha256=payload['prereg_sha256'],
            preregistered_unix=prereg['created_unix'],target_cpu_proof_sha256=payload['target_cpu_proof_sha256'],
            actual_target_cpu_tests=payload['actual_target_cpu_tests'],source_map_sha256=payload['source_map_sha256'],
            parent_actual_unstarted_verified=True,child_actual_unstarted_verified=True,actual_stage=status['stage'],
            scope='Actual source/manifests/allocation and strict no trainer claim/log/Adam ledger/update/checkpoint; no model/GPU/healthy admission'))
    content=(json.dumps(method,indent=2,allow_nan=False)+'\n').encode()
    paths=[namespace/'preparation'/NAMES['method'] for namespace in (parent,base)]
    common.require(all(not path.exists() and not path.is_symlink() for path in paths),
        'Method already exists/partial; inspect the same handle and never replay freeze')
    for path in paths:_exclusive_bytes(path,content)
    return dict(method_bytes_base64=base64.b64encode(content).decode(),method_sha256=common.sha(content),
        actual_current_zero_write_verified=True,checked_unix=time.time(),role='prospective_method_identity_freeze_not_admission',
        production_admission=False,healthy_checkpoint_accepted=False,physical_model_execution_authenticated=False,
        scope='Actual metadata and source only; no tensor inspection, tail choice or production permission')


def native_code():
    """Embed only authenticated metadata functions; imports perform no GPU work."""
    helpers='\n'.join(inspect.getsource(fn) for fn in (common.canonical,common.sha,common.require,common.strict_json))
    source='\n'.join(inspect.getsource(fn) for fn in
        (discovery._digest,discovery._verify_metadata_file,discovery._verify_original_mapping,discovery._job_fields))
    own='\n'.join(inspect.getsource(fn) for fn in (_exclusive_bytes,_native_freeze))
    constants='\n'.join(name+'='+repr(globals()[name]) for name in ('NAMES','METHOD_ROLE'))
    return ('import base64,hashlib,json,os,stat,sys,time,re\nfrom pathlib import Path\nfrom types import SimpleNamespace\n'+helpers+
        '\ncommon=SimpleNamespace(canonical=canonical,sha=sha,require=require,strict_json=strict_json)\n'+source+
        '\ndiscovery=SimpleNamespace(_digest=_digest,_verify_metadata_file=_verify_metadata_file,'
        '_verify_original_mapping=_verify_original_mapping,_job_fields=_job_fields,BIRTH_FIELDS='+repr(discovery.BIRTH_FIELDS)+')\n'+
        constants+'\n'+own+'\np=json.load(sys.stdin);print(json.dumps(_native_freeze(p)))\n')


def verify_receipt(receipt,inputs):
    raw=base64.b64decode(receipt['method_bytes_base64'],validate=True);value=common.strict_json(raw)
    common.require(receipt['method_sha256']==common.sha(raw) and receipt['role']=='prospective_method_identity_freeze_not_admission' and
        receipt['actual_current_zero_write_verified'] is True and receipt['production_admission'] is False and
        receipt['healthy_checkpoint_accepted'] is False and receipt['physical_model_execution_authenticated'] is False,
        'Actual freeze receipt lacks the sealed non-admission method bytes')
    binding=value['prospective_binding'];identity=inputs['actual_zero_identity']
    common.require(value['role']==METHOD_ROLE and value['source_commit']==inputs['source_commit'] and
        value['parent_production_run_id']==identity['parent_run_id'] and value['discovery_run_id']==identity['child_run_id'] and
        value['negative_tail_limit'] is None and value['before_completed_discovery_update'] is True and
        value['before_any_optimizer_write'] is True and value['production_admission'] is False and value['healthy_checkpoint_accepted'] is False and
        discovery._number(value['frozen_unix']) and value['frozen_unix']>=inputs['observed']['observation_finished_unix'] and
        discovery._number(receipt['checked_unix']) and receipt['checked_unix']>=value['frozen_unix'],
        'Returned actual new method identity or prospective time differs')
    for name in ('health_rule_document','negative_tail_selection','fixed_candidates','sampling_distribution'):
        common.require(common.canonical(value[name])==common.canonical(inputs['prereg'][name]),'Returned root criterion bytes differ: '+name)
    expected=dict(launch_sha256=inputs['launch_sha256'],native_job_id=identity['native_job_id'],
        discovery_base=inputs['launch']['base'],parent_base=inputs['launch']['parent_base'],
        child_manifest_sha256=inputs['observed']['child_manifest_sha256'],parent_manifest_sha256=inputs['observed']['parent_run_manifest']['manifest_sha256'],
        zero_update_observation_sha256=inputs['observation_sha256'],preregistration_sha256=inputs['prereg_sha256'],
        target_cpu_proof_sha256=inputs['target_cpu_proof_sha256'],actual_target_cpu_tests=inputs['actual_target_cpu_tests'],source_map_sha256=inputs['source_map_sha256'],
        preregistered_unix=inputs['prereg']['created_unix'])
    common.require(all(binding[name]==value for name,value in expected.items()) and
        binding['parent_actual_unstarted_verified'] is True and binding['child_actual_unstarted_verified'] is True,
        'Returned actual source/CPU/launch/zero/preregistration binding differs')
    return raw


def freeze(*,prep=PREP,**kwargs):
    """One explicitly authorized future attempt; unknown/partial blocks replay."""
    prep=Path(prep).resolve()
    common.require(not any((prep/name).exists() or (prep/name).is_symlink() for name in NAMES.values()),
        'v5 freeze already reserved; inspect original handle and never repeat')
    inputs=freeze_inputs(prep=prep,**kwargs)
    intent=dict(inputs,created_unix=time.time(),external_orchestrator_sha256=_sha_file(__file__),
        external_discovery_helper_sha256=_sha_file(discovery.__file__),external_common_helper_sha256=_sha_file(common.__file__),
        production_admission=False,healthy_checkpoint_accepted=False,scope='One metadata reread/freeze attempt; no health or threshold choice')
    _exclusive_bytes(prep/NAMES['intent'],(json.dumps(intent,indent=2,allow_nan=False)+'\n').encode())
    receipt=common.remote('h100-8',native_code(),inputs,timeout=60,interpreter=inputs['launch']['base']+'/env/bin/python')
    content=verify_receipt(receipt,inputs)
    _exclusive_bytes(prep/NAMES['method'],content)
    public={key:value for key,value in receipt.items() if key!='method_bytes_base64'}
    _exclusive_bytes(prep/NAMES['receipt'],(json.dumps(public,indent=2,allow_nan=False)+'\n').encode())
    return public


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-preregistration',action='store_true');group.add_argument('--freeze',action='store_true')
    parser.add_argument('--source-commit',required=True);parser.add_argument('--target-cpu-proof',required=True)
    parser.add_argument('--target-cpu-proof-sha256',required=True);parser.add_argument('--expected-cpu-tests',required=True,type=int)
    parser.add_argument('--preregistration',required=True);parser.add_argument('--preregistration-sha256',required=True)
    parser.add_argument('--launch');parser.add_argument('--launch-sha256');parser.add_argument('--observation');parser.add_argument('--observation-sha256')
    args=parser.parse_args();kw=dict(source_commit=args.source_commit,target_cpu_proof_path=args.target_cpu_proof,
        target_cpu_proof_sha256=args.target_cpu_proof_sha256,expected_cpu_tests=args.expected_cpu_tests,
        preregistration_path=args.preregistration,preregistration_sha256=args.preregistration_sha256)
    if args.check_preregistration:
        common.require(not any((args.launch,args.launch_sha256,args.observation,args.observation_sha256)),
            'Preregistration check has no actual native job/zero binding or remote call')
        checked=preregistration_inputs(**kw)
        result=dict(role='v5_root_preregistration_local_structure_check',source_commit=checked['source_commit'],
            source_map_sha256=checked['source_map_sha256'],preregistration_sha256=checked['prereg_sha256'],
            actual_target_cpu_tests=checked['actual_target_cpu_tests'],production_admission=False,healthy_checkpoint_accepted=False,
            physical_execution_authenticated=False,scope='Original CPU/Git/criteria bytes only; no preregistration write or native call')
    else:
        common.require(all((args.launch,args.launch_sha256,args.observation,args.observation_sha256)),
            'Freeze requires explicit actual new launch/zero original bytes and SHA')
        result=freeze(**kw,launch_path=args.launch,launch_sha256=args.launch_sha256,
                      observation_path=args.observation,observation_sha256=args.observation_sha256)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
