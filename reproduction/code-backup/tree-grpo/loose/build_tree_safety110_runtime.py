"""Build only a new code/metadata package; never upload, submit or change STATUS."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

WORK = Path(__file__).resolve().parent
TEMPLATE = WORK/'tree-safety110-runtime'
OLD = WORK/'tree-recovery-runtime'
EXPECTED_PARENT_SHA = 'e59d8011df3b1248c0c548afeb9d4ad6e564c5f9b0f84f2147bb3f99ad182366'
EXPECTED_RECEIPT_SHA = '2296c2f3e3a6e288845f0106c84e47d0f4a2c6055c144b555d0656fad19c6a76'
EXPECTED_NATIVE_SHA = '104086becfaf0177ab8d030fc0b13d1a166d134050fd995ecca6dfde7c0b5959'
SAFETY_KEYS = ('target_old_policy_kl', 'negative_tail_limit', 'initial_old_logprob_max_abs_delta')
SAMPLING_OVERRIDES = ['actor_rollout_ref.rollout.sampling_consistency.enabled=true',
                      'actor_rollout_ref.rollout.sampling_consistency.enforce=false',
                      'actor_rollout_ref.rollout.sampling_consistency.max_abs_delta=0.01']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()

def validate_inputs(value, source):
    require(value['status'] == 'SOURCE_INTERFACE_REVIEWED_FOR_ZERO_WRITE_CALIBRATION',
            'Draft inputs cannot freeze: source hook and ready schema must first be reviewed')
    require(value['stage'] == 'calibration110', 'Only zero-write healthy110 calibration is supported')
    commit = value['source_commit']
    require(isinstance(commit, str) and len(commit) == 40 and git(source, 'rev-parse', 'HEAD') == commit,
            'Owner must provide exact current source commit')
    require(not git(source, 'status', '--porcelain'), 'Cannot freeze a dirty candidate source')
    thresholds = value['policy_safety_thresholds']
    require(set(thresholds) == set(SAFETY_KEYS), 'Missing explicit safety trial inputs')
    for key in SAFETY_KEYS:
        number = thresholds[key]
        require(isinstance(number, (int, float)) and not isinstance(number, bool) and
                math.isfinite(number) and float(number) == .01,
                'This calibration package requires explicit 0.01 trial values, not changed training thresholds')
    contract = value['probability_calibration']
    require(contract['source_interface_reviewed'] is True, 'First-real-outer zero-write hook is unreviewed')
    require(contract['environment'] == {'TREE_PPO_CALIBRATION_ONLY': '1'}, 'Calibration-only environment changed')
    require(contract['configuration_overrides'] == SAMPLING_OVERRIDES, 'Actual sampling diagnostic trial inputs changed')
    require(contract['ready_relative_path'] == 'output/probability-calibration/ready.json' and
            contract['ready_status'] == 'CALIBRATION_OBSERVED_ZERO_WRITES', 'Ready interface changed')
    source_contract = contract['source_contract_files']
    required = {'search_r1/baselines/sampling_consistency.py',
                'search_r1/baselines/policy_safety_runtime.py',
                'verl/workers/fsdp_workers.py', 'verl/trainer/ppo/ray_trainer.py',
                'verl/workers/rollout/vllm_rollout/vllm_rollout.py',
                'verl/trainer/config/ppo_tree_safety300.yaml'}
    require(isinstance(source_contract, dict) and required.issubset(source_contract),
            'Pin all reviewed sampling/calibration module, measure runtime, worker, driver, rollout and config source hashes')
    for relative, expected in source_contract.items():
        parts = PurePosixPath(relative)
        require(not parts.is_absolute() and '..' not in parts.parts, 'Invalid source contract path')
        require(sha(source/relative) == expected, 'Source contract hash changed: '+relative)
    return value

def freeze(value, source, destination):
    require(not destination.exists(), 'Build destination exists; preserve earlier frozen candidates')
    parent = json.loads((TEMPLATE/'HEALTHY110_PARENT.json').read_bytes())
    require(parent['sha256'] == EXPECTED_PARENT_SHA and parent['identity']['completed_step'] == 110,
            'Accepted healthy110 parent changed')
    require(sha(TEMPLATE/'INCREMENTAL110_RECEIPT.json') == EXPECTED_RECEIPT_SHA,
            'Independent healthy110 receipt changed')
    require(sha(TEMPLATE/'ACTUAL3262_RECEIPT.json') == EXPECTED_NATIVE_SHA,
            'Independent synthetic safety receipt changed')
    destination.mkdir(parents=True)
    for path in TEMPLATE.iterdir():
        require(path.is_file(), 'Template unexpectedly contains a mutable directory: '+str(path))
        if path.name not in ('manifest.json', 'PRE_DISPATCH_RECEIPT.json',
                             'BUILD_INPUTS.template.json', 'test_contract_schema.py'):
            target_name = 'TEMPLATE_PREPARATION_EVIDENCE.json' if path.name == 'DRAFT_STATUS.json' else path.name
            shutil.copy2(path, destination/target_name)
    readme = destination/'README.md'
    readme.write_text('This is a newly frozen code/metadata candidate with a manifest and source snapshot. '
                      'It remains unsubmitted and requires its own independent pre-dispatch receipt. '
                      'The following documentation describes the source template; its draft-state statements '
                      'are provenance, not current package state.\n\n'+readme.read_text())
    archive_destination = destination/'source/repo'
    archive_destination.mkdir(parents=True)
    with tempfile.TemporaryFile() as archive:
        subprocess.run(['git', '-C', str(source), 'archive', '--format=tar', value['source_commit']],
                       stdout=archive, check=True)
        require(archive.tell() < 32*1024**2, 'Unexpected source archive size; assets must stay on h100-9')
        archive.seek(0)
        with tarfile.open(fileobj=archive) as bundle:
            members = bundle.getmembers()
            for member in members:
                part = PurePosixPath(member.name)
                require(not part.is_absolute() and '..' not in part.parts and
                        (member.isfile() or member.isdir()), 'Unexpected Git archive entry')
            bundle.extractall(archive_destination)
    (destination/'BUILD_INPUTS.json').write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    old = json.loads((OLD/'manifest.json').read_bytes())
    overrides = [f'actor_rollout_ref.actor.policy_safety.{key}={value["policy_safety_thresholds"][key]}'
                 for key in SAFETY_KEYS] + SAMPLING_OVERRIDES
    files = {str(path.relative_to(destination)): sha(path) for path in sorted(destination.rglob('*'))
             if path.is_file() and path != destination/'manifest.json'}
    package = dict(code_commit=value['source_commit'], stage='calibration110',
                   scope='FROZEN_PENDING_INDEPENDENT_HEALTHY110_PRE_DISPATCH_REVIEW',
                   common_data_identity=old['common_data_identity'], model_identity=old['model_identity'],
                   external_files=old['external_files'], files=files,
                   parent_manifest={key: parent[key] for key in ('path','sha256','identity')},
                   healthy110_metadata=parent, configuration_overrides=overrides,
                   policy_safety_thresholds=value['policy_safety_thresholds'],
                   probability_calibration=value['probability_calibration'],
                   parent_reviews={'INCREMENTAL110_RECEIPT.json': EXPECTED_RECEIPT_SHA},
                   synthetic_safety_review={'ACTUAL3262_RECEIPT.json': EXPECTED_NATIVE_SHA,
                                           'scope': 'Accepted source0980 synthetic8 only; new real3B calibration hook needs independent review'},
                   actual_3b_training_accepted=False, calibrated_training_thresholds_accepted=False,
                   long_training_ready=False,
                   historical_failure_preserved=dict(job_id='3234', completed_outer=128,
                                                     selected_healthy_branch_parent=110, failed_outer=129))
    (destination/'manifest.json').write_text(json.dumps(package, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status='FROZEN_ZERO_WRITE_CALIBRATION_CANDIDATE_UNREVIEWED',
                         destination=str(destination), manifest_sha256=sha(destination/'manifest.json'),
                         source_commit=value['source_commit'], files=len(files),
                         uploaded=False, submitted=False, long_training_ready=False), sort_keys=True))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=WORK/'tree-ppo-three-guards')
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--validate-inputs-only', action='store_true')
    args = parser.parse_args()
    value = validate_inputs(json.loads(args.inputs.read_bytes()), args.source.resolve())
    if args.validate_inputs_only:
        print(json.dumps(dict(status='INPUTS_VALIDATED_NOT_READY_FOR_DISPATCH', submitted=False)))
        return
    require(args.destination is not None, 'Provide a new isolated destination directory')
    freeze(value, args.source.resolve(), args.destination.resolve())

if __name__ == '__main__':
    main()
