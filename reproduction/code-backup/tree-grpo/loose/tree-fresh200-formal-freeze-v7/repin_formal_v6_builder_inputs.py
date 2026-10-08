"""Local JSON rebinder draft; never build, prepare, query, submit or read assets.

Only human-provided actual four-observation descriptors can fill production
inputs. Other V5 evidence/prospective-path sections remain historical metadata,
not current V7 acceptance or destinations. This source has not been executed.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import types


LOCAL_ROOT = Path(__file__).resolve().parents[2]
RUNTIME = LOCAL_ROOT / 'work/tree-fresh200-train-runtime-v7'
SOURCE_DIR = LOCAL_ROOT / 'work/tree-fresh200-adam-offload-source'
REMOTE_ROOT = PurePosixPath('/dataset/shaoshitong/search-r1-rl/tree-grpo')
COMMIT = '53082e1667a6ab131a036ca2ef66e004989ed1a9'
BASE_PATH = LOCAL_ROOT / 'work/tree-fresh200-formal-v5-preparation-draft/BUILDER_INPUTS_TEMPLATE.json'
BASE_SHA = '24ce55c3bc765ec094164c345c5534a76a2461587cd87d3df25b22305ad032f3'
INVENTORY_SHA = '42d225fb38510e9726107617e72e06b9deead8539acfa858998ce7830d5ca55e'
INTERFACE_SHA = 'e52577f068bc7b19ef90dbff1b98ae3e803aa97cbaf4a43bf037fce82d439242'
SPEC_SHA = 'ad6599be178c75b199b072e128d641977334c03c5e1f39d59043c68bcde5b6cb'
CONTRACT_SHA = '8ff97890a326f99a5b823d13681a068bbd9a61fee3eacb5d56692423cd3204f1'
WALL_POLICY_PATH = LOCAL_ROOT / 'work/tree-fresh200-3317-readonly-performance/ACTUAL_LOCAL_GPU_PARTITION_POLICY.stdout'
WALL_POLICY_SHA = 'ba2f015fba1d6853882449fd538113a7f407a2e7d6bab277d47b2dc6370c5dc0'
INPUT_NAMES = ('actual_calibration_ready', 'actual_calibration_acceptance',
               'threshold_acceptance', 'source_context_free_review')
RUNTIME_ENTRIES = ('run_tree.sh', 'validation_cost_entry.py',
                   'validation_cost_hooks_instruct300.py', 'sitecustomize.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_path(value):
    require(type(value) is str, 'Local path must be text')
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts
            and str(path) == value and str(path.resolve()).startswith('/Users/'),
            'Only canonical local /Users paths are accepted')
    return path


def read_local_bound(path, expected):
    require(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected),
            'Exact lowercase SHA256 required')
    path = local_path(str(path))
    observed = path.lstat()
    require(stat.S_ISREG(observed.st_mode), 'Pinned local file must be ordinary')
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode)
                and (observed.st_dev, observed.st_ino) == (before.st_dev, before.st_ino),
                'Pinned local path changed before opening')
        raw = stream.read()
        after = os.fstat(stream.fileno())
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        require(tuple(getattr(before, key) for key in fields)
                == tuple(getattr(after, key) for key in fields), 'Pinned local file changed')
    require(hashlib.sha256(raw).hexdigest() == expected, 'Pinned local SHA differs')
    return raw


def validate_descriptors(observations):
    require(type(observations) is dict
            and set(observations) == {'production_metadata', 'production_inputs'},
            'Exactly local production_metadata and runtime production_inputs required')
    for kind in ('production_metadata', 'production_inputs'):
        entries = observations[kind]
        require(type(entries) is dict and set(entries) == set(INPUT_NAMES),
                'All four actual descriptors required: ' + kind)
        for name in INPUT_NAMES:
            entry = entries[name]
            require(type(entry) is dict and set(entry) == {'path', 'sha256'},
                    'Actual non-null descriptor required: ' + kind + '.' + name)
            require(type(entry['sha256']) is str
                    and re.fullmatch('[0-9a-f]{64}', entry['sha256']), 'Actual SHA required')
            if kind == 'production_metadata':
                local_path(entry['path'])
            else:
                require(type(entry['path']) is str, 'Runtime path must be text')
                path = PurePosixPath(entry['path'])
                require(path.is_absolute() and '..' not in path.parts
                        and str(path) == entry['path'] and path.is_relative_to(REMOTE_ROOT),
                        'Runtime descriptor must remain inside the own h100-9 ROOT')
    require(all(observations['production_metadata'][name]['sha256']
                == observations['production_inputs'][name]['sha256'] for name in INPUT_NAMES),
            'Local and runtime actual JSON SHA values differ')


def rebound_inputs(observations, once_intent, train_wall_hours):
    """Validate real JSON observations and return data; create no output here."""
    require(type(train_wall_hours) is int and train_wall_hours > 0,
            'Root must explicitly provide finite positive integer train wall hours')
    validate_descriptors(observations)
    intent = PurePosixPath(once_intent)
    require(type(once_intent) is str and str(intent) == once_intent
            and intent.parent == REMOTE_ROOT / 'instruct300'
            and re.fullmatch(r'TREE_FRESH200_TRAIN_ONCE-[A-Za-z0-9_-]+\.json', intent.name),
            'Root must provide a new formal once-intent path; it is never created or queried')
    value = json.loads(read_local_bound(BASE_PATH, BASE_SHA))
    inventory = json.loads(read_local_bound(RUNTIME / 'FILE_HASHES_V7.json', INVENTORY_SHA))
    interface = json.loads(read_local_bound(RUNTIME / 'SOURCE_INTERFACE_COMPATIBLE_RECEIPT.json', INTERFACE_SHA))
    spec = json.loads(read_local_bound(RUNTIME / 'CONFIGURATION_SPEC.json', SPEC_SHA))
    require(inventory['wrapper_count'] == len(inventory['wrapper_files']) == 33
            and inventory['wrapper_files']['CONFIGURATION_SPEC.json'] == SPEC_SHA
            and inventory['wrapper_files']['production_contract.py'] == CONTRACT_SHA,
            'Frozen V7 wrapper dependencies differ')
    primary = interface['source_files']; additional = interface['additional_changed_runtime_files']
    require(interface['code_commit'] == COMMIT and interface['source_interface_count'] == 16
            and len(primary) == 16 and len(additional) == 4
            and not set(primary).intersection(additional)
            and interface['total_source_interface_count'] == 20
            and 'verl/utils/fsdp_utils.py' in primary, 'Frozen V7 source closure differs')
    contract_path = RUNTIME / 'production_contract.py'
    contract = types.ModuleType('formal_v7_rebinding_production_contract')
    contract.__file__ = str(contract_path)
    exec(compile(read_local_bound(contract_path, CONTRACT_SHA), str(contract_path), 'exec'), contract.__dict__)
    values = {name: contract.read_bound(observations['production_metadata'][name]['path'],
                                      observations['production_metadata'][name]['sha256'])
              for name in INPUT_NAMES}
    review = values['source_context_free_review'][0]
    require(review.get('status') == 'CONTEXT_FREE_FRESH200_SOURCE_ACCEPTED'
            and review.get('fork_turns') == 'none' and review.get('code_commit') == COMMIT
            and review.get('blocking_findings') == [],
            'Actual current-source compact review required; helper/owner review is insufficient')
    package = dict(code_commit=COMMIT, common_data_identity=value['assets']['common_data_identity'],
                   model_identity=value['assets']['model_identity'])
    thresholds = contract.observed_production_inputs(package, values)
    contract.configuration_overrides(spec, thresholds)
    result = copy.deepcopy(value)
    source = result['source']
    source.update(code_commit=COMMIT, path=str(SOURCE_DIR), required_source_interface_keys=16,
                  additional_runtime_interface_count=4, total_source_interface_count=20,
                  source_interface_receipt_sha256=INTERFACE_SHA,
                  counter_interface_current_scope='V7 primary16 plus runtime4 source-byte identity; count protocol unchanged; no package or training release')
    source['source_interface_manifest'] = source['source_interface_receipt'] = {
        'path': str(RUNTIME / 'SOURCE_INTERFACE_COMPATIBLE_RECEIPT.json'), 'sha256': INTERFACE_SHA}
    result['wrapper_inventory'] = {'path': str(RUNTIME / 'FILE_HASHES_V7.json'), 'sha256': INVENTORY_SHA}
    result['configuration_spec_path'] = str(RUNTIME / 'CONFIGURATION_SPEC.json')
    result['configuration_spec_sha256'] = SPEC_SHA
    result['runtime_entries'] = {}
    for name in RUNTIME_ENTRIES:
        expected = inventory['wrapper_files'][name]
        read_local_bound(RUNTIME / name, expected)
        result['runtime_entries'][name] = {'path': str(RUNTIME / name), 'sha256': expected}
    result['production_metadata'] = copy.deepcopy(observations['production_metadata'])
    result['production_inputs'] = copy.deepcopy(observations['production_inputs'])
    result['bootstrap_thresholds'] = dict(thresholds)
    result['dispatch']['once_intent'] = once_intent
    result['dispatch']['train_wall_hours'] = train_wall_hours
    result['dispatch']['wall_authority'] = dict(
        kind='explicit_root_provided_finite_train_wall_hours',
        chosen_formal_hours=train_wall_hours,
        original_instruction_time=None,
        original_instruction_time_status='unknown',
        source='Root explicitly supplied --train-wall-hours for the current complete 200-outers plus full 51713-row final goal',
        current_goal=dict(complete_outers=200, final_evaluation_rows=51713, full_final_evaluation=True),
        scheduler_policy=dict(path=str(WALL_POLICY_PATH), sha256=WALL_POLICY_SHA),
        actual_duration_not_inferred=True,
        inherited_wall_authority_used=False,
        wall_default_used=False,
        budget_sufficiency_accepted=False,
        actual_release=False,
        slurm_submitted=False)
    result['status'] = 'LOCAL_FORMAL_V7_BUILDER_INPUTS_REBOUND_NOT_RELEASED'
    result['metadata_rebinding'] = dict(base_template_path=str(BASE_PATH), base_template_sha256=BASE_SHA,
        historical_base_sections_only=['actual_barriers', 'current_zero_observation', 'existing_source_context_free_evidence',
                                      'final_review', 'historical_raw_cpu_preflight_inputs', 'local_preparation',
                                      'review_inputs', 'wrapper_owner_receipt'],
        scope='Metadata only; preserved V5 evidence and prospective paths do not authorize V7 execution; no builder, preparation or dispatch invoked')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('observations', 'observations-sha', 'once-intent', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--train-wall-hours', required=True, type=int)
    args = parser.parse_args()
    observations = json.loads(read_local_bound(args.observations, args.observations_sha))
    result = rebound_inputs(observations, args.once_intent, args.train_wall_hours)
    output = local_path(args.output)
    require(output.suffix == '.json' and not output.exists() and not output.is_symlink(),
            'New no-clobber local JSON output required')
    raw = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + b'\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(status=result['status'], path=str(output),
                         sha256=hashlib.sha256(raw).hexdigest(), builder_executed=False, training_release=False)))


if __name__ == '__main__':
    main()
