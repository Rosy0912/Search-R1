"""Fixed first15 evaluation failure only. Unknown worker waits remain unknown."""
import importlib
from pathlib import Path

from recovery_common import checked_json, checked, require, same_ref

FAILURE_SHA = 'a337e552d34592ed812b88cb0d5d7068ea682a78d4f5ac65ead7d4cb4ce4c51e'
CP150_SHA = 'fda38c2a8a60fc13512c8a3b29ae2a2c5b4ba02aa732104a02257cc478ba4050'
FAILURE_SCHEMA = 'opsd-signal-first15-original-failure-recovery-metadata-not-an-execution-request-v1'


def validate_observed_failure(value, config, ref):
    require(value['schema'] == FAILURE_SCHEMA and value['variant'] == 'G_POS'
            and value['run_id'] == config['run_id'] and value['completed_outers'] == 15
            and value['actual_optimizer_updates'] == 150 and value['cumulative_pair_exposure'] == 4568
            and value['raw_checkpoint']['sha256'] == CP150_SHA
            and value['native_endpoint_accepted'] is False
            and value['missing_original_wave1_actual_wait_evidence'] is True,
            'Not the fixed completed15/Adam150 raw evaluation failure')
    require(value['original_evaluator_actual_parent_wait_returncode'] == 1
            and value['original_controller_actual_supervisor_wait_returncode'] == 1
            and value['original_driver_actual_same_pane_exit_returncode'] == 1,
            'Previous actual evaluator/controller/supervisor waits must stay1')
    pane = value['same_server_socket_and_pane']['pane']
    require(pane['pane_dead'] == '1' and pane['pane_dead_status'] == '1'
            and pane['pane_id'] == '%45' and pane['session_id'] == '$45',
            'Original exact supervisor pane terminal differs')
    parent = value['original_native_evaluator_parent_terminal']
    controllers = value['original_group_controller_terminal']
    require(len(parent) == len(controllers) == 1
            and parent[0]['value']['config_sha256'] == ref['sha256']
            and parent[0]['value']['run_id'] == config['run_id']
            and parent[0]['value']['phase'] == 'evaluation-running'
            and len(parent[0]['value']['child_exit_evidence']) == 1
            and parent[0]['value']['child_exit_evidence'][0]['actual_wait_returncode'] == 1
            and controllers[0]['actual_wait_returncode'] == 1,
            'Real recorded parent/controller failure evidence differs')
    require(len(value['eight_old_worker_processes_actual_absent']) == 8
            and len({(x['pid'], x['starttick']) for x in value['eight_old_worker_processes_actual_absent']}) == 8
            and all(x['state'] == 'ABSENT' and x['wait_returncode'] == 'UNKNOWN'
                    and x['uid'] == config['owner']['uid'] and x['run_id'] == config['run_id']
                    for x in value['eight_old_worker_processes_actual_absent']),
            'Old orphan identities/unknown waits must not become successful formal exits')
    stack = '\n'.join(value['filtered_failure_stack'])
    require('gates.Hold: Receipt hash changed:' in stack and 'memory-live.json' in stack
            and 'memory_observer_admission' in stack, 'Different failure needs a new explicitly audited recovery')
    return value


def prove_old_absent(value, gates):
    identities = [x['expected'] for x in value['old_controller_and_supervisor_processes']]
    identities += value['eight_old_worker_processes_actual_absent']
    identities += [value['original_native_evaluator_parent_terminal'][0]['value']['child_exit_evidence'][0]['identity']]
    for identity in identities:
        try:
            actual = gates.process(identity['pid'])
        except (FileNotFoundError, ProcessLookupError):
            continue
        require(not gates.same_process(actual, identity), 'Old actual owner/worker still lives; never signal it')
    return identities


def fresh_resources(config, gates):
    from evaluation_retry import retry_hold
    publication = retry_hold(lambda: gates.publication_admission(config, gates.scoped(config, gates.bound(config['gates']['publication']))), gates.Hold)()
    storage = retry_hold(lambda: gates.storage_admission(config, gates.bound(config['gates']['storage'])), gates.Hold)()
    oracle = retry_hold(lambda: gates.retriever_admission(config, gates.bound(config['gates']['retriever_admission'])), gates.Hold)()
    observer = retry_hold(lambda: gates.memory_observer_admission(config), gates.Hold)()
    devices, apps = gates.inventory(); selected = [d for d in devices if d['index'] in config['owner']['physical_gpus']]
    require([d['index'] for d in selected] == [4, 5, 6, 7]
            and [d['uuid'] for d in selected] == config['owner']['gpu_uuids']
            and all(d['actual_free_mib'] >= config['resources']['startup_minimum_free_mib'] for d in selected),
            'Current actual GPU UUID/free admission differs')
    relevant = [a for a in apps if a['gpu_uuid'] in config['owner']['gpu_uuids']]
    require(all(a['pid'] == oracle['owner']['pid'] and a['gpu_uuid'] == config['owner']['gpu_uuids'][-1]
                and a['used_mib'] <= 1024 for a in relevant), 'Old/foreign GPU ownership remains')
    return dict(publication=publication, storage=storage, oracle=oracle, observer=observer,
                devices=selected, only_admitted_oracle=True)


def untouched_future_arms(configs):
    for candidate, _ in configs.values():
        if candidate['variant'] != 'G_POS':
            require(not Path(candidate['paths']['output_root']).exists(),
                    'Future arm already started: audit a new recovery instead of reusing fixed15')


def validate_recovery_start(repair, config, ref, gates, configs):
    require(repair['evaluation_failure']['sha256'] == FAILURE_SHA, 'Only exact released failure metadata allowed')
    value = validate_observed_failure(checked_json(repair['evaluation_failure']), config, ref)
    untouched_future_arms(configs)
    require(repair['resume_checkpoint'] == value['raw_checkpoint'], 'Recovery raw rich checkpoint changed')
    phase = checked_json(value['original_native_evaluator_parent_terminal'][0]['reference'])
    require(phase == value['original_native_evaluator_parent_terminal'][0]['value'], 'Actual evaluator wait1 bytes differ')
    controller = checked_json(value['original_group_controller_terminal'][0]['reference'])
    require(controller['actual_wait_returncode'] == 1
            and controller['controller'] == value['original_group_controller_terminal'][0]['controller'],
            'Actual controller wait1 bytes differ')
    prefix = checked_json(repair['completed_prefix'])
    require(prefix['schema'] == 'opsd-signal-debug-retention-v1' and prefix['accepted'] is True
            and prefix['mode'] == 'prefix' and prefix['currentN'] == prefix['complete_outers'] == 15
            and prefix['full_study_complete'] is False and prefix['sampling_workers'] == 120
            and same_ref(prefix['configs']['G_POS'], ref)
            and same_ref(prefix['per_outer'][-1]['end_checkpoint'], repair['resume_checkpoint']),
            'Previously complete15 retained prefix/raw endpoint identity differs')
    native = importlib.import_module('checkpoint'); prepare = importlib.import_module('prepare')
    evaluation = importlib.import_module('evaluation')
    cfg = dict(config, _config_sha256=ref['sha256'])
    raw, _ = native.check_checkpoint(repair['resume_checkpoint'], cfg, cfg['run_id'], 150, 14)
    require(raw['cursor']['next_pair_cursor'] == raw['cursor']['usable_pair_count'] == 308
            and raw['cumulative_pair_exposure'] == 4568, 'Completed CP150 cursor/exposure differs')
    output = Path(config['paths']['output_root'])
    require(not (output/'outers/outer-00015/outer-manifest.json').exists(), 'Later outer started: new recovery audit required')
    for path in (output/'checkpoints').glob('*/manifest.json'):
        candidate = checked_json(dict(path=str(path), sha256=gates.sha(path)))
        require(candidate['optimizer_updates'] <= 150, 'Later Adam exists: cannot reuse fixed15 failure')
    prepare.verify_runtime(config); prepare.verify_common(config)
    fresh_identity, _ = evaluation.input_identity(cfg, 15, repair['resume_checkpoint'])
    require(len(fresh_identity['source_files']) == 12, 'Actual fresh source100 payload inventory incomplete')
    preservation = checked_json(repair['preservation'])
    require(preservation['schema'] == 'opsd-signal-failed-evaluation-preservation-v2'
            and preservation['accepted'] is True and preservation['formal_endpoint_accepted'] is False
            and same_ref(preservation['failure_metadata'], repair['evaluation_failure'])
            and preservation['original_root'] == str(output/'evaluations/outer-15')
            and not Path(preservation['original_root']).exists(),
            'Failed canonical attempt not independently preserved/released')
    for row in preservation['files']:
        checked(row['preserved'])
        require(row['original']['sha256'] == row['preserved']['sha256']
                and row['original']['bytes'] == row['preserved']['bytes'], 'Preserved failure asset bytes differ')
    require(len(preservation['files']) == preservation['file_count'], 'Failure mapping incomplete')
    preserved_root = Path(preservation['preserved_root']).resolve()
    require(preserved_root.is_relative_to(Path(config['paths']['study_base']).resolve()/'preserved')
            and len({r['preserved']['path'] for r in preservation['files']}) == len(preservation['files'])
            and {str(p.resolve()) for p in preserved_root.rglob('*') if p.is_file()}
                == {r['preserved']['path'] for r in preservation['files']}, 'Failure tree mapping is not complete')
    required_metadata = required_mutable_paths(config)
    require({r['original']['path'] for r in preservation['metadata_files']} == required_metadata,
            'Overwritten failure status/registry metadata preservation incomplete')
    for row in preservation['metadata_files']:
        checked(row['preserved'])
        require(row['original']['sha256'] == row['preserved']['sha256']
                and row['original']['bytes'] == row['preserved']['bytes'], 'Preserved metadata bytes differ')
    for key in ('failed_status', 'failure_actual_exits'):
        ref = value[key]['reference']
        require(any(same_ref(r['original'],ref) for r in preservation['metadata_files']),
                'Original failed status/actual exit identity was not preserved')
    prove_old_absent(value, gates); resources = fresh_resources(cfg, gates)
    return dict(accepted=True, fixed_completed_outer=15, actual_adam_updates=150, no_adam_repeated=True,
                old_failure_waits=[1, 1, 1], old_worker_waits_remain_unknown=True,
                fresh_source100_payload_files=12, fresh_runtime_train_val_commoninit_raw_state=True,
                preservation=repair['preservation'], resources=resources, no_signals=True)


def required_mutable_paths(config):
    return {str(Path(config['paths']['output_root'])/name) for name in ('status.json','failure-actual-exits.json')} | {
        str(Path(config['paths']['study_base'])/'study-status.json'),
        str(Path(config['gates']['scoped_cpu_processes_path']))}
