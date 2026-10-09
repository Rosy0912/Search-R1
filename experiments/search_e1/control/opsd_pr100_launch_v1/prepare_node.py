"""Prepare the explicitly authorized PR100 run on HKUST; no models or GPU work."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid


def proof(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return proof(path)


def main():
    parser = argparse.ArgumentParser()
    for name in ('base', 'training-helper', 'periodic-helper', 'publication-evidence',
                 'previous-terminal', 'previous-terminal-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    base = Path(args.base).resolve()
    if not str(base).startswith('/data/shaoshitong/search-e1-opsd-pr100-'):
        raise ValueError('Only the new HKUST PR100 output namespace is supported')
    if os.getuid() != 1005 or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('HKUST UID1005 CPU-only preparation is required')
    training = Path(args.training_helper).resolve()
    periodic = Path(args.periodic_helper).resolve()
    sys.path.insert(0, str(training))
    prepare = importlib.import_module('prepare')
    gates = importlib.import_module('gates')
    cohorts = importlib.import_module('cohort_v1')
    published = json.loads(Path(args.publication_evidence).read_text())
    commit = published['source_commit']
    if published.get('accepted') is not True or published['remote_readback_sha'] != commit:
        raise ValueError('Actual published source readback is required')
    terminal_ref = proof(args.previous_terminal)
    if terminal_ref['sha256'] != args.previous_terminal_sha256:
        raise ValueError('Previous driver terminal changed')
    terminal = json.loads(Path(args.previous_terminal).read_text())
    successful = terminal.get('accepted') is True and terminal.get('actual_exit_code') == 0
    failed_and_released = (
        terminal.get('schema') == 'pr100-prior-driver-observed-terminal-v1'
        and terminal.get('original_driver_pid') == 1894801
        and terminal.get('original_driver_absent') is True
        and terminal.get('original_driver_actual_foreground_exit_code') == 2
        and terminal.get('pane', {}).get('raw') == '$74\t%74\t1894801\t1\t2\tbash\n'
        and terminal.get('gpu_compute', {}).get('actual_command_returncode') == 0
        and not Path('/proc/1894801').exists())
    if not (successful or failed_and_released):
        raise ValueError('Previous driver must be actually terminal; fresh GPU admission below still applies')
    receipts = base / 'receipts'
    receipts.mkdir(parents=True, exist_ok=True)
    old = json.loads(Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-P/receipts/authorized-launch-20261009/config.json').read_text())
    config = prepare.template(str(base), 'PR', 'pr100-' + uuid.uuid4().hex, str(base))
    config['run_id'] = uuid.uuid4().hex
    config['source_commit'] = commit
    config['paths']['helper_root'] = str(training)
    config['paths']['runtime'] = old['paths']['runtime']
    for key in ('source_model_manifest', 'runtime_manifest', 'train_identity', 'validation_identity',
                'common_initial_adapter_manifest', 'common_initial_adapter_tensor_sha256'):
        config[key] = old[key]
    config['paths']['train'] = old['paths']['train']
    config['paths']['validation'] = old['paths']['validation']
    config['helper_source_manifest'] = proof(training / 'source-manifest.json')
    frozen = cohorts.freeze_from_files(config['train_identity'], config['validation_identity'])
    config['common_cohort_manifest'] = save(receipts / 'common-100-cohorts.json', frozen)
    old_cohorts = json.loads(Path(old['common_cohort_manifest']['path']).read_text())
    if frozen['cohorts'][:10] != old_cohorts['cohorts']:
        raise ValueError('First ten cohorts must retain the previous common seed42 identity')
    config['authorization']['pr100_periodic'] = dict(
        human_observed_unix=time.time(), source='Latest human PR100 execution goal',
        text='HKUST1 eight GPUs, source100 fresh Adam, C1+P+R, 100 online outers, fixed512 every10; first10 operating acceptance then auto100',
        output_policy='Latest human: key checkpoints and necessary logs remotely; no large Mac payloads')
    config['authorization']['eight_gpu_execution'] = dict(cluster='HKUST', node='hkust1', uid=1005)
    config['retention']['merged_cache_cleanup_authorized'] = True
    scope = dict(study_id=config['study_id'], variant='PR', run_id=config['run_id'], output_root=config['paths']['output_root'])
    publication = dict(published, schema='opsd-pr100-scoped-publication-v1', scope=scope,
        source_manifest_sha256=config['helper_source_manifest']['sha256'])
    config['gates']['publication'] = save(receipts / 'training-publication.json', publication)
    storage = dict(schema='opsd-pr100-remote-storage-v1', accepted=True, cluster='HKUST',
        mode='human_hkust_local_exception', scope=scope, output_root=config['paths']['output_root'],
        human_authorization=dict(explicit_exception=True, observed_unix=time.time(),
            thread_id='01a1028b-b626-7f91-93db-0099ff1623c5',
            text='HKUST local storage is authorized; latest human permits remote account cleanup if needed and requests only key checkpoint/log outputs'),
        backup_policy=dict(durable_target=str(base / 'preserved/periodic-checkpoint-copies'),
            synchronize_completed_checkpoints=True, checkpoint_scope='Ten evaluation boundary checkpoints; protected originals remain on HKUST',
            Mac_payloads=False, same_node_not_disaster_recovery=True))
    config['gates']['storage'] = save(receipts / 'training-storage.json', storage)
    devices, applications = gates.inventory()
    oracle = dict(json.loads(Path(old['gates']['retriever_admission']['path']).read_text()), scope=scope, observed_unix=time.time())
    actual_oracle = gates.retriever_admission(config, oracle)
    if [d['uuid'] for d in devices] != config['owner']['gpu_uuids']:
        raise ValueError('Eight GPU UUID allocation changed')
    if any(app['pid'] != oracle['pid'] for app in applications):
        raise ValueError('Previous GPU work or foreign compute remains')
    resource = dict(schema='opsd-pr100-resource-transition-v1', accepted=True, cluster='HKUST',
        scope=scope, gpu_uuids=config['owner']['gpu_uuids'], previous_terminal=terminal_ref,
        actual_devices=devices, actual_applications=applications, observed_unix=time.time())
    config['gates']['resource_transition'] = save(receipts / 'resource-transition.json', resource)
    config['gates']['retriever_admission'] = save(receipts / 'oracle18004-admission.json', oracle)
    for p in (Path(config['paths']['base']), Path(config['paths']['receipts']), base / 'logs', base / 'preserved'):
        p.mkdir(parents=True, exist_ok=True)
    for name in config['resources']['shared_gpu_lease_paths'] + [config['resources']['study_lease_path']]:
        p = Path(name)
        if not p.parent.is_dir():
            raise ValueError('Existing shared lease parent is missing')
        p.touch(exist_ok=True)
    prepare.validate_config(config, launch=True)
    config_ref = save(receipts / 'training-config.json', config)
    summary = dict(accepted=True, cpu_only=True, training_config=config_ref,
        training_run_id=config['run_id'], study_id=config['study_id'], source_commit=commit,
        cohort_questions=frozen['selected_unique_question_count'], first_ten_common_prefix_equal=True,
        disk_free_bytes=shutil.disk_usage(base).free, remote_realpaths={
            key:str(Path(config['paths'][key]).resolve()) for key in ('base', 'output_root', 'runtime', 'model')},
        previous_terminal=terminal_ref, oracle_actual_owner=actual_oracle['owner'],
        periodic_helper=str(periodic), large_payloads_on_Mac=False)
    save(receipts / 'training-preparation.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
