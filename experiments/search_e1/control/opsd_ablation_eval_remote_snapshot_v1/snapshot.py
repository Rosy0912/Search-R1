"""Independent same-HKUST copies of closed native evaluator artifacts.

This CPU-only consumer never creates a model, takes a training lock, sends HTTP,
signals a process, changes a producer file, or copies payloads to the Mac.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
BASE = Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008')
DRIVER = BASE / 'eval-study-driver-stats-v2-8b641cb51b4c478684f596f75c603851/runs'
STUDY = BASE / 'receipts/stats-v2-handoff-8b641cb51b4c478684f596f75c603851/inputs/study-manifest.json'
STUDY_SHA = '4965dee13af70a02ef6a486d4b9dfb6518a8a2b1183a4c26591fb0fa19379aae'
RUN = '697c6bdf9cad41458cfed8acc78617e9'
ORDER = ('zero', 'C0', 'C1', 'M', 'R', 'W', 'P', 'J')
LABELS = {'bf16_zero', 'fp16_zero', 'C0', 'C1', 'M', 'R', 'W', 'P', 'J', 'P_fp16', 'J_fp16'}
COPY_SHA = '40e05812069c118b14df6605128204bd1a013286674c4cacab768a91ec48c211'


def require(value, message):
    if not value:
        raise ValueError(message)


def load_module(path, expected):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and all(not p.is_symlink() for p in (path, *path.parents)), 'Canonical source required')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'Published source byte SHA differs')
    namespace = dict(__name__='_closed_remote_copy_dependency', __file__=str(path))
    exec(compile(raw, str(path), 'exec'), namespace)
    return namespace


def setup(args):
    require(os.getuid() == os.geteuid() == 1005 and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules,
            'Same HKUST UID1005/CVD empty/no Torch required')
    copy = load_module(args.copy_primitives, COPY_SHA)
    manifest_ref, raw = copy['ref'](Path(__file__).parent / 'source-manifest.json')
    manifest = json.loads(raw)
    for name in ('snapshot.py', 'native_scope.py'):
        actual, _ = copy['ref'](Path(__file__).parent / name)
        require(actual['sha256'] == manifest['files'][name]['sha256'] and actual['bytes'] == manifest['files'][name]['bytes'], 'Own source manifest differs')
    native = load_module(Path(__file__).parent / 'native_scope.py', manifest['files']['native_scope.py']['sha256'])
    transport = native['primitives'](native['NODE_DEPENDENCY'])
    source = dict(own_manifest=manifest_ref, copy_primitives=copy['ref'](args.copy_primitives)[0],
                  native_scope=copy['ref'](Path(__file__).parent / 'native_scope.py')[0],
                  original_native_primitives=copy['ref'](native['NODE_DEPENDENCY'])[0])
    return copy, native, transport, source


def json_ref(copy, reference):
    actual, raw = copy['ref'](reference['path'])
    require(all(actual[k] == reference[k] for k in ('path', 'sha256')) and ('bytes' not in reference or actual['bytes'] == reference['bytes']), 'Exact referenced file SHA/bytes differs')
    return json.loads(raw), actual


def new_output(copy, output):
    output = Path(output)
    require(output.is_absolute() and '..' not in output.parts and output.resolve().is_relative_to(BASE / 'analysis') and not output.exists(),
            'New independent same-HKUST analysis output required')
    require(all(not p.is_symlink() for p in (output, *output.parents)), 'No output aliases or symlinks')
    output.mkdir(parents=True, exist_ok=False)
    return output


def scope(copy, native, group):
    root, parent = native['roots'](group)
    versions = native['versions'](group)
    paths = dict(config=parent.parent / 'evaluation-config.json', summary=root / 'summary.json',
                 controller=root / 'controller-complete.json', accepted=parent / 'result-accepted.json',
                 parent_exit=parent / 'actual-child-exit.json', parent_release=parent / 'actual-child-release.json')
    pins = {key: copy['ref'](path)[0] for key, path in paths.items()}
    cfg, _ = json_ref(copy, pins['config'])
    waited, _ = json_ref(copy, pins['parent_exit'])
    phases = {version: dict(phase=copy['ref'](root / version / 'phase-complete.json')[0],
                            terminal=copy['ref'](root / version / 'actual-terminal.json')[0]) for version in versions}
    return dict(schema='opsd-ablation-closed-native-scope-v1', accepted=True, study_id=native['STUDY_ID'],
                variant=group, run_id=cfg['run_id'], pins=pins, phases=phases, versions=versions,
                parent_identity=waited['identity'])


def identity_refs(copy, original):
    """Retain scalar manifests/config/identity, not their model/optimizer payloads."""
    cfg, _ = json_ref(copy, original['pins']['config'])
    train, train_ref = json_ref(copy, cfg['training']['config'])
    refs = [train_ref]
    for value in (cfg.get('helper_source_manifest'), cfg.get('gates', {}).get('publication'),
                  train.get('source_model_manifest'), train.get('runtime_manifest'), train.get('helper_source_manifest'),
                  train.get('common_cohort_manifest'), train.get('common_initial_adapter_manifest'),
                  cfg['training'].get('raw10_checkpoint'), cfg['training'].get('terminal')):
        if value:
            _, proof = json_ref(copy, value)
            refs.append(proof)
    # Identity manifests contain the already accepted source model/data payload
    # hashes. They are preserved; large weights/data remain in protected originals.
    identity = dict(source_model=train['source_model_manifest'], runtime=train['runtime_manifest'],
                    train=train['train_identity'], validation=train['validation_identity'],
                    common_cohort=train['common_cohort_manifest'], common_initial=train['common_initial_adapter_manifest'],
                    raw10_checkpoint=cfg['training']['raw10_checkpoint'], evaluation_science=cfg['scientific'],
                    helper_source=cfg['helper_source_manifest'], scientific_source= train['helper_source_manifest'],
                    retained_original_payloads_not_copied=True)
    return list({r['path']: r for r in refs}.values()), identity


def same_metadata(copy, row):
    actual = copy['metadata'](Path(row['path']).stat())
    return actual == {key: row[key] for key in actual}


def transfer(copy, output, files, exclusions, scope_value, closure, identities, source):
    """Whole SHA both directions and independent inode for every exact file."""
    require(files and len(files) == len(set(files)), 'Exact nonempty unique source set required')
    require(shutil.disk_usage(output).free >= sum(p.stat().st_size for p in files) + (10 << 30), 'Payload plus 10GiB remote free space required')
    destination = output / 'artifacts'
    destination.mkdir(exist_ok=False)
    original_rows, copy_rows = [], []
    for index, path in enumerate(files):
        # The identity manifests may live in the protected source100 directory;
        # both that directory and this destination are the same HKUST local disk.
        require(path.is_absolute() and path.is_relative_to('/data/shaoshitong') and '..' not in path.parts
                and all(not p.is_symlink() for p in (path, *path.parents)), 'Source must be an exact same-HKUST regular path')
        relative = path.relative_to('/').as_posix()
        original, replica = copy['copied'](path, destination / relative, relative)
        original_rows.append(original)
        copy_rows.append(replica)
        if index % 500 == 0:
            print(json.dumps(dict(event='remote_closed_eval_copy_progress', files=index + 1, expected_files=len(files))), flush=True)
    for row in original_rows:
        require(same_metadata(copy, row), 'Closed producer file changed before copy closure')
    inventory = copy['save'](output / 'source-inventory.private.json', dict(schema='opsd-ablation-remote-eval-source-inventory-v1',
        accepted=True, files=original_rows, file_count=len(files), bytes=sum(r['bytes'] for r in original_rows),
        source_scope=scope_value, original_closure=closure, identity_manifests=identities, source=source,
        excluded_subtrees=exclusions, node_originals_retained=True, all_source_pinned_metadata_stable=True))
    accepted = copy['save'](output / 'copy-accepted.private.json', dict(accepted=True, files=copy_rows, file_count=len(files),
        bytes=sum(r['bytes'] for r in original_rows), source_inventory=inventory, independent_inode_copy=True,
        all_source_and_destination_sha_equal=True, all_destination_metadata_stable=True,
        no_symlinks_or_special_files=True, storage_cluster='HKUST', Mac_payload_copy=False))
    return inventory, accepted, original_rows


def snapshot(args, copy, native, transport, source):
    groups = args.groups.split(',')
    require(groups and len(set(groups)) == len(groups) and groups == [g for g in ORDER if g in groups], 'Explicit ordered closed group subset required')
    output = new_output(copy, args.output)
    started = time.time()
    results = {}
    for group in groups:
        original = scope(copy, native, group)
        closure = native['closure'](transport, original)
        files, exclusions = native['enumerate_files'](transport, original)
        identities, model_identity = identity_refs(copy, original)
        files = sorted(set(files + [Path(ref['path']) for ref in identities]))
        stage = output / group
        stage.mkdir(exist_ok=False)
        inventory, accepted, original_rows = transfer(copy, stage, files, exclusions, original, closure, identities, source)
        require(closure == native['closure'](transport, original), 'Original native closure changed while copying')
        after_files, after_exclusions = native['enumerate_files'](transport, original)
        after_ids, after_identity = identity_refs(copy, original)
        require(files == sorted(set(after_files + [Path(r['path']) for r in after_ids])) and exclusions == after_exclusions
                and identities == after_ids and model_identity == after_identity, 'Exact closed artifact/identity set changed')
        require(all(same_metadata(copy, row) for row in original_rows), 'Producer metadata changed after closure reread')
        result = dict(schema='opsd-ablation-remote-eval-group-copy-v1', accepted=True, group=group, run_id=original['run_id'],
            primary_slots=closure['primary_slots'], models=closure['models'], source=inventory, replica=accepted,
            exact_closed_scope=original, scalar_model_data_runtime_identity=model_identity, excluded_subtrees=exclusions,
            files=len(files), bytes=sum(r['bytes'] for r in original_rows), actual_parent_wait_returncode=0,
            all_original_phase_waits_zero=True, independent_inode_copy=True, source_and_replica_full_sha=True,
            no_inference_rescoring_GPU_HTTP_lock_signal=True, node_originals_retained=True, checked_unix=time.time())
        results[group] = copy['save'](stage / 'complete.private.json', result)
        print(json.dumps(dict(event='remote_closed_eval_group_complete', group=group, files=result['files'],
                              bytes=result['bytes'], primary_slots=result['primary_slots'], complete=results[group])), flush=True)
    summary = dict(schema='opsd-ablation-remote-eval-copy-batch-v1', accepted=True, groups=results,
        source=source, started_unix=started, ended_unix=time.time(), storage_cluster='HKUST', Mac_payload_copy=False,
        whole_eleven_model_goal_complete=False, original_GPU_driver_unchanged=True, backup_never_training_dependency=True)
    result = copy['save'](output / 'complete.private.json', summary)
    print(json.dumps(dict(event='remote_closed_eval_batch_complete', accepted=True, complete=result,
                          group_count=len(groups), whole_eleven_model_goal_complete=False)), flush=True)


def finalize(args, copy, native, transport, source):
    # These references are captured by Root from the actual original retained
    # foreground. This helper does not manufacture a successful wait from absence.
    input_ref, raw = copy['ref'](args.final_inputs)
    require(input_ref['sha256'] == args.final_inputs_sha256, 'Exact final copy inputs SHA required')
    value = json.loads(raw)
    require(value['schema'] == 'opsd-ablation-remote-eval-final-copy-inputs-v1' and value['state'] == 'READY', 'Actual final foreground evidence required')
    study, sp = json_ref(copy, value['study_manifest'])
    require(sp['path'] == str(STUDY) and sp['sha256'] == STUDY_SHA and study['run_id'] == RUN, 'Wrong original current study')
    start, start_ref = json_ref(copy, value['driver_start'])
    done, done_ref = json_ref(copy, value['driver_complete'])
    terminal, terminal_ref = json_ref(copy, value['foreground_terminal'])
    waited, wait_ref = json_ref(copy, value['actual_wait_evidence'])
    released, release_ref = json_ref(copy, value['full_owned_release'])
    require(start_ref['path'] == str(DRIVER / 'driver-start.json') and done_ref['path'] == str(DRIVER / 'driver-complete.json')
        and done['accepted'] is True and done['primary_slots'] == 5632 and done['model_count'] == 11
        and done['actual_group_train_and_eval_exit0'] is True and done['input_manifest']['sha256'] == sp['sha256']
        and start['input_manifest']['sha256'] == sp['sha256'] and start['owner']['run_id'] == RUN
        and start['owner']['pid'] == 1894801 and start['owner']['uid'] == 1005 and start['owner']['starttick'] == 927990515,
        'Original driver11/5632/start/input identity is not actually complete')
    require(terminal['accepted'] is True and terminal['actual_exit_code'] == 0
        and terminal['signal_termination_not_exit0'] is False and terminal['owner'] == start['owner']
        and terminal['run_id'] == RUN and terminal['input_manifest']['sha256'] == sp['sha256']
        and terminal['actual_wait_evidence']['sha256'] == wait_ref['sha256'], 'Bound actual foreground terminal not success')
    require(waited['identity'] == start['owner'] and waited['actual_wait_returncode'] == 0
        and waited['exit_status_source'] in ('Popen.wait', 'retained_same_pid_foreground_pane'), 'Actual original foreground wait0 required; absence is insufficient')
    require(released.get('accepted_no_owned_or_opaque') is True or released.get('accepted_no_owned_or_foreign') is True,
        'Actual full owned release required, not a vanished PID')
    final, final_ref = json_ref(copy, done['summary'])
    require(final_ref['path'] == str(DRIVER / 'eleven-model-summary.json')
        and final['schema'] == 'opsd-ablation-eleven-fixed-model-summary-v1' and final['accepted'] is True
        and final['primary_slots'] == 5632 and set(final['models']) == LABELS and len(final['inputs']) == 11,
        'Exact eleven accepted original native512 models required')
    groups = {}
    batch_proofs = []
    for batch in value['copy_batches']:
        complete, bp = json_ref(copy, batch['complete'])
        # Actual external execution and each batch remain distinct, never a
        # synthesized common process/wait history.
        execution, ep = json_ref(copy, batch['actual_execution'])
        require(complete['schema'] == 'opsd-ablation-remote-eval-copy-batch-v1' and complete['accepted'] is True,
            'Original copy batch not accepted')
        require(execution.get('actual_wait_returncode') == 0 and execution.get('complete', {}).get('sha256') == bp['sha256'],
            'Each original copy batch needs its own actual wait0 bound completion')
        for group, ref in complete['groups'].items():
            require(group in ORDER and group not in groups, 'Duplicate or unknown closed group copy')
            copied, cp = json_ref(copy, ref)
            require(copied['schema'] == 'opsd-ablation-remote-eval-group-copy-v1' and copied['accepted'] is True and copied['group'] == group,
                    'Group copy closure differs')
            original = scope(copy, native, group)
            require(copied['exact_closed_scope'] == original, 'Final original group differs from accepted copied scope')
            closure = native['closure'](transport, original)
            inventory, invref = json_ref(copy, copied['source'])
            replica, repref = json_ref(copy, copied['replica'])
            require(inventory['original_closure'] == closure and replica['source_inventory']['sha256'] == invref['sha256']
                and len(inventory['files']) == len(replica['files']) == copied['files'], 'Copied/native closure or inventory changed')
            # The copying process already read both whole payload SHA streams.
            # Recheck all stable pinned source and destination metadata now.
            require(all(same_metadata(copy, row) for row in inventory['files']), 'Original source changed since accepted copy')
            for row in replica['files']:
                require(copy['metadata'](Path(row['path']).stat()) == row['metadata'], 'Accepted remote replica changed')
            groups[group] = dict(complete=cp, files=copied['files'], bytes=copied['bytes'], primary_slots=copied['primary_slots'],
                                 source=invref, replica=repref, models=copied['models'])
        batch_proofs.append(dict(complete=bp, actual_execution=ep))
    require(set(groups) == set(ORDER) and sum(r['primary_slots'] for r in groups.values()) == 5632,
            'All eight evaluator roots/eleven models must be copied before whole completion')
    output = new_output(copy, args.output)
    # Only immutable named original final records; never enumerate a still-live
    # driver/observer directory. Complete group/config/worker records are in the
    # explicitly closed independent batches above.
    final_paths = {Path(r['path']) for r in (input_ref, sp, start_ref, done_ref, terminal_ref, wait_ref, release_ref, final_ref)}
    extra = DRIVER / 'eleven-model-inputs.json'
    final_paths.add(extra)
    final_scope = dict(study_manifest=sp, driver_start=start_ref, driver_complete=done_ref, foreground_terminal=terminal_ref,
                       actual_wait_evidence=wait_ref, full_owned_release=release_ref, eleven_model_summary=final_ref)
    inventory, accepted, rows = transfer(copy, output, sorted(final_paths), [], final_scope, final_scope, [], source)
    require(all(same_metadata(copy, row) for row in rows), 'Final original records changed while preserving')
    result = dict(schema='opsd-ablation-eleven-evaluation-remote-persistence-v1', accepted=True, models=11, primary_slots=5632,
        groups=groups, separately_waited_copy_batches=batch_proofs, original_driver_exit0=True,
        exact_original_driver_terminal=terminal_ref, actual_wait_evidence=wait_ref, full_owned_release=release_ref,
        original_driver_complete=done_ref, original_final_summary=final_ref, final_original_record_inventory=inventory,
        final_original_record_replica=accepted, files=sum(g['files'] for g in groups.values()) + len(rows),
        bytes=sum(g['bytes'] for g in groups.values()) + sum(r['bytes'] for r in rows), storage_cluster='HKUST',
        physical_disaster_recovery_not_claimed=True, source_and_replica_full_sha=True, independent_inode_copy=True,
        Mac_raw_payloads=False, weights_optimizer_cache_payloads_copied=False, Node_originals_retained=True,
        no_inference_rescoring_GPU_HTTP_lock_signal=True, source=source, checked_unix=time.time())
    complete_ref = copy['save'](output / 'complete.private.json', result)
    public = dict(schema='opsd-ablation-eleven-evaluation-persistence-public-v1', accepted=True, models=11, primary_slots=5632,
        evaluator_roots=8, files=result['files'], bytes=result['bytes'], storage_cluster='HKUST', actual_driver_exit0=True,
        separately_waited_copy_batches=len(batch_proofs), independent_inode_copy=True, full_source_and_replica_SHA=True,
        final_complete_sha256=complete_ref['sha256'], scope='Closed native results/traces/HTTP/generatedIDs/args/phase waits/config/scalar identity manifests and exact original final terminal. Protected original checkpoints/model/data remain remote; FP16 rebuildable model trees excluded.',
        physical_disaster_recovery_not_claimed=True, Mac_raw_payloads=False)
    public_ref = copy['save'](output / 'public-summary.json', public)
    print(json.dumps(dict(event='all_eleven_evaluation_remote_persistence_complete', accepted=True, models=11,
        primary_slots=5632, complete=complete_ref, public_summary=public_ref)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('snapshot', 'finalize'))
    parser.add_argument('--copy-primitives', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--groups')
    parser.add_argument('--final-inputs')
    parser.add_argument('--final-inputs-sha256')
    args = parser.parse_args()
    copy, native, transport, source = setup(args)
    if args.mode == 'snapshot':
        require(args.groups is not None and args.final_inputs is None, 'Explicit closed subset only')
        snapshot(args, copy, native, transport, source)
    else:
        require(args.groups is None and args.final_inputs and args.final_inputs_sha256, 'Explicit original final evidence required')
        finalize(args, copy, native, transport, source)


if __name__ == '__main__':
    main()
