"""Serial four-arm CPU supervisor. No models, signals, or implicit Adam reset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import uuid

from checkpoint import checked_json, checked_file, fileproof, exclusive_json, atomic_json, require
from prepare import validate_config, VARIANTS
import gates


def actual_identity(child, run_id):
    for _ in range(30):
        if child.poll() is not None:
            raise RuntimeError('Child exited before actual identity was bound: ' + str(child.returncode))
        try:
            identity = gates.process(child.pid)
            require(identity['run_id'] == run_id and identity['uid'] == 1001, 'Child run/UID differs')
            return identity
        except (FileNotFoundError, ProcessLookupError):
            time.sleep(.2)
    raise RuntimeError('Unable to bind actual running child identity')


def cpu_role(name, identity, input_ref, helper, manifest_ref, publication_ref):
    return dict(role=name, identity=identity, no_gpu_or_training=True,
        argv_sha256=hashlib.sha256(json.dumps(identity['argv'], sort_keys=True,
            separators=(',', ':'), ensure_ascii=False).encode()).hexdigest(),
        source=fileproof(Path(helper) / ('study_runner.py' if name == 'study_driver' else 'memory_observer.py')),
        input_config=input_ref, helper_source_manifest=manifest_ref, publication=publication_ref)


def run(study_ref, resume=False):
    study, sr = checked_json(study_ref)
    require(os.getuid() == 1001 and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Authorized CPU-only HKUST4 supervisor required')
    require('torch' not in sys.modules, 'Supervisor must never import Torch')
    require([g['variant'] for g in study['groups']] == list(VARIANTS), 'Predetermined four-arm execution order changed')
    root = Path(study['study_base']).resolve()
    require(root == Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010'), 'Study output is outside authorized remote root')
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root / 'study-sequence.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = gates.process(os.getpid())
    require(owner['run_id'] and len(owner['run_id']) > 0, 'Independent CPU supervisor marker required')
    helper = Path(study['helper_root']).resolve()
    checked_file(study['helper_source_manifest']); checked_file(study['publication'])
    require(helper / 'study_runner.py' == Path(__file__).resolve(), 'Supervisor source path differs from published input')
    driver = cpu_role('study_driver', owner, sr, helper, study['helper_source_manifest'], study['publication'])
    completions = []
    for group in study['groups']:
        config, cf = checked_json(group['config'])
        config['_config_sha256'] = cf['sha256']
        validate_config(config, launch=True)
        require(config['variant'] == group['variant'] and config['study_id'] == study['study_id'], 'Group study identity differs')
        require(config['paths']['helper_root'] == str(helper), 'Group helper differs from same frozen publication')
        output = Path(config['paths']['output_root'])
        status_path = output / 'status.json'
        if status_path.exists():
            prior = json.loads(status_path.read_text())
            require(prior['run_id'] == config['run_id'] and prior['config_sha256'] == cf['sha256'], 'Prior run/config differs')
            previous_owner = prior.get('owner')
            if previous_owner:
                try:
                    previous_process = gates.process(previous_owner['pid'])
                except (FileNotFoundError, ProcessLookupError):
                    previous_process = None
                require(previous_process is None or not gates.same_process(previous_process, previous_owner),
                    'Previous controller is still actually live; never start a duplicate resume')
            # Even a TRAIN_COMPLETE status is not a child return code. Resume
            # the same controller contract to revalidate committed states and
            # existing evaluations without repeating Adam or formal slots.
            require(resume, 'Existing partial group cannot be fresh-restarted')
        registry = Path(config['gates']['scoped_cpu_processes_path'])
        registry.parent.mkdir(parents=True, exist_ok=True)
        def register(roles):
            return atomic_json(registry, dict(schema='opsd-ablation-scoped-cpu-processes-v1', accepted=True,
                run_id=config['run_id'], config_sha256=cf['sha256'], roles=roles))
        register([driver])
        gates.require_launch_without_memory(config, config['run_id'], owner)
        observer_marker = 'signal-observer-' + uuid.uuid4().hex
        observer_log = Path(config['paths']['receipts']) / ('memory-observer-' + str(time.time_ns()) + '.log')
        observer_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', E1_RUN_ID=observer_marker, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        observer_command = [config['paths']['python'], str(helper / 'memory_observer.py'), '--config', cf['path'],
            '--output', str(Path(config['paths']['receipts']) / 'memory-live.json'),
            '--binding-output', config['gates']['memory_binding_path'], '--interval', '2']
        binding = Path(config['gates']['memory_binding_path'])
        observer = None; observer_handle = None; observer_identity = None
        if binding.exists():
            require(resume, 'Existing observer binding cannot be overwritten by a fresh launch')
            old = json.loads(binding.read_text())
            require(old['experiment_run_id'] == config['run_id'] and old['config_sha256'] == cf['sha256'], 'Old observer is from another config/run')
            try:
                actual = gates.process(old['observer_identity']['pid'])
            except (FileNotFoundError, ProcessLookupError):
                actual = None
            if actual is not None:
                require(gates.same_process(actual, old['observer_identity']) and actual['run_id'] == old['observer_identity']['run_id'], 'Old observer PID was reused')
                gates.memory_observer_admission(config)
                observer_identity = actual
            else:
                preserved = Path(config['paths']['receipts']) / 'preserved-observers' / str(time.time_ns())
                preserved.mkdir(parents=True)
                refs = []
                for name in ('memory-binding.json', 'memory-live.json', 'memory-live.samples.jsonl', 'memory-terminal.json', 'study-driver-release.json'):
                    source = Path(config['paths']['receipts']) / name
                    if source.exists():
                        original = fileproof(source); target = preserved / name
                        shutil.copyfile(source, target)
                        require(fileproof(target)['sha256'] == original['sha256'], 'Observer evidence preservation differs')
                        refs.append(dict(original=original, preserved=fileproof(target)))
                        source.unlink()  # Exact inactive monitor files only; immutable copies verified.
                exclusive_json(preserved / 'absence-and-preservation.json', dict(accepted=True, previous_binding=old,
                    kernel_process_absent=True, previous_wait_returncode='UNKNOWN', preserved=refs, no_signals_sent=True))
        if observer_identity is None:
            observer_handle = observer_log.open('x')
            observer = subprocess.Popen(observer_command, env=observer_env, stdout=observer_handle, stderr=subprocess.STDOUT, start_new_session=True)
            observer_identity = actual_identity(observer, observer_marker)
        register([driver, cpu_role('memory_observer', observer_identity, cf, helper,
            config['helper_source_manifest'], config['gates']['publication'])])
        for _ in range(120):
            if binding.exists():
                value = json.loads(binding.read_text())
                if value['observer_identity']['pid'] == observer_identity['pid']:
                    break
            require(observer is None or observer.poll() is None, 'Memory observer failed before startup; preserve output')
            time.sleep(.5)
        else:
            raise RuntimeError('Observer binding did not become ready')
        gates.memory_observer_admission(config)
        run_dir = root / 'sequence-attempts' / (config['variant'] + '-' + uuid.uuid4().hex)
        run_dir.mkdir(parents=True)
        command = [config['paths']['python'], str(helper / 'controller.py'), '--config', cf['path'], '--config-sha256', cf['sha256'], '--run-id', config['run_id']]
        if status_path.exists():
            command.append('--resume')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', E1_RUN_ID=config['run_id'], PYTHONDONTWRITEBYTECODE='1',
            PYTHONUNBUFFERED='1', HF_HOME=str(root / 'cache/hf'), TMPDIR=str(root / 'tmp'))
        log = (run_dir / 'controller.log').open('x')
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        identity = actual_identity(child, config['run_id'])
        exclusive_json(run_dir / 'started.json', dict(accepted=True, group=group, actual_controller=identity,
            observer=observer_identity, command=command, study_manifest=sr, supervisor=owner))
        while child.poll() is None:
            atomic_json(root / 'study-status.json', dict(schema='opsd-signal-study-status-v1', phase='GROUP_RUNNING',
                updated_unix=time.time(), supervisor=owner, active_group=config['variant'], controller=identity,
                finished_groups=completions, goal_outer_count=100, planned_test_evaluations=8))
            time.sleep(10)
        code = child.wait(); log.flush(); os.fsync(log.fileno()); log.close()
        exclusive_json(run_dir / 'actual-controller-exit.json', dict(actual_wait_returncode=code, controller=identity, log=fileproof(log.name)))
        require(code == 0, 'Nonzero group controller exit preserved; resume only after cause/state audit')
        final = json.loads(status_path.read_text())
        require(final['phase'] == 'TRAIN_COMPLETE' and final['outer_iterations'] == 25
            and final['run_id'] == config['run_id'] and final['config_sha256'] == cf['sha256'], 'Actual group did not complete25')
        # Controller publication follows actual full release; external driver independently rescans.
        release = gates.require_external_study_release(config, config['run_id'], owner, known_children=[identity])
        summary, sp = checked_json(fileproof(output / 'training-em-25-complete.json'))
        require(summary['accepted'] and len(summary['rows']) == 25, 'All25 TRAIN EM summaries absent')
        for iteration in (15,25):
            parent, _ = checked_json(fileproof(output / 'evaluations' / ('outer-%02d' % iteration) / 'parent-complete.json'))
            require(parent['accepted'] and parent['completed_outer'] == iteration, 'Formal15/25 parent evidence absent')
        completions.append(dict(variant=config['variant'], summary=sp, actual_wait_returncode=code, release=release,
            actual_terminal=fileproof(run_dir / 'actual-controller-exit.json')))
        ack = Path(config['paths']['receipts']) / 'study-driver-release.json'
        if ack.exists():
            old_ack = fileproof(ack)
            target = run_dir / 'previous-driver-ack.json'
            shutil.copyfile(ack, target)
            require(fileproof(target)['sha256'] == old_ack['sha256'], 'Previous driver ACK preservation differs')
        ack_ref = atomic_json(ack,
            dict(schema='opsd-signal-study-driver-release-v1', accepted=True,
                 run_id=config['run_id'], config_sha256=cf['sha256'], actual_wait_returncode=0,
                 release=release, driver_identity=owner,
                 controller_terminal=fileproof(run_dir / 'actual-controller-exit.json')))
        for _ in range(30):
            if observer is not None:
                if observer.poll() is not None:break
            else:
                try:
                    current = gates.process(observer_identity['pid'])
                    require(gates.same_process(current, observer_identity), 'Reused monitor identity changed')
                except (FileNotFoundError, ProcessLookupError):break
            time.sleep(1)
        if observer is not None:
            require(observer.poll() == 0, 'Completed observer must naturally exit0; no signal or foreign process takeover')
            observer_handle.close()
            observer_exit = dict(actual_wait_returncode=observer.wait(), observer=observer_identity, log=fileproof(observer_log))
        else:
            terminal, tp = checked_json(fileproof(Path(config['paths']['receipts']) / 'memory-terminal.json'))
            require(terminal['accepted'] is True and terminal['run_id'] == config['run_id']
                and terminal['config_sha256'] == cf['sha256']
                and gates.same_process(terminal['observer_identity'], observer_identity)
                and terminal['observer_identity']['run_id'] == observer_identity['run_id']
                and terminal['study_driver_release_ack']['sha256'] == ack_ref['sha256']
                and terminal['study_driver_release_ack']['path'] == ack_ref['path']
                and terminal['actual_driver_identity'] == owner,
                'Reused monitor terminal belongs to another config, process, or driver ACK')
            try:
                gates.process(observer_identity['pid'])
            except (FileNotFoundError, ProcessLookupError):pass
            else:raise RuntimeError('Reused monitor has not actually released')
            observer_exit = dict(actual_wait_returncode='UNKNOWN', observer=observer_identity,
                actual_kernel_absent=True, natural_terminal=tp, note='Reused process was not this supervisor child; no inferred wait0')
        exclusive_json(run_dir / 'actual-observer-exit.json', observer_exit)
    receipt = exclusive_json(root / 'receipts' / ('four-group-terminal-' + str(time.time_ns()) + '.json'),
        dict(accepted=True, groups=completions, completed_groups=4, completed_outers=100, full_training_em_rows=100,
            actual_cpu_supervisor=owner, study_manifest=sr, analysis_still_required=True))
    atomic_json(root / 'study-status.json', dict(phase='FOUR_GROUPS_TRAIN_AND_EVAL_COMPLETE', receipt=receipt,
        updated_unix=time.time(), final_analysis_still_required=True))
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-manifest', required=True); parser.add_argument('--study-manifest-sha256', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(dict(path=args.study_manifest, sha256=args.study_manifest_sha256), args.resume), sort_keys=True))
