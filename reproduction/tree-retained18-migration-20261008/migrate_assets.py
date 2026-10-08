"""Copy the preserved own Tree assets from H100 NFS to retained18 local disk.

Run only on h100-10. Source tensors stay inside Wuhan H100. This does not start
training, release/cancel allocations, or claim an exact executor resume.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

SOURCE = Path('/mnt/shared/workspace/h100-11_from_h100-9/shaoshitong/search-r1-rl/tree-grpo')
DESTINATION = Path('/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008')
PARENT = Path('runs/tree-current200-continue-3331/checkpoints/training_state/rolling-completed/global_step_4')
PARENT_SHA = '84cf98caf0e7e67b52edf72a3cd57455d58c7e08ef020c4e665d81cf969a3b73'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def publish(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(pending, path)


def main():
    if socket.gethostname().split('.')[0] != 'h100-10':
        raise RuntimeError('Migration is restricted to retained18 h100-10')
    if digest(SOURCE/PARENT/'manifest.json') != PARENT_SHA:
        raise RuntimeError('Preserved original rolling4 identity changed')
    DESTINATION.mkdir(parents=True, exist_ok=True)
    control = DESTINATION/'.node-migration'
    control.mkdir(exist_ok=True)
    lock = (control/'lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    intent = control/'intent.json'
    if intent.exists():
        raise RuntimeError('Existing migration intent: observe, do not launch twice')
    # Preserve room for permanent5/50/100/150/200 states, rolling states and HF.
    if shutil.disk_usage(DESTINATION).free < 450 * 1024**3:
        raise RuntimeError('Insufficient local space for assets and required saves')
    state = dict(status='COPYING', pid=os.getpid(), started_epoch=time.time(),
                 source=str(SOURCE), destination=str(DESTINATION),
                 parent_manifest_sha256=PARENT_SHA, completed_groups=[],
                 native_parent_loaded=False, exact_executor_resume=False,
                 training_submitted=False, foreign_jobs_modified=False)
    intent.write_text(json.dumps(state, indent=2)+'\n')
    result_path = control/'result.json'
    publish(result_path, state)
    groups = [
        ('env', []), ('repo', []), ('tools', ['--exclude=__pycache__']),
        ('retriever-gpu-lib', ['--exclude=__pycache__']),
        ('instruct300/assets/qwen2.5-3b-instruct-aa8e725', []),
        ('instruct300/common-data', []),
        ('assets-root/assets/e5-base-v2', ['--exclude=*partial*', '--exclude=*.aria2']),
        ('assets-root/assets/wiki/e5_Flat.index', []),
        ('assets-root/assets/wiki/wiki-18.jsonl', []),
        ('runs/tree-current200-continue-3331', ['--exclude=*.pt',
             '--exclude=*.safetensors', '--exclude=cache/', '--exclude=runtime/']),
        ('runs/tree-fresh200-train-3328', ['--exclude=*.pt',
             '--exclude=*.safetensors', '--exclude=cache/', '--exclude=runtime/']),
        (str(PARENT), []),
    ]
    try:
        for relative, exclusions in groups:
            source, target = SOURCE/relative, DESTINATION/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            directory = source.is_dir()
            if directory:
                target.mkdir(exist_ok=True)
            args = ['rsync', '-a', '--partial-dir=.rsync-partial', *exclusions,
                    str(source)+('/' if directory else ''),
                    str(target)+('/' if directory else '')]
            state['current_group'] = relative
            publish(result_path, state)
            subprocess.run(args, check=True)
            state['completed_groups'].append(relative)
            publish(result_path, state)
        if digest(DESTINATION/PARENT/'manifest.json') != PARENT_SHA:
            raise RuntimeError('Copied native manifest differs')
        manifest = json.loads((DESTINATION/PARENT/'manifest.json').read_text())
        state['parent_files_verified'] = []
        records = [manifest['driver'], *manifest['worker_states']]
        original_root = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
        for record in records:
            path = DESTINATION/Path(record['path']).relative_to(original_root)
            expected = record['sha256']
            if digest(path) != expected:
                raise RuntimeError('Copied saved state checksum mismatch: '+path.name)
            state['parent_files_verified'].append(path.name)
            publish(result_path, state)
        # A task-only compatibility alias preserves saved-state absolute paths.
        # It is created only after all required local copies and parent hashes pass.
        alias = original_root
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.exists() or alias.is_symlink():
            if alias.resolve() != DESTINATION.resolve():
                raise RuntimeError('Original task alias already points elsewhere')
        else:
            alias.symlink_to(DESTINATION, target_is_directory=True)
        state.update(status='LOCAL_ASSETS_AND_PARENT_FILES_COPIED_VERIFIED',
                     finished_epoch=time.time(), compatibility_alias=str(alias),
                     native_parent_loaded=False, training_submitted=False)
        publish(result_path, state)
    except BaseException as error:
        state.update(status='FAILED_PARTIAL_COPY_RETAINED', error_type=type(error).__name__,
                     error=str(error), finished_epoch=time.time())
        publish(result_path, state)
        raise


if __name__ == '__main__':
    main()
