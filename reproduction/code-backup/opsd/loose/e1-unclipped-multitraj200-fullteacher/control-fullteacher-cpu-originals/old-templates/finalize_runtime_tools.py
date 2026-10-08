"""Local-only source binding. No remote mutation, job ID, GPU, signing or launch."""
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE_ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006')
FORMAL_FILES = ('formal_job.sh', 'formal_supervisor.py', 'owned_runtime.py',
                'resource_sample.py', 'retriever_entry.py', 'ray_cpu_check.py',
                'hardware_rank.py', 'cpu_import_check.py')
SOURCE_FILES = ('reproduction/baselines/run_opsd_e1_multitraj200.sh',
                'reproduction/baselines/e1_import_bootstrap.py',
                'reproduction/h20/retriever_local.py',
                'search_r1/search/retrieval_server.py',
                'verl/trainer/config/ppo_opsd_e1_multitraj200.yaml',
                'search_r1/baselines/main.py', 'search_r1/baselines/preflight.py',
                'search_r1/baselines/e1_profile.py', 'search_r1/baselines/controller.py',
                'search_r1/baselines/actor.py', 'search_r1/baselines/distillation.py', 'search_r1/baselines/trajectory_archive.py', 'search_r1/baselines/rollout.py', 'reproduction/baselines/e1-multitraj200/microbatch_probe.py',
                'search_r1/baselines/full_state.py', 'search_r1/baselines/checkpoint_gate.py',
                'verl/workers/fsdp_workers.py', 'verl/trainer/ppo/ray_trainer.py',
                'verl/utils/torch_functional.py')


def fail(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + '\n').encode()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def prepared_files(directory, names, replacements):
    files = {}
    for name in names:
        raw = (directory / name).read_bytes()
        for before, after in replacements.items():
            raw = raw.replace(before.encode(), after.encode())
        fail(b'BLOCKED_' not in raw, 'Unbound file: ' + name)
        if name.endswith('.py'):
            compile(raw, name, 'exec', dont_inherit=True)
        files[name] = raw
    return files


def publish(kind, files, manifest, manifest_name='manifest.json'):
    fail(all(Path(name).name==name and name not in ('.','..',manifest_name) for name in files),
         'Bundle members must be unique fixed basenames')
    raw = encode(manifest)
    pin = sha(raw)
    directory = HERE / (kind + '-' + pin[:12])
    remote = REMOTE_ROOT / 'tools' / directory.name
    if directory.exists():
        fail(not directory.is_symlink() and directory.resolve(strict=True)==directory
             and directory.stat().st_uid==os.getuid() and directory.stat().st_mode&0o777==0o555,
             'Existing immutable directory identity/mode differs')
        fail((directory / manifest_name).read_bytes() == raw, 'Existing manifest differs')
        fail({p.name for p in directory.iterdir()} == set(files) | {manifest_name},
             'Existing directory inventory differs')
        identities=[]
        for name in set(files)|{manifest_name}:
            member=directory/name
            fail(not member.is_symlink() and member.stat().st_uid==os.getuid()
                 and member.stat().st_mode&0o777==0o444,'Existing immutable member identity/mode differs')
            identities.append((member.stat().st_dev,member.stat().st_ino))
        fail(len(set(identities))==len(identities),'Immutable members share an inode')
        for name, value in files.items():
            fail((directory / name).read_bytes() == value, 'Existing immutable file changed')
    else:
        fail(not directory.is_symlink(),'Output directory symlink refused')
        directory.mkdir()
        for name, value in files.items():
            with (directory / name).open('xb') as f:
                f.write(value);f.flush();os.fchmod(f.fileno(),0o444);os.fsync(f.fileno())
        with (directory / manifest_name).open('xb') as f:
            f.write(raw);f.flush();os.fchmod(f.fileno(),0o444);os.fsync(f.fileno())
        os.chmod(directory,0o555)
        fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:os.fsync(fd)
        finally:os.close(fd)
        for name,value in {**files,manifest_name:raw}.items():
            fail((directory/name).read_bytes()==value,'Published bytes differ')
        fd=os.open(HERE,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:os.fsync(fd)
        finally:os.close(fd)
    return {'local_directory': str(directory), 'remote_directory': str(remote),
            'manifest_name': manifest_name, 'manifest_sha256': pin,
            'files': {name: sha(value) for name, value in files.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--source', required=True)
    parser.add_argument('--revision', default='')
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    fail(not args.revision or re.fullmatch('[a-z0-9-]+',args.revision) is not None,
         'Simple actual tool revision required')
    fail(re.fullmatch('[0-9a-f]{40}', args.source) is not None, 'Actual full source SHA required')
    fail(git(repo, 'rev-parse', 'HEAD') == args.source and not git(repo, 'status', '--porcelain'),
         'Committed clean source required; candidate cannot finalize')
    source_pins = {name: sha((repo / name).read_bytes()) for name in SOURCE_FILES}
    replacements = {'BLOCKED_UNBOUND_SOURCE_COMMIT': args.source}

    formal_files = prepared_files(HERE / 'formal-tools-candidate', FORMAL_FILES, replacements)
    formal_manifest = {'version': 1, 'source_commit': args.source, 'source_files': source_pins,
                       'scope': 'SOURCE_BOUND_SOFTWARE_ONLY_NO_ACTUAL_JOB_OR_ACCEPTANCE',
                       'common_data_manifest_sha256': '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',
                       'official_model_manifest_sha256': '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44',
                       'retrieval_asset_manifest_sha256': '87b5e85df12cbf4635cb787b08161950ddc23e2e5b850e68fb7249668573cc87',
                       'native_import_closure_sha256': 'EXTERNAL_ENV_REQUIRED_NOT_GUESSED',
                       'public_key_sha256': 'EXTERNAL_ENV_REQUIRED_NOT_GUESSED',
                       'tools': {name: sha(raw) for name, raw in formal_files.items()}}

    native_files = prepared_files(HERE / 'first5-native-candidate',
                                 ('inspect_payloads.py', 'test_inspector.py'), replacements)
    native_manifest = {'version': 1, 'source_commit': args.source,
                       'scope': 'MULTITRAJ200_CPU_STORED_STATE_SOFTWARE_NO_ACTUAL_ACCEPTANCE',
                       'profile': 'multitraj200', 'horizon': 200,
                       'warmup_steps': 57, 'saved_adam_steps': 50,
                       'saved_lr': 1e-6*(5/57),
                       'tools': {name: sha(raw) for name, raw in native_files.items()}}
    native = publish('first5-native', native_files, native_manifest, 'candidate-manifest.json')

    replacements.update({'BLOCKED_NATIVE_TOOLS_PATH': native['remote_directory'],
                         'BLOCKED_NATIVE_INSPECTOR_SHA': native['files']['inspect_payloads.py'],
                         'BLOCKED_NATIVE_MANIFEST_SHA': native['manifest_sha256'],
                         'BLOCKED_GATE_SOURCE_SHA': source_pins['search_r1/baselines/checkpoint_gate.py']})
    receipt_files = prepared_files(HERE / 'first5-receipt-candidate',
                                  ('receipt_payload.py', 'hf_cpu_reload.py', 'inspect_first5_cpu.py'),
                                  replacements)
    receipt_manifest = {'version': 1, 'source_commit': args.source, 'source_files': source_pins,
                        'scope': 'SOURCE_BOUND_SOFTWARE_NO_ACTUAL_FIRST5_ACCEPTANCE_NO_SIGNING_OR_RELEASE',
                        'native_payload_inspector_sha256': native['files']['inspect_payloads.py'],
                        'native_payload_tools_manifest_sha256': native['manifest_sha256'],
                        'tools': {name: sha(raw) for name, raw in receipt_files.items()}}
    formal = publish('formal', formal_files, formal_manifest)
    receipt = publish('first5', receipt_files, receipt_manifest)
    fail(git(repo, 'rev-parse', 'HEAD') == args.source and not git(repo, 'status', '--porcelain'),
         'Source drifted during binding')
    fail(source_pins == {name: sha((repo / name).read_bytes()) for name in SOURCE_FILES},
         'Source file changed during binding')
    result = {'version': 1, 'scope': 'LOCAL_BOUND_TOOL_PREPARATION_NO_ACTUAL_JOB_ID_NO_GPU_NO_SIGNATURE',
              'source_commit': args.source, 'remote_root': str(REMOTE_ROOT),
              'remote_repo': str(REMOTE_ROOT / ('repo-' + args.source[:12])),
              'formal': formal, 'native': native, 'first5': receipt,
              'source_files': source_pins}
    output = HERE / ('runtime-tools-bound-' + args.source[:12] +
                     ('-'+args.revision if args.revision else '') + '.json')
    raw = encode(result)
    if output.exists():
        fail(output.read_bytes() == raw, 'Existing binding result differs')
    else:
        with output.open('xb') as f:
            f.write(raw)
    print(raw.decode(), end='')


if __name__ == '__main__':
    main()
