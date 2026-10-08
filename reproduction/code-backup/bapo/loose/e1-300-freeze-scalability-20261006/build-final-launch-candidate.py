#!/usr/bin/env python3
"""Local candidate inventory builder. Never signs or grants launch permission."""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE = 'b7119791b86951727cea36a1881e8a525c932494'
ROOT = '/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006'
CPU_SHA = '8a32af877fdc8393c8d13378756d606e2e3b89cb04b416c8025a7c011e300095'
CONFIG_SHA = '4a985fce68eaa90548eea96f1c9613cb3e3d37784a93958c0566cc0ac5ff1064'
DATA_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
MODEL_SHA = '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), f'Not an ordinary non-symlink file: {p}')
    return hashlib.sha256(p.read_bytes()).hexdigest()

def inventory(directory):
    p = Path(directory)
    require(p.is_dir() and not p.is_symlink(), 'Tools must be an ordinary directory')
    entries = sorted(p.iterdir())
    require(all(not f.is_symlink() for f in entries), 'Symlink in tools inventory')
    return {f.name: sha(f) for f in entries if f.is_file()}

def encode(obj):
    return (json.dumps(obj, sort_keys=True, indent=2) + '\n').encode()

def build(tools, cpu_record, config, report_local=None, report_remote=None):
    tools = Path(tools)
    hashes = inventory(tools)
    require(len(hashes) >= 22, 'Incomplete tool inventory')
    for name in ('submit.sh', 'hardware.sbatch', 'formal300.sbatch', 'hardware.sh',
                 'verify_hardware_receipt.py', 'verify_launch.py', 'cpu-tests.py',
                 'freeze_scalability_gpu_fixture.py'):
        require(name in hashes, f'Missing launch dependency: {name}')
    require(sha(cpu_record) == CPU_SHA, 'Actual CPU record SHA differs')
    cpu = json.loads(Path(cpu_record).read_text())
    require(cpu['source_commit'] == SOURCE, 'CPU source differs')
    for name, expected in (('passed', 395), ('failed', 0), ('skipped', 0), ('exit_code', 0)):
        require(type(cpu.get(name)) is int and cpu[name] == expected, 'CPU result differs: ' + name)
    require(cpu['scope'] == 'ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE', 'CPU scope differs')
    require(cpu['sys_prefix'] == ROOT + '/env', 'CPU environment differs')
    require(cpu['test_launcher_sha256'] == hashes['cpu-tests.py'], 'CPU launcher changed since actual tests')
    for event in ('before', 'after'):
        require(cpu['import_origins'][event]['source_commit'] == SOURCE and
                cpu['import_origins'][event]['repo'] == ROOT + '/repo', 'CPU module origin differs')
    require(sha(config) == CONFIG_SHA, 'Configuration differs')
    require(bool(report_local) == bool(report_remote), 'Provide both report locations or neither')
    review = {'path': None, 'sha256': None, 'pending_independent_review': True}
    if report_local:
        remote = Path(report_remote)
        require(remote.is_absolute() and Path(ROOT) in remote.parents and '..' not in remote.parts,
                'Review destination must be inside the new root')
        review = {'path': str(remote), 'sha256': sha(report_local), 'pending_independent_review': True}
    manifest = {'source_commit': SOURCE, 'files': hashes}
    approval = {
        'status': 'CANDIDATE_REQUIRES_INDEPENDENT_E1_LAUNCH_REVIEW',
        'scope': 'UNSIGNED_LOCAL_CANDIDATE_NOT_LAUNCH_OR_GPU_PERMISSION',
        'source_commit': SOURCE, 'root': ROOT, 'tools_file_count': len(hashes),
        'tools_manifest_sha256': hashlib.sha256(encode(manifest)).hexdigest(), 'tool_files': hashes,
        'config_yaml_sha256': CONFIG_SHA, 'data_manifest_sha256': DATA_SHA,
        'model_manifest_sha256': MODEL_SHA, 'cpu_gate_record_sha256': CPU_SHA,
        'cpu_expected_passed': 395, 'independent_review': review,
        'actual_gpu': False, 'real_hardware_restore_accepted': False,
        'production_training_permission_signed': False,
    }
    head = {name: hashes[name] for name in ('submit.sh', 'hardware.sbatch', 'formal300.sbatch')}
    require(inventory(tools) == hashes, 'Tools changed during candidate construction')
    return manifest, approval, head

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tools', required=True)
    p.add_argument('--cpu-record', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--report-local')
    p.add_argument('--report-remote')
    p.add_argument('--output', required=True, help='Fresh local candidate directory; never logs/ or scheduler-head/')
    a = p.parse_args()
    target = Path(a.output)
    require(not target.exists(), 'Output must be fresh; no overwrite')
    require(str(target.resolve()) != ROOT and Path(ROOT) not in target.resolve().parents,
            'Builder must not write deployed root')
    manifest, approval, head = build(a.tools, a.cpu_record, a.config, a.report_local, a.report_remote)
    target.mkdir(parents=True)
    for name, obj in (('tools-manifest.json', manifest), ('launch-approval.UNSIGNED-CANDIDATE.json', approval),
                      ('approved-launch-hashes.UNSIGNED-CANDIDATE.json', head)):
        with (target / name).open('xb') as stream:
            stream.write(encode(obj))
    print(json.dumps({'status': 'UNSIGNED_CANDIDATE_ONLY', 'output': str(target.resolve()),
                      'tools_file_count': len(manifest['files']),
                      'sha256': {f.name: sha(f) for f in target.iterdir() if f.is_file()}}, sort_keys=True))

if __name__ == '__main__':
    main()
