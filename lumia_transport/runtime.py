"""Admit a bound Slurm host change without editing the training configuration.

The frozen trainer, optimizer, sampler, restoration and all numerical code are
imported from the original checkout. Only the physical-host admission receives
the new allocated host. This does not promise bitwise rollouts across hardware.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import socket

PREFIX = 'RECURSIVE_LUMIA_TRANSPORT_'
HOSTS = {'node6': 'gpu-4090-3', 'node9': 'gpu-ada6000-2',
         'node10': 'gpu-l40s-1', 'node11': 'gpu-l40s-2'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(binding, record, env, uid, host):
    if (binding['version'] != 1 or uid != 1066
            or not env.get('SLURM_JOB_ID', '').isdigit()
            or binding['target_node'] != env.get('SLURM_JOB_NODELIST')
            or HOSTS.get(binding['target_node']) != host
            or binding['target_host'] != host
            or binding['parent_host'] != record['host']
            or binding['target_host'] == binding['parent_host']
            or binding['frozen_source_commit'] != record['source_commit']
            or env.get('RECURSIVE_SOURCE_COMMIT', env.get('SEARCH_R1_BASELINE_COMMIT')) != record['source_commit']
            or any(binding[k] != record[k] for k in ('run_id', 'group', 'config_sha256', 'data_identity'))
            or str(binding['parent_job']) == env['SLURM_JOB_ID']):
        raise ValueError('Transport requires the exact frozen run and new allocated physical host')
    return True


def load():
    path = Path(os.environ[PREFIX + 'BINDING'])
    if path.is_symlink() or sha(path) != os.environ[PREFIX + 'BINDING_SHA256']:
        raise ValueError('Transport binding bytes changed')
    binding = json.loads(path.read_text())
    configuration = Path(binding['configuration'])
    if sha(configuration) != binding['configuration_record_sha256']:
        raise ValueError('Frozen configuration receipt changed')
    record = json.loads(configuration.read_text())
    if sha(record['config']) != record['config_sha256']:
        raise ValueError('Frozen training configuration changed')
    if Path(__file__).resolve().parents[1] != Path(binding['transport_source']).resolve():
        raise ValueError('Transport module did not load from the published adapter checkout')
    validate(binding, record, os.environ, os.getuid(), socket.gethostname().split('.')[0])
    return binding, record


def admit_config(config, expected_old_host, target_host):
    """Make a validation-only copy; the trainer keeps the original exact config."""
    if config.baselines.local_runtime_host != expected_old_host:
        raise ValueError('Runtime configuration host is not the bound parent host')
    result = copy.deepcopy(config)
    result.baselines.local_runtime_host = target_host
    return result


def install():
    binding, record = load()
    from search_r1.baselines import local_runtime
    frozen = Path(binding['frozen_source']).resolve()
    if not Path(local_runtime.__file__).resolve().is_relative_to(frozen):
        raise ValueError('Training runtime did not import from the frozen checkout')
    if getattr(local_runtime.check_local_runtime, '_bound_transport', False):
        return
    original = local_runtime.check_local_runtime

    def check(config, ray_nodes=None):
        load()  # Every actual runtime admission rechecks physical allocation.
        return original(admit_config(config, record['host'], binding['target_host']), ray_nodes)

    check._bound_transport = True
    local_runtime.check_local_runtime = check


def worker_setup():
    install()
    from release.recursive_bran_a.ablation150_v1.bootstrap import install as native_install
    native_install()
