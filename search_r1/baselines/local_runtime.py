"""Opt-in fail-closed single-host runtime boundary for this experiment."""
import ipaddress
import os
import socket
from urllib.parse import urlparse


def loopback(value):
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return value == 'localhost'


def check_local_runtime(config, ray_nodes=None):
    if not config.baselines.get('local_runtime_only', False):
        return
    expected = config.baselines.local_runtime_host
    if not expected or socket.gethostname().split('.')[0] != expected:
        raise ValueError('Runtime is not on the expected local host')
    if os.environ.get('TREE_SHARED_TRAINING_MODE') == '1':
        # Explicit reviewed own systemd protocol verifies its actual unit cgroup;
        # it never claims a Slurm node/device reservation.
        from shared_identity import verify_cgroup
        verify_cgroup()
    elif os.environ.get('SLURM_JOB_NODELIST') != expected or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Local runtime requires the matching Slurm allocation')
    parsed = urlparse(config.retriever.url)
    if (parsed.scheme != 'http' or not loopback(parsed.hostname or '') or parsed.username
            or parsed.password or parsed.path != '/retrieve'):
        raise ValueError('Retriever must be a local loopback HTTP service')
    address = os.environ.get('RAY_ADDRESS', '')
    if not loopback(urlparse('http://' + address).hostname or ''):
        raise ValueError('Ray must use an explicit local loopback address')
    proxies = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')
    if any(os.environ.get(name) for name in proxies):
        raise ValueError('Online local runtime must not use an HTTP/SOCKS proxy')
    if ray_nodes is not None:
        local = {'127.0.0.1', '::1'}
        local.update(record[4][0] for record in socket.getaddrinfo(socket.gethostname(), None))
        alive = [node for node in ray_nodes if node.get('Alive')]
        if len(alive) != 1 or alive[0].get('NodeManagerAddress') not in local:
            raise ValueError('Ray contains a remote or unexpected alive node')
