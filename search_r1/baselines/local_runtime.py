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


def local_addresses():
    """Actual host interfaces, including Ray's route-selected local IPv4."""
    local = {'127.0.0.1', '::1'}
    local.update(record[4][0] for record in socket.getaddrinfo(socket.gethostname(), None))
    # DNS alone may resolve the hostname to loopback on HKUST5. Do not
    # allow a remote address: read only addresses assigned to this host.
    import fcntl, struct
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        for _, name in socket.if_nameindex():
            try:
                value = fcntl.ioctl(probe.fileno(), 0x8915, struct.pack('256s', name.encode()[:15]))
            except OSError:
                continue # Interfaces without an IPv4 address are expected.
            local.add(socket.inet_ntoa(value[20:24]))
    return local


def check_local_runtime(config, ray_nodes=None):
    if not config.baselines.get('local_runtime_only', False):
        return
    expected = config.baselines.local_runtime_host
    if not expected or socket.gethostname().split('.')[0] != expected:
        raise ValueError('Runtime is not on the expected local host')
    direct=os.environ.get('SEARCH_R1_EXECUTION_BACKEND')=='hkust_direct'
    if direct:
        identities={'xielab1':1005,'xielab5':1001,'xie-lab-4':1001}
        if identities.get(expected)!=os.getuid() or not os.environ.get('SEARCH_R1_BASELINE_RUN_ID'):
            raise ValueError('Direct HKUST runtime identity changed')
        from .preflight import validate_inherited_gpu_leases
        validate_inherited_gpu_leases(config.trainer.n_gpus_per_node)
    elif os.environ.get('SLURM_JOB_NODELIST') != expected or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Local runtime requires the matching Slurm allocation')
    parsed = urlparse(config.retriever.url)
    local_retriever = loopback(parsed.hostname or '') or (direct and expected in ('xielab5','xie-lab-4') and parsed.hostname=='10.123.4.11' and parsed.port==18003)
    if (parsed.scheme != 'http' or not local_retriever or parsed.username
            or parsed.password or parsed.path != '/retrieve'):
        raise ValueError('Retriever must be the approved HKUST HTTP service')
    address = os.environ.get('RAY_ADDRESS', '')
    if not (direct and address=='local') and not loopback(urlparse('http://' + address).hostname or ''):
        raise ValueError('Ray must use an explicit local loopback address')
    proxies = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')
    if any(os.environ.get(name) for name in proxies):
        raise ValueError('Online local runtime must not use an HTTP/SOCKS proxy')
    if ray_nodes is not None:
        local = local_addresses()
        alive = [node for node in ray_nodes if node.get('Alive')]
        if len(alive) != 1 or alive[0].get('NodeManagerAddress') not in local:
            raise ValueError('Ray contains a remote or unexpected alive node')
