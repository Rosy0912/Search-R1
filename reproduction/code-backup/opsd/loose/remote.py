import os
import pathlib
import shlex
import subprocess
import sys

opts = ['-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o',
        'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2', '-o',
        'ControlMaster=no', '-o', 'ControlPath=none']
command = sys.argv[1]
worker = 'env -u LD_LIBRARY_PATH ssh ' + shlex.join(opts + [
    'h100-10', command])
head = 'env -u LD_LIBRARY_PATH ssh ' + shlex.join(opts + [
    '-o', 'HostName=10.33.0.16', '-o', 'HostKeyAlias=100.108.173.4', '-o', 'StrictHostKeyChecking=yes', 'h100-16', worker])
socket_path = pathlib.Path(__file__).with_name('ssh-control-path.txt').read_text()
args = ['ssh'] + opts + ['-o', 'ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p', 'mgmt-host', head]
try:
    result = subprocess.run(args, timeout=int(os.environ.get("OPSD_REMOTE_TIMEOUT", "60")))
    sys.exit(result.returncode)
except subprocess.TimeoutExpired:
    print('Remote command exceeded 60s; inspect remote state before retrying a mutation.', file=sys.stderr)
    sys.exit(124)
