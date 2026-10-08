"""Task-local SSH route; never edits SSH, VPN or authentication configuration."""
import argparse
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=Path)
parser.add_argument('--command')
parser.add_argument('--timeout', type=int, default=90)
args = parser.parse_args()
if bool(args.file) == bool(args.command):
    parser.error('Choose exactly one command or script file')
inner = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'h100-16']
inner += ['/bin/bash', '--noprofile', '--norc'] if args.file else [args.command]
command = 'env -u LD_LIBRARY_PATH ' + shlex.join(inner)
route = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
         '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
         '-o', 'ProxyCommand=nc -X 5 -x 127.0.0.1:7892 %h %p', 'mgmt-host', command]
result = subprocess.run(route, input=args.file.read_bytes() if args.file else None,
                        timeout=args.timeout)
raise SystemExit(result.returncode)
