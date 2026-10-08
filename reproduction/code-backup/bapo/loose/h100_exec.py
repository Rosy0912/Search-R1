"""BAPO task SSH transport; uses existing identities without reading keys."""
import argparse
import shlex
import subprocess
from pathlib import Path

def route(node, command, socks=False, via_worker9=False):
    options = '-T -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 '
    worker = command if node == 'h100-16' else 'env -u LD_LIBRARY_PATH ssh ' + options + node + ' ' + shlex.quote(command)
    head = 'env -u LD_LIBRARY_PATH ssh ' + options + 'h100-16 ' + shlex.quote(worker)
    if via_worker9:
        head = 'env -u LD_LIBRARY_PATH ssh ' + options + 'root@h100-9 ' + shlex.quote(head)
    proxy = ['-o', 'ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p'] if socks else []
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3', *proxy, 'mgmt-host', head]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', default='h100-5', choices=['h100-5', 'h100-16'])
    parser.add_argument('--stdin-file', type=Path)
    parser.add_argument('--stdin-offset', type=int, default=0)
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--socks', action='store_true')
    parser.add_argument('--via-worker9', action='store_true')
    parser.add_argument('command')
    args = parser.parse_args()
    with args.stdin_file.open('rb') if args.stdin_file else open('/dev/null', 'rb') as stream:
        if args.stdin_offset:
            assert args.stdin_file and args.stdin_offset >= 0
            stream.seek(args.stdin_offset)
        result = subprocess.run(route(args.node, args.command, args.socks, args.via_worker9), stdin=stream, timeout=args.timeout)
    raise SystemExit(result.returncode)
