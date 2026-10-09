"""Read an authorized HKUST3 staged archive through direct campus SSH."""
import subprocess
import sys

if __name__ == '__main__':
    command = sys.stdin.read()
    result = subprocess.run(['ssh', '-T', '-B', 'en0',
                             '-o', 'ProxyCommand=none', '-o', 'ProxyJump=none',
                             '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                             '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
                             'hkust3', command], timeout=43200)
    raise SystemExit(result.returncode)
