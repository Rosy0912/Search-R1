"""Run an authorized h100-11 command through the verified retained-18 route."""
import subprocess
import sys
import argparse

REMOTE = r'''import json, shlex, subprocess, sys
from pathlib import Path
cfg=json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())
target='h100-11'
ip,gateway,user=cfg['targets'][target],cfg['gateway_ip'],cfg['user']
prefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ConnectionAttempts=1','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2']
args=prefix.copy()
if ip!=gateway:args+=['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',user+'@'+gateway])]
keys=cfg.get('trusted_target_host_keys',{}).get(target)
if keys:args+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\n'.join(keys)])]
raise SystemExit(subprocess.run(args+[user+'@'+ip,COMMAND],timeout=TIMEOUT).returncode)
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    code = REMOTE.replace('COMMAND', repr(sys.stdin.read())).replace('TIMEOUT', str(args.timeout))
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
                             'mgmt-host','env -u LD_LIBRARY_PATH python3 -'],
                            input=code,text=True,timeout=args.timeout+50)
    raise SystemExit(result.returncode)
