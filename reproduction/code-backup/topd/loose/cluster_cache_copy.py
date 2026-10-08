import subprocess,shlex
from pathlib import Path
script=r'''#!/bin/bash
set -euo pipefail
source_root=/dataset/shaoshitong/search-r1-rl/cosearch
target_root=/dataset/shaoshitong/search-r1-rl/topd
transport='env -u LD_LIBRARY_PATH ssh -T -S /dev/shm/topd-worker-connect-v2.sock -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3'
$transport h100-11 "mkdir -p $target_root/cache/source-site-packages $target_root/logs"
printf 'READ_ONLY_CLUSTER_SOURCE_ASSETS_START\n'
nice -n 19 rsync -rlt --size-only --partial --append-verify -e "$transport" "$source_root/assets-root/assets/" "h100-11:$target_root/assets-root/assets/"
printf 'ASSET_COPY_DONE\n'
nice -n 19 rsync -rlt --size-only --partial --append-verify --exclude='__pycache__/' --exclude='*.pyc' -e "$transport" "$source_root/env/lib/python3.10/site-packages/" "h100-11:$target_root/cache/source-site-packages/"
printf 'ENVIRONMENT_SITE_PACKAGES_COPY_DONE\n'
$transport h100-11 "printf '0\n' > $target_root/logs/cluster-cache-copy.exit"
'''
folder='/dev/shm/topd-01a0fcbe';path=folder+'/cluster-cache-copy.sh'
head='mkdir -p '+shlex.quote(folder)+' && cat > '+shlex.quote(path)
route=['ssh','-S','none','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-o','ProxyCommand=nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host']
def command(c):return route+['env -u LD_LIBRARY_PATH ssh -S /dev/shm/topd-01a0fcbe/head-connect-v2.sock -o BatchMode=yes -o ConnectTimeout=15 h100-16 '+shlex.quote(c)]
subprocess.run(command(head),input=script,text=True,check=True)
start='tmux new-session -d -s topd-cluster-cache-20261003 '+shlex.quote('bash '+path+' > '+folder+'/cluster-cache-copy.log 2>&1')
subprocess.run(command(start),check=True)
print('OWN_CLUSTER_CACHE_COPY_STARTED_READ_ONLY_SOURCE',flush=True)
