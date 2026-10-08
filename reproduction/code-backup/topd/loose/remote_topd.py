import subprocess, shlex, sys
command = sys.stdin.read()
worker = 'env -u LD_LIBRARY_PATH ssh -S /dev/shm/topd-worker-connect-v2.sock -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=2 h100-11 ' + shlex.quote(command)
head = 'env -u LD_LIBRARY_PATH ssh -S none -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=2 h100-16 ' + shlex.quote(worker)
r = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-S','none','-o','ProxyCommand=nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host',head],timeout=3600)
sys.exit(r.returncode)
