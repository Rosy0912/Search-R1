import subprocess,shlex,sys
c=sys.stdin.read()
r=subprocess.run(['ssh','-T','-S','none','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-o','ProxyCommand=nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host','env -u LD_LIBRARY_PATH ssh -T -S none -o BatchMode=yes -o ConnectTimeout=15 h100-16 '+shlex.quote(c)],timeout=3600)
sys.exit(r.returncode)
