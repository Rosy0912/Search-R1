import shlex,subprocess,sys
opts=['-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','ControlMaster=no','-o','ControlPath=none']
head='env -u LD_LIBRARY_PATH ssh '+shlex.join(opts+['-o','HostName=10.33.0.16','-o','HostKeyAlias=100.108.173.4','-o','StrictHostKeyChecking=yes','h100-16',sys.argv[1]])
r=subprocess.run(['ssh']+opts+['-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host',head],timeout=60)
sys.exit(r.returncode)
