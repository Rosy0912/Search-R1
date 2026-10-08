import shlex, subprocess
port='18892'
opts=['-T','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-R',f'127.0.0.1:{port}:127.0.0.1:{port}']
worker='env -u LD_LIBRARY_PATH '+shlex.join(['ssh']+opts+['h100-9','printf TREE_PROXY_READY\\n; sleep 86400'])
head='env -u LD_LIBRARY_PATH '+shlex.join(['ssh']+opts+['h100-16',worker])
outer=['ssh','-T','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-o','ProxyCommand=nc -X 5 -x 127.0.0.1:7892 %h %p','-R',f'127.0.0.1:{port}:127.0.0.1:7892','mgmt-host',head]
raise SystemExit(subprocess.call(outer))
