import subprocess,shlex
opts=['-T','-A','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ControlMaster=no','-o','ControlPath=none']
command=shlex.join(['env','-u','LD_LIBRARY_PATH','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','UserKnownHostsFile=/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/peer-known-hosts','-o','StrictHostKeyChecking=yes','root@10.33.0.13','hostname'])
worker='env -u LD_LIBRARY_PATH ssh '+shlex.join(opts+['h100-10',command])
head='env -u LD_LIBRARY_PATH ssh '+shlex.join(opts+['-o','HostName=10.33.0.16','-o','HostKeyAlias=100.108.173.4','-o','StrictHostKeyChecking=yes','h100-16',worker])
program='import subprocess; subprocess.run('+repr(['ssh-add','/home/lichen/.ssh/id_ed25519'])+',check=True); subprocess.run('+repr(['env','-u','LD_LIBRARY_PATH','ssh']+opts+['-o','HostName=10.33.0.16','-o','HostKeyAlias=100.108.173.4','-o','StrictHostKeyChecking=yes','h100-16',worker])+',check=True)'
mgmt='ssh-agent python3 -c '+shlex.quote(program)
subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host',mgmt],check=True)
