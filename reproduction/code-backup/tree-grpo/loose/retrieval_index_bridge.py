import pathlib,subprocess,json,time,shlex
opts=['-T','-N','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
worker=['env','-u','LD_LIBRARY_PATH','ssh']+opts+['-R','127.0.0.1:18083:127.0.0.1:38082','root@h100-9']
outer=['ssh']+[x for x in opts if x!='-N']+['-o','ProxyCommand=nc -X 5 -x 127.0.0.1:7892 %h %p','-R','127.0.0.1:38082:127.0.0.1:28082','mgmt-host',shlex.join(worker)]
p=pathlib.Path('outputs/evidence/index-bridge-started.json')
with p.open('x') as f:json.dump({'epoch':time.time(),'scope':'own loopback index tunnel18083; no credentials/global settings changes'},f)
with pathlib.Path('outputs/evidence/h1009-index-forward.log').open('x') as out,open('/dev/null') as dev:
 child=subprocess.Popen(outer,stdin=dev,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
p.write_text(json.dumps({'epoch':time.time(),'local_pid':child.pid,'scope':'own loopback index tunnel18083 -> existing own A800-local28082'},indent=2)+'\n');print(child.pid)
