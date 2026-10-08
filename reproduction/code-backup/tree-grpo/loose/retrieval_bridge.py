"""Two owned SSH forwards; existing identities only, all listeners loopback."""
import pathlib,socket,subprocess,json,time,os,shlex
root=pathlib.Path.cwd();logs=root/'outputs/evidence'
for port in [28082]:
 with socket.socket() as s:s.bind(('127.0.0.1',port))
opts=['-T','-N','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
a800=['ssh']+opts+['-o','HostName=110.43.160.159','-o','HostKeyAlias=[ssh.platform-multi.baai.ac.cn]:2222','-o','ProxyCommand=none','-L','127.0.0.1:28082:127.0.0.1:18082','baai-a800']
worker=['env','-u','LD_LIBRARY_PATH','ssh']+opts+['-R','127.0.0.1:18082:127.0.0.1:38082','root@h100-9']
outer=['ssh']+[x for x in opts if x!='-N']+['-o','ProxyCommand=nc -X 5 -x 127.0.0.1:7892 %h %p','-R','127.0.0.1:38082:127.0.0.1:28082','mgmt-host',shlex.join(worker)]
marker=logs/'retrieval-bridge-started.json'
with marker.open('x') as f:json.dump({'epoch':time.time(),'scope':'only loopback SSH forward for own auxiliary retrieval; no credentials copied or global settings changed'},f)
children=[]
for name,command in [('a800-local',a800),('h1009-remote',outer)]:
 with (logs/(name+'-retrieval-forward.log')).open('x') as out,open('/dev/null') as dev:
  child=subprocess.Popen(command,stdin=dev,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  children.append({'name':name,'pid':child.pid,'start_epoch':time.time()})
 marker.write_text(json.dumps({'scope':'only loopback SSH forward for own auxiliary retrieval','processes':children},indent=2)+'\n')
print(json.dumps(children))
