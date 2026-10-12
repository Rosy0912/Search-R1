"""Open one task-scoped management SSH connection; never modify global SSH."""
import json,pathlib,subprocess,tempfile,time,sys
root=pathlib.Path(tempfile.mkdtemp(prefix='rba-h100-management-'));root.chmod(0o700);socket=root/'control.sock';attempts=[]
for index in range(20):
 args=['ssh','-M','-N','-f','-S',str(socket),'-oControlMaster=yes','-oControlPersist=3600','-oServerAliveInterval=20','-oServerAliveCountMax=3','-oProxyCommand=none','-oProxyJump=none','-oBatchMode=yes','-oConnectTimeout=8','mgmt-host']
 r=subprocess.run(args,capture_output=True,text=True,timeout=15);attempts.append({'time':time.time(),'rc':r.returncode,'stderr':r.stderr});print(json.dumps(attempts[-1]),flush=True)
 if r.returncode==0:
  check=subprocess.run(['ssh','-S',str(socket),'-O','check','mgmt-host'],capture_output=True,text=True,timeout=5)
  live=subprocess.run(['ssh','-S',str(socket),'-oControlMaster=no','mgmt-host','hostname'],capture_output=True,text=True,timeout=15)
  proof={'socket':str(socket),'check_rc':check.returncode,'check':check.stderr,'hostname_rc':live.returncode,'hostname':live.stdout,'attempts':attempts}
  pathlib.Path(sys.argv[1]).write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True);raise SystemExit(0 if check.returncode==live.returncode==0 else 1)
 time.sleep(3)
pathlib.Path(sys.argv[1]).write_text(json.dumps({'socket':str(socket),'attempts':attempts,'connected':False},indent=2));raise SystemExit(1)
