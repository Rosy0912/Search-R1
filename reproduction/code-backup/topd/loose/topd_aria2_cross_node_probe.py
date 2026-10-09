"""Bounded remote aria2 throughput probes. Secrets only travel over stdin."""
import concurrent.futures, json, pathlib, shlex, subprocess, sys

REMOTE = r'''
import json,os,pathlib,subprocess,sys,time,urllib.request
token=sys.stdin.readline().strip()
node,endpoint,port=NODE,ENDPOINT,PORT
root=pathlib.Path('/data/shaoshitong/topd-hf-route-probes-20261009')/('aria2-'+endpoint)
root.mkdir(parents=True,exist_ok=True)
url='https://'+('hf-mirror.com' if endpoint=='mirror' else 'huggingface.co')+'/Sutoonq/TMP_MODEL/resolve/main/topd-migration-20261009/archives/teacher_model-00029-of-00029.tar.zst'
env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
args=['aria2c','--enable-rpc=true','--rpc-listen-all=false','--rpc-listen-port='+str(port),'--all-proxy=','--disk-cache=0','--quiet=true','--log=/dev/null','--file-allocation=none','--max-tries=2','--retry-wait=2','--connect-timeout=8','--timeout=12','--auto-file-renaming=false','-x','8','-s','8','-j','1','-k','1M','-c','-i','-']
p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,cwd=str(root),env=env,start_new_session=True)
p.stdin.write((url+'\n  header=Authorization: Bearer '+token+'\n  out=teacher29.tar.zst\n').encode());p.stdin.close()
def rpc(method):
 req=urllib.request.Request('http://127.0.0.1:'+str(port)+'/jsonrpc',data=json.dumps({'jsonrpc':'2.0','id':'probe','method':'aria2.'+method,'params':[]}).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=2) as r:return json.load(r).get('result',[])
samples=[];start=time.time()
try:
 for i in range(12):
  time.sleep(4)
  rows=rpc('tellActive') if p.poll() is None else []
  samples.append({'epoch':time.time(),'rows':[{k:x.get(k) for k in ('gid','status','totalLength','completedLength','downloadSpeed','connections','errorCode')} for x in rows]})
  if p.poll() is not None:break
finally:
 if p.poll() is None:
  import signal
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=8)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
out={'node':node,'endpoint':endpoint,'connections_requested':8,'seconds':time.time()-start,'samples':samples,'exit':p.returncode,'mac_data_used':False,'mac_proxy_used':False,'partial_preserved':True}
root.joinpath('probe.json').write_text(json.dumps(out));print(json.dumps(out))
'''
def main():
 token=sys.stdin.readline().strip();assert token
 def run(case):
  node,endpoint,port=case
  code=REMOTE.replace('NODE',repr(node)).replace('ENDPOINT',repr(endpoint)).replace('PORT',str(port))
  args=['ssh','-T','-B','en0','-b','10.4.162.158','-oProxyCommand=none','-oProxyJump=none','-oConnectTimeout=8','-oBatchMode=yes',node,'python3 -c '+shlex.quote(code)]
  p=subprocess.run(args,input=token+'\n',text=True,capture_output=True,timeout=75)
  if p.returncode:return {'node':node,'endpoint':endpoint,'error':'remote_probe_failed','exit':p.returncode}
  return json.loads(p.stdout)
 with concurrent.futures.ThreadPoolExecutor(3) as pool:
  out=list(pool.map(run,[('hkust1','mirror',18619),('hkust1','official',18620),('hkust2','mirror',18621)]))
 pathlib.Path('outputs/hpc3-migration/aria2-cross-node-probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
