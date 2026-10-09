"""IPv4 real private-file probe, credentials via remote curl stdin only."""
import concurrent.futures,json,pathlib,shlex,subprocess,sys
code=r'''
import subprocess,sys,json,time
token=sys.stdin.readline().strip();t=time.time()
url='https://'+HOST+'/Sutoonq/TMP_MODEL/resolve/main/topd-migration-20261009/archives/teacher_model-00028-of-00029.tar.zst?download=true'
config='url = "'+url+'"\nheader = "Authorization: Bearer '+token+'"\nrange = "0-16777215"\n'
p=subprocess.run(['curl','-4','--noproxy','*','-L','--silent','--output','/dev/null','--connect-timeout','4','--max-time','24','--write-out','%{http_code} %{size_download} %{speed_download} %{time_connect} %{time_appconnect} %{time_starttransfer} %{num_redirects}','--config','-'],input=config,text=True,capture_output=True,timeout=28)
print(json.dumps({'host':HOST,'exit':p.returncode,'metrics':p.stdout,'seconds':time.time()-t,'mac_used':False}))
'''
token=sys.stdin.readline().strip()
def run(c):
 node,host=c;p=subprocess.run(['ssh','-T','-B','en0','-b','10.4.162.158','-oProxyCommand=none','-oProxyJump=none','-oConnectTimeout=8','-oBatchMode=yes',node,'python3 -c '+shlex.quote(code.replace('HOST',repr(host)))],input=token+'\n',capture_output=True,text=True,timeout=36)
 return {'node':node,'result':json.loads(p.stdout) if not p.returncode else {'exit':p.returncode}}
with concurrent.futures.ThreadPoolExecutor(3) as pool:out=list(pool.map(run,[('hkust1','hf-mirror.com'),('hkust2','hf-mirror.com'),('hkust1','huggingface.co')]))
pathlib.Path('outputs/hpc3-migration/curl4-probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
