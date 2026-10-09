"""Recover failed TLS download with an independently probed own remote exit."""
import pathlib, subprocess, shlex, json
with open('/dev/tty') as f:
    token=f.readline().strip()
code=r'''import sys,json,pathlib,time,requests,hashlib,os,signal,subprocess,fcntl
r=pathlib.Path('/data/shaoshitong/cosearch-hf-migration-20261009');t=sys.stdin.readline().strip();script=sys.stdin.read();e=json.loads((r/'own-exit.json').read_text());s=requests.Session();s.trust_env=False;start=time.time()
try:
 response=s.get('https://huggingface.co/Sutoonq/TMP_MODEL/resolve/220f71043c64b4e5663769be6f2b96a89075a036/cosearch-migration-20261009/archives/retriever-supplement.tar.zst',headers={'Authorization':'Bearer '+t,'Range':'bytes=0-1048575'},proxies={'https':'http://127.0.0.1:'+str(e['port'])},timeout=(8,20));assert response.status_code==206 and response.headers['Content-Range'].startswith('bytes 0-1048575/');assert len(response.content)==1048576
except Exception as err:
 print(json.dumps({'benchmark':'FAILED','error_type':type(err).__name__}));sys.exit(1)
benchmark={'epoch':time.time(),'bytes':len(response.content),'seconds':time.time()-start,'sample_sha256':hashlib.sha256(response.content).hexdigest(),'route':'HKUST2_own_exit','mac_data_relay':False};(r/'own-exit-benchmark.json').write_text(json.dumps(benchmark));print(json.dumps(benchmark),flush=True)
old=json.loads((r/'controller.json').read_text())['pid'];p=pathlib.Path('/proc')/str(old)
if p.exists():
 assert b'cosearch_hf_download_hkust2.py' in (p/'cmdline').read_bytes();os.kill(old,signal.SIGTERM)
 for i in range(50):
  if not p.exists():break
  time.sleep(.1)
l=(r/'downloader.lock').open('a');fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB);(r/'cosearch_hf_download_hkust2.py').write_text(script);fcntl.flock(l,fcntl.LOCK_UN)
child=subprocess.Popen(['python3','-u',str(r/'cosearch_hf_download_hkust2.py')],stdin=subprocess.PIPE,stdout=(r/'downloader.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True);child.stdin.write((t+chr(10)).encode());child.stdin.close();print(json.dumps({'old_pid':old,'pid':child.pid,'exit_port':e['port'],'token_persisted':False}))
'''
args=['ssh','-T','-B','en0','-oProxyCommand=none','-oProxyJump=none','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=8','hkust2']
r=subprocess.run(args+['python3 -c '+shlex.quote(code)],input=token+chr(10)+pathlib.Path(__file__).with_name('cosearch_hf_download_hkust2.py').read_text(),capture_output=True,text=True,timeout=45)
print(r.stdout);print('exit',r.returncode)
pathlib.Path('outputs/evidence/native-x-hpc-hf-download-exit-recovery-20261009.json').write_text(r.stdout)
