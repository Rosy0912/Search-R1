import sys,json,shlex,subprocess,time,uuid,base64,hashlib
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation')
def remote(host,code,python='python3',timeout=55):
 compile(code,'remote','exec')
 r=subprocess.run(control.route(host,python+' -c '+shlex.quote(code)),capture_output=True,text=True,timeout=timeout)
 assert r.returncode==0,r.stderr
 return json.loads(r.stdout)
proof=remote('h100-8',r'''import json,os,subprocess,time;from pathlib import Path
b=Path('/dataset/shaoshitong/Search-G1-common8');stage=json.loads((b/'logs/pipeline-status.json').read_text());cleanup=json.loads((b/'logs/cleanup-status.json').read_text());ledger=json.loads((b/'logs/process-ledger.json').read_text())
assert stage['stage']=='FAILED' and cleanup['cleaned']
print(json.dumps(dict(stage=stage,cleanup=cleanup,ledger=ledger,jobs=subprocess.run(['sacct','-j','3238','--format=JobID,State,ExitCode','--noheader'],capture_output=True,text=True).stdout,checked_unix=time.time())))
''')
(prep/'attempt1-terminal-observation.json').write_text(json.dumps(proof,indent=2))
h=json.loads((prep/'original-retrieval-export-handle.json').read_text())
code='import json,os,signal,time,hashlib;from pathlib import Path\nh='+repr(h)+'\n'+r'''
b=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export');p=Path('/proc')/str(h['pid']);s=b/'serve_original_retrieval.py'
assert hashlib.sha256(s.read_bytes()).hexdigest()==h['source_sha256']
if p.exists():
 st=(p/'stat').read_text().rsplit(')',1)[1].split()
 assert (int(st[19]),int(st[3]),p.stat().st_uid)==(h['start_ticks'],h['sid'],h['uid'])
 assert str(s).encode() in (p/'cmdline').read_bytes()
 fd=os.pidfd_open(h['pid']);signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)
 for _ in range(50):
  if not p.exists() or (p/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
  time.sleep(.1)
 assert not p.exists() or (p/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
proof=dict(cleaned=True,handle=h,checked_unix=time.time(),reason='attempt1 failed before training; own source service no longer needed')
(b/'cleanup.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''
p=remote('h100-9',code);(prep/'original-retrieval-source-cleanup.json').write_text(json.dumps(p,indent=2))
base='/dataset/shaoshitong/Search-G1-common8-attempt2-'+uuid.uuid4().hex[:12]
code='import json,os,shutil,hashlib,subprocess,time;from pathlib import Path\nb=Path('+repr(base)+')\n'+r'''
old=Path('/dataset/shaoshitong/Search-G1-common8');b.mkdir(exist_ok=False)
for n in ('preparation','data','logs','tmp','ray','cache','artifacts'):(b/n).mkdir()
for n in ('env','bootstrap-assets','bootstrap-tools','assets','retriever-gpu-lib'):(b/n).symlink_to(old/n,target_is_directory=True)
shutil.copytree(old/'repo',b/'repo',ignore=shutil.ignore_patterns('__pycache__','.git','.pytest_cache'))
commit='4b1a06243e915c73b439fca8c62ba5f928ca812b'
run=json.loads((old/'logs/run-manifest.json').read_text())
assert run['source_commit']==commit
for n,h in run['source_sha256'].items():assert hashlib.sha256((b/'repo'/n).read_bytes()).hexdigest()==h
for n in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):os.link(old/'data'/n,b/'data'/n)
shutil.copy2(old/'preparation/official-model-manifest-original.json',b/'preparation/official-model-manifest-original.json')
assert not (b/'data/calibration-300.parquet').exists() and not (b/'logs/run-manifest.json').exists()
proof=dict(base=str(b),source_commit=commit,parent_attempt_run_id=run['run_id'],actual_updates=0,shared_asset_semantics='same-cluster original assets and repaired installed env only; no prior model checkpoint/probe/counters',canonical_data_links='only six original shared products; separate derivatives',free_bytes=shutil.disk_usage(b).free,checked_unix=time.time())
(b/'preparation/attempt-preparation.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''
p=remote('h100-8',code);(prep/'attempt2-preparation.json').write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))
