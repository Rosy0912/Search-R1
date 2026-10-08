"""Integration gate: four disjoint streams, injected disconnect, retained partial, SHA gate."""
import hashlib,json,subprocess,sys,tempfile,time
from pathlib import Path
source_code=Path(__file__).with_name('migrate_cosearch_hpc.py').read_text()
with tempfile.TemporaryDirectory() as tmp:
 b=Path(tmp);(b/'outputs/evidence').mkdir(parents=True);src=b/'src';dst=b/'dst';src.mkdir();dst.mkdir()
 rows=[]
 for i in range(4):
  data=bytes([i+1])*(8*1024*1024+127);name='global_step_203/file'+str(i)
  (src/'global_step_203').mkdir(exist_ok=True);(src/name).write_bytes(data)
  rows.append({'kind':'file','path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
 e=b/'outputs/evidence';(e/'native-x-hpc-migration-file-manifest.jsonl').write_text('\n'.join(json.dumps(x) for x in rows+[{'kind':'complete','files':4}]))
 code=source_code.replace("BASE=Path(__file__).resolve().parents[1]",'BASE=Path('+repr(str(b))+')').replace("SOURCE='/dataset/from_h100-16/shaoshitong/search-r1-rl/cosearch'",'SOURCE='+repr(str(src))).replace("TARGET='/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/source'",'TARGET='+repr(str(dst))).replace('64*1024*1024','4')
 code=code.replace('ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)',"from types import SimpleNamespace\nctl=SimpleNamespace(route=lambda host,command:['/bin/sh','-c',command])")
 # Exercise genuine append/resume after a receiver disconnect mid-stream.
 injected='''from pathlib import Path
import sys,time
p=Path(PATH);p.parent.mkdir(parents=True,exist_ok=True)
marker=p.with_suffix('.injected')
with p.open('ab',buffering=0) as f:
 for chunk in iter(lambda:sys.stdin.buffer.read(1048576),b''):
  f.write(chunk);time.sleep(.08)
  if not marker.exists():marker.write_text('once');raise RuntimeError('injected disconnect')
'''
 code=code.replace('from cosearch_transfer_io import receiver_code','def receiver_code(path):return '+repr(injected)+".replace('PATH',repr(path))")
 run=b/'run.py';run.write_text(code);out=b/'log'
 with out.open('w') as log:p=subprocess.Popen([sys.executable,str(run)],stdout=log,stderr=log)
 maximum=0;deadline=time.time()+45
 while p.poll() is None and time.time()<deadline:
  try:st=json.loads((e/'native-x-hpc-migration-progress.json').read_text());maximum=max(maximum,len(st.get('active_files',[])))
  except (FileNotFoundError,json.JSONDecodeError):pass
  time.sleep(.05)
 if p.poll() is None:p.kill();raise AssertionError('test timed out')
 assert p.returncode==0,out.read_text()
 st=json.loads((e/'native-x-hpc-migration-progress.json').read_text());assert st['status']=='asset_transfer_complete_all_sha_verified';assert st['file_retry_count']==4;assert maximum==4
 for rec in rows:assert hashlib.sha256((dst/rec['path']).read_bytes()).hexdigest()==rec['sha256']
 # Completed corrupted target must stop at integrity gate, not silently overwrite.
 (dst/rows[0]['path']).write_bytes(b'corrupted')
 with out.open('a') as log:r=subprocess.run([sys.executable,str(run)],stdout=log,stderr=log,timeout=30)
 assert r.returncode!=0
 st=json.loads((e/'native-x-hpc-migration-progress.json').read_text());assert 'AssertionError' in st['error'];assert (dst/rows[0]['path']).read_bytes()==b'corrupted'
 print(json.dumps({'parallel_streams_observed':maximum,'injected_disconnects_retried':4,'all_files_final_SHA_equal':True,'corrupt_completed_target_hard_stop':True}))
