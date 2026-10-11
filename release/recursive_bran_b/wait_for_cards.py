"""CPU-only wait for the authorized four cards, then launch exactly once."""
import argparse,hashlib,json,os,socket,subprocess,time,urllib.request
from pathlib import Path
UUIDS=['GPU-ed24344f-9cd8-6d29-e8b9-d7582a8cfd65','GPU-79de0d51-c2a2-897e-f76a-ec84f03b38e2','GPU-3a89b647-73fb-27bc-c8a8-e9f663b96d5f','GPU-21352a4a-5b55-8adf-8e82-c70b7a94ecf3']
def process(pid):
 p=Path('/proc',str(pid));st=(p/'stat').read_text().split(') ',1)[1].split()
 return dict(pid=pid,uid=p.stat().st_uid,start_tick=int(st[19]),state=st[0])
def check():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
 selected=[]
 for line in rows.splitlines():
  i,u,f,util=[x.strip() for x in line.split(',')]
  if i in ('4','5','6','7'):selected.append(dict(index=int(i),uuid=u,free_mib=float(f),utilization=float(util)))
 if [x['uuid'] for x in selected]!=UUIDS:raise RuntimeError('Selected hardware identity changed')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=20)
 foreign=[];oracle=None
 for line in apps.splitlines():
  u,p,m=[x.strip() for x in line.split(',')]
  if u not in UUIDS:continue
  ident=process(int(p));ident.update(uuid=u,memory_mib=float(m))
  if u==UUIDS[-1] and ident['pid']==1749337 and ident['uid']==1001 and ident['start_tick']==641133475 and float(m)<=1024:oracle=ident
  else:foreign.append(ident)
 ready=oracle is not None and not foreign and all(x['free_mib']>=40000 and x['utilization']<=15 for x in selected)
 return dict(ready=ready,gpus=selected,foreign=foreign,oracle=oracle)
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--configuration',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);a=p.parse_args()
 b=a.base.resolve();source=a.source.resolve();assert source.is_relative_to(b) and os.getuid()==1001 and socket.gethostname().split('.')[0]=='xielab5'
 config=json.loads(a.configuration.read_text());proof=json.loads(a.proof.read_text())
 assert config['source']==str(source) and config['source_commit']==proof['source_commit']==(source/'SOURCE_COMMIT').read_text().strip()
 assert hashlib.sha256(Path(config['config']).read_bytes()).hexdigest()==config['config_sha256']
 for name,sha in proof['files'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha,name
 if (b/'receipts/start.json').exists():raise RuntimeError('A launch already happened; explicit state recovery required')
 r=b/'receipts';record=dict(time=time.time(),identity=process(os.getpid()),run_id=config['run_id'],configuration=str(a.configuration),source=str(source),source_commit=config['source_commit'],waiter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 with (r/'card-waiter-binding.json').open('x') as f:json.dump(record,f)
 while True:
  try:
   snapshot=check()
  except (subprocess.TimeoutExpired,subprocess.CalledProcessError,FileNotFoundError) as exc:
   snapshot=dict(ready=False,observation_error=repr(exc))
  snapshot.update(time=time.time(),waiter_identity=record['identity'],stage='WAIT_FOUR_AUTHORIZED_GPUS')
  with (r/'card-waiter-events.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n');f.flush()
  tmp=r/'card-waiter-live.json.tmp';tmp.write_text(json.dumps(snapshot));os.replace(tmp,r/'card-waiter-live.json')
  print(json.dumps(snapshot),flush=True)
  if snapshot['ready']:break
  time.sleep(30)
 env=dict(os.environ,RECURSIVE_BRAN_SOURCE=str(source),RECURSIVE_BRAN_CONFIGURATION=str(a.configuration))
 # Launcher rechecks after leases; any launch refusal terminates, never blindly retries.
 result=subprocess.run(['bash',str(source/'release/recursive_bran_b/launch_hkust4.sh')],env=env,cwd=source)
 with (r/'card-waiter-launch-terminal.json').open('x') as f:json.dump(dict(time=time.time(),actual_child_rc=result.returncode,binding=record),f)
 raise SystemExit(result.returncode)
if __name__=='__main__':main()
