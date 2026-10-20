from pathlib import Path
import hashlib,json,os,subprocess,time
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def headroom():
 host=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
 f=Path('/sys/fs/cgroup/memory.max')
 if f.exists() and f.read_text().strip()!='max':
  limit=int(f.read_text());m=dict((k,int(v)) for k,v in (l.split() for l in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines()));return min(host,limit-sum(m.get(k,0) for k in ['anon','kernel','file_dirty','file_writeback']))
 return host
r=Path(B['root']);pf=Path(SUB['plan']);assert sha(pf)==B['plan_sha256'];p=json.loads(pf.read_text())
assert all(json.loads((r/'submission.json').read_text())[k]==SUB[k] for k in ['job_id','job','plan_sha256','group','source'])
rows=[l.split('|') for l in subprocess.check_output(['sacct','-X','-n','-P','-j',B['job_id'],'--format=JobIDRaw,User,JobName%100,State,ExitCode,Start,End,AllocTRES%250'],text=True).splitlines() if l.strip()]
rows=[x for x in rows if x[0]==B['job_id']];assert len(rows)==1 and rows[0][1:3]==['jmzhang',B['job_name']];row=rows[0]
tres=dict(x.split('=',1) for x in row[7].split(',') if '=' in x);s=dict(B,at=time.time(),state=row[3],scheduler_exit=row[4],resources=dict(gpus=int(tres.get('gres/gpu',0)),cpus=int(tres.get('cpu',0)),memory_gib=int(tres.get('mem','0G').removesuffix('G'))),scheduler=row,worker_exit=(r/'exit.code').read_text().strip() if (r/'exit.code').exists() else None,parent_wait_rc=None,entry_exit_rc=None,complete_exists=False,manifest_exists=False,terminal_clock_order_verified=False,current_entry_bound=False,configuration_projection_verified=False,eight_uuid_entry=False,cpu_headroom_bytes=headroom(),all_payload_files_present=False,stable_payload_metadata=False)
small={}
for n in ['entry-once.json','receipts/current-entry.json','receipts/configuration.json','receipts/resolved.yaml','actual-parent-wait.json','exit.code','submission.json','submission-intent.json']:
 f=r/n
 if f.is_file():assert f.stat().st_size<2*1024**2;small[n]=f
if (r/'receipts/current-entry.json').exists():
 e=json.loads((r/'receipts/current-entry.json').read_text());assert e['allocation']==B['job_id'] and e['run_id']==B['run_id'] and e['group']==B['group'] and e['output']==B['output']
 s['current_entry_bound']=True;s['resolved_sha256']=sha(r/'receipts/resolved.yaml')
 cfg=json.loads((r/'receipts/configuration.json').read_text());assert cfg['config_sha256']==s['resolved_sha256']==e['config_sha256'] and cfg['fresh_authorized'] and cfg['parent_checkpoint'] is None
 import sys;sys.path.insert(0,p['source']);from omegaconf import OmegaConf
 from reproduction.sevenb_ablation_lumia_resume_v1.fresh_contract import BASE,project_config
 old=OmegaConf.to_container(OmegaConf.load(BASE/'parent-small'/(B['group']+'-resolved.yaml')),resolve=True)
 assert OmegaConf.to_container(OmegaConf.load(r/'receipts/resolved.yaml'),resolve=True)==project_config(old,B['group'],e['host']);s['configuration_projection_verified']=True
 output=Path(B['output']);ent=json.loads((r/'entry-once.json').read_text());s['eight_uuid_entry']=ent['job_id']==B['job_id'] and ent['plan_sha256']==B['plan_sha256'] and len(ent['gpus'])==8 and len({v.split(',')[1].strip() for v in ent['gpus']})==8 and all(float(v.split(',')[-1])>=44000 for v in ent['gpus'])
 for n in ['training-complete.json','entry-exit.json','progress.json']:
  f=output/n
  if f.is_file():assert f.stat().st_size<2*1024**2;small['output/'+n]=f
 if (output/'entry-exit.json').exists():s['entry_exit_rc']=json.loads((output/'entry-exit.json').read_text())['rc']
 if (r/'actual-parent-wait.json').exists():s['parent_wait_rc']=json.loads((r/'actual-parent-wait.json').read_text())['rc']
 if (output/'training-complete.json').exists():s['training_complete']=json.loads((output/'training-complete.json').read_text());s['complete_exists']=True
 if s['complete_exists'] and s['entry_exit_rc'] is not None and s['parent_wait_rc'] is not None:
  from datetime import datetime,timezone,timedelta
  if row[3]=='COMPLETED':
   start=datetime.fromisoformat(row[5]).replace(tzinfo=timezone(timedelta(hours=8))).timestamp();end=datetime.fromisoformat(row[6]).replace(tzinfo=timezone(timedelta(hours=8))).timestamp()
   s['terminal_clock_order_verified']=start<=ent['at']<=e['time']<=s['training_complete']['time']<=json.loads((output/'entry-exit.json').read_text())['at']<=json.loads((r/'actual-parent-wait.json').read_text())['at']<=end+2
f=Path(B['native'])/'manifest.json'
if f.exists():
 assert f.stat().st_size<2*1024**2;s.update(manifest_exists=True,manifest=json.loads(f.read_text()),manifest_sha256=sha(f));small['manifest.json']=f
 members=[Path(x['path']) for x in s['manifest']['worker_states']+[s['manifest']['driver']]]
 s['all_payload_files_present']=len(members)==9 and all(x.is_file() and not x.is_symlink() and x.stat().st_size>0 and x.parent==Path(B['native']) for x in members)
 if s['all_payload_files_present']:
  before={str(x):(x.stat().st_ino,x.stat().st_size,x.stat().st_mtime_ns) for x in members};time.sleep(1);s['stable_payload_metadata']=all(before[str(x)]==(x.stat().st_ino,x.stat().st_size,x.stat().st_mtime_ns) for x in members) and sha(f)==s['manifest_sha256']
s['small_paths']={k:str(v) for k,v in small.items()}
print('READINESS='+json.dumps(s),flush=True)
