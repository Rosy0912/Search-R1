import sys,json,time,os,hashlib,subprocess
from pathlib import Path
p=json.load(sys.stdin);n=Path(p['namespace']);q=Path('/proc')/str(p['identity']['pid']);current=None
if q.exists():
 s=(q/'stat').read_text();z=s[s.rfind(')')+2:].split();current={'pid':p['identity']['pid'],'start_ticks':int(z[19]),'sid':int(z[3]),'uid':q.stat().st_uid,'state':z[0],'cwd':os.readlink(q/'cwd')};current['birth_matches']=all(current[k]==p['identity'][k] for k in ('pid','start_ticks','sid','uid'));current['alive']=current['birth_matches'] and current['state'] not in ('Z','X')
state=None;sf=n/'reservation.json'
if sf.exists():
 raw=sf.read_bytes();assert len(raw)<65536;state=json.loads(raw);assert state['role']=='only_capacity_reservation' and state['identity']['pid']==p['identity']['pid'] and state['source_sha256']==p['source_sha256']
 state={k:state[k] for k in ('role','identity','status','physical_indices','bytes_per_gpu','source_sha256','source_backup_sha256','milestones','allocated_devices','visible_mapping','ready_samples','failure','release')}
 state['record_sha256']=hashlib.sha256(raw).hexdigest()
 backup=n/'source.py';assert backup.is_file() and hashlib.sha256(backup.read_bytes()).hexdigest()==p['source_sha256']
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=30);mapping={}
for row in raw.splitlines():
 v=[x.strip() for x in row.split(',')];mapping[v[1]]={'index':int(v[0]),'used_mib':int(v[2]),'util_percent':int(v[3])}
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=30);owned=[];foreign_training=[]
for row in apps.splitlines():
 v=[x.strip() for x in row.split(',')];i=mapping[v[0]]['index']
 if int(v[1])==p['identity']['pid']:owned.append({'gpu_index':i,'pid':int(v[1]),'used_mib':int(v[2]),'gpu_uuid':v[0]})
 elif i in [0,1,2,3]:foreign_training.append({'gpu_index':i,'pid':int(v[1]),'used_mib':v[2]})
for item in foreign_training:
 f=Path('/proc')/str(item['pid'])
 try:
  stat=(f/'stat').read_text();v=stat[stat.rfind(')')+2:].split()
  item['start_ticks']=int(v[19]);item['sid']=int(v[3]);item['uid']=f.stat().st_uid;item['state']=v[0]
  item['cwd_sha256']=hashlib.sha256(os.readlink(f/'cwd').encode()).hexdigest()
 except (FileNotFoundError,ProcessLookupError):item['proc_missing']=True
log=Path(p['log']);assert log.stat().st_size<=65536
last=None
for row in log.read_text(errors='replace').splitlines():
 if row.startswith('{'):
  try:last=json.loads(row)
  except ValueError:pass
out={'role':'actual_owned_capacity_reservation_bounded_observation','checked_unix':time.time(),'process':current,'state':state,'owned_cuda_applications':owned,'foreign_training_cuda_applications':foreign_training,'last_operator_log_json':last,'log_bytes':log.stat().st_size,'signals_sent':False,'trainer_started':False,'native_algorithm_accepted':False,'reservation_peak_is_training_peak':False}
if state and state['status']=='READY':
 assert current and current['alive'] and current['birth_matches'] and len(state['ready_samples'])==2 and state['physical_indices']==[1]
 assert sorted(x['gpu_index'] for x in owned)==[1] and all(x['used_mib']>=61440 for x in owned)
 out['actual_owned_60GiB_each_verified']=True
else:out['actual_owned_60GiB_each_verified']=False
other=p['other_owned_reservation'];op=Path('/proc')/str(other['identity']['pid']);other_current=None
if op.exists():
 s=(op/'stat').read_text();z=s[s.rfind(')')+2:].split();other_current={'pid':other['identity']['pid'],'start_ticks':int(z[19]),'sid':int(z[3]),'uid':op.stat().st_uid,'state':z[0]}
 other_current['birth_matches']=all(other_current[k]==other['identity'][k] for k in ('pid','start_ticks','sid','uid'));other_current['alive']=other_current['birth_matches'] and other_current['state'] not in ('Z','X')
out['other_owned_process']=other_current
oraw=(Path(other['namespace'])/'reservation.json').read_bytes();assert len(oraw)<65536;ost=json.loads(oraw);assert ost['source_sha256']==other['source_sha256'] and ost['identity']['pid']==other['identity']['pid'];assert hashlib.sha256((Path(other['namespace'])/'source.py').read_bytes()).hexdigest()==other['source_sha256']
out['other_owned_state']={k:ost[k] for k in ('status','source_sha256','physical_indices','bytes_per_gpu','ready_samples','allocated_devices','failure','release')};out['other_owned_state']['record_sha256']=hashlib.sha256(oraw).hexdigest()
otherapps=[x for x in foreign_training if x['pid']==other['identity']['pid']]
out['other_owned_cuda_applications']=otherapps
if other_current and other_current['alive'] and ost['status']=='READY':
 assert sorted(x['gpu_index'] for x in otherapps)==[0,3] and all(int(x['used_mib'])>=61440 for x in otherapps)
 out['other_actual_owned_60GiB_each_verified']=True
 for x in otherapps:assert all(x[k]==other['identity'][k] for k in ('pid','start_ticks','sid','uid'))
 out['foreign_training_cuda_applications']=[x for x in foreign_training if x['pid']!=other['identity']['pid']]
else:out['other_actual_owned_60GiB_each_verified']=False
out['failed_training_pipeline_birth_present']=False
fq=Path('/proc')/'2253031'
if fq.exists():
 fs=fq.joinpath('stat').read_text();fz=fs[fs.rfind(')')+2:].split();out['failed_training_pipeline_birth_present']=int(fz[19])==600558078 and fz[0] not in ('Z','X')
print(json.dumps(out))
