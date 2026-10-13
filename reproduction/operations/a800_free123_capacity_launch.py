import sys,json,time,os,subprocess,hashlib
from pathlib import Path
if sys.flags.optimize:raise RuntimeError('Launch guards require optimization disabled')
p=json.load(sys.stdin);assert p['physical_indices']==[1,2,3]
source=Path(p['remote_source']);root=source.parent;assert source.resolve()==source and source.stat().st_uid==os.getuid() and hashlib.sha256(source.read_bytes()).hexdigest()==p['source_sha256']
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.total,memory.free','--format=csv,noheader,nounits'],text=True,timeout=30);gpus={}
for row in inventory.splitlines():
 v=[x.strip() for x in row.split(',')];i=int(v[0])
 if i in p['physical_indices']:gpus[v[1]]={'index':i,'uuid':v[1],'pci_bus_id':v[2],'total_mib':int(v[3]),'free_mib':int(v[4])}
assert sorted(x['index'] for x in gpus.values())==[1,2,3] and all(x['free_mib']>=62464 for x in gpus.values())
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=30)
for row in apps.splitlines():
 v=[x.strip() for x in row.split(',')];assert v[0] not in gpus,'Selected capacitycard occupied; noPopen/noCUDAclaim'
preflight={'role':'actual_free123_capacity_prelaunch_resource_read','checked_unix':time.time(),'gpus':list(gpus.values()),'selected_compute_apps':[],'GPU_operation_performed':False}
claim=root/('launch-'+p['id']+'.json');log=root/('reservation-'+p['id']+'.log');assert not log.exists() and not Path(p['namespace']).exists()
with claim.open('x') as f:json.dump({'role':'exclusive_only_capacity_reservation_launch_claim','intent':p,'claimed_unix':time.time(),'no_retry':True,'actual_preflight':preflight},f,indent=2);f.flush();os.fsync(f.fileno())
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,2,3',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTORCH_CUDA_ALLOC_CONF='backend:native',PYTORCH_ALLOC_CONF='backend:native')
with log.open('xb') as out:proc=subprocess.Popen(['/data/search-g1/env/bin/python','-I',str(source),'--namespace',p['namespace'],'--expected-source-sha256',p['source_sha256']],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
q=Path('/proc')/str(proc.pid);s=(q/'stat').read_text();z=s[s.rfind(')')+2:].split();identity={'pid':proc.pid,'start_ticks':int(z[19]),'sid':int(z[3]),'uid':q.stat().st_uid,'pgid':int(z[2]),'euid':int(next(x for x in (q/'status').read_text().splitlines() if x.startswith('Uid:')).split()[2])};assert identity['pid']==identity['sid'] and identity['uid']==os.getuid()
d={'role':'actual_only_capacity_reservation_process_launch','identity':identity,'source_path':str(source),'source_sha256':p['source_sha256'],'namespace':p['namespace'],'log':str(log),'physical_indices':p['physical_indices'],'bytes_per_gpu':p['bytes_per_gpu'],'claimed_unix':time.time(),'reservation_ready':False,'training_started':False,'native_algorithm_accepted':False,'no_retry':True,'actual_preflight':preflight,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'hostname':__import__('socket').gethostname(),'launcher_optimize':sys.flags.optimize,'environment':{k:env[k] for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF')},'capacity_exclusivity':'NONEXCLUSIVE','native_training_admission':False};receipt=root/('launch-receipt-'+p['id']+'.json')
with receipt.open('x') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
fd=os.open(root,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
try:os.fsync(fd)
finally:os.close(fd)
print(json.dumps({**d,'receipt':str(receipt)}))
