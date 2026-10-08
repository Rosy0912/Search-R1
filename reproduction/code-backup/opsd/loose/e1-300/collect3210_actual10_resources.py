"""Authorized actual10 fresh resource/source/local HTTP evidence; stored checkpoint remains5."""
from pathlib import Path
import subprocess,sys,shlex,json,base64,hashlib,datetime
H=Path(__file__).resolve().parent
code=r'''
import pathlib,json,hashlib,base64,subprocess,os,socket,csv,io,datetime,time,gzip,re,urllib.request
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004');run=root/'runs/formal-3210';repo=root/'repo-a0968c2d24be';state=run/'full_state/global_step_5';mp=state/'manifest.json';sha=lambda b:hashlib.sha256(b).hexdigest();m=json.loads(mp.read_text());assert m['job_id']=='3210' and m['source_commit']=='a0968c2d24beeaa0400f60e4e34677fe6deef9c9' and m['driver_pid']==591546 and m['actual_outer_updates']==m['global_step']==5 and m['next_global_step']==6
assert socket.gethostname()=='h100-10' and os.getuid()==0
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==m['source_commit'] and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
files={}
def store(p):
 b=p.read_bytes();assert p.resolve()==p and p.stat().st_uid==0;files[str(p)]={'base64':base64.b64encode(b).decode(),'sha256':sha(b),'bytes':len(b),'uid':p.stat().st_uid};return b
store(mp)
reports=[]
for rank in range(8):
 rec=m['files'][f'actor/training_state/rank_{rank}.json'];p=pathlib.Path(rec['path']);b=store(p);assert sha(b)==rec['sha256'];r=json.loads(b);assert r['rank']==rank and r['status']=='EXACT_FULL_STATE_RELOAD_OK' and r['step']==5 and r['world_size']==8;reports.append(r)
for n in ('train.log','resolved-profile.json','bindings/manifest.json','evidence/service.json','evidence/http-top3-24.json','evidence/leases.json'):
 p=run/n
 if p.exists():store(p)
service=json.loads((run/'evidence/service.json').read_text())['owner'];wanted={r['pid'] for r in reports}|{m['driver_pid'],service['pid']};assert len(wanted)==10

def proc(pid):
 p=pathlib.Path('/proc')/str(pid);status=(p/'status').read_text();cg=(p/'cgroup').read_text();fields=dict(l.split(':',1) for l in status.splitlines() if ':' in l);stat=(p/'stat').read_text().rsplit(')',1)[1].split();uids=fields['Uid'].split();assert uids==['0']*4 and re.search(r'(?:^|/)job_3210(?:/|$)',cg,re.M) and fields['State'].strip()[0] not in ('Z','X')
 return {'pid':pid,'uid':uids[0],'uids':uids,'rss_kib':int(fields.get('VmRSS','0 kB').split()[0]),'peak_rss_kib':int(fields.get('VmHWM','0 kB').split()[0]),'cpu_user_ticks':int(stat[11]),'cpu_system_ticks':int(stat[12]),'process_start_ticks':stat[19],'in_slurm_job':True,'cgroup':cg,'name':fields['Name'].strip(),'state':fields['State'].strip()[0],'cmdline':(p/'cmdline').read_bytes().decode().replace('\0',' ')}
samples=[];rawsnapshots=[]
for i in range(2):
 row={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'3210','scope':'Unfiltered full current actual10 active NVML snapshot; no silent query-error/PID filtering'}
 try:
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=10);apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
  row['gpus']=[{'index':r[0].strip(),'uuid':r[1].strip(),'util':float(r[2]),'used_mib':float(r[3]),'total_mib':float(r[4])} for r in csv.reader(gpu.splitlines())];row['gpu_processes']=[]
  for a in csv.reader(apps.splitlines()):
   assert len(a)==3
   # Every returned GPU application row is retained; any unowned PID fails.
   row['gpu_processes'].append({'uuid':a[0].strip(),'gpu_used_mib':a[2].strip(),**proc(int(a[1]))})
  row['job_processes']=[proc(p) for p in sorted(wanted)];rawsnapshots.append({'time':row['time'],'gpu_raw':gpu,'apps_raw':apps})
 except Exception as e:row['query_error']=type(e).__name__;row['query_error_message']=str(e)
 samples.append(row)
 if i==0:time.sleep(2)
assert all('query_error' not in r for r in samples)
for r in samples:
 assert len(r['gpus'])==8 and len({v['uuid'] for v in r['gpus']})==8 and {p['pid'] for p in r['gpu_processes']}=={v['pid'] for v in reports}|{service['pid']}
 assert len([p for p in r['gpu_processes'] if p['pid']==service['pid']])==8
 assert next(p for p in r['job_processes'] if p['pid']==service['pid'])['process_start_ticks']==service['start']
 for report in reports:assert len([p for p in r['gpu_processes'] if p['pid']==report['pid']])==1
raw=(run/'resources.ndjson').read_bytes();compressed=gzip.compress(raw,mtime=0);assert gzip.decompress(compressed)==raw
history={'source_path':str(run/'resources.ndjson'),'raw_bytes':len(raw),'raw_sha256':sha(raw),'gzip_bytes':len(compressed),'gzip_sha256':sha(compressed),'gzip_base64':base64.b64encode(compressed).decode(),'last_line_complete':raw.endswith(b'\n'),'scope':'Lossless entire original prefix through captured read, no lines filtered; sampler remains running'}
label=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=run/'evidence'/('independent-actual10-'+label);out.mkdir(mode=0o700)
pause=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in samples).encode()
for name,b in [('pause-resources.ndjson',pause),('resources-history.ndjson.gz',compressed),('nvml-original.json',json.dumps(rawsnapshots,sort_keys=True).encode())]:
 with (out/name).open('xb') as f:f.write(b)
 store(out/name)
metadata={k:v for k,v in history.items() if k!='gzip_base64'}
with (out/'history-metadata.json').open('x') as f:json.dump(metadata,f,sort_keys=True,indent=2)
store(out/'history-metadata.json')
current={str(p):proc(p) for p in sorted(wanted)}
assert current[str(m['driver_pid'])]['process_start_ticks']=='1819179500'
body=json.dumps(dict(queries=json.loads((run/'evidence/http-top3-24.json').read_text())['queries'],topk=3,return_scores=True)).encode();httpstart=time.perf_counter();httpbefore=proc(service['pid'])
req=urllib.request.Request('http://127.0.0.1:18084/retrieve',data=body,headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=30) as response:httpraw=response.read()
httpafter=proc(service['pid']);assert httpafter['process_start_ticks']==httpbefore['process_start_ticks']==service['start']
http={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'3210','url':'http://127.0.0.1:18084/retrieve','queries':json.loads(body)['queries'],'topk':3,'response':json.loads(httpraw),'raw_response_sha256':sha(httpraw),'seconds':time.perf_counter()-httpstart,'service_before':httpbefore,'service_after':httpafter,'scope':'Fresh actual10 same-node request; API conformance not accuracy proof'}
with (out/'fresh-http-top3-24.json').open('x') as f:json.dump(http,f,sort_keys=True,indent=2)
store(out/'fresh-http-top3-24.json')
store(run/'acceptance-5-receipt.json')
# Resolve current direct stdout from actual driver FD1, not an assumed historical path.
direct=pathlib.Path(os.readlink('/proc/'+str(m['driver_pid'])+'/fd/1'));store(direct)
current_config_sha256=sha((run/'resolved-profile.json').read_bytes())
record={'node':socket.gethostname(),'uid':os.getuid(),'job_id':'3210','source_commit':m['source_commit'],'source_clean':True,'slurm':subprocess.check_output(['scontrol','show','job','3210','-o'],text=True),'checkpoint_manifest_sha256':sha(mp.read_bytes()),'files':files,'rank_reports':reports,'samples':samples,'raw_nvml_snapshots':rawsnapshots,'current_processes':current,'history':history,'remote_evidence_dir':str(out),'receipt_exists':True,'resolved_profile_raw_sha256':current_config_sha256,'direct_stdout_path':str(direct),'scope':'Fresh actual10 resource/source/PID/API collection; stored native/HF checkpoint remains5, no rawAdam200/cursor5120stored claim'}
print(json.dumps(record,sort_keys=True))
'''
p=subprocess.run([sys.executable,str(H.parent/'remote.py'),'python3 -c '+shlex.quote(code)],capture_output=True,timeout=75)
label=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=H/('formal3210-actual10-resource-independent-'+label);out.mkdir();(out/'transport.stdout').write_bytes(p.stdout);(out/'transport.stderr').write_bytes(p.stderr)
assert p.returncode==0,p.stderr.decode();v=json.loads(p.stdout);(out/'collection-original.json').write_bytes(p.stdout)
for path,r in v['files'].items():
 b=base64.b64decode(r['base64']);assert hashlib.sha256(b).hexdigest()==r['sha256'];local=(out/'run'/path.split('/runs/formal-3210/',1)[1]) if '/runs/formal-3210/' in path else (out/'direct'/Path(path).name);local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(b)
print(json.dumps({'collection':str(out),'remote_evidence_dir':v['remote_evidence_dir'],'manifest_sha256':v['checkpoint_manifest_sha256'],'history_raw_bytes':v['history']['raw_bytes'],'source':v['source_commit'],'actual10_scope':True}))
