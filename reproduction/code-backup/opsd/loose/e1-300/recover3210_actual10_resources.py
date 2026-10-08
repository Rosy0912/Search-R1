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
out=run/'evidence/independent-actual10-20261004T183936Z'
samples=[json.loads(l) for l in (out/'pause-resources.ndjson').read_text().splitlines()];rawsnapshots=json.loads((out/'nvml-original.json').read_text());history=json.loads((out/'history-metadata.json').read_text())
for name in ('pause-resources.ndjson','nvml-original.json','history-metadata.json','fresh-http-top3-24.json'):store(out/name)
store(run/'acceptance-5-receipt.json')
current={str(p):proc(p) for p in sorted(wanted)};direct=pathlib.Path(os.readlink('/proc/'+str(m['driver_pid'])+'/fd/1'));store(direct)
record={'node':socket.gethostname(),'uid':os.getuid(),'job_id':'3210','source_commit':m['source_commit'],'source_clean':True,'slurm':subprocess.check_output(['scontrol','show','job','3210','-o'],text=True),'checkpoint_manifest_sha256':sha(mp.read_bytes()),'files':files,'rank_reports':reports,'samples':samples,'raw_nvml_snapshots':rawsnapshots,'current_processes':current,'history':history,'remote_evidence_dir':str(out),'receipt_exists':True,'resolved_profile_raw_sha256':sha((run/'resolved-profile.json').read_bytes()),'direct_stdout_path':str(direct),'scope':'Recovered successful original fresh actual10 evidence after transport timeout; snapshots/history/HTTP not rerun; current scalar process/source/slurm recovery read'}
print(json.dumps(record,sort_keys=True))
'''
p=subprocess.run([sys.executable,str(H.parent/'remote.py'),'python3 -c '+shlex.quote(code)],capture_output=True,timeout=75)
label=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=H/('formal3210-actual10-resource-recovered-'+label);out.mkdir();(out/'transport.stdout').write_bytes(p.stdout);(out/'transport.stderr').write_bytes(p.stderr)
assert p.returncode==0,p.stderr.decode();v=json.loads(p.stdout);(out/'collection-original.json').write_bytes(p.stdout)
for path,r in v['files'].items():
 b=base64.b64decode(r['base64']);assert hashlib.sha256(b).hexdigest()==r['sha256'];local=(out/'run'/path.split('/runs/formal-3210/',1)[1]) if '/runs/formal-3210/' in path else (out/'direct'/Path(path).name);local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(b)
print(json.dumps({'collection':str(out),'remote_evidence_dir':v['remote_evidence_dir'],'manifest_sha256':v['checkpoint_manifest_sha256'],'history_raw_bytes':v['history'].get('raw_bytes'),'source':v['source_commit'],'actual10_scope':True}))
