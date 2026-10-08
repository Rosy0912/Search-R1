"""Explicit retained-controller proof, never synthesized scontrol terminal output."""
import hashlib,re
from datetime import datetime,timezone,timedelta

def verify_controller_exit(data,job,commit,pilot,own_file,digest,loads):
 assert set(data)=={'proof_kind','source_commit','job_id','node','cpus','controller_host','source_log_path','controller_origin','start_snapshot','head_config','source_commit_evidence'}
 assert data['proof_kind']=='controller_normal_exit' and data['job_id']==job
 assert data['source_commit']==commit and data['node']=='h100-10' and data['cpus']==192
 assert data['controller_host']=='h100-16' and data['source_log_path']=='/var/log/slurm/slurmctld.log'
 def retained(field):
  item=data[field];assert set(item)=={'path','sha256'}
  assert re.fullmatch('[0-9a-f]{64}',item['sha256'])
  p=own_file(item['path'],pilot)
  assert digest(p)==item['sha256'],('Controller proof auxiliary SHA mismatch',field)
  return p.read_text()
 assert retained('source_commit_evidence').strip()==commit
 start=retained('start_snapshot');config=retained('head_config')
 def unique(text,key,value):
  assert re.findall(r'\b'+re.escape(key)+r'\s*=\s*([^\s]+)',text)==[value],key
 for key,value in (('JobId',job),('JobState','RUNNING'),('ExitCode','0:0'),('NumNodes','1'),('NumCPUs','192'),('Partition','gpu'),('NodeList','h100-10'),('BatchHost','h100-10'),('Restarts','0'),('TresPerNode','gres:gpu:h100:8')):unique(start,key,value)
 for key,value in (('MinJobAge','300'),('AccountingStorageType','accounting_storage/none'),('JobCompType','jobcomp/none'),('SlurmctldLogFile','/var/log/slurm/slurmctld.log')):unique(config,key,value)
 assert re.search(r'^\s*SlurmctldHost(?:\[0\])?\s*=\s*h100-16(?:\([^\n]*\))?\s*$',config,re.M)
 origin=loads(retained('controller_origin'))
 assert origin['controller_host']==data['controller_host'] and origin['job_id']==job
 assert set(origin['files'])=={data['source_log_path']}
 source=origin['files'][data['source_log_path']]
 assert type(source['source_bytes']) is int and source['source_bytes']>0
 lines=source['matched_lines'];assert type(lines) is list and len(lines)==4
 assert source['matched_lines_sha256']==hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest()
 j=re.escape(job)
 patterns=[r'_slurm_rpc_submit_batch_job: JobId='+j+r' InitPrio=\d+ usec=\d+',r'sched: Allocate JobId='+j+r' NodeList=h100-10 #CPUs=192 Partition=gpu',r'_job_complete: JobId='+j+r' WEXITSTATUS 0',r'_job_complete: JobId='+j+r' done']
 times=[]
 for line,pattern in zip(lines,patterns):
  match=re.fullmatch(r'\[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3})\] '+pattern,line)
  assert match,('Unexpected/missing/conflicting controller event',line)
  times.append(datetime.fromisoformat(match[1]))
 assert times[0]<times[1]<times[2]<=times[3]
 for key,instant in (('SubmitTime',times[0]),('StartTime',times[1])):unique(start,key,instant.isoformat(timespec='seconds'))
 captured=datetime.fromisoformat(origin['retrieved_utc'])
 assert captured.utcoffset()==timedelta(0) and captured>=times[3].replace(tzinfo=timezone(timedelta(hours=8)))
 return {'proof_kind':'controller_normal_exit','job_id':job,'source_log_path':data['source_log_path'],'completion_controller_timestamp':times[3].isoformat(timespec='milliseconds'),'scontrol_terminal_record_present':False}
