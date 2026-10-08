
import base64,datetime,hashlib,json,os,re,stat,subprocess
from pathlib import Path
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'scope':'OWN_JOB3261_CONTROLLER_LOG_OBSERVATION_NOT_RECONSTRUCTED_SCONTROL','configuration_selected_original':[],'files':{},'commands':{}}
p=subprocess.run(['scontrol','show','config'],capture_output=True,timeout=20);result['commands']['config']={'returncode':p.returncode,'stderr_base64':base64.b64encode(p.stderr).decode()}
values={}
for line in p.stdout.decode(errors='replace').splitlines():
 k=line.split('=',1)[0].strip()
 if k in ['SlurmctldLogFile','SlurmdLogFile','SlurmctldHost','StateSaveLocation','MinJobAge','AccountingStorageType']:
  result['configuration_selected_original'].append(line);values[k]=line.split('=',1)[1].strip()
pattern=re.compile(rb'(?:JobId=|job_id=|JobID=|Job |job |JobId |jobid=)3261(?:\D|$)|\b3261\.(?:batch|extern|[0-9]+)\b')
log=Path(values.get('SlurmctldLogFile','/var/log/slurm/slurmctld.log'))
for q in [log,Path(str(log)+'.1')]:
 try:
  assert q.is_absolute() and q.resolve()==q and q.stat().st_uid==0 and stat.S_ISREG(q.stat().st_mode)
  fd=os.open(q,os.O_RDONLY|os.O_NOFOLLOW);h=hashlib.sha256();lines=[];offset=0
  with os.fdopen(fd,'rb') as f:
   b=os.fstat(f.fileno())
   for raw in f:
    h.update(raw)
    if pattern.search(raw):lines.append({'offset':offset,'bytes':len(raw),'line_base64':base64.b64encode(raw).decode()})
    offset+=len(raw)
   a=os.fstat(f.fileno())
  result['files'][str(q)]={'observed_prefix_bytes':offset,'sha256':h.hexdigest(),'own_job_matched_lines':lines,'uid':b.st_uid,'device':b.st_dev,'inode':b.st_ino,'size_before':b.st_size,'size_after':a.st_size}
 except Exception as e:result['files'][str(q)]={'error_type':type(e).__name__,'error':str(e)}
print(json.dumps(result,sort_keys=True))
