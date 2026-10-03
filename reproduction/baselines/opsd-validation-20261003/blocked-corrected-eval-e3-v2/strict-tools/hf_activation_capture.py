"""CPU/process/lease capture for a NEW reviewed launcher; never loads a model.
This standalone preview entry remains exit78. No production launcher was changed.
"""
import fcntl,json,os,re,socket
from pathlib import Path
from training_evidence import digest
from runtime_capture import process_identity,checker_identity

def lock_records(raw,stat):
 wanted=(os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino);records=[]
 for line in raw.splitlines():
  fields=line.split()
  if len(fields)<8 or fields[1:4]!=['FLOCK','ADVISORY','WRITE']:continue
  bits=fields[5].split(':')
  if len(bits)!=3:continue
  try:identity=(int(bits[0],16),int(bits[1],16),int(bits[2]))
  except ValueError:continue
  if identity==wanted:records.append(line)
 if len(records)!=1:raise ValueError('Exactly one actual own lock inode FLOCK WRITE record required')
 return records

def capture_lease(fd,lock_path,proc=Path('/proc')):
 """Reassert inherited open-file-description flock; reject foreign held descriptor.
No unlock. Must be called by same own launcher lineage with inherited fd9.
"""
 path=Path(lock_path).resolve();opened=os.fstat(fd);target=path.stat()
 if (opened.st_dev,opened.st_ino)!=(target.st_dev,target.st_ino):raise ValueError('Inherited lease fd is not own gpu-job.lock')
 before=(proc/'locks').read_text();lock_records(before,target)
 # Reasserting this already locked descriptor proves the inherited fd owns it.
 # If a foreign descriptor owns the lock, this fails instead of accepting inode labels.
 fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 after=(proc/'locks').read_text();records=lock_records(after,target)
 return dict(lock_path=str(path),fd=fd,device=target.st_dev,inode=target.st_ino,raw_locks_before=before,raw_locks_after=after,matched_locks=records,reasserted_same_descriptor=True)

def capture_hf_activation(launcher_pid,checker_path,job,selected_uuid,lock_path,fd=9,proc=Path('/proc')):
 if not re.fullmatch('[0-9]+',job) or os.environ.get('SLURM_JOB_ID')!=job:raise ValueError('Actual own Slurm job required')
 own=process_identity(os.getpid(),proc);identity=checker_identity(launcher_pid,checker_path,job,proc)
 for record in (own,identity['launcher'],identity['checker']):
  if record['uid']!=os.getuid() or any(uid!=os.getuid() for uid in record['uids']) or not re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',record['cgroup']):raise ValueError('Launcher/checker/capture actual UID/cgroup mismatch')
 if not selected_uuid or not selected_uuid.startswith('GPU-'):raise ValueError('Actual selected UUID missing')
 lease=capture_lease(fd,lock_path,proc)
 return dict(schema='actual-hf500-activation-v1',job_id=job,node=socket.gethostname().split('.')[0],uid=os.getuid(),gpu_uuid=selected_uuid,checker_pid=identity['checker']['pid'],checker_start_ticks=identity['checker']['process_start_ticks'],own_cgroup=own['cgroup'],capture_process=own,launcher=identity['launcher'],checker=identity['checker'],checker_sha256=identity['checker_sha256'],lease=lease,capture_source_sha256=digest(Path(__file__)))

def validate_process_raw(record,job,uid):
 fields=record['raw_stat'].rsplit(')',1)[1].split();status=dict(line.split(':',1) for line in record['raw_status'].splitlines() if ':' in line)
 if not record['raw_stat'].startswith(str(record['pid'])+' ') or fields[19]!=record['process_start_ticks'] or int(fields[1])!=record['ppid']:raise ValueError('PID/start/parent differs from raw stat')
 raw_uid=[int(value) for value in status['Uid'].split()]
 if raw_uid!=record['uids'] or any(value!=uid for value in raw_uid) or record['uid']!=uid:raise ValueError('UID differs from raw process status')
 if record['cgroup']!=record['raw_cgroup'] or not re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',record['raw_cgroup']):raise ValueError('Raw process outside own job')
 command=[part.decode() for part in bytes.fromhex(record['raw_cmdline_hex']).split(b'\0') if part]
 if not command or command!=record['cmdline']:raise ValueError('Command differs from raw process bytes')

def validate_activation_audit(audit,job,checker_path,lock_path,capture_sha):
 if audit['schema']!='actual-hf500-activation-v1' or audit['job_id']!=job or audit['node']!='h100-10' or audit['capture_source_sha256']!=capture_sha:raise ValueError('Activation audit job/node/source mismatch')
 uid=audit['uid']
 for name in ('capture_process','launcher','checker'):validate_process_raw(audit[name],job,uid)
 checker=audit['checker']
 if audit['checker_pid']!=checker['pid'] or audit['checker_start_ticks']!=checker['process_start_ticks'] or str(Path(checker_path).resolve()) not in checker['cmdline'] or audit['checker_sha256']!=digest(checker_path):raise ValueError('Actual checker identity/source mismatch')
 lease=audit['lease'];target=Path(lock_path).resolve();stat=target.stat()
 if lease['lock_path']!=str(target) or lease['fd']!=9 or lease['device']!=stat.st_dev or lease['inode']!=stat.st_ino or lease['reasserted_same_descriptor'] is not True:raise ValueError('Own lease identity mismatch')
 if lock_records(lease['raw_locks_after'],stat)!=lease['matched_locks']:raise ValueError('Kernel lock raw record mismatch')
 lock_records(lease['raw_locks_before'],stat)
 return True

if __name__=='__main__':
 print('BLOCKED CAPTURE PREVIEW: integrate into new independently reviewed HF launcher before activation',file=__import__('sys').stderr)
 raise SystemExit(78)

def emit_activation(audit,evidence):
 directory=Path(evidence);raw=json.dumps(audit,sort_keys=True,allow_nan=False)
 with (directory/'hf-activation.json').open('x') as handle:handle.write(raw+'\n')
 with (directory/'hf-activation.raw.log').open('x') as handle:handle.write('ACTUAL_HF500_ACTIVATION_AUDIT_JSON='+raw+'\n')
 return audit
