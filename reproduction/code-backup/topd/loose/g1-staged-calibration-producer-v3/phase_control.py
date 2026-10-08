"""Private CPU phase requests before actual CUDA starts/controlled exits."""
import hashlib,json,os,pathlib,time

def notify(operation,phase,config_path,directory):
 root=os.environ.get('G1_GUARDIAN_CONTROL')
 if not root:raise RuntimeError('Actual supervised phase control required')
 root=pathlib.Path(root);config_path=pathlib.Path(config_path);d=json.loads(config_path.read_text())
 stat=pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
 record=dict(operation=operation,phase=phase,outer_step=d['outer_step'],pid=os.getpid(),ppid=os.getppid(),starttime_ticks=int(stat[19]),job=os.environ['SLURM_JOB_ID'],cgroup=pathlib.Path('/proc/self/cgroup').read_text(),config_path=str(config_path),config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),directory=str(directory),epoch=time.time())
 name='%d-%s-%s-%d'%(d['outer_step'],phase,operation,os.getpid());request=root/(name+'.request.json');ack=root/(name+'.ack.json')
 temporary=request.with_suffix('.tmp')
 with temporary.open('x') as f:os.chmod(temporary,0o600);json.dump(record,f);f.flush();os.fsync(f.fileno())
 os.link(temporary,request);temporary.unlink()
 fd=os.open(root,os.O_RDONLY);os.fsync(fd);os.close(fd)
 began=time.monotonic()
 while not ack.exists():
  if time.monotonic()-began>45:raise TimeoutError('No actual phase guardian acknowledgement')
  time.sleep(.05)
 reply=json.loads(ack.read_text())
 if reply.get('status')!='AUTHORIZED_TRANSITION' or reply.get('request_sha256')!=hashlib.sha256(request.read_bytes()).hexdigest():raise RuntimeError('Unknown guardian phase acknowledgement')
