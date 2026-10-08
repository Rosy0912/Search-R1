import hashlib,json,os,pathlib,sys,threading,time
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import phase_control

def local_proc_fixture(monkeypatch):
 # Mac local fixture supplies only /proc text; native Linux reads actual own /proc.
 if pathlib.Path('/proc/self/stat').exists():return
 real=pathlib.Path.read_text
 def read(self,*args,**kwargs):
  if str(self)=='/proc/self/stat':return '%d (CPUfixture) '%os.getpid()+' '.join(['S',str(os.getppid())]+['0']*17+['123'])
  if str(self)=='/proc/self/cgroup':return '0::/CPU_CONTRACT_NO_GPU'
  return real(self,*args,**kwargs)
 monkeypatch.setattr(pathlib.Path,'read_text',read)

@pytest.mark.parametrize('valid',[True,False])
def test_actual_request_atomic_publication_and_unknown_ack(tmp_path,monkeypatch,valid):
 local_proc_fixture(monkeypatch);monkeypatch.setenv('SLURM_JOB_ID','CPUfixture');monkeypatch.setenv('G1_GUARDIAN_CONTROL',str(tmp_path));config=tmp_path/'config.json';config.write_text(json.dumps(dict(outer_step=0)))
 records=[]
 def ack():
  began=time.monotonic()
  while not list(tmp_path.glob('*.request.json')):
   if time.monotonic()-began>2:return
   time.sleep(.01)
  p=next(tmp_path.glob('*.request.json'));r=json.loads(p.read_text());records.append(r)
  a=p.with_name(p.name.replace('.request.json','.ack.json'));tmp=a.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status='AUTHORIZED_TRANSITION' if valid else 'UNKNOWN',request_sha256=hashlib.sha256(p.read_bytes()).hexdigest())));os.replace(tmp,a)
 thread=threading.Thread(target=ack);thread.start()
 if valid:phase_control.notify('BEGIN','A',config,tmp_path)
 else:
  with pytest.raises(RuntimeError,match='Unknown guardian'):phase_control.notify('BEGIN','A',config,tmp_path)
 thread.join(2);assert not thread.is_alive() and len(records)==1 and records[0]['pid']==os.getpid()
 assert records[0]['config_sha256']==hashlib.sha256(config.read_bytes()).hexdigest()
 assert not list(tmp_path.glob('*.request.tmp'))
