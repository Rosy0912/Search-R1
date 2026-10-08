import json,os,pathlib,socket,tempfile,threading,types,sys
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import terminal as t
from ipc import identity,read,send
from checkpoint_inventory import sha

@pytest.mark.skipif(sys.platform!='linux',reason='actual Linux SO_PEERCRED/proc native contract')
@pytest.mark.parametrize('authorized',[True,False])
def test_real_owned_unix_held_original_fit_waits_for_parent_ack(monkeypatch,authorized):
 monkeypatch.setenv('SLURM_JOB_ID','cpu-fixture')
 with tempfile.TemporaryDirectory(prefix='g1-terminal-') as directory:
  root=pathlib.Path(directory);outer=root/'actual-outer.json';outer.write_text('{"export_step":2}')
  (root/'effective-runtime-config.json').write_text('{}')
  server=socket.socket(socket.AF_UNIX);server.bind(str(root/'s'));server.listen(1)
  binding=root/'binding.json';binding.write_text(json.dumps({'socket':str(root/'s'),'capability':'private-capability','server_identity':identity(os.getpid())}))
  config={'evidence_root':str(root),'actual_outer_receipt_path':str(outer)};trainer=types.SimpleNamespace(alive=True);called=[];result=[];errors=[]
  def original(self):assert self is trainer and self.alive;called.append(1);return 42
  def worker():
   try:result.append(t.held_fit(original,trainer,config,binding));trainer.alive=False
   except BaseException as error:errors.append(error)
  thread=threading.Thread(target=worker);thread.start()
  connection,_=server.accept();request=read(connection)
  assert request.pop('capability')=='private-capability' and request['operation']=='TERMINAL_HELD'
  assert called==[1] and thread.is_alive() and trainer.alive and result==[]
  assert json.loads((root/'trainer-terminal-held.json').read_text())==request
  assert 'private-capability' not in (root/'trainer-terminal-held.json').read_text()
  send(connection,{'status':'AUTHORIZED_OWN_SHUTDOWN' if authorized else 'UNKNOWN','held_sha256':sha(root/'trainer-terminal-held.json')})
  thread.join(3);connection.close();server.close();assert not thread.is_alive()
  if authorized:assert result==[42] and not errors and not trainer.alive
  else:assert errors and not result and trainer.alive

def test_terminal_join_late_error_prevents_shutdown_authorization():
 errors=[];calls=[]
 class Guard:
  def finish(self):calls.append('finish')
  def close(self):calls.append('close')
  def check(self):
   if errors:raise RuntimeError('late actual protected error')
 class Monitor:
  def join(self,seconds):errors.append('late actual protected error')
  def is_alive(self):return False
 with pytest.raises(RuntimeError,match='late actual protected error'):
  t.drain_monitors(Guard(),threading.Event(),Monitor(),[])
 assert calls==['finish','close']

def test_hold_is_inside_original_fit_frame_before_ray_shutdown():
 source=pathlib.Path(t.__file__).with_name('train_entry.py').read_text()
 assert 'class HeldTrainer(original_entry.RayPPOTrainer)' in source
 assert 'return held_fit(original_fit,self,config,binding)' in source
 assert source.index('original_entry.RayPPOTrainer=HeldTrainer')<source.index('main_task._function(resolved)')<source.index('ray.shutdown()')
