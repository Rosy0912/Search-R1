"""Hold original fit frame and Ray workers until authenticated parent drain."""
import json,os,pathlib,socket,struct,time
from ipc import identity,read,send
from checkpoint_inventory import sha
from pipeline import atomic

def held_fit(original,trainer,config,binding):
 result=original(trainer)
 root=pathlib.Path(config['evidence_root']);b=json.loads(pathlib.Path(binding).read_text())
 outer=pathlib.Path(config['actual_outer_receipt_path'])
 if not outer.is_file():raise RuntimeError('Terminal original outer receipt absent')
 request=dict(operation='TERMINAL_HELD',identity=identity(os.getpid()),job=os.environ['SLURM_JOB_ID'],actual_outer_evidence={'path':str(outer),'sha256':sha(outer)},held_epoch=time.time(),effective_config_sha256=sha(root/'effective-runtime-config.json'))
 atomic(root/'trainer-terminal-held.json',request)
 sock=socket.socket(socket.AF_UNIX);sock.settimeout(14400)
 try:
  sock.connect(b['socket']);pid,uid,_=struct.unpack('3i',sock.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
  if uid!=os.getuid() or identity(pid)!=b['server_identity']:raise RuntimeError('Terminal actual parent peer changed')
  send(sock,dict(request,capability=b['capability']))
  response=read(sock)
  if response.get('status')!='AUTHORIZED_OWN_SHUTDOWN' or response.get('held_sha256')!=sha(root/'trainer-terminal-held.json'):raise RuntimeError('Unknown terminal authorization; no retry')
  atomic(root/'trainer-terminal-authorization-received.json',dict(response,received_epoch=time.time()))
 finally:sock.close()
 return result

def drain_monitors(guardian,sampler_stop,monitor,sampler_errors):
 guardian.finish();guardian.close()
 sampler_stop.set();monitor.join(45)
 if monitor.is_alive() or sampler_errors:raise RuntimeError('Terminal resource monitor did not drain cleanly')
 guardian.check()
 return {'scope':'ACTUAL_MONITORS_DRAINED_BEFORE_OWNED_SHUTDOWN','completed_epoch':time.time()}
