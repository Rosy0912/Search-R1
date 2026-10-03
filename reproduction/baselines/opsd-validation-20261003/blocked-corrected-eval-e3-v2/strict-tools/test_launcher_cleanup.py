"""CPU real subprocesses only. No CUDA, Slurm, or cluster leases are used."""
import io, subprocess, sys, threading, time
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import hf_launcher as launcher

def spawn(body):
 return subprocess.Popen([sys.executable,'-u','-c',body],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)

class BrokenOutput:
 def write(self,data):raise OSError('CPU fixture stdout write failed')
 def flush(self):pass

@pytest.mark.parametrize('failure',['stdout','log'])
def test_stream_failure_stops_own_full_pipe_child_bounded(tmp_path,failure):
 child=spawn("import os,signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);os.write(1,b'READY\\n');\nwhile True:os.write(1,b'x'*65536)")
 assert child.stdout.readline()==b'READY\n'
 errors=[];failed=threading.Event()
 log=tmp_path/'reload.log'
 if failure=='log':log.write_bytes(b'EXISTING ORIGINAL EVIDENCE')
 output=BrokenOutput() if failure=='stdout' else io.BytesIO()
 reader=threading.Thread(target=launcher.stream_child,args=(child,log,output,failed,errors),daemon=True)
 reader.start();started=time.monotonic()
 try:
  assert failed.wait(3)
  with pytest.raises((OSError,FileExistsError)):
   launcher.wait_child(child,failed,errors)
 finally:launcher.finish_child(child,reader,timeout=.2)
 assert time.monotonic()-started<3
 assert child.poll() is not None and not reader.is_alive()
 assert len(errors)==1
 if failure=='log':assert log.read_bytes()==b'EXISTING ORIGINAL EVIDENCE'
 else:assert 'stdout write failed' in str(errors[0]) and log.stat().st_size>0

def test_terminate_escalates_only_own_child():
 child=spawn("import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('READY',flush=True);time.sleep(60)")
 assert child.stdout.readline()==b'READY\n'
 try:
  started=time.monotonic();code=launcher.stop_own_child(child,timeout=.2)
  assert code<0 and time.monotonic()-started<2
 finally:
  if child.poll() is None:child.kill();child.wait(timeout=2)
  child.stdout.close()

def test_success_preserves_every_raw_byte(tmp_path):
 child=spawn("import os;os.write(1,b'RAW\\x00\\xff\\n'*20000)")
 output=io.BytesIO();errors=[];failed=threading.Event();log=tmp_path/'reload.log'
 reader=threading.Thread(target=launcher.stream_child,args=(child,log,output,failed,errors),daemon=True);reader.start()
 try:assert launcher.wait_child(child,failed,errors)==0
 finally:launcher.finish_child(child,reader,timeout=2)
 assert not errors and log.read_bytes()==output.getvalue()==b'RAW\x00\xff\n'*20000
