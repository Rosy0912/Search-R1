"""Existing three-hop route: UTF-8 Python source on stdin, never bulk weights."""
import shlex,subprocess,time
MAX_SOURCE_BYTES=4*1024*1024

def route():
 options='-T -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=2'
 worker='env -u LD_LIBRARY_PATH ssh '+options+' -S /dev/shm/topd-worker-connect-v2.sock h100-11 '+shlex.quote('env -u LD_LIBRARY_PATH python3 -')
 head='env -u LD_LIBRARY_PATH ssh '+options+' -S none h100-16 '+shlex.quote(worker)
 return ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-S','none','-o','ProxyCommand=nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host',head]

def send(source,*,runner=subprocess.run,attempts=1,retry_safe=False,timeout=900,pause=time.sleep):
 raw=source.encode('utf-8') if isinstance(source,str) else source
 if not isinstance(raw,bytes) or not raw or len(raw)>MAX_SOURCE_BYTES or b'\x00' in raw:raise ValueError('Only bounded UTF-8 Python source allowed')
 text=raw.decode('utf-8');compile(text,'owned-source-stdin','exec')
 if attempts<1 or attempts>4 or attempts>1 and not retry_safe:raise ValueError('Retries require caller immutable/idempotent submission record')
 for attempt in range(attempts):
  try:r=runner(route(),input=raw,capture_output=True,timeout=timeout)
  except (subprocess.TimeoutExpired,OSError):
   if attempt+1==attempts:raise
   pause(3);continue
  if r.returncode!=255 or attempt+1==attempts:return r
  pause(3)
 raise AssertionError('unreachable')
