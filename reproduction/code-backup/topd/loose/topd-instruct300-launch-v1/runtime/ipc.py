"""Bounded metadata-only local owned supervisor IPC. No pickle/tensors/capability logging."""
import hashlib,json,os,pathlib,socket,struct
MAX_FRAME=2*1024**2
def identity(pid):
 p=pathlib.Path('/proc')/str(pid)
 return {'pid':pid,'starttime_ticks':int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),'cgroup':(p/'cgroup').read_text()}
def send(sock,value):
 b=json.dumps(value,sort_keys=True).encode()
 if len(b)>MAX_FRAME:raise ValueError('metadata frame too large')
 sock.sendall(struct.pack('!I',len(b))+b)
def read(sock):
 def exact(n):
  data=b''
  while len(data)<n:
   chunk=sock.recv(n-len(data))
   if not chunk:raise EOFError('unknown supervisor ACK; no retry')
   data+=chunk
  return data
 n=struct.unpack('!I',exact(4))[0]
 if n>MAX_FRAME:raise ValueError('metadata frame too large')
 return json.loads(exact(n))
def client(binding,request):
 b=json.loads(pathlib.Path(binding).read_text());sock=socket.socket(socket.AF_UNIX);sock.settimeout(14400)
 try:
  sock.connect(b['socket']);pid,uid,gid=struct.unpack('3i',sock.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
  actual=identity(pid)
  if uid!=os.getuid() or actual!=b['server_identity']:raise ValueError('supervisor actual peer changed')
  send(sock,dict(request,capability=b['capability']));response=read(sock)
  if response.get('status')!='COMPLETE' or response.get('outer_step')!=request['outer_step']:raise RuntimeError('failed/unknown supervisor ACK; no retry')
  return response
 finally:sock.close()
