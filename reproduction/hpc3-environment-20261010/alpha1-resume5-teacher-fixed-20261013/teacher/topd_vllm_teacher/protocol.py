import json,os,socket,struct,pathlib
LIMIT=512*1024*1024

def exact(sock,size):
    data=bytearray()
    while len(data)<size:
        part=sock.recv(size-len(data))
        if not part:raise EOFError('Teacher IPC truncated')
        data.extend(part)
    return bytes(data)
def send(sock,meta,payload=b''):
    header=json.dumps(meta,sort_keys=True,allow_nan=False).encode()
    if len(header)>65536 or len(payload)>LIMIT:raise ValueError('Teacher IPC frame too large')
    sock.sendall(struct.pack('!IQ',len(header),len(payload))+header+payload)
def receive(sock):
    head,size=struct.unpack('!IQ',exact(sock,12))
    if not 0<head<=65536 or size>LIMIT:raise ValueError('Teacher IPC frame bounds')
    return json.loads(exact(sock,head)),exact(sock,size)
def peer(sock,job):
    pid,uid,_=struct.unpack('3i',sock.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
    if uid!=os.getuid():raise ValueError('Foreign teacher socket UID')
    cg=pathlib.Path('/proc',str(pid),'cgroup').read_text()
    if not any(('job_'+str(job)) in line.split(':',2)[-1].split('/') for line in cg.splitlines()):raise ValueError('Foreign teacher socket job')
    stat=pathlib.Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,uid=uid,starttime=stat[19],cgroup=cg.splitlines())
def write(path,value):
    p=pathlib.Path(path);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

