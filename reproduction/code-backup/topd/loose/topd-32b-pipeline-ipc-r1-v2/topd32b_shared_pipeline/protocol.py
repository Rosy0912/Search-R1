"""Bounded Unix JSON+safetensors framing; no pickle, network endpoint, or gold."""
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import struct

MAX_FRAME=128*1024*1024
MAX_META=64*1024
PROTOCOL='TOPD_SHARED32B_AFUNIX_V1'
IDENTITY=('run_id','code_commit','config_sha256','data_identity','teacher_receipt_sha256','source_sha256')


def require(condition,message):
    if not condition:raise RuntimeError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_exact(connection,length):
    require(type(length) is int and 0<=length<=MAX_FRAME,'Frame length exceeds bound')
    result=[]
    while length:
        part=connection.recv(min(length,1024*1024));require(part,'Incomplete frame');result.append(part);length-=len(part)
    return b''.join(result)


def receive(connection):
    meta_length,payload_length=struct.unpack('!QQ',read_exact(connection,16))
    require(meta_length<=MAX_META and payload_length<=MAX_FRAME,'Oversized metadata/payload')
    def reject_constant(value):raise RuntimeError('Nonfinite JSON constant')
    def unique(pairs):
        result={}
        for key,value in pairs:
            require(key not in result,'Duplicate JSON field');result[key]=value
        return result
    metadata=json.loads(read_exact(connection,meta_length),parse_constant=reject_constant,object_pairs_hook=unique)
    require(type(metadata) is dict,'JSON metadata object required')
    return metadata,read_exact(connection,payload_length)


def send(connection,metadata,payload=b''):
    encoded=canonical(metadata);require(len(encoded)<=MAX_META and len(payload)<=MAX_FRAME,'Output frame exceeds bound')
    connection.sendall(struct.pack('!QQ',len(encoded),len(payload))+encoded+payload)


def process_identity(pid):
    require(type(pid) is int and pid>0,'Actual PID required')
    directory=Path('/proc')/str(pid);raw=(directory/'stat').read_text()
    fields=raw[raw.rfind(')')+2:].split()
    return dict(pid=pid,starttime=int(fields[19]),cgroups=[line.split(':',2)[-1] for line in (directory/'cgroup').read_text().splitlines()])


def in_parent_scope(paths,parent):
    require(parent.startswith('/') and parent not in ('/','/system.slice'),'Specific job parent scope required')
    return any(path==parent or path.startswith(parent+'/') for path in paths)


def peer(connection,expected,parent_scope):
    require(hasattr(socket,'SO_PEERCRED'),'Linux actual SO_PEERCRED required')
    pid,uid,gid=struct.unpack('3i',connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
    actual=process_identity(pid)
    require(pid==expected['pid'] and uid==expected['uid']==os.getuid()
            and actual['starttime']==expected['starttime'] and in_parent_scope(actual['cgroups'],parent_scope),'Foreign Unix PID/UID/starttime/jobscope')
    return dict(actual,uid=uid,gid=gid)


def private_directory(path):
    path=Path(path)
    require(path.is_dir() and not path.is_symlink(),'Private directory missing/symlink')
    value=path.stat();require(value.st_uid==os.getuid() and stat.S_IMODE(value.st_mode)==0o700,'Private directory owner/mode differs')
    return path.resolve()


def artifact(directory,name):
    require(type(name) is str and Path(name).name==name and name not in ('.','..'),'Nonlocal artifact basename')
    root=private_directory(directory);path=root/name
    require(not path.is_symlink() and path.resolve().parent==root,'Artifact path escapes run directory')
    return path


def write_private_json(directory,name,value):
    path=artifact(directory,name);raw=canonical(value)+b'\n'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    directory_fd=os.open(str(Path(directory)),os.O_DIRECTORY)
    try:os.fsync(directory_fd)
    finally:os.close(directory_fd)
    return dict(name=name,sha256=digest(raw))


def read_private_json(directory,item):
    require(set(item)=={'name','sha256'},'Artifact reference fields differ')
    path=artifact(directory,item['name']);value=path.stat()
    require(value.st_uid==os.getuid() and stat.S_IMODE(value.st_mode)==0o600,'Artifact owner/mode differs')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:raw=stream.read()
    require(digest(raw)==item['sha256'],'Actual artifact SHA differs')
    return json.loads(raw)
