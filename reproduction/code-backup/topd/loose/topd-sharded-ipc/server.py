"""Own private Unix CPU prototype server; fail-closed lifetime after any error."""
import os
from pathlib import Path
import socket
import stat

from protocol import artifact,peer,process_identity,private_directory,receive,require,send


class SharedUnixServer:
    def __init__(self,session):
        self.session=session;self.manifest=session.manifest;self.directory=private_directory(session.directory)
        self.path=artifact(self.directory,Path(self.manifest['service']['socket']).name)
        require(str(self.path)==self.manifest['service']['socket'] and len(os.fsencode(str(self.path)))<108,'Own socket boundary/length differs')
        require(not self.path.exists(),'Refuse existing own socket')
        self.listener=None;self.created_inode=None

    def start(self):
        self.listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.listener.settimeout(self.manifest['timeout_seconds']);self.listener.bind(str(self.path));os.chmod(self.path,0o600)
        self.created_inode=self.path.stat().st_ino;self.listener.listen(1)

    def connection(self,connection):
        connection.settimeout(self.manifest['timeout_seconds'])
        actual=peer(connection,self.manifest['driver'],self.manifest['parent_cgroup'])
        self.session.peer_identity=actual;self.session.actual_service_identity=process_identity(os.getpid())
        while True:
            metadata,payload=receive(connection);command=metadata.get('command')
            if command=='BEGIN_SCORE':send(connection,self.session.begin(metadata,payload))
            elif command=='INPUT_CHUNK':send(connection,self.session.accept_chunk(metadata,payload))
            elif command=='COMMIT_SCORE':
                result,raw=self.session.commit(metadata,payload);send(connection,result,raw)
            elif command=='ACK_RESULT':send(connection,self.session.acknowledge(metadata,payload));return
            else:raise RuntimeError('Unknown shared scorer command')

    def serve(self):
        require(self.listener is not None,'Socket not started')
        try:
            while self.session.next_step<=self.manifest['outer_updates']:
                with self.listener.accept()[0] as connection:
                    try:self.connection(connection)
                    except Exception:
                        self.session.abort()
                        try:send(connection,dict(status='FAILED',error_code='SHARED_SESSION_ABORTED_NO_RETRY'))
                        except Exception:pass
                        raise
        finally:self.close()

    def close(self):
        if self.listener is not None:self.listener.close();self.listener=None
        if self.created_inode is not None:
            try:value=self.path.lstat()
            except FileNotFoundError:pass
            else:
                require(stat.S_ISSOCK(value.st_mode) and value.st_uid==os.getuid() and value.st_ino==self.created_inode,'Socket replaced; refuse cleanup')
                self.path.unlink()
        self.created_inode=None
