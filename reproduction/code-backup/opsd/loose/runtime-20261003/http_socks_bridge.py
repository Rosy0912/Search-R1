"""Task-local HTTPS CONNECT bridge to the user's existing SOCKS route."""
import select,socket,socketserver,struct

def readn(sock,n):
    data=b''
    while len(data)<n:
        part=sock.recv(n-len(data))
        if not part: raise ConnectionError('proxy disconnected')
        data+=part
    return data

class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        remote=None
        try:
            self.request.settimeout(20); data=b''
            while b'\r\n\r\n' not in data:
                part=self.request.recv(4096)
                if not part: return
                data+=part
                if len(data)>65536: raise ValueError('headers too large')
            header,remaining=data.split(b'\r\n\r\n',1)
            method,target,_=header.split(b'\r\n',1)[0].split()
            if method!=b'CONNECT': raise ValueError('HTTPS CONNECT only')
            host,port=target.decode('ascii').rsplit(':',1);name=host.encode('idna')
            if int(port)!=443 or len(name)>255: raise ValueError('invalid HTTPS target')
            remote=socket.create_connection(('127.0.0.1',18984),20)
            remote.sendall(b'\x05\x01\x00');assert readn(remote,2)==b'\x05\x00'
            remote.sendall(b'\x05\x01\x00\x03'+bytes([len(name)])+name+struct.pack('!H',int(port)))
            version,status,_,kind=readn(remote,4)
            if version!=5 or status!=0: raise ConnectionError('SOCKS CONNECT failed')
            length={1:4,4:16}.get(kind)
            if kind==3: length=readn(remote,1)[0]
            if length is None: raise ValueError('invalid SOCKS reply')
            readn(remote,length+2)
            self.request.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n')
            if remaining: remote.sendall(remaining)
            self.request.settimeout(None);remote.settimeout(None)
            while True:
                readable,_,_=select.select([self.request,remote],[],[],180)
                if not readable: break
                for source in readable:
                    chunk=source.recv(1024*1024)
                    if not chunk: return
                    (remote if source is self.request else self.request).sendall(chunk)
        except (OSError,ValueError,ConnectionError,AssertionError):
            pass
        finally:
            if remote is not None: remote.close()
class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address=True
    daemon_threads=True
with Server(('127.0.0.1',18985),Handler) as server:
    print('OPSD_LOOPBACK_HTTPS_PROXY_READY',flush=True)
    server.serve_forever()
