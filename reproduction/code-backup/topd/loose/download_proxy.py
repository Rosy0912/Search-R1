"""Project-only HTTPS CONNECT relay. Loopback, no credentials or URL logging."""
import select
import socket
import socketserver
import threading

LIMIT=threading.BoundedSemaphore(4)
class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        with LIMIT:
            line=self.rfile.readline(8192).decode('ascii',errors='replace').strip()
            fields=line.split()
            if len(fields)!=3 or fields[0]!='CONNECT':
                self.wfile.write(b'HTTP/1.1 405 Method Not Allowed\r\n\r\n');return
            host,port=fields[1].rsplit(':',1)
            if port!='443' or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-' for c in host):
                self.wfile.write(b'HTTP/1.1 403 Forbidden\r\n\r\n');return
            while self.rfile.readline(8192).strip():pass
            try:
                upstream=socket.create_connection((host,443),timeout=15)
            except OSError:
                self.wfile.write(b'HTTP/1.1 502 Bad Gateway\r\n\r\n');return
            with upstream:
                self.wfile.write(b'HTTP/1.1 200 Connection established\r\n\r\n');self.wfile.flush()
                sockets=[self.connection,upstream]
                while True:
                    ready,_,_=select.select(sockets,[],[],120)
                    if not ready:return
                    for source in ready:
                        data=source.recv(256*1024)
                        if not data:return
                        (upstream if source is self.connection else self.connection).sendall(data)
class Server(socketserver.ThreadingTCPServer):
    daemon_threads=True
    allow_reuse_address=False
Server(('127.0.0.1',18975),Handler).serve_forever()
