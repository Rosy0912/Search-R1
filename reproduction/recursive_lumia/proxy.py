"""Task-owned loopback transport; forward retrieval payloads without modification."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import os
from pathlib import Path
import sys
import urllib.request

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != '/retrieve':
            self.send_error(404)
            return
        raw = self.rfile.read(int(self.headers['Content-Length']))
        req = urllib.request.Request(self.server.remote+'/retrieve', data=raw,
                                     headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=1200) as response:
                data = response.read()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(502, str(e))
    def log_message(self, *args):
        pass

def main():
    base=Path(sys.argv[1])
    configuration=json.loads((base/'runs'/os.environ['RECURSIVE_ARM']/'receipts/configuration.json').read_text())
    path=Path(configuration['retriever_receipt'])
    assert path.parent==base/'receipts' and not path.is_symlink()
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==configuration['retriever_receipt_sha256']
    r=json.loads(raw)
    assert r['url']==configuration['retriever_url']
    assert r['root']==str(base) and r['encoder_dtype']=='float32' and r['index_dtype']=='float32'
    server=ThreadingHTTPServer(('127.0.0.1',29146), Handler)
    server.remote=r['url']
    print(json.dumps(dict(proxy_pid=os.getpid(),target=server.remote)),flush=True)
    server.serve_forever()

if __name__ == '__main__':main()
