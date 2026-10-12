"""One private, resumable same-cluster transfer of the two public retrieval assets."""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--assets',required=True);p.add_argument('--token-file',required=True)
    p.add_argument('--host',default='10.123.4.11');p.add_argument('--port',type=int,default=18106)
    a=p.parse_args();root=Path(a.assets).resolve();token=Path(a.token_file).read_text().strip()
    allowed={'/index/e5_Flat.index','/corpus/wiki-18.jsonl','/manifest.json'}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                self.send_error(403);return
            if self.path not in allowed:self.send_error(404);return
            path=root/self.path[1:];size=path.stat().st_size;start=0
            field=self.headers.get('Range')
            if field:
                try:
                    if not field.startswith('bytes=') or not field.endswith('-'):raise ValueError()
                    start=int(field[6:-1])
                    if not 0<=start<size:raise ValueError()
                except ValueError:self.send_error(416);return
            self.send_response(206 if field else 200)
            self.send_header('Content-Length',str(size-start));self.send_header('Accept-Ranges','bytes')
            if field:self.send_header('Content-Range',f'bytes {start}-{size-1}/{size}')
            self.end_headers()
            with path.open('rb') as f:
                self.connection.sendfile(f,offset=start,count=size-start)
        def log_message(self,*args):pass
    print('ASSET_TRANSFER_LISTENING',a.host,a.port,flush=True)
    ThreadingHTTPServer((a.host,a.port),Handler).serve_forever()


if __name__=='__main__':main()
