"""Export only the hashed same-cluster asset allowlist to h100-8."""
import http.server,json,os,socketserver,time,urllib.parse
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
BASE=Path('/dataset/shaoshitong/g1-common8-asset-export')
ALLOWED=set(json.loads((BASE/'source-hashes.json').read_text()))
class Handler(http.server.BaseHTTPRequestHandler):
 def do_HEAD(self): self.send_asset(False)
 def do_GET(self): self.send_asset(True)
 def send_asset(self,body):
  name=urllib.parse.unquote(urllib.parse.urlsplit(self.path).path).lstrip('/')
  if self.client_address[0]!='10.33.0.8' or name not in ALLOWED:
   self.send_error(403); return
  path=ROOT/name; size=path.stat().st_size; start=0
  header=self.headers.get('Range')
  if header:
   if not header.startswith('bytes=') or not header.endswith('-'):
    self.send_error(400); return
   try: start=int(header[6:-1])
   except ValueError: self.send_error(400); return
   if start<0 or start>=size: self.send_error(416); return
  self.send_response(206 if header else 200)
  self.send_header('Content-Length',str(size-start))
  if header: self.send_header('Content-Range',f'bytes {start}-{size-1}/{size}')
  self.end_headers()
  if body:
   try:
    with path.open('rb') as f:
     f.seek(start)
     while True:
      chunk=f.read(1024*1024)
      if not chunk: break
      self.wfile.write(chunk)
   except (BrokenPipeError,ConnectionResetError): pass
class Server(socketserver.ThreadingMixIn,http.server.HTTPServer):
 daemon_threads=True
with Server(('10.33.0.13',19873),Handler) as server:
 pid=os.getpid(); ticks=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
 (BASE/'handle.json').write_text(json.dumps(dict(pid=pid,start_ticks=int(ticks),port=19873,started=time.time())))
 server.serve_forever()
