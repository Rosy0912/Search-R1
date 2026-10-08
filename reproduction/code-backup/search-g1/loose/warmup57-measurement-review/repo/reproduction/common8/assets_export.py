"""Bounded, allowlisted H1009→8 streaming; no GPU or model/data copy on Mac."""
import http.server,json,os,socketserver,threading,time
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
BASE=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export')
FILES={'assets-root/assets/wiki/e5_Flat.index':64559075373,'assets-root/assets/wiki/wiki-18.jsonl':14393579520}
SNAPSHOT={n:(ROOT/n).stat() for n in FILES}
assert all(not (ROOT/n).is_symlink() and SNAPSHOT[n].st_size==size for n,size in FILES.items())
slots=threading.BoundedSemaphore(2)
class Handler(http.server.BaseHTTPRequestHandler):
 def do_HEAD(self):self.asset(False)
 def do_GET(self):self.asset(True)
 def asset(self,body):
  name=self.path.lstrip('/')
  if self.client_address[0]!='10.33.0.8' or name not in FILES:self.send_error(403);return
  if not slots.acquire(blocking=False):self.send_error(503);return
  try:
   p=ROOT/name;now=p.stat();before=SNAPSHOT[name]
   assert not p.is_symlink() and (now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns)==(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
   self.connection.settimeout(120)
   self.send_response(200);self.send_header('Content-Length',str(FILES[name]));self.end_headers()
   if body:
    with p.open('rb') as f:
     for chunk in iter(lambda:f.read(1024*1024),b''):self.wfile.write(chunk)
  except (BrokenPipeError,ConnectionResetError,TimeoutError):pass
  finally:slots.release()
class Server(socketserver.ThreadingMixIn,http.server.HTTPServer):daemon_threads=True
with Server(('10.33.0.13',19874),Handler) as server:
 pid=os.getpid();fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
 handle=dict(pid=pid,start_ticks=int(fields[19]),sid=int(fields[3]),uid=os.getuid(),port=19874,started_unix=time.time(),files=FILES,scope='tiny bounded same-cluster stream server; not GPU work or target transfer acceptance')
 with (BASE/'handle.json').open('x') as f:json.dump(handle,f,indent=2);f.flush();os.fsync(f.fileno())
 server.serve_forever()
