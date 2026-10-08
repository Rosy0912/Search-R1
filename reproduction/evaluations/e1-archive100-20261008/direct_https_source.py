"""Ephemeral authenticated TLS exporter for an immutable 12-file checkpoint."""
import hashlib, hmac, http.server, json, re, ssl, sys, threading, time, urllib.parse
from pathlib import Path

BASE=Path(__file__).resolve().parent
CFG=json.loads((BASE/'private-config.json').read_text())
SPEC=json.loads((BASE/'manifest.json').read_text())
SOURCE=Path(CFG['source']).resolve()
FILES={row['name']:row for row in SPEC['files']}
assert len(FILES)==12
for name,row in FILES.items():
 p=SOURCE/name
 assert '/' not in name and p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes']

class Handler(http.server.BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def log_message(self,*args): pass
 def respond(self,code,body,kind='application/json'):
  self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.end_headers()
  if self.command!='HEAD':self.wfile.write(body)
 def do_HEAD(self):self.handle_request()
 def do_GET(self):self.handle_request()
 def handle_request(self):
  self.connection.settimeout(90)
  if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+CFG['token']):
   self.respond(403,b'{"error":"denied"}');return
  u=urllib.parse.urlsplit(self.path)
  if u.path=='/health':self.respond(200,json.dumps({'ready':True,'files':len(FILES)}).encode());return
  mode,_,name=u.path.lstrip('/').partition('/')
  name=urllib.parse.unquote(name)
  if name not in FILES or mode not in ['file','prefix']:
   self.respond(404,b'{"error":"not allowed"}');return
  path=SOURCE/name;size=FILES[name]['bytes']
  before=path.stat()
  if before.st_size!=size or path.is_symlink():self.respond(409,b'{"error":"source changed"}');return
  if mode=='prefix':
   try:length=int(urllib.parse.parse_qs(u.query)['length'][0]);assert 0<=length<=size
   except Exception:self.respond(400,b'{"error":"bad length"}');return
   h=hashlib.sha256();remaining=length
   with path.open('rb') as f:
    while remaining:
     chunk=f.read(min(8*1024*1024,remaining));assert chunk;h.update(chunk);remaining-=len(chunk)
   assert before.st_mtime_ns==path.stat().st_mtime_ns
   self.respond(200,json.dumps({'name':name,'length':length,'sha256':h.hexdigest()}).encode());return
  start,end=0,size-1;code=200
  if self.headers.get('Range'):
   m=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers['Range'])
   if not m:self.respond(416,b'');return
   start=int(m.group(1));end=int(m.group(2)) if m.group(2) else end;code=206
   if not 0<=start<=end<size:self.respond(416,b'');return
  self.send_response(code);self.send_header('Content-Type','application/octet-stream');self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes')
  if code==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
  self.end_headers()
  if self.command=='HEAD':return
  with path.open('rb') as f:
   f.seek(start);remaining=end-start+1
   while remaining:
    chunk=f.read(min(1024*1024,remaining));assert chunk;self.wfile.write(chunk);remaining-=len(chunk)
  assert path.stat().st_mtime_ns==before.st_mtime_ns

server=http.server.ThreadingHTTPServer((CFG['bind_ip'],CFG['port']),Handler)
server.daemon_threads=True
ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);ctx.minimum_version=ssl.TLSVersion.TLSv1_2
ctx.load_cert_chain(str(BASE/'source.crt'),str(BASE/'source.key'))
server.socket=ctx.wrap_socket(server.socket,server_side=True)
threading.Timer(CFG.get('lifetime_seconds',7200),server.shutdown).start()
print(json.dumps({'phase':'ready','pid':__import__('os').getpid(),'files':len(FILES),'port':CFG['port'],'time':time.time()}),flush=True)
try:server.serve_forever()
finally:server.server_close()
