import http.server, json, pathlib, re, shutil, urllib.parse
ROOT=pathlib.Path('/data/search-r1/assets')
ALLOWED={a['path'] for a in json.loads(pathlib.Path(__file__).with_name('assets.json').read_text())}
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def serve(self,body):
  name=urllib.parse.unquote(urllib.parse.urlsplit(self.path).path).lstrip('/')
  if name not in ALLOWED: self.send_error(404); return
  p=ROOT/name
  if not p.is_file(): self.send_error(404); return
  size=p.stat().st_size; start=0; end=size-1; partial=False
  header=self.headers.get('Range')
  if header:
   match=re.fullmatch(r'bytes=(\d+)-(\d*)',header)
   if not match: self.send_error(416); return
   start=int(match[1]); end=min(int(match[2]),size-1) if match[2] else size-1; partial=True
   if not 0<=start<=end<size: self.send_error(416); return
  self.send_response(206 if partial else 200)
  self.send_header('Content-Type','application/octet-stream'); self.send_header('Content-Length',str(end-start+1)); self.send_header('Accept-Ranges','bytes')
  if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
  self.end_headers()
  if body:
   try:
    with p.open('rb') as f:
     f.seek(start); remaining=end-start+1
     while remaining:
      chunk=f.read(min(1024*1024,remaining))
      if not chunk: raise IOError('source changed during transfer')
      self.wfile.write(chunk); remaining-=len(chunk)
   except (BrokenPipeError,ConnectionResetError):pass
 def do_GET(self):self.serve(True)
 def do_HEAD(self):self.serve(False)
http.server.ThreadingHTTPServer(('127.0.0.1',19992),Handler).serve_forever()
