"""ROOT invokes after independent source review; source stdin, only noGRES CPU sbatch."""
import argparse,base64,hashlib,json,pathlib,sys
from stream_transport import send
HERE=pathlib.Path(__file__).resolve().parent
ITEMS=[('g1-staged-calibration-producer-v7','2469d111013469bc7393f0e96220db2fbd023f928d854885d8498b6f3958b05d','671ec8d06f7b081119f7eda47deaf0db72ce33a721cf617799da80e88d3ef64b'),('g1-production-pipeline-v12','9e6bce802cf3507d699563851db5f7a7049c8010347efe07bdbea27b54c7ad24','881a5c396655de3ac304b6eeefa468ce59bcc02cbb6b1e0d6a387a678febf846'),('g1-known-run-tools-v2','85d90e5cebce64a9ecf30930467f9529e0ce5862b1c62bcb99e8edd7a17f99c8','e14c95bd26ccf5c748f849e6f2f2b0e60904e1f91e24a92174462cbd4202157b')]
def render():
 own_manifest=HERE/'files.json';manifest=json.loads(own_manifest.read_text())
 for name,r in manifest['files'].items():
  p=HERE/name
  if p.is_symlink() or p.stat().st_size!=r['size'] or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Local helper source differs')
 archives=list(ITEMS);own_archive=HERE.with_name(HERE.name+'-tools.tar.gz')
 # Self manifest/archive hashes are supplied through the independently reviewed
 # local freeze; not placed inside the circular source files manifest.
 freeze=json.loads((HERE.parent.parent/'outputs/g1-native-cpu-deploy-v4-freeze.json').read_text())
 archives.append((HERE.name,freeze['archive_sha256'],freeze['manifest_sha256']))
 payload=[]
 for name,archive_sha,manifest_sha in archives:
  raw=(HERE.parent/(name+'-tools.tar.gz')).read_bytes()
  if hashlib.sha256(raw).hexdigest()!=archive_sha:raise ValueError('Approved local source archive drift')
  payload.append({'name':name,'archive_sha256':archive_sha,'manifest_sha256':manifest_sha,'base64':base64.b64encode(raw).decode()})
 source=(HERE/'remote_bootstrap.py').read_text().replace('PAYLOADS=[] # local approved deploy tool replaces this literal, never remote argv','PAYLOADS='+repr(payload)).replace('ENTRY_MANIFEST_SHA256=None','ENTRY_MANIFEST_SHA256='+repr(freeze['manifest_sha256']))
 compile(source,'frozen-cpu-bootstrap-stdin','exec');return source

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--render-only',action='store_true');a=ap.parse_args();source=render()
 if a.render_only:print(json.dumps({'scope':'LOCAL_SOURCE_RENDER_ONLY_NOT_SUBMITTED','bytes':len(source.encode()),'sha256':hashlib.sha256(source.encode()).hexdigest()}));return
 r=send(source,attempts=3,retry_safe=True,timeout=900)
 sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);raise SystemExit(r.returncode)
if __name__=='__main__':main()
