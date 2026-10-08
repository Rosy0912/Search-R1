"""Exact isolated sourceV3 identity shared by preparation and actual launch."""
import pathlib,json,hashlib
SOURCE=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/tools/search-r1-32b-grpo100-source-v3')
SOURCE_SHA='1cc5b8bbbe499edbc752acb52d8a91ff2c59b87b312925afba9a0c4216f8687b'
UPSTREAM='e623724532d06e975b61f3b35ec7925dc790c939'
NATIVE={'path':'/dataset/shaoshitong/search-r1-rl/topd/logs/r1-latest-source-v3-cpu-3199/result.json','sha256':'afae7133f27967e32d7f2d2365e92166124c2d10c7270b8a1ec1d503b3b0927a'}
def validate_native_result(r):
 if not (r.get('job')=='3199' and r.get('tests')==22 and r.get('skips')==0 and r.get('failures')==0 and r.get('exit_code')==0 and r.get('cuda_initialized') is False and r.get('source_manifest_sha256')==SOURCE_SHA):raise ValueError('Actual latest sourceV3 native22 proof differs')
 return r

def check(c,read_native=False):
 ref=c['source_manifest']
 if pathlib.Path(c['candidate'])!=SOURCE or pathlib.Path(ref['path'])!=SOURCE/'r1-files.json' or ref['sha256']!=SOURCE_SHA:raise ValueError('Exact isolated streamed sourceV3 binding required')
 if c.get('source_upstream_commit')!=UPSTREAM or c.get('source_native_cpu_evidence')!=NATIVE:raise ValueError('Latest sourceV3 upstream/native binding required')
 if read_native:
  raw=pathlib.Path(NATIVE['path']).read_bytes()
  if hashlib.sha256(raw).hexdigest()!=NATIVE['sha256']:raise ValueError('Native22 result bytes differ')
  validate_native_result(json.loads(raw))
 return SOURCE_SHA
