"""Local-only generation checks; generated requests are never executed."""
import ast, base64, hashlib, importlib.util, json, zlib
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('generator',HERE/'generate_formal_rayip_v2_requests.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
def literals(path):
 return {n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('PLAN','PLAN_SHA','PART','PART_SHA','PAYLOAD')}
def test_dynamic_payload_hashes_and_bounded_requests(tmp_path):
 out=g.generate(HERE/'formal-rayip-v2',tmp_path/'requests');receipt=json.loads((out/'request-manifest.json').read_bytes());plan=receipt['plan']
 assert len(plan['old33'])==33
 assert len(plan['files'])==19
 parts={}
 for p in out.glob('*-install-part.py'):
  values=literals(p);raw=base64.b64decode(values['PAYLOAD'],validate=True)
  assert hashlib.sha256(raw).hexdigest()==values['PART_SHA'];parts[values['PART']]=raw
  assert len(p.read_bytes())<65536
 packed=b''.join(parts[p] for p in plan['parts'])
 assert hashlib.sha256(packed).hexdigest()==plan['compressed_sha256']
 decoded=zlib.decompress(packed);assert len(decoded)<=1048576
 assert hashlib.sha256(decoded).hexdigest()==plan['uncompressed_sha256']
 bundle=json.loads(decoded);assert set(bundle)==set(plan['files'])
 for name,item in plan['files'].items():
  raw=base64.b64decode(bundle[name],validate=True)
  assert raw==(HERE/'formal-rayip-v2'/name).read_bytes()
  assert hashlib.sha256(raw).hexdigest()==item['sha256']
 for name,digest in receipt['requests'].items():
  p=out/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest;compile(p.read_bytes(),name,'exec')
def test_existing_output_refused(tmp_path):
 out=tmp_path/'requests';out.mkdir();(out/'unknown').write_text('retain')
 with pytest.raises(FileExistsError):g.generate(HERE/'formal-rayip-v2',out)
 assert (out/'unknown').read_text()=='retain'
def test_symlink_payload_refused(tmp_path):
 import shutil
 candidate=tmp_path/'candidate';shutil.copytree(HERE/'formal-rayip-v2',candidate)
 p=candidate/'test_retry_namespace.py';p.unlink();p.symlink_to(HERE/'formal-rayip-v2/test_retry_namespace.py')
 with pytest.raises(ValueError,match='ordinary candidate'):g.build(candidate)
def test_static_no_launch_or_global_write(tmp_path):
 out=g.generate(HERE/'formal-rayip-v2',tmp_path/'requests');verify=(out/'verify-actual-linux.py').read_text()
 assert "'actual_admission'" in verify and "'29 passed'" in verify
 assert "'claim'" not in verify and "['sbatch'" not in verify and "['srun'" not in verify
 assert "p.open('rb')" in verify and 'LOCK_NB' in verify
 assert "CUDA_VISIBLE_DEVICES=''" in verify and "HF_HUB_OFFLINE='1'" in verify
 assert "sys.prefix==" in verify and "not torch.cuda.is_initialized()" in verify
 assert "actual-linux-cpu-verification-intent.json" in verify

def test_fixed_decompression_bound_and_inventory(tmp_path):
 out=g.generate(HERE/'formal-rayip-v2',tmp_path/'requests')
 finish=next(out.glob('*-install-finish.py')).read_text()
 assert 'codec.decompress(packed,1048577)' in finish
 assert 'len(decoded)<=1048576' in finish and 'codec.eof' in finish
 assert 'not codec.unconsumed_tail' in finish and 'not codec.unused_data' in finish
 assert "set(bundle)==set(PLAN['files'])" in finish
 assert "not list(D.iterdir())" in finish
 # Exercise the bounded zlib admission primitives with bomb and trailing streams.
 codec=zlib.decompressobj();decoded=codec.decompress(zlib.compress(b'x'*1048577),1048577)
 assert len(decoded)>1048576
 codec=zlib.decompressobj();codec.decompress(zlib.compress(b'{}')+b'unknown',1048577)
 assert codec.unused_data==b'unknown'
