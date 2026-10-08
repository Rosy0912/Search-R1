"""Isolated exact sourceV2 repin; never modifies prior sealed namespaces."""
from pathlib import Path
import shutil,re,hashlib,json
W=Path('work');OLD='50fcaec86dcc0f7266f49cdf482a17d3b414b54a439f43a10617e49573320a25';NEW='9040e2ee665e346f2c4c37d3319e65c32b67612668c4f7088c5f6c68c72106d0'
names=['qwen25-32b-trained-eval64','topd-32b-native-score-gate','topd-32b-shared-owner','topd-32b-pipeline-ipc','topd-shared32b-supervisor']
assert hashlib.sha256((W/'search-r1-32b-grpo100-source-v2/r1-files.json').read_bytes()).hexdigest()==NEW
for n in names:
 src=W/(n+'-r1-v4');dst=W/(n+'-r1-v5');assert not dst.exists();shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','*.pyc'))
 for p in dst.rglob('*'):
  if p.is_file() and p.suffix in ('.py','.json','.md','.sbatch','.sha256'):
   p.write_text(p.read_text().replace('-r1-v4','-r1-v5').replace(OLD,NEW))
assert not (W/'r1-32b-hf-semantic-review-v4').exists();shutil.copytree(W/'r1-32b-hf-semantic-review-v3',W/'r1-32b-hf-semantic-review-v4',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','*.pyc'))
for p in (W/'r1-32b-hf-semantic-review-v4').iterdir():
 if p.is_file() and p.name not in ('original_semantic_gate.py','checkpoint_inventory.py'):
  p.write_text(p.read_text().replace('semantic-review-v3','semantic-review-v4').replace(OLD,NEW))
p=W/'test_standard_teacher_identity_v5_chain.py';assert not p.exists();p.write_text((W/'test_standard_teacher_identity_v4_chain.py').read_text().replace('-r1-v4','-r1-v5'))
p=W/'qwen25-32b-trained-eval64-r1-v5/test_training_identity.py';p.write_text(p.read_text()+'''\ndef test_old_engineering_source_fails_even_when_self_consistent():
 a=records();old='50fcaec86dcc0f7266f49cdf482a17d3b414b54a439f43a10617e49573320a25'
 a[0]['source_manifest_sha256']=old;a[3]['source_manifest']['sha256']=old
 with pytest.raises(RuntimeError,match='engineering-patch'):m.check_standard100(*a)
''')
p=W/'test_standard_teacher_identity_v5_chain.py';p.write_text(p.read_text()+'''\ndef test_exact_sourcev2_pin_not_arbitrary_source():
 import importlib.util
 path=E/'training_identity.py';s=importlib.util.spec_from_file_location('v5_engineering_contract',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 expected='9040e2ee665e346f2c4c37d3319e65c32b67612668c4f7088c5f6c68c72106d0'
 assert m.SOURCE_MANIFEST==expected
 assert sha(W/'search-r1-32b-grpo100-source-v2/r1-files.json')==expected
 assert expected in (W/'r1-32b-hf-semantic-review-v4/review.py').read_text()
 assert (S/'worker.py').read_bytes()==(W/'topd-32b-native-score-gate-r1-v4/worker.py').read_bytes()
 assert (E/'eval64.py').read_bytes()==(W/'qwen25-32b-trained-eval64-r1-v4/eval64.py').read_bytes()
''')
# Original helpers unchanged, re-pin changed training_identity in assets gate.
p=W/'qwen25-32b-trained-eval64-r1-v5/trained_assets.py';s=p.read_text().replace('568ea251e12ae8bbe7620b2772e5521f97ed5c07b1cbe5d0aec9d687f6d30966',hashlib.sha256((p.parent/'training_identity.py').read_bytes()).hexdigest());p.write_text(s)
# Clone existing deterministic pin publisher into the new namespaces.
p=W/'refresh_standard_teacher_identity_v5_pins.py';p.write_text((W/'refresh_standard_teacher_identity_v4_pins.py').read_text().replace('-r1-v4','-r1-v5'))
