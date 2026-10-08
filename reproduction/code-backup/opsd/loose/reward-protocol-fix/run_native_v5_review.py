from pathlib import Path
import json,hashlib,tarfile,subprocess,os,sys
root=Path('/mnt/data/shaoshitong/search-r1-rl/opsd');review=root/'tools/corrected-integration-review-9fb';archive=review/'v5-tools.tar.gz';dest=review/'tools-v5';assert not dest.exists();dest.mkdir()
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='2a23f1c541ace1b2a6b4a61fb87c45b51317f1f558c3caefe3a5bddb99a2384b'
with tarfile.open(archive) as t:
 for member in t.getmembers():assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
 t.extractall(dest)
m=dest/'candidate-manifest.json';assert hashlib.sha256(m.read_bytes()).hexdigest()=='a3a58a5967deac7aa045953849af0fbbdec67da53cf471338556e5227e9c3af3'
for name,sha in json.loads(m.read_text())['files'].items():assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==sha
repo=root/'repo';assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()=='8e60976d364ab7d7ae51714415813c1359dafe9a'
env=dict(os.environ,PYTHONPATH=str(review/'source-v3')+':'+str(root/'tools/pytest-lib'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',SEARCH_R1_REQUIRE_REAL_DATAPROTO='1',HF_HUB_OFFLINE='1');python=str(root/'env/bin/python');log=review/'v5-native-cpu.txt'
with log.open('wb') as f:r=subprocess.run([python,'-m','pytest','-q',*sorted(p.name for p in dest.glob('test_*.py'))],cwd=dest,env=env,stdout=f,stderr=subprocess.STDOUT)
print(json.dumps(dict(scope='native CPU tools only; no500/HF/GPU receipt',candidate_manifest_sha256=hashlib.sha256(m.read_bytes()).hexdigest(),returncode=r.returncode,log=str(log),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),tail=log.read_text()[-2000:])))
if r.returncode:raise SystemExit(r.returncode)
sys.path.insert(0,str(dest));import base_asset_ancestry as ba
base=root/'assets-root/assets/qwen2.5-3b';pins={}
for record in ba.original_base_records():pins[record['file']]=ba.verify_original_record(base/record['file'],record)
proof=dict(scope='CPU original source8e base content ancestry only; no model load/GPU evaluation',directory=str(base),original_assets_sha256=ba.TRAINING_ASSETS_SHA,revision=ba.MODEL_REVISION,files=pins)
out=review/'actual-base-original-content-pins.json'
with out.open('x') as f:json.dump(proof,f,indent=2)
assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()==''
print(json.dumps(dict(actual_base_original_content_pins=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),files=len(pins),original_training_source_unchanged=True)))
