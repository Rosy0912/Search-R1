from pathlib import Path
import io,tarfile,json,hashlib,base64
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006/probability-diagnostic-3307-v2-20261006')
run=r/'runs/hf-zero-adam-replay-5e1669e5ef1fd15fb01093604ac7efe3-3308'
files=list(run.iterdir())+[r/'logs'/x for x in ['slurm-3308.out','slurm-3308.err','diagnostic-approval.json','target-tests.actual.json','target-tests.actual.log','diagnostic-cpu-precheck.json','runtime-executable.json','tools-manifest.json','INDEPENDENT-ZERO-ADAM-DIAGNOSTIC-TOOLS-CODE-REVIEW.json']]+list((r/'scheduler-head').glob('*.json'))
if any(not p.is_file() or p.is_symlink() for p in files):raise RuntimeError('regular raw evidence only')
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 for p in sorted(files):t.add(p,arcname=str(p.relative_to(r)),recursive=False)
raw=buf.getvalue()
record={'root':str(r),'job_id':3308,'files':{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'archive_sha256':hashlib.sha256(raw).hexdigest(),'archive_bytes':len(raw),'base64':base64.b64encode(raw).decode()}
with (r/'logs/terminal3308-raw.tar.gz').open('xb') as f:f.write(raw)
print(json.dumps(record))
