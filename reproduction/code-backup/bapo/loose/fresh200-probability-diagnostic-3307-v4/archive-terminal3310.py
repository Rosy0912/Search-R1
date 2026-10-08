from pathlib import Path
import io,tarfile,json,hashlib,base64
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006/probability-diagnostic-3307-v4-20261006')
run=r/'runs/hf-zero-adam-replay-04348f67c0da16f3fab3320bff61ab95-3310'
files=list(run.iterdir())+[r/'logs'/x for x in ['slurm-3310.out','slurm-3310.err','diagnostic-approval.json','target-tests.actual.json','target-tests.actual.log','diagnostic-cpu-precheck.json','runtime-executable.json','driver-library.actual.json','tools-manifest.json','INDEPENDENT-V4-DIAGNOSTIC-TOOLS-CODE-REVIEW.json']]+list((r/'scheduler-head').glob('*.json'))
if any(not p.is_file() or p.is_symlink() for p in files):raise RuntimeError('regular raw evidence only')
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 for p in sorted(files):t.add(p,arcname=str(p.relative_to(r)),recursive=False)
raw=buf.getvalue()
record={'root':str(r),'job_id':3310,'files':{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'archive_sha256':hashlib.sha256(raw).hexdigest(),'archive_bytes':len(raw),'base64':base64.b64encode(raw).decode()}
with (r/'logs/terminal3310-raw.tar.gz').open('xb') as f:f.write(raw)
print(json.dumps(record))
