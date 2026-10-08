from pathlib import Path
import io,tarfile,json,hashlib,base64
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006/numeric-diagnostic-3310-20261006')
run=r/'runs/numeric-zero-adam-replay-2a9da77d45febc79ca59183205e24e9e-3311'
allpaths=list(run.rglob('*'));assert not any(p.is_symlink() for p in allpaths)
files=[p for p in allpaths if p.is_file()]+[r/'logs'/x for x in ['slurm-3311.out','slurm-3311.err','diagnostic-approval.json','target-tests.actual.json','target-tests.actual.log','diagnostic-cpu-precheck.json','runtime-executable.json','driver-library.actual.json','numeric-cpu-plan.json','tools-manifest.json','INDEPENDENT-NUMERIC-WRAPPER-CODE-REVIEW.json']]+list((r/'scheduler-head').glob('*.json'))
assert all(p.is_file() and not p.is_symlink() for p in files)
assert (run/'original-exit-code.txt').read_text().strip()=='0' and (run/'cleanup-exit-code.txt').read_text().strip()=='0'
assert (run/'numeric-probe').is_file() and (run/'numeric-probe.selected-logits.pt').is_file()
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz')as t:
 for p in sorted(files):t.add(p,arcname=str(p.relative_to(r)),recursive=False)
raw=buf.getvalue()
with (r/'logs/terminal3311-raw.tar.gz').open('xb')as f:f.write(raw)
print(json.dumps(dict(root=str(r),job_id=3311,files={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},archive_sha256=hashlib.sha256(raw).hexdigest(),archive_bytes=len(raw),base64=base64.b64encode(raw).decode())))
