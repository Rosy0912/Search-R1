import pathlib,json,os
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005');r=root/'runs/bapo-instruct300-hardware-3239'
assert r==r.resolve() and (r/'source-commit.txt').read_text().strip()=='7de6494bff08f35ffeefbf23ed9a9e1b73e5612e'
out={'job_id':3239,'run':str(r),'files':sorted(p.name for p in r.iterdir())}
for name in ['hardware.log','train.log','resource-watch.jsonl','resource-watch.log','own-process-cleanup.json']:
 p=r/name
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4000));out[name]=f.read().decode(errors='replace')
p=root/'logs/slurm-hardware-3239.err';out['stderr']=p.read_text()[-2500:]
print(json.dumps(out))
