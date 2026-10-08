import pathlib,json,os,sys
job=sys.argv[1];assert job.isascii()and job.isdigit()and int(job)>0
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006');r=root/('runs/bapo-instruct300-hardware-'+job)
assert r==r.resolve() and (r/'source-commit.txt').read_text().strip()=='b7119791b86951727cea36a1881e8a525c932494'
out={'job_id':int(job),'run':str(r),'files':sorted(p.name for p in r.iterdir())}
for name in ['hardware.log','train.log','resource-watch.jsonl','resource-watch.log','own-process-cleanup.json']:
 p=r/name
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4000));out[name]=f.read().decode(errors='replace')
p=root/('logs/slurm-hardware-'+job+'.err');out['stderr']=p.read_text()[-2500:]
print(json.dumps(out))
