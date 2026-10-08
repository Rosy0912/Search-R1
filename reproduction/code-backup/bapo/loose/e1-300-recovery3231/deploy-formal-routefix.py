import json,pathlib,hashlib,sys
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005');base=root/'scheduler-head';d=root/'scheduler-formal-routefix';p=json.load(sys.stdin)
assert base==base.resolve()and not base.is_symlink()
assert not (base/'formal300-job-id.txt').exists()and not (base/'formal300-submit-intent.txt').exists()
assert hashlib.sha256((base/'formal300.sbatch').read_bytes()).hexdigest()==p['original_sbatch_sha256']
if not d.exists():d.mkdir()
assert d==d.resolve()and not d.is_symlink()and not (d/'formal300-submit-intent.txt').exists()and not (d/'formal300-job-id.txt').exists()
allowed={'submit-formal-routefix.py','FORMAL-ROUTEFIX-INDEPENDENT-REVIEW.md','approved-formal-routefix-hashes.json','formal-routefix-independent-approval.json','watch-terminal-formal-routefix.py'}
assert set(p['files'])==allowed
for name,item in p['files'].items():
 raw=item['text'].encode();assert hashlib.sha256(raw).hexdigest()==item['sha256']
 path=d/name
 if path.exists():assert path.is_file()and not path.is_symlink()and path.read_bytes()==raw
 else:
  with path.open('xb')as f:f.write(raw)
print(json.dumps({'status':'NEW_HEAD_ONLY_FORMAL_SOURCE_ROUTE_CONTROLLER_DEPLOYED_NOT_SUBMITTED','files':len(allowed)}))
