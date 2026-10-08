"""Deploy only new head retry records; never changes original head/worker tools."""
import hashlib,json,pathlib,sys
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005')
base=root/'scheduler-head';dest=root/'scheduler-recovery3237'
payload=json.load(sys.stdin)
assert payload['prior_job']==3237 and payload['source']=='7de6494bff08f35ffeefbf23ed9a9e1b73e5612e'
assert base==base.resolve() and not base.is_symlink()
for name,key in [('hardware.sbatch','sbatch'),('hardware-3237-terminal.txt','prior_terminal')]:
 assert hashlib.sha256((base/name).read_bytes()).hexdigest()==payload['hashes'][key]
assert (base/'hardware-job-id.txt').read_text().strip()=='3237'
if not dest.exists():dest.mkdir()
assert dest==dest.resolve() and not dest.is_symlink()
assert not (dest/'hardware-submit-intent.txt').exists() and not (dest/'hardware-job-id.txt').exists()
for name,item in payload['files'].items():
 assert pathlib.PurePosixPath(name).name==name
 raw=item['text'].encode();assert hashlib.sha256(raw).hexdigest()==item['sha256']
 p=dest/name
 if p.exists():assert p.is_file() and not p.is_symlink() and p.read_bytes()==raw
 else:
  with p.open('xb') as f:f.write(raw)
 p.chmod(0o755 if name.endswith('.sh') else 0o644)
print(json.dumps({'status':'NEW_RETRY_CONTROLLER_DEPLOYED_NOT_SUBMITTED','dest':str(dest),'files':len(payload['files'])}))
