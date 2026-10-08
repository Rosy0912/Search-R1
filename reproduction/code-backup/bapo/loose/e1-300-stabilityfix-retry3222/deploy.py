"""Stage reviewed retry wrapper on scheduler head; never launch or alter original tools."""
import json,sys,base64,hashlib,pathlib
D=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005/scheduler-retry-after3222')
assert D.parent.is_dir() and not D.parent.is_symlink() and D.parent.resolve()==D.parent
payload=json.load(sys.stdin);assert set(payload)=={'retry-submit.py','retry-approval.json','INDEPENDENT-REVIEW.md'}
data={}
for name,item in payload.items():
 assert set(item)=={'base64','sha256'}
 b=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==item['sha256'];data[name]=b
approval=json.loads(data['retry-approval.json']);assert approval['status']=='PASS_INDEPENDENT_RETRY_REVIEW'
assert approval['retry_dir']==str(D) and approval['root']==str(D.parent)
assert approval['script_sha256']==hashlib.sha256(data['retry-submit.py']).hexdigest()
assert approval['report_sha256']==hashlib.sha256(data['INDEPENDENT-REVIEW.md']).hexdigest()
D.mkdir(exist_ok=True);assert D.resolve()==D and not D.is_symlink()
assert not (D/'formal300-submit-intent.json').exists() and not (D/'formal300-job-id.txt').exists(),'Already submitted: do not deploy'
for name,b in data.items():
 p=D/name
 if p.exists():assert p.is_file() and not p.is_symlink() and p.read_bytes()==b
 else:
  with p.open('xb') as f:f.write(b)
print('REVIEWED_RETRY_FILES_STAGED_NOT_SUBMITTED')
