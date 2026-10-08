import base64,json
from pathlib import Path
review=json.loads(Path('outputs/bapo-rewardfix-training-verifiers-code-review.json').read_text())
print(list(review))
r=Path('work/rewardfix/training-tools')
files={n:base64.b64encode((r/n).read_bytes()).decode() for n in review['tools_sha256']}
reports={n:base64.b64encode((Path('outputs')/n).read_bytes()).decode() for n in ['bapo-rewardfix-training-verifiers-code-review.md','bapo-rewardfix-training-verifiers-code-review.json']}
common='''import pathlib,base64,json,hashlib
records=RECORDS
review=REVIEW
def install(p,b):
 assert not p.is_symlink()
 if p.exists():assert p.read_bytes()==b,str(p)
 else:p.write_bytes(b)
'''.replace('RECORDS',repr(files)).replace('REVIEW',repr(review))
worker=common+'''root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097')
for name,encoded in records.items():
 b=base64.b64decode(encoded);assert hashlib.sha256(b).hexdigest()==review['tools_sha256'][name]
 install(root/'tools'/name,b)
for name,encoded in REPORTS.items():install(root/'logs'/name,base64.b64decode(encoded))
install(root/'logs/training-launch-approved-hashes.json',(json.dumps(review['tools_sha256'],indent=2,sort_keys=True)+'\\n').encode())
print('INDEPENDENT_NEW_TRAINING_TOOLS_DEPLOYED')
'''.replace('REPORTS',repr(reports))
head=common+'''root=pathlib.Path('/dev/shm/bapo-rewardfix-b097-scheduler-20261004')
assert root.is_dir() and not root.is_symlink()
for name,encoded in records.items():
 if not (name.endswith('.sbatch') or name=='submit-training.sh'):continue
 b=base64.b64decode(encoded);assert hashlib.sha256(b).hexdigest()==review['tools_sha256'][name]
 install(root/name,b)
install(root/'training-submit-approved-hashes.json',(json.dumps(review['tools_sha256'],indent=2,sort_keys=True)+'\\n').encode())
print('INDEPENDENT_HEAD_SUBMISSION_TOOLS_DEPLOYED_NO_JOB_SUBMITTED')
'''
Path('work/rewardfix/deploy-training-worker.py').write_text(worker);Path('work/rewardfix/deploy-training-head.py').write_text(head)
