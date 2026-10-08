"""Publish only an independently accepted actual-five receipt for live job3154."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154')
gate=root/'output/acceptance-step5'
source=Path(os.environ['TREE_FIVE_REVIEW_PATH'])
expected=os.environ['TREE_FIVE_REVIEW_SHA256']
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
 return h.hexdigest()
signed_bytes=source.read_bytes()
assert hashlib.sha256(signed_bytes).hexdigest()==expected
approval=json.loads(signed_bytes);ready=json.loads((gate/'ready.json').read_bytes())
assert ready['job_id']=='3154' and ready['completed_outer_updates']==5 and ready['next_outer_update']==6
assert approval['status']=='PASS' and approval['ready_sha256']==sha(gate/'ready.json')
for key in ['job_id','code_commit','config_sha256','data_identity','model_identity']:
 assert approval.get(key)==ready[key],key
for key in ['full_state_reload_accepted','full_hf_reload_accepted','rank_tree_numerics_accepted']:
 assert approval.get(key) is True,key
assert approval['full_state_manifest_sha256']==ready['full_state_manifest_sha256']==sha(Path(ready['full_state_manifest']))
assert approval['hf_identity_sha256']==ready['hf_identity']['sha256']
manifest=json.loads(Path(ready['full_state_manifest']).read_bytes())
for record in manifest['worker_states']+[manifest['driver']]:assert sha(Path(record['path']))==record['sha256']
assert not (gate/'released.json').exists()
assert not list((root/'output/tree-rank-audit').glob('step-000006-rank-*.json'))
slurm=subprocess.check_output(['scontrol','show','job','3154'],text=True)
assert 'JobState=RUNNING' in slurm and 'Restarts=0' in slurm
# Publish precisely the signed bytes that were parsed; no second source path lookup.
pending=gate/('approved.pending-'+str(os.getpid()))
try:
 with pending.open('xb') as stream:
  stream.write(signed_bytes);stream.flush();os.fsync(stream.fileno())
 os.link(pending,gate/'approved.json')  # Atomic no-clobber; trainer repeats state/HF checks.
finally:
 pending.unlink(missing_ok=True)
print(json.dumps(dict(status='INDEPENDENT_ACTUAL5_APPROVAL_PUBLISHED_NOT_YET_RELEASE_PROOF',job=3154,receipt_sha256=expected)))
