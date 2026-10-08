"""Publish only independently accepted recovery101..105 for live job3234."""
import hashlib,json,os,re,subprocess
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234')
gate=root/'output/acceptance-step105'
source=Path(os.environ['TREE_RECOVERY105_REVIEW_PATH'])
expected=os.environ['TREE_RECOVERY105_REVIEW_SHA256']
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
 return h.hexdigest()
signed_bytes=source.read_bytes()
assert hashlib.sha256(signed_bytes).hexdigest()==expected
approval=json.loads(signed_bytes)
ready_raw=(gate/'ready.json').read_bytes();ready=json.loads(ready_raw)
ready_sha=hashlib.sha256(ready_raw).hexdigest()
assert ready['job_id']=='3234' and ready['completed_outer_updates']==105 and ready['next_outer_update']==106
assert ready['code_commit']=='502e9fafa8e3aba722dbfcb493abc2b9d3ece326'
assert approval.get('recovery_parent_completed_step')==100
assert approval.get('recovery_parent_manifest_sha256')=='c20922f74f7dc77e7a233fcdcf329f2914bfaff335ed34d02209e1e6d09a3e86'
assert approval.get('actual_recovery_outer_updates')==5
assert approval.get('optimizer_calls_per_rank_per_outer')==10
assert approval.get('complete_trajectory_logging_accepted') is True
assert approval.get('restored_fixed512_quality_accepted') is True
assert approval['status']=='PASS' and approval['ready_sha256']==ready_sha
for key in ['job_id','code_commit','config_sha256','data_identity','model_identity']:
 assert approval.get(key)==ready[key],key
for key in ['full_state_reload_accepted','full_hf_reload_accepted','rank_tree_numerics_accepted']:
 assert approval.get(key) is True,key
manifest_raw=Path(ready['full_state_manifest']).read_bytes()
assert approval['full_state_manifest_sha256']==ready['full_state_manifest_sha256']==hashlib.sha256(manifest_raw).hexdigest()
assert approval['hf_identity_sha256']==ready['hf_identity']['sha256']
manifest=json.loads(manifest_raw)
for record in manifest['worker_states']+[manifest['driver']]:assert sha(Path(record['path']))==record['sha256']
assert not (gate/'released.json').exists()
assert not list((root/'output/tree-rank-audit').glob('step-000106-rank-*.json'))
slurm=subprocess.check_output(['scontrol','show','job','3234'],text=True)
fields=dict(re.findall(r'(\S+?)=(\S+)',slurm))
assert fields['JobId']=='3234' and fields['JobState']=='RUNNING' and fields['Restarts']=='0'
assert fields['NodeList']==fields['BatchHost']=='h100-9'
assert (gate/'ready.json').read_bytes()==ready_raw
assert Path(ready['full_state_manifest']).read_bytes()==manifest_raw
# Publish precisely the signed bytes that were parsed; no second source path lookup.
pending=gate/('approved.pending-'+str(os.getpid()))
try:
 with pending.open('xb') as stream:
  stream.write(signed_bytes);stream.flush();os.fsync(stream.fileno())
 os.link(pending,gate/'approved.json')  # Atomic no-clobber; trainer repeats state/HF checks.
finally:
 pending.unlink(missing_ok=True)
print(json.dumps(dict(status='INDEPENDENT_RECOVERY105_APPROVAL_PUBLISHED_NOT_YET_RELEASE_PROOF',job=3234,receipt_sha256=expected)))
