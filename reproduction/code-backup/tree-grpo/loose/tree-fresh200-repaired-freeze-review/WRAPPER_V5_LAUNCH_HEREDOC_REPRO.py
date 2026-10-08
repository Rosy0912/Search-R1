import os,pathlib,hashlib,json
root=pathlib.Path(os.environ['TREE_INSTRUCT_TOOLS'])
manifest=root/'manifest.json'; receipt=root/'PRE_DISPATCH_RECEIPT.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==os.environ['TREE_INSTRUCT_PACKAGE_SHA256']
assert hashlib.sha256(receipt.read_bytes()).hexdigest()==os.environ['TREE_INSTRUCT_REVIEW_SHA256']
r=json.loads(receipt.read_bytes()); m=json.loads(manifest.read_bytes())
assert r['status']=='READY_FOR_FRESH200_BEHAVIOR_MU_ZERO_WRITE_CALIBRATION'
assert r['manifest_sha256']==os.environ['TREE_INSTRUCT_PACKAGE_SHA256']
assert r['code_commit']==m['code_commit']
assert r['actual_3b_training_accepted'] is False and r['long_training_ready'] is False
assert r['requires_zero_optimizer_writes'] is True and r['requires_actual_behavior_mu_binding'] is True and r['original_sampler_actor_gate_accepted'] is False
assert m['stage']=='fresh_mu_calibration'
assert m['training_protocol']=='tree_fresh200_five_leaf_v1'
assert m['production_thresholds_accepted'] is False
assert not any(k in m for k in ('parent_manifest','healthy110_metadata'))
assert m['probability_calibration']['environment']=={'TREE_FRESH_INITIALIZATION_ONLY':'1','TREE_PPO_CALIBRATION_ONLY':'1'}
assert m['scope']=='FROZEN_PENDING_INDEPENDENT_FRESH200_MU_PRE_DISPATCH_REVIEW'
for relative,expected in m['files'].items():
 assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected,relative
