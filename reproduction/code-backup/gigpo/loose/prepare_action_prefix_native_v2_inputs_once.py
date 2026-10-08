from pathlib import Path
import json,hashlib,copy,os
O=Path('outputs/gigpo-e1-policy-safety').resolve();R=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004');P=R/'logs/action-prefix-native-proof-v2'
h=lambda b:hashlib.sha256(b).hexdigest()
plan=json.loads((O/'ACTION-PREFIX-NATIVE-STARTUP-FIX-DEPENDENCY-PLAN-v2.json').read_text())
def spec(p,target):
 p=Path(p).resolve();b=p.read_bytes();assert len(b)<16*1024*1024
 return dict(local_path=str(p),path=str(target),bytes=len(b),sha256=h(b))
roles=copy.deepcopy(plan['source_roles_preserved']);roles['native_helper_manifest']=plan['native_helper_manifest_role'];roles['native_helper_review']=spec(O/'ACTION-PREFIX-NATIVE-STARTUP-FIX-INDEPENDENT-REVIEW-v2.json',P/'ACTION-PREFIX-NATIVE-STARTUP-FIX-INDEPENDENT-REVIEW-v2.json')
assert roles['native_helper_review']['sha256']=='1ce5ab17794cd7943ae1cb5e9689e04ef333bd918520d4704110e6732a90cfe8'
extra=copy.deepcopy(plan['historical47_dependencies_preserved']+plan['former_helper_roles_retained_as_additional_evidence']+[plan['actual_v1_binding_existing_reuse']]+plan['new_additional_evidence'])
for name in ('ACTION-PREFIX-NATIVE-DEPLOYMENT-STARTUP-FIX-INDEPENDENT-REVIEW-v2.json','ACTION-PREFIX-NATIVE-FAILURE3297-RELEASE-v1-TARGET-ORIGINAL.json','ACTION-PREFIX-NATIVE-FAILURE3297-SLURMCTLD-v2-TARGET-ORIGINAL.json','ACTION-PREFIX-NATIVE-STARTUP-FIX-DEPENDENCY-PLAN-v2.json'):
 extra.append(spec(O/name,P/name))
for s in list(roles.values())+extra:
 b=Path(s['local_path']).read_bytes();assert len(b)==s['bytes'] and h(b)==s['sha256'];assert set(s)=={'local_path','path','bytes','sha256'}
assert len({s['path'] for s in list(roles.values())+extra})==len(roles)+len(extra)
out=O/'ACTION-PREFIX-NATIVE-PROOF-INPUTS-v2.json'
with out.open('x') as f:f.write(json.dumps(dict(roles=roles,additional_evidence=extra),indent=2)+'\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256':h(out.read_bytes()),'roles':len(roles),'additional_evidence':len(extra),'fresh_proofs':len([s for s in list(roles.values())+extra if str(s['path']).startswith(str(P)+'/')])}))
