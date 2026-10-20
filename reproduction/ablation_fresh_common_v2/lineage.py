"""Exact fresh training/step proofs. No old stopped/native or reference aliases."""
from pathlib import Path
import contextlib,io,json
from .scope import BASE,SOURCE,SCIENCE,slot
from reproduction.official_eval_v2.protocol import sha
from . import native_core_milestone as mn, native_core_final as an

ROUTES={
 'native_milestone':'01d88924c47c63edeef07d1a676b4bba2a0336aa9511c1cf2f06d6d9e85cc6d5',
 'HF_milestone':'91884485e0ea3ff0d61dd778baeb67880221afdf83f7e74a0f7f1cad80be68d0',
 'native_final':'e2eeb901932547f718699eb38f1f546be1e5f20a0f92faa937a95719cbc5b0c6',
 'HF_final':'fd26eebd569767dcb6fefe3f2af353fe64530b3c4c2bc96452251a17b52abe71'}
EVIDENCE=('native.json','hf.json','native-claim.json','hf-claim.json','manifest.json','training-plan.json','training-submission.json')

def validate(values, hashes, group, step):
 assert set(values)==set(hashes)==set(EVIDENCE)
 n,h,nc,hc,m,p,s=(values[k] for k in EVIDENCE)
 b=an.binding(s,p) if step==150 else mn.milestone_binding(s,p,step)
 assert s['group']==group and p['group']==group and b['job_id']==n['job_id']==h['job_id']==nc['job_id']==hc['job_id']
 assert n['group']==h['group']==nc['group']==hc['group']==group
 assert n['entry_source']==h['entry_source']==SOURCE and n['science_source']==h['science_source']==SCIENCE
 assert n['completed_actual_rounds']==step and n['plan_sha256']==h['plan_sha256']==s['plan_sha256']==hashes['training-plan.json']
 assert n['manifest_sha256']==nc['manifest_sha256']==h['native_manifest_sha256']==hashes['manifest.json']
 assert h['native_local_acceptance_sha256']==hc['native_local_acceptance_sha256']==hashes['native.json']
 assert n['local_raw_sha_verified'] and h['local_raw_sha_verified']
 if step==150:
  assert n['status']=='NEW_FRESH_ABLATION_FINAL150_FULL_NATIVE_CPU_ACCEPTED'
  assert n['post_terminal_recheck_unchanged'] and h['post_terminal_recheck_unchanged']
  assert nc['route_sha256']==ROUTES['native_final'] and hc['route_sha256']==ROUTES['HF_final']
 else:
  assert n['status']=='NEW_FRESH_ABLATION_MILESTONE_FULL_NATIVE_CPU_ACCEPTED'
  assert n['actual_training_limit_remains']==150 and n['native_boundary']=='post-evaluation'
  assert n['post_parent_and_payload_recheck_passed'] and h['post_parent_and_payload_recheck_passed']
  assert nc['step']==hc['step']==step and h['completed_actual_rounds']==h['step']==step
  assert nc['route_sha256']==ROUTES['native_milestone'] and hc['route_sha256']==ROUTES['HF_milestone']
 assert len(n['workers'])==len(n['physical_rank_files'])==8
 assert {v['rank'] for v in n['workers']}==set(range(8))
 assert set(n['physical_rank_files'])=={'rank-%02d.pt'%r for r in range(8)}
 assert n['not51713_prediction_acceptance'] and h['not51713_prediction_acceptance']
 assert n['hf_identity']==h['hf_identity'] and h['hf']==b['hf']
 assert m['status']=='COMPLETE_SAME_WORLD_STATE_SAVED' and m['identity']['completed_step']==step and m['identity']['job_id']==s['job_id']
 assert m['identity']['actual_update_limit']==150 and m['identity']['code_commit']==SCIENCE and m['identity']['data_identity']==b['data_identity']
 assert m['hf_checkpoint']==b['hf'] and m['hf_identity']==h['hf_identity']
 assert len(m['worker_states'])==8 and {v['identity']['rank'] for v in m['worker_states']}==set(range(8))
 for v in m['worker_states']:
  rank=v['identity']['rank'];f=n['physical_rank_files']['rank-%02d.pt'%rank]
  assert v['identity']['world_size']==8 and all(v['identity'][k]==x for k,x in m['identity'].items())
  assert v['path']==f['path']==str(Path(b['native'])/('rank-%02d.pt'%rank))
  assert v['sha256']==f['sha256'] and v['bytes']==f['bytes']>0 and v['full_worker_state_finite']
 assert m['driver']['path']==str(Path(b['native'])/'driver.pt') and m['driver']['next_step']==step+1
 return b

def fresh_boundary(values,hashes,group,step):
 b=validate(values,hashes,group,step);s=values['training-submission.json']
 assert sha(Path(b['native'])/'manifest.json')==hashes['manifest.json']
 assert sha(s['plan'])==hashes['training-plan.json']
 actual=json.loads((Path(b['root'])/'submission.json').read_text())
 assert all(actual[k]==s[k] for k in ('group','job','job_id','plan_sha256','source','science_source','root'))
 name='native_readiness_final.py' if step==150 else 'native_readiness_milestone.py'
 ns={'SUB':s,'B':b}
 with contextlib.redirect_stdout(io.StringIO()):exec(compile((Path(__file__).parent/name).read_text(),name,'exec'),ns)
 snapshot=ns['s'];core=an if step==150 else mn
 assert core.ready(snapshot,b) and snapshot['manifest_sha256']==hashes['manifest.json']
 return snapshot
