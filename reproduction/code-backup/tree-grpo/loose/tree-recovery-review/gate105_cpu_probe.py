import os,sys,pathlib,json,hashlib,tempfile
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0,str(pathlib.Path('work/tree-recovery-runtime/source/repo').resolve()))
from search_r1.baselines import training_state as st
from tests.baselines.test_instruct300_driver_gate import real_gate_prefix
from omegaconf import OmegaConf
proof=[]
for mode in ('pass','bad_ready','bad_config','false_acceptance','tampered_shard','bad_hf','timeout'):
 with tempfile.TemporaryDirectory() as td:
  root=pathlib.Path(td);gate=root/'gate';gate.mkdir()
  records=[]
  for rank in range(8):
   f=root/f'rank-{rank}.pt';f.write_bytes(f'fixture-{rank}'.encode());records.append({'path':str(f),'sha256':st.file_sha(f)})
  f=root/'driver.pt';f.write_bytes(b'driver');driver={'path':str(f),'sha256':st.file_sha(f)}
  manifest={'worker_states':records,'driver':driver};mp=root/'manifest.json';mp.write_text(json.dumps(manifest))
  ready=dict(job_id='999',code_commit='502e9fafa8e3aba722dbfcb493abc2b9d3ece326',config_sha256='cfg',data_identity='data',model_identity='base',completed_outer_updates=105,next_outer_update=106,full_state_manifest=str(mp),full_state_manifest_sha256=st.file_sha(mp),hf_checkpoint=str(root/'hf'),hf_identity={'sha256':'hfgood'})
  raw=json.dumps(ready,sort_keys=True,allow_nan=False).encode();approval={k:ready[k] for k in ('job_id','code_commit','config_sha256','data_identity','model_identity')}
  approval.update(status='PASS',ready_sha256=hashlib.sha256(raw).hexdigest(),full_state_reload_accepted=True,full_hf_reload_accepted=True,rank_tree_numerics_accepted=True,full_state_manifest_sha256=st.file_sha(mp),hf_identity_sha256='hfgood')
  if mode=='bad_ready':approval['ready_sha256']='bad'
  if mode=='bad_config':approval['config_sha256']='bad'
  if mode=='false_acceptance':approval['rank_tree_numerics_accepted']=1
  if mode=='tampered_shard':pathlib.Path(records[7]['path']).write_bytes(b'changed')
  if mode=='bad_hf':approval['hf_identity_sha256']='changed'
  if mode!='timeout':(gate/'approved.json').write_text(json.dumps(approval))
  with patch('search_r1.baselines.select_teacher.model_manifest',return_value={'sha256':'hfgood'}):
   try:st.wait_for_acceptance(gate,ready,timeout_seconds=-1)
   except (ValueError,TimeoutError) as e:
    assert mode!='pass';assert not (gate/'released.json').exists();proof.append({'case':mode,'rejected':True,'error':str(e)})
   else:
    assert mode=='pass';released=json.loads((gate/'released.json').read_text());assert released['completed_outer_updates']==105 and released['next_outer_update']==106;proof.append({'case':mode,'released105_to106':True})
# Execute the real fit's prefix with actual106 and check no next fetch before approval.
with tempfile.TemporaryDirectory() as td:
 os.environ['SEARCH_R1_BASELINE_OUTPUT']=td
 sampler=st.CheckpointBatchSampler(169028,640,42,'data');it=iter(sampler)
 for _ in range(85):next(it)
 assert sampler.cursor==54400
 t=SimpleNamespace(global_steps=106,acceptance_ready={'completed_outer_updates':105},config=OmegaConf.create({'trainer':{'acceptance_gate_enabled':True,'acceptance_step':105,'acceptance_gate_timeout_seconds':0}}))
 fetch=real_gate_prefix()
 with patch('search_r1.baselines.training_state.wait_for_acceptance',side_effect=TimeoutError('pending independent review')):
  try:fetch(t,it);raise AssertionError('fetched before approval')
  except TimeoutError:pass
 assert sampler.cursor==54400 and not getattr(t,'acceptance_released',False)
 with patch('search_r1.baselines.training_state.wait_for_acceptance',return_value={'status':'PASS'}):fetch(t,it)
 assert sampler.cursor==55040 and t.acceptance_released
 proof.append(dict(case='REAL_FIT106_PREFIX',blocked_before_fetch=True,cursor_before=54400,cursor_after_release=55040))
pathlib.Path('work/tree-recovery-review/GATE105_CPU_PROOF.json').write_text(json.dumps({'status':'CPU_GATE105_PROBE_PASS','cases':proof,'scope':'Synthetic local files/mocked HF identity; no actual acceptance publication or hardware'},indent=2))
print('GATE105 CPU PASS',len(proof))
