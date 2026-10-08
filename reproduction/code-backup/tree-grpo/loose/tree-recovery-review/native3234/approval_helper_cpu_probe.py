import pathlib,tempfile,json,hashlib,ast,os
from unittest.mock import patch
src=pathlib.Path('work/publish_recovery3234_105_approval.py');raw=src.read_bytes();cases=[]
for mode in ('pass','not105','wrong_parent','four_updates','nine_calls','missing_trajectory','quality_false','bad_ready_sha','bad_identity','state_tamper','already106','stopped','wrong_job','wrong_node','restart','alreadyapproved','changed_signed_source_after_read'):
 with tempfile.TemporaryDirectory() as td:
  root=pathlib.Path(td);gate=root/'output/acceptance-step105';gate.mkdir(parents=True);(root/'output/tree-rank-audit').mkdir()
  records=[]
  for rank in range(8):
   f=root/f'rank-{rank}.pt';f.write_bytes(f'rank{rank}'.encode());records.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
  f=root/'driver.pt';f.write_bytes(b'driver');mp=root/'manifest.json';mp.write_text(json.dumps({'worker_states':records,'driver':{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}}))
  msha=hashlib.sha256(mp.read_bytes()).hexdigest();identity={'job_id':'3234','code_commit':'502e9fafa8e3aba722dbfcb493abc2b9d3ece326','config_sha256':'cfg','data_identity':'data','model_identity':'model'}
  ready=dict(identity,completed_outer_updates=105,next_outer_update=106,full_state_manifest=str(mp),full_state_manifest_sha256=msha,hf_identity={'sha256':'hf'})
  if mode=='not105':ready['completed_outer_updates']=5
  rp=gate/'ready.json';rp.write_text(json.dumps(ready));rsha=hashlib.sha256(rp.read_bytes()).hexdigest()
  approval=dict(identity,status='PASS',ready_sha256=rsha,recovery_parent_completed_step=100,recovery_parent_manifest_sha256='c20922f74f7dc77e7a233fcdcf329f2914bfaff335ed34d02209e1e6d09a3e86',actual_recovery_outer_updates=5,optimizer_calls_per_rank_per_outer=10,complete_trajectory_logging_accepted=True,restored_fixed512_quality_accepted=True,full_state_reload_accepted=True,full_hf_reload_accepted=True,rank_tree_numerics_accepted=True,full_state_manifest_sha256=msha,hf_identity_sha256='hf')
  changes={'wrong_parent':('recovery_parent_completed_step',5),'four_updates':('actual_recovery_outer_updates',4),'nine_calls':('optimizer_calls_per_rank_per_outer',9),'missing_trajectory':('complete_trajectory_logging_accepted',1),'quality_false':('restored_fixed512_quality_accepted',False),'bad_ready_sha':('ready_sha256','bad'),'bad_identity':('config_sha256','bad')}
  if mode in changes:approval[changes[mode][0]]=changes[mode][1]
  if mode=='state_tamper':pathlib.Path(records[7]['path']).write_bytes(b'changed')
  if mode=='already106':(root/'output/tree-rank-audit/step-000106-rank-00.json').write_text('{}')
  if mode=='alreadyapproved':(gate/'approved.json').write_text('preserved')
  signed=root/'review.json';signed.write_text(json.dumps(approval));signedbytes=signed.read_bytes();signedsha=hashlib.sha256(signedbytes).hexdigest()
  slurm={'JobId':'3234','JobState':'RUNNING','Restarts':'0','NodeList':'h100-9','BatchHost':'h100-9'}
  for key,value in {'stopped':('JobState','COMPLETED'),'wrong_job':('JobId','32340'),'wrong_node':('NodeList','h100-8'),'restart':('Restarts','1')}.items():
   if mode==key:slurm[value[0]]=value[1]
  tree=ast.parse(raw)
  for n in tree.body:
   if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='root':n.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(root))],keywords=[])
  read_orig=pathlib.Path.read_bytes;reads=[0]
  def read_once(path):
   value=read_orig(path)
   if path==signed:
    reads[0]+=1
    if mode=='changed_signed_source_after_read':signed.write_text('changed after first validated read')
   return value
  with patch.dict(os.environ,{'TREE_RECOVERY105_REVIEW_PATH':str(signed),'TREE_RECOVERY105_REVIEW_SHA256':signedsha}),patch('pathlib.Path.read_bytes',read_once),patch('subprocess.check_output',return_value=' '.join(k+'='+v for k,v in slurm.items())):
   try:exec(compile(ast.fix_missing_locations(tree),str(src),'exec'),{'__name__':'fixture'})
   except (AssertionError,FileExistsError):assert mode not in ('pass','changed_signed_source_after_read');assert (gate/'approved.json').read_text()=='preserved' if mode=='alreadyapproved' else not (gate/'approved.json').exists()
   else:assert mode in ('pass','changed_signed_source_after_read') and read_orig(gate/'approved.json')==signedbytes
  assert reads[0]==1
  cases.append({'case':mode,'passed':True,'signed_read_count':1})
p=pathlib.Path('work/tree-recovery-review/native3234/APPROVAL_HELPER_CPU_PROOF.json');p.write_text(json.dumps({'status':'CPU_APPROVAL_HELPER_CASES_PASS','helper_sha256':hashlib.sha256(raw).hexdigest(),'cases':cases,'scope':'Temporary fixtures only; no actual3234 approval or release'},indent=2))
print('MOCK CASES PASS',len(cases))
