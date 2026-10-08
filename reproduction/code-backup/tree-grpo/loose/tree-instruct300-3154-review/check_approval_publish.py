from pathlib import Path
import ast,hashlib,json,os,tempfile,types
src=Path('work/publish_instruct3154_five_approval.py');t=ast.parse(src.read_text());t.body=[n for n in t.body if not isinstance(n,ast.Assign) or not any(isinstance(z,ast.Name) and z.id=='root' for z in n.targets)];code=compile(t,str(src),'exec');results={};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for mode in ['normal','source_changed_after_read','bad_receipt_sha','wrong_identity','string_flag','changed_shard','not_running','restarted','rank6','already_released','already_approved']:
 with tempfile.TemporaryDirectory() as d:
  root=Path(d);gate=root/'output/acceptance-step5';gate.mkdir(parents=True);state=root/'fullstate';state.mkdir();files=[]
  for i in range(9):
   p=state/str(i);p.write_bytes(('state'+str(i)).encode());files.append({'path':str(p),'sha256':sha(p)})
  m=state/'manifest.json';m.write_text(json.dumps({'worker_states':files[:8],'driver':files[8]}))
  ready={'job_id':'3154','completed_outer_updates':5,'next_outer_update':6,'code_commit':'f994','config_sha256':'cfg','data_identity':'data','model_identity':'model','full_state_manifest':str(m),'full_state_manifest_sha256':sha(m),'hf_identity':{'sha256':'hf'}};rp=gate/'ready.json';rp.write_text(json.dumps(ready))
  approval={k:ready[k] for k in ['job_id','code_commit','config_sha256','data_identity','model_identity']};approval.update(status='PASS',ready_sha256=sha(rp),full_state_reload_accepted=True,full_hf_reload_accepted=True,rank_tree_numerics_accepted=True,full_state_manifest_sha256=sha(m),hf_identity_sha256='hf')
  if mode=='wrong_identity':approval['model_identity']='wrong'
  if mode=='string_flag':approval['full_state_reload_accepted']='true'
  source=root/'signed.json';source.write_text(json.dumps(approval));signed=source.read_bytes();pin=sha(source)
  if mode=='changed_shard':Path(files[0]['path']).write_text('damage')
  if mode=='rank6':q=root/'output/tree-rank-audit';q.mkdir();(q/'step-000006-rank-00.json').write_text('{}')
  if mode=='already_released':(gate/'released.json').write_text('{}')
  if mode=='already_approved':(gate/'approved.json').write_text('existing')
  def check_output(*a,**kw):
   if mode=='source_changed_after_read':source.write_text('changed after read')
   return 'JobState=PENDING Restarts=0' if mode=='not_running' else 'JobState=RUNNING Restarts=1' if mode=='restarted' else 'JobState=RUNNING Restarts=0'
  # Use real stdlib except the one scheduler subprocess. Restore env afterwards.
  old={k:os.environ.get(k) for k in ['TREE_FIVE_REVIEW_PATH','TREE_FIVE_REVIEW_SHA256']};os.environ.update(TREE_FIVE_REVIEW_PATH=str(source),TREE_FIVE_REVIEW_SHA256='0'*64 if mode=='bad_receipt_sha' else pin)
  fake=types.ModuleType('subprocess');fake.check_output=check_output
  import sys
  actual=sys.modules.get('subprocess');sys.modules['subprocess']=fake
  try:
   try:exec(code,{'root':root});result='PUBLISHED'
   except (AssertionError,FileExistsError):result='REJECT'
  finally:
   if actual is None:sys.modules.pop('subprocess',None)
   else:sys.modules['subprocess']=actual
   for k,v in old.items():
    if v is None:os.environ.pop(k,None)
    else:os.environ[k]=v
  if mode in ['normal','source_changed_after_read']:assert result=='PUBLISHED' and (gate/'approved.json').read_bytes()==signed
  elif mode=='already_approved':assert result=='REJECT' and (gate/'approved.json').read_text()=='existing'
  else:assert result=='REJECT' and not (gate/'approved.json').exists()
  assert not list(gate.glob('approved.pending-*'));results[mode]=result
Path('work/tree-instruct300-3154-review/APPROVAL_PUBLISH_CPU_CHECKS.json').write_text(json.dumps({'results':results,'scope':'Actual source executed against temporary local paths, mocked Slurm only. No real approval or GPU operation'},indent=2)+'\n');print(results)
