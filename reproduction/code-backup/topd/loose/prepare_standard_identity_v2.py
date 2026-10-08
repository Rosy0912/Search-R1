from pathlib import Path
import shutil,hashlib
W=Path('work');names=['qwen25-32b-trained-eval64-r1','topd-32b-native-score-gate-r1','topd-32b-shared-owner-r1','topd-32b-pipeline-ipc-r1','topd-shared32b-supervisor-r1']
for n in names:
 d=W/(n+'-v2');assert not d.exists();shutil.copytree(W/(n+'-v1'),d,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
 for p in d.rglob('*'):
  if p.is_file() and p.suffix in ('.py','.json','.md','.sbatch'):
   s=p.read_text()
   for old in names:s=s.replace(old+'-v1',old+'-v2')
   p.write_text(s)
E=W/(names[0]+'-v2');p=E/'trained_assets.py';p.write_text(p.read_text().replace("outer['export_path']","outer['checkpoint_directory']"))
p=E/'training_identity.py';s=p.read_text().replace("outer.get('export_path')","outer.get('checkpoint_directory')");s=s.replace("need(config.get('mode')", "need(result.get('export',{}).get('directory')==model_directory,'Actual trainer export directory differs')\n need(config.get('mode')");p.write_text(s)
h=hashlib.sha256(p.read_bytes()).hexdigest();p=E/'trained_assets.py';s=p.read_text().replace('7f68c5359226d55dfff9d6ef35c77ffecac00cfc21997e29eb9c96d9dfb94b2d',h);p.write_text(s)
p=E/'test_trained_assets.py';p.write_text(p.read_text().replace('export_path','checkpoint_directory'))
p=E/'test_training_identity.py';s=p.read_text();s=s.replace("outer=dict(actual_completed_outer_steps=list(range(1,101)),export_step=100,export_path=model)","outer=original_outer(model)").replace("cleanup_passed=True,error=None,actual_update_evidence", "cleanup_passed=True,error=None,export={'directory':model},actual_update_evidence").replace("outer['export_path']","outer['checkpoint_directory']")
s=s.replace('def records():', '''def original_outer(model):
 # Execute the actual unchanged producer record_outer body; publication itself
 # uses a temporary CPU path. This is not a claim of real actor updates.
 import ast,json,math,time,tempfile
 source=pathlib.Path(__file__).with_name('original_r1_observation.py')
 tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='record_outer')
 namespace={'json':json,'pathlib':pathlib,'time':time,'math':math,'atomic':lambda p,v:p.write_text(json.dumps(v))}
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
 with tempfile.TemporaryDirectory() as tmp:
  output=pathlib.Path(tmp)/'actual-outer.json'
  for step in range(1,101):namespace['record_outer'](output,step,{'actor/grad_norm':0.,'actor/lr':1e-6},str(pathlib.Path(model).parents[1]))
  return json.loads(output.read_text())
def records():''')
p.write_text(s);shutil.copyfile(W/'search-r1-32b-grpo100-tools-v3/r1_observation.py',E/'original_r1_observation.py')
for d in [W/(n+'-v2') for n in names]:
 for p in d.glob('*.md'):
  s=p.read_text().replace('G1-100','standard Search-R1-100').replace('source_reference4e7448d…','source_reference0d8a9214dc6e19f10e21549b56f6c01b45d67100').replace('export_path exact','checkpoint_directory exact').replace('all7 runtime candidate files','all8 runtime candidate files (including training_identity.py)').replace('isolated v1','isolated v2')
  p.write_text(s)
# Keep v1 generation and archive frozen; refresh scripts separate.
for old,new in [('refresh_standard_teacher_identity_pins.py','refresh_standard_teacher_identity_v2_pins.py'),('freeze_standard_teacher_identity.py','freeze_standard_teacher_identity_v2.py'),('test_standard_teacher_identity_chain.py','test_standard_teacher_identity_v2_chain.py')]:
 s=(W/old).read_text()
 for n in names:s=s.replace(n+'-v1',n+'-v2')
 s=s.replace('identity-v1','identity-v2').replace('test_standard_teacher_identity_chain.py','test_standard_teacher_identity_v2_chain.py')
 (W/new).write_text(s)
