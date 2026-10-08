from pathlib import Path
import json,re,hashlib,ast,subprocess
b=Path('work/e1-300-ray-titlefix-20261006');raw=Path('work/evidence/ray-titlefix-target-cpu-independent-raw');repo=Path('work/bapo-e1-300-ray-titlefix');ROOT='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-20261006';SOURCE='484c0b30fa73fa5354bdb3452f007134ec5db13f';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();inventory=json.loads((raw/'inventory.json').read_text())
for name,item in inventory.items():
 p=raw/name.replace('/','__');assert sha(p)==item['sha256']and p.stat().st_size==item['bytes']
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==SOURCE
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip()
assert (raw/'source-expected.txt').read_text()==SOURCE+'\n'
text=(raw/'logs__cpu-gate-durable.log').read_text();summary=re.findall(r'(?m)^(\d+) passed, (\d+) warnings in ([0-9.]+)s$',text);assert len(summary)==1 and int(summary[0][0])==395
markers={}
for label,prefix in [('before','CPU_IMPORT_ORIGINS_BEFORE_JSON='),('after','CPU_IMPORT_ORIGINS_AFTER_JSON=')]:
 rows=[json.loads(line.split(prefix,1)[1])for line in text.splitlines()if prefix in line];assert len(rows)==1;markers[label]=rows[0]
 for item in rows:
  assert item['repo']==ROOT+'/repo'and item['source_commit']==SOURCE
  for name,v in item['modules'].items():
   path=Path(v['path']);assert str(path).startswith(ROOT+'/repo/');local=repo/path.relative_to(Path(ROOT)/'repo');assert local.is_file()and sha(local)==v['sha256']
  for name,v in item['functions'].items():
   module=v['function_module'];assert name.startswith(module+'.');local=repo/Path(v['path']).relative_to(Path(ROOT)/'repo');assert sha(local)==v['sha256']and v['path']==v['co_filename'];tree=ast.parse(local.read_text());function=name[len(module)+1:];container=tree
   for part in function.split('.'):
    container=next(n for n in container.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==part)
   first=min([container.lineno]+[d.lineno for d in container.decorator_list])if isinstance(container,ast.FunctionDef)else container.lineno
   assert v['co_firstlineno']==first and re.fullmatch('[0-9a-f]{64}',v['code_sha256'])
assert markers['before']['functions']==markers['after']['functions']
assert all(markers['after']['modules'][k]==v for k,v in markers['before']['modules'].items())
stability=[json.loads(line.split('STABILITY_TEST_FILES_JSON=',1)[1])for line in text.splitlines()if 'STABILITY_TEST_FILES_JSON='in line];assert len(stability)==2 and stability[0]==stability[1]
for name,v in stability[0].items():assert sha(repo/name)==v
actual=json.loads((raw/'logs__cpu-gate-durable-exit.json').read_text());assert type(actual['exit_code'])is int and actual['exit_code']==0 and actual['source_commit']==SOURCE and actual['scope']=='ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE'and actual['training_submitted']is False;assert actual['log_sha256']==sha(raw/'logs__cpu-gate-durable.log')
runtime=json.loads((raw/'logs__runtime-executable.json').read_text());capture=json.loads((raw/'logs__runtime-capture.log').read_text());assert capture['record']==runtime and capture['path']==ROOT+'/logs/runtime-executable.json'and capture['bytes']==(raw/'logs__runtime-executable.json').stat().st_size and capture['sha256']==sha(raw/'logs__runtime-executable.json');assert runtime['source_commit']==SOURCE and runtime['root']==ROOT and runtime['sys_prefix']==ROOT+'/env'and runtime['host']=='h100-5'and runtime['scope']=='ACTUAL_LINUX_INTERPRETER_IDENTITY_NOT_GPU_OR_LAUNCH_ACCEPTANCE'
manifest=json.loads((raw/'logs__tools-manifest.json').read_text());assert manifest['source_commit']==SOURCE and len(manifest['files'])==35
assert manifest['files']['cpu-tests.py']==sha(b/'deployment-tools/cpu-tests.py')
record=dict(import_origins=markers,source_commit=SOURCE,source_marker_sha256=sha(raw/'source-expected.txt'),exit_code=actual['exit_code'],wall_seconds=None,pytest_reported_wall_seconds=float(summary[0][2]),log_sha256=sha(raw/'logs__cpu-gate-durable.log'),exit_record_sha256=sha(raw/'logs__cpu-gate-durable-exit.json'),test_launcher_sha256=manifest['files']['cpu-tests.py'],sys_prefix=runtime['sys_prefix'],scope='ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE',passed=395,skipped=0,failed=0,primary_log_path=ROOT+'/logs/cpu-gate-durable.log',original_exit_path=ROOT+'/logs/cpu-gate-durable-exit.json',first_observation_exit_verified=False)
(b/'cpu-gate-durable-derived-record-candidate.json').write_text(json.dumps(record,indent=2)+'\n')
result=dict(status='PASS_INDEPENDENT_DURABLE_SECOND_RUN_SOURCE_CPU_RAW_CHECK',source_commit=SOURCE,passed=395,skipped=0,failed=0,exit_code=0,pytest_reported_seconds=float(summary[0][2]),actual_wall_seconds=None,first_observation_exit_verified=False,before_modules=len(markers['before']['modules']),after_modules=len(markers['after']['modules']),functions=len(markers['before']['functions']),raw_inventory=inventory,derived_record_sha256=sha(b/'cpu-gate-durable-derived-record-candidate.json'),actual_gpu_accepted=False,launch_authorized=False)
(b/'actual-durable-cpu-independent-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='raw_inventory'}))
