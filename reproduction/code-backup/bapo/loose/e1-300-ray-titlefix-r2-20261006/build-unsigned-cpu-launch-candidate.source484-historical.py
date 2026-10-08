"""Build new R2 unsigned launch from exact actual originals; no deployment/permission."""
import argparse,ast,base64,hashlib,importlib.util,json,pathlib,re,subprocess
SOURCE='484c0b30fa73fa5354bdb3452f007134ec5db13f';ROOT='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-r2-20261006'
MANIFEST='cf44f194751afb0e8c012b1ef1fbabf5d397d9badda3b483468bc0017c2ceaf8'
CONFIG='4a985fce68eaa90548eea96f1c9613cb3e3d37784a93958c0566cc0ac5ff1064';DATA='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24';MODEL='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
def need(v,msg):
 if not v:raise ValueError(msg)
def sha(p):
 p=pathlib.Path(p);need(p.is_file()and not p.is_symlink(),'Ordinary original required');return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 def pairs(items):
  d={}
  for k,v in items:need(k not in d,'Duplicate raw key');d[k]=v
  return d
 return json.loads(pathlib.Path(p).read_bytes(),object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite original')))
def encode(v):return(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def binding(p,name):return dict(path=ROOT+'/logs/'+name,bytes=pathlib.Path(p).stat().st_size,sha256=sha(p))
def validate_title(record,runtime,tools,manifest):
 need(record['root']==ROOT and record['source_commit']==SOURCE and record['sys_prefix']==ROOT+'/env','Actual R2 title identity differs')
 need(record['status']=='PASS_ACTUAL_CPU25_TITLES_NOT_GPU_OR_PERMISSION'and record['scope']=='ACTUAL_LINUX_RAY_VENDORED_TITLE_CPU_ONLY_NOT_GPU_OR_TRAINING','Actual title status differs')
 need(record['tool_manifest_sha256']==MANIFEST and all(type(record[k])is int and record[k]==0 for k in ('signals_sent','child_exit_code'))and record['child_stderr']==''and record['ray_init_called']is False and record['actual_gpu_accepted']is False,'Actual title safety differs')
 for key,name in [('self_sha256','actual-title-dependency-preflight.py'),('fixture_sha256','identity-linux-fixture.py')]:need(record[key]==manifest['files'][name]==sha(tools/name),'Actual title tool differs')
 dep=record['ready']['dependency'];need(dep==record['dependency_after']and dep['ray_version']=='2.40.0'and dep['extension_bytes']==69104 and type(dep['extension_bytes'])is int and dep['extension_sha256']=='9429a267fba07b2ec523ae906878e154dbe053065c4e89c7e450037866c2557b'and dep['same_as_ray_worker']is True and dep['function_builtin']is True and dep['function_module']==dep['function_name']=='setproctitle','Actual Ray native dependency differs')
 need(dep['ray_init_path']=='/dataset/shaoshitong/search-r1-rl/bapo/env/lib/python3.10/site-packages/ray/__init__.py'and dep['ray_init_sha256']=='516137346e27e7465277760a413c204c63447366c4f4450d3d45533bd2cf0183'and dep['extension_path']==str(pathlib.Path(dep['ray_init_path']).parent/'thirdparty_files/setproctitle.cpython-310-x86_64-linux-gnu.so'),'Exact observed Ray bootstrap/native origin differs')
 spec=importlib.util.spec_from_file_location('r2_bound_watch',tools/'resource_watch.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w);titles=sorted(w.RAY_APPROVED_TITLES);need(len(titles)==25,'Exact title set required');original=record['original_process'];need(type(original['pid'])is int and original['pid']==record['ready']['pid']and original['pid']>0,'Actual child identity differs')
 need([s for s in original['command']if s]==[ROOT+'/env/bin/python','-B',ROOT+'/tools/actual-title-dependency-preflight.py','--child'],'Original actual child argv differs')
 def snapshot(s):
  need(all(type(s[k])is int and s[k]>=(0 if k=='uid'else 1)for k in ('pid','start_ticks','uid')),'Snapshot scalar type differs');e=s['executable'];need({k:e[k]for k in ('resolved_path','device','inode')}==runtime['executable']and e['sha256']==runtime['executable_sha256'],'Actual child physical executable differs');need(all(type(e[k])is int for k in ('device','inode')),'Executable scalar differs');need(base64.b64decode(s['cmdline_base64'],validate=True)==('\0'.join(s['command'])).encode(),'Original proc bytes differ')
 snapshot(original);acks=record['acks'];need(len(acks)==25,'Actual25 acknowledgements required')
 for i,(ack,title)in enumerate(zip(acks,titles)):
  need(ack['type']=='ack'and type(ack['sequence'])is int and ack['sequence']==i and ack['title']==title,'Actual title sequence differs');s=ack['snapshot'];snapshot(s);need(all(s[k]==original[k]for k in ('pid','start_ticks','uid','executable'))and s['command'][0]==title and not any(s['command'][1:]),'Actual25 stable proc identity differs')
def build(tools,raw,repo,config,transfer):
 tools,raw,repo=map(pathlib.Path,(tools,raw,repo));need(sha(config)==CONFIG,'Fixed algorithm config differs');entries=list(tools.iterdir());need(len(entries)==36 and all(p.is_file()and not p.is_symlink()for p in entries),'Exact36 ordinary tools required');manifestp=raw/'logs/tools-manifest.json';need(sha(manifestp)==MANIFEST,'Actual R2 manifest differs');manifest=load(manifestp);need(manifest['source_commit']==SOURCE and len(manifest['files'])==36 and {p.name:sha(p)for p in entries}==manifest['files'],'Actual R2 tools differ')
 captured=load(transfer);need(captured['root']==ROOT,'Actual original transfer ROOT differs');inventory=captured['files']
 for name,row in inventory.items():
  p=raw/name;need(not pathlib.Path(name).is_absolute()and '..'not in pathlib.Path(name).parts,'Safe original transfer path');original=base64.b64decode(row['base64'],validate=True);need(p.read_bytes()==original and sha(p)==row['sha256']and p.stat().st_size==row['bytes'],'Original transfer differs '+name)
 need((raw/'source-expected.txt').read_text()==SOURCE+'\n','R2 source marker differs');need(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'Fixed local source differs')
 log=raw/'logs/cpu-gate-durable.log';exitp=raw/'logs/cpu-gate-durable-exit.json';text=log.read_text();summary=re.findall(r'(?m)^(\d+) passed, (\d+) warnings in ([0-9.]+)s$',text);need(len(summary)==1 and summary[0][0]=='395','Actual395 source tests required')
 actual=load(exitp);need(type(actual['exit_code'])is int and actual['exit_code']==0 and actual['source_commit']==SOURCE and actual['scope']=='ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE'and actual['training_submitted']is False and actual['log_sha256']==sha(log)and actual['tools_manifest_sha256']==MANIFEST,'Actual R2 CPU durable exit differs')
 markers={}
 for label in ('before','after'):
  prefix='CPU_IMPORT_ORIGINS_'+label.upper()+'_JSON=';rows=[json.loads(line.split(prefix,1)[1])for line in text.splitlines()if prefix in line];need(len(rows)==1,'Exact CPU origin marker required');v=rows[0];need(v['repo']==ROOT+'/repo'and v['source_commit']==SOURCE,'Actual R2 import identity differs');markers[label]=v
  for item in v['modules'].values():
   local=repo/pathlib.Path(item['path']).relative_to(pathlib.Path(ROOT)/'repo');need(sha(local)==item['sha256'],'Actual imported module differs')
  for name,item in v['functions'].items():
   local=repo/pathlib.Path(item['path']).relative_to(pathlib.Path(ROOT)/'repo');need(sha(local)==item['sha256']and item['co_filename']==item['path'],'Actual imported function differs');node=ast.parse(local.read_text());need(name.startswith(item['function_module']+'.'),'Function module differs')
   for part in name[len(item['function_module'])+1:].split('.'):node=next(n for n in node.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name==part)
   first=min([node.lineno]+[n.lineno for n in node.decorator_list])if isinstance(node,ast.FunctionDef)else node.lineno;need(first==item['co_firstlineno']and re.fullmatch('[a-f0-9]{64}',item['code_sha256']),'Actual function code location differs')
 need(markers['before']['functions']==markers['after']['functions']and all(markers['after']['modules'][k]==v for k,v in markers['before']['modules'].items()),'Actual CPU functions/modules changed')
 stability=[json.loads(line.split('STABILITY_TEST_FILES_JSON=',1)[1])for line in text.splitlines()if 'STABILITY_TEST_FILES_JSON='in line];need(len(stability)==2 and stability[0]==stability[1]and all(sha(repo/n)==v for n,v in stability[0].items()),'Actual stability test inputs differ')
 runtimep=raw/'logs/runtime-executable.json';runtime=load(runtimep);need(runtime['source_commit']==SOURCE and runtime['root']==ROOT and runtime['sys_prefix']==ROOT+'/env'and runtime['host']=='h100-5'and runtime['scope']=='ACTUAL_LINUX_INTERPRETER_IDENTITY_NOT_GPU_OR_LAUNCH_ACCEPTANCE','Actual R2 runtime differs')
 capture=load(raw/'logs/runtime-capture.log');need(capture['record']==runtime and capture['path']==ROOT+'/logs/runtime-executable.json'and capture['bytes']==runtimep.stat().st_size and capture['sha256']==sha(runtimep),'Runtime capture differs')
 titlep=raw/'logs/title-dependency-preflight.json';validate_title(load(titlep),runtime,tools,manifest)
 titleexit=load(raw/'logs/title-dependency-preflight-exit.json');need(type(titleexit['exit_code'])is int and titleexit['exit_code']==0 and titleexit['log_sha256']==sha(raw/'logs/title-dependency-preflight.log'),'Actual title natural exit differs')
 glog=raw/'logs/guardian-tests-durable.log';gexit=load(raw/'logs/guardian-tests-durable-exit.json');gtext=glog.read_text();need(re.findall(r'Ran (\d+) tests in ([0-9.]+)s',gtext)==re.findall(r'Ran (112) tests in ([0-9.]+)s',gtext)and len(re.findall(r'Ran 112 tests in ([0-9.]+)s',gtext))==1 and gtext.rstrip().endswith('OK'),'Actual112 guardian tests required')
 need(type(gexit['exit_code'])is int and gexit['exit_code']==0 and gexit['source_commit']==SOURCE and gexit['log_sha256']==sha(glog)and gexit['tools_manifest_sha256']==MANIFEST and gexit['training_submitted']is False,'Actual guardian original exit differs')
 cpu=dict(import_origins=markers,source_commit=SOURCE,source_marker_sha256=sha(raw/'source-expected.txt'),exit_code=0,wall_seconds=None,pytest_reported_wall_seconds=float(summary[0][2]),log_sha256=sha(log),exit_record_sha256=sha(exitp),test_launcher_sha256=manifest['files']['cpu-tests.py'],sys_prefix=ROOT+'/env',scope='ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE',passed=395,skipped=0,failed=0,primary_log_path=ROOT+'/logs/cpu-gate-durable.log',original_exit_path=ROOT+'/logs/cpu-gate-durable-exit.json',first_observation_exit_verified=True,tools_manifest_sha256=MANIFEST)
 approval=dict(status='CANDIDATE_REQUIRES_INDEPENDENT_E1_LAUNCH_REVIEW',scope='UNSIGNED_LOCAL_CANDIDATE_NOT_LAUNCH_OR_GPU_PERMISSION',source_commit=SOURCE,root=ROOT,tools_file_count=36,tools_manifest_sha256=MANIFEST,tool_files=manifest['files'],config_yaml_sha256=CONFIG,data_manifest_sha256=DATA,model_manifest_sha256=MODEL,cpu_gate_record_sha256=hashlib.sha256((json.dumps(cpu,indent=2)+'\n').encode()).hexdigest(),cpu_expected_passed=395,cpu_gate_log=binding(log,'cpu-gate-durable.log'),cpu_gate_exit_record=binding(exitp,'cpu-gate-durable-exit.json'),runtime_executable=binding(runtimep,'runtime-executable.json'),title_dependency_preflight=binding(titlep,'title-dependency-preflight.json'),guardian_actual_linux_record=binding(raw/'logs/guardian-tests-durable-exit.json','guardian-tests-durable-exit.json'),guardian_actual_linux_log=binding(glog,'guardian-tests-durable.log'),actual_cpu_raw_transfer_sha256=sha(transfer),independent_review=dict(path=ROOT+'/logs/final-launch-independent-review.md',sha256=None,pending_independent_review=True),actual_gpu=False,actual_gpu_accepted=False,real_hardware_restore_accepted=False,production_training_permission_signed=False,launch_authorized=False)
 approval['shared_input_bindings']=dict(data=dict(path=ROOT+'/common-data/manifest.json',sha256=DATA),model=dict(path=ROOT+'/assets/qwen2.5-3b-instruct-aa8e725/VERIFIED-MODEL-MANIFEST.json',sha256=MODEL),source_config=dict(path=ROOT+'/repo/verl/trainer/config/ppo_bapo_e1_300.yaml',sha256=CONFIG),shared_inputs_verifier=dict(path=ROOT+'/tools/verify_shared_inputs.py',sha256=manifest['files']['verify_shared_inputs.py']),complete25_prefix=dict(path=ROOT+'/recovery-inputs/train-prefix1-25.log',bytes=959681,sha256='57b05f48a27ab5fd5c81e3e524f71569c6675f8c3ca234ac105b0c677e2c3dc0'),expected_existing_budget=dict(total_outer=300,scheduler_warmup=114,stage_questions=64,complete_restore_step=25,next_step=26))
 need({p.name:sha(p)for p in tools.iterdir()}==manifest['files'],'Tools changed during build');return manifest,cpu,approval

def main():
 p=argparse.ArgumentParser();p.add_argument('--tools',required=True);p.add_argument('--transfer',required=True);p.add_argument('--raw',required=True);p.add_argument('--repo',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=pathlib.Path(a.output);need(not out.exists()and pathlib.Path(ROOT)not in out.resolve().parents,'Fresh local candidate output only');m,c,r=build(a.tools,a.raw,a.repo,a.config,a.transfer);out.mkdir(parents=True)
 for name,obj in [('tools-manifest.json',m),('cpu-gate.json',c),('launch-approval.UNSIGNED-CANDIDATE.json',r),('approved-launch-hashes.UNSIGNED-CANDIDATE.json',{n:m['files'][n]for n in ('submit.sh','hardware.sbatch','formal300.sbatch')})]:
  data=(json.dumps(c,indent=2)+'\n').encode()if name=='cpu-gate.json'else encode(obj)
  if name=='tools-manifest.json':data=(pathlib.Path(a.raw)/'logs/tools-manifest.json').read_bytes()
  with(out/name).open('xb')as f:f.write(data)
 for p in pathlib.Path(a.raw).rglob('*'):
  if p.is_file():
   with(out/('original__'+str(p.relative_to(pathlib.Path(a.raw))).replace('/','__'))).open('xb')as f:f.write(p.read_bytes())
 print(json.dumps(dict(status='UNSIGNED_NOT_AUTHORIZATION',output=str(out.resolve()),files={p.name:sha(p)for p in out.iterdir()})))
if __name__=='__main__':main()
