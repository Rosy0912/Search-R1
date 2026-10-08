"""Authorized external immutable-tools CPU diagnostic; source89 untouched/noGPU."""
from pathlib import Path
import sys,json,base64,hashlib,subprocess,shlex
H=Path(__file__).resolve().parent;D=H/'entropy-eager-fix';m=json.loads((D/'manifest.json').read_text());pin='1dd79b7561d335c59e5b7cafe189f460046653831c992b13844bb9b3896cf4cd';sha=lambda b:hashlib.sha256(b).hexdigest();assert sha((D/'manifest.json').read_bytes())==pin
files={n:base64.b64encode((D/n).read_bytes()).decode() for n in m['files']}
for n,b in files.items():assert sha(base64.b64decode(b))==m['files'][n]
files['manifest.json']=base64.b64encode((D/'manifest.json').read_bytes()).decode()
code=r'''
import pathlib,base64,hashlib,json,os,subprocess,ast,socket
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004');repo=root/'repo-89beac273f78';tools=root/'tools/entropy-eager-diagnostic-1dd79b7561d3';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert os.getuid()==0 and socket.gethostname()=='h100-10';assert repo.resolve()==repo
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='89beac273f783d8e48d7d3e34da040f753514d90';assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
tools.mkdir(mode=0o700)
for name,b64 in FILES.items():
 assert pathlib.Path(name).name==name
 with (tools/name).open('xb') as f:f.write(base64.b64decode(b64))
assert tools.resolve()==tools and tools.stat().st_uid==0 and sha(tools/'manifest.json')=='1dd79b7561d335c59e5b7cafe189f460046653831c992b13844bb9b3896cf4cd'
m=json.loads((tools/'manifest.json').read_text())
for name,pin in m['files'].items():assert (tools/name).stat().st_uid==0 and sha(tools/name)==pin
assert (repo/'search_r1/baselines/actor.py').read_bytes()==(tools/'actor_89be_original.py').read_bytes()
evidence=tools/'evidence';evidence.mkdir(mode=0o700);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo));native='/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python';results=[]
for leg in ('legacy','candidate'):
 output=evidence/(leg+'.json');q=subprocess.run([native,str(tools/'cpu_entropy_probe.py'),'--repo',str(repo),'--candidate',str(tools/'actor_candidate.py'),'--leg',leg,'--output',str(output)],env=env,capture_output=True,timeout=45)
 with (evidence/(leg+'-stdout.txt')).open('xb') as f:f.write(q.stdout)
 with (evidence/(leg+'-stderr.txt')).open('xb') as f:f.write(q.stderr)
 with (evidence/(leg+'-exit.json')).open('x') as f:json.dump({'returncode':q.returncode},f)
 assert q.returncode==0,q.stderr.decode();v=json.loads(output.read_text());assert v['torch_version'].split('+')[0]=='2.4.0' and v['cuda_is_initialized'] is False and v['entropy_values_exact'] is True and v['entropy_gradients_exact'] is True
 assert bool(v['strict_guard_error'])==(leg=='legacy') and v['eager_selection']==(leg=='candidate')
 if leg=='candidate':assert v['tamper_rejected'] is True and not v['generated_functions']
 results.append({'leg':leg,'output_sha256':sha(output),'output':v,'stdout_sha256':sha(evidence/(leg+'-stdout.txt')),'stderr_sha256':sha(evidence/(leg+'-stderr.txt')),'exit_sha256':sha(evidence/(leg+'-exit.json'))})
old=ast.parse((tools/'actor_89be_original.py').read_text());new=ast.parse((tools/'actor_candidate.py').read_text());imports=[n for n in new.body if isinstance(n,ast.Import) and any(x.name=='verl.utils.torch_functional' and x.asname=='verl_F' for x in n.names)];assert len(imports)==1;new.body.remove(imports[0]);cls=next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name=='DistillationActor');init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__');assert ast.dump(init.body[1])==ast.dump(ast.parse('self.compute_entropy_from_logits = verl_F.entropy_from_logits').body[0]);init.body.pop(1);assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
for name,pin in m['files'].items():assert sha(tools/name)==pin
record={'node':socket.gethostname(),'uid':os.getuid(),'source_commit':'89beac273f783d8e48d7d3e34da040f753514d90','source_clean_before_after':True,'tools':str(tools),'manifest_sha256':sha(tools/'manifest.json'),'literal_other_actor_ast_unchanged':True,'cases':results,'scope':'Actual nativeTorch24 CPU constructor/provenance with mockteacher only; noGPU/newjob/full3B/first5 acceptance'}
with (evidence/'native-cpu-result.json').open('x') as f:json.dump(record,f,sort_keys=True,indent=2)
print(json.dumps(record,sort_keys=True))
'''.replace('FILES',repr(files))
q=subprocess.run([sys.executable,str(H.parent/'remote.py'),'python3 -c '+shlex.quote(code)],capture_output=True,timeout=75)
(H/'entropy-candidate-native-actual-original.json').write_bytes(q.stdout);(H/'entropy-candidate-native-actual-stderr.txt').write_bytes(q.stderr)
assert q.returncode==0,q.stderr.decode();v=json.loads(q.stdout);print(json.dumps({'actual_native_cpu_constructor_pass':True,'legs':[r['leg'] for r in v['cases']],'source_clean':v['source_clean_before_after'],'scope':v['scope']}))
