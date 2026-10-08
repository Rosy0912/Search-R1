"""Exact V8 allocation predicate and subprocess/guard inheritance, CPU only."""
import ast,hashlib,importlib.util,json,os
from pathlib import Path
from unittest.mock import patch
root=Path.cwd();d=root/'work/tree-fresh200-f7-v8-v6-context-free-review/actual-cpu-stage12-independent';sp=root/'work/tree-fresh200-mu-runtime-v8/supervisor.py';source=sp.read_text();tree=ast.parse(source);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
# Execute actual source statements that validate the controller allocation response, never the controller query or service launch.
selected=[n for n in main.body if 87<=n.lineno<=90]
assert len(selected)==4,[n.lineno for n in selected]
module=ast.Module(body=selected,type_ignores=[]);ast.fix_missing_locations(module)
def require(ok,message):
 if not ok:raise RuntimeError(message)
import re
cases=[]
for label,response in [('matching_local','JobId=777 Cluster=wuhan-h100 NodeList=h100-9 JobState=RUNNING OverSubscribe=NO TresPerNode=gres:gpu:8 CPUs/Task=32 NumCPUs=32 MinMemoryNode=256G TimeLimit=03:00:00 Requeue=0'),('mismatched_job_cluster_node_and_limits','JobId=888 Cluster=foreign NodeList=h100-16 JobState=RUNNING OverSubscribe=NO TresPerNode=gres:gpu:8 CPUs/Task=320 NumCPUs=320 MinMemoryNode=1G TimeLimit=99:00:00 Requeue=1')]:
 env={'require':require,'re':re,'allocation':response};error=None
 try:exec(compile(module,str(sp),'exec'),env)
 except Exception as exc:error=repr(exc)
 cases.append({'name':label,'mock_allocation_response':response,'actual_v8_predicate_accepted':error is None,'allocated_cpus':env.get('allocated_cpus'),'error':error})
assert cases[1]['actual_v8_predicate_accepted'] is True
submit=root/'work/tree-fresh200-dispatch-v6/submit_tree_fresh200_once.py';spec=importlib.util.spec_from_file_location('zero_v6_env',submit);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
with patch.dict(os.environ,{'SLURM_CLUSTERS':'foreign','SLURM_CONF':'/foreign/slurm.conf','SCONTROL_FEDERATION':'1','SCANCEL_NAME':'another-job'},clear=True):
 cleaned=s.local_slurm_environment()
assert 'SLURM_CLUSTERS' not in cleaned and 'SLURM_CONF' not in cleaned and 'SCONTROL_FEDERATION' not in cleaned and cleaned['SCANCEL_NAME']=='another-job'
query=next(n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='check_output' and isinstance(n.args[0],ast.List) and isinstance(n.args[0].elts[0],ast.Constant) and n.args[0].elts[0].value=='scontrol')
guard=root/'work/resource_guard_identity.py';gt=ast.parse(guard.read_text());cancel=next(n for n in ast.walk(gt) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='run' and isinstance(n.args[0],ast.List) and isinstance(n.args[0].elts[0],ast.Constant) and n.args[0].elts[0].value=='scancel')
result={'status':'V8_ZERO_RUNTIME_IDENTITY_ADMISSION_COUNTEREXAMPLE_OBSERVED_CPU_ONLY','source':{'path':str(sp),'sha256':hashlib.sha256(sp.read_bytes()).hexdigest(),'allocation_predicate_lines':[n.lineno for n in selected]},'allocation_cases':cases,'submit_environment_actual_source':{'path':str(submit),'sha256':hashlib.sha256(submit.read_bytes()).hexdigest(),'cleaned_synthetic_env':cleaned,'old_SLURM_CLUSTERS_SLURM_CONF_SCONTROL_override_removed':True},'allocation_query':{'line':query.lineno,'source':ast.unparse(query),'explicit_env':any(k.arg=='env' for k in query.keywords),'has_local_option':False},'external_guard':{'path':str(guard),'sha256':hashlib.sha256(guard.read_bytes()).hexdigest(),'cancel_line':cancel.lineno,'source':ast.unparse(cancel),'explicit_env':any(k.arg=='env' for k in cancel.keywords)},'limits':['No Slurm command, scancel, remote, GPU or official assets executed','Mismatched reply is synthetic; no actual wrong allocation/controller or foreign cancellation observed','V6 submit sanitizer prevents the original pre-submit SLURM_CONF/CLUSTERS production injection; it is not evidence of a rechecked runtime cluster/job identity','SCANCEL_NAME is a documented cancellation filter, not a foreign cluster route; do not claim SCANCEL_CLUSTERS exists or is a supported routing override','Current default local cluster/federation configuration after scheduling was not revalidated by V8 runtime']}
(d/'ZERO_RUNTIME_IDENTITY_COUNTEREXAMPLE.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps({'cases':cases,'proof_sha256':hashlib.sha256((d/'ZERO_RUNTIME_IDENTITY_COUNTEREXAMPLE.json').read_bytes()).hexdigest()},indent=2))
