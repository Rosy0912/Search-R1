"""Independent local CPU review. All scheduler/GPU/process interfaces are fixtures."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
WRAPPER = WORK/'tree-fresh200-mu-runtime-v10'
spec = importlib.util.spec_from_file_location('independent_v10_contract', WRAPPER/'calibration_contract.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
RAW = (WORK/'resource_guard_identity.py').read_bytes()
GUARD_SHA = hashlib.sha256(RAW).hexdigest()
assert GUARD_SHA == '284c014cd6e273816c112ea8a7b515f3afa4e55bef5c74d23f319f93f146ff88'
ALLOCATION = 'JobId=777 Cluster=wuhan-h100 JobState=RUNNING NodeList=h100-9 BatchHost=h100-9 NumNodes=1 OverSubscribe=NO TresPerNode=gres:gpu:8 CPUs/Task=32 NumCPUs=192 MinMemoryNode=256G TimeLimit=03:00:00 Requeue=0 Restarts=0'
CGROUP='0::/slurm/job_777/step_batch\n'
ENV = dict(SLURM_JOB_ID='777', SLURM_JOB_NODELIST='h100-9', SLURM_CLUSTER_NAME='wuhan-h100', SLURM_CONF='foreign', SLURM_CLUSTERS='foreign', SCANCEL_NAME='other', SCONTROL_FEDERATION='1', KEEP='own', PYTHONOPTIMIZE='0')
cases = []

def record(name, value):
    cases.append(dict(name=name, passed=True, observation=value))

def rejection(fn):
    try: fn()
    except BaseException as err: return dict(type=type(err).__name__, message=str(err))
    raise AssertionError('Expected rejection')

def client(outputs=None):
    outputs = ['ClusterName = wuhan-h100\n', '', ALLOCATION] if outputs is None else outputs
    calls=[]
    def run(argv, **kw):
        calls.append(dict(argv=argv, kwargs=kw))
        return SimpleNamespace(returncode=0, stderr='', stdout=outputs[len(calls)-1])
    return run, calls

run, calls = client(); identity = c.verify_zero_runtime(ENV, 'h100-9', run, cgroup_text=CGROUP)
assert [x['argv'] for x in calls] == [['scontrol','--local','show','config'],['scontrol','--local','show','federation'],['scontrol','--local','show','job','777']]
assert all(x['kwargs']['env']=={'SLURM_JOB_ID':'777','SLURM_JOB_NODELIST':'h100-9','KEEP':'own','PYTHONOPTIMIZE':'0'} for x in calls)
assert identity['requested_cpus']==32 and identity['allocated_cpus']==192 and identity['wall_hours']==3
record('exact local query and sanitized ENV', calls)

bad = ALLOCATION.replace('JobId=777','JobId=888').replace('Cluster=wuhan-h100','Cluster=foreign').replace('NodeList=h100-9','NodeList=h100-16').replace('CPUs/Task=32','CPUs/Task=320').replace('MinMemoryNode=256G','MinMemoryNode=1G').replace('TimeLimit=03:00:00','TimeLimit=99:00:00').replace('Requeue=0','Requeue=1')
record('original Z01 wrong allocation rejected', rejection(lambda:c.zero_allocation_identity(bad,'777')))
field_rejections={}
for key, new in [('JobId','888'),('Cluster','foreign'),('NodeList','h100-16'),('BatchHost','h100-16'),('NumNodes','2'),('JobState','PENDING'),('OverSubscribe','YES'),('TresPerNode','gres:gpu:4'),('CPUs/Task','320'),('MinMemoryNode','1G'),('TimeLimit','99:00:00'),('Requeue','1'),('Restarts','1'),('NumCPUs','31')]:
    old=dict(x.split('=',1) for x in ALLOCATION.split())[key]
    field_rejections[key]=rejection(lambda raw=ALLOCATION.replace(key+'='+old,key+'='+new):c.zero_allocation_identity(raw,'777'))
record('individual exact fields including actual CPU lower bound', field_rejections)
record('duplicate ambiguous field rejected', rejection(lambda:c.zero_allocation_identity(ALLOCATION+' JobId=777','777')))
for name, env, host in [('foreign cluster',dict(ENV,SLURM_CLUSTER_NAME='foreign'),'h100-9'),('foreign node',ENV,'h100-16'),('unknown job',dict(ENV,SLURM_JOB_ID='unknown'),'h100-9')]:
    run,calls=client(); err=rejection(lambda:c.verify_zero_runtime(env,host,run,cgroup_text=CGROUP)); assert calls==[]
    record(name+' rejected before query',err)
for name, outputs, expected_calls in [('foreign default',['ClusterName = foreign\n'],1),('federation',['ClusterName = wuhan-h100\n','FederationName = other\n'],2)]:
    run,calls=client(outputs); err=rejection(lambda:c.verify_zero_runtime(ENV,'h100-9',run,cgroup_text=CGROUP)); assert len(calls)==expected_calls
    record(name+' never queries job/sibling',dict(error=err,calls=calls))
record('client failure rejected',rejection(lambda:c.verify_zero_runtime(ENV,'h100-9',lambda *a,**kw:SimpleNamespace(returncode=1,stderr='failure',stdout=''),cgroup_text=CGROUP)))

def guard_fixture(root, job='777', swapped=False, altered=False, symlink=False):
    tools=root/'package';tools.mkdir(); (tools/'calibration_contract.py').write_bytes((WRAPPER/'calibration_contract.py').read_bytes())
    source=root/'tools/resource_guard_identity.py';source.parent.mkdir();source.write_bytes(RAW)
    run=root/'runs'/('tree-fresh200-'+job);run.mkdir(parents=True)
    package=dict(stage='fresh_mu_calibration',scope='FROZEN_PENDING_INDEPENDENT_FRESH200_MU_PRE_DISPATCH_REVIEW',files={'calibration_contract.py':c.sha(tools/'calibration_contract.py')},external_files={str(source):GUARD_SHA})
    manifest=tools/'manifest.json';manifest.write_text(json.dumps(package));digest=c.sha(manifest)
    local=dict(identity,job_id=job,allocation_stdout=ALLOCATION.replace('JobId=777','JobId='+job))
    (run/'scheduler-runtime-identity.json').write_text(json.dumps(dict(local,manifest_sha256=digest,review_sha256='b'*64,run_id=run.name)))
    if altered:source.write_bytes(RAW+b'\n# changed\n')
    if symlink:
        source.rename(source.with_suffix('.real'));source.symlink_to(source.with_suffix('.real'))
    uuids=['GPU-'+str(i) for i in range(8)]
    nv=ModuleType('pynvml');nv.nvmlInit=lambda:None;nv.nvmlDeviceGetCount=lambda:8
    nv.nvmlDeviceGetHandleByIndex=lambda i:i;nv.nvmlDeviceGetUUID=lambda i:uuids[i]
    nv.nvmlDeviceGetMemoryInfo=lambda h:SimpleNamespace(total=100,used=16)
    nv.nvmlDeviceGetUtilizationRates=lambda h:SimpleNamespace(gpu=1)
    nv.nvmlDeviceGetComputeRunningProcesses=lambda h:[SimpleNamespace(pid=42,usedGpuMemory=16)]
    nv.nvmlDeviceGetGraphicsRunningProcesses=lambda h:[];nv.nvmlDeviceGetProcessUtilization=lambda h,t:[]
    nv.NVMLError_NotFound=type('NVMLError_NotFound',(Exception,),{})
    original_read=Path.read_text; original_open=os.open; forwarded=[];opened=[]
    def read(path,*a,**kw):
        text=str(path)
        if text=='/proc/self/cgroup':return '0::/slurm/job_777/step_batch\n'
        if text=='/dataset/shaoshitong/search-r1-rl/tree-grpo/tools/expected-gpu-uuids.json':return json.dumps(uuids)
        if text=='/proc/42/cgroup':return '0::/slurm/job_999/step_batch\n'
        if text=='/proc/42/stat':return '42 (fixture) S '+' '.join(['0']*18+['101']+['0']*4)
        return original_read(path,*a,**kw)
    def opened_source(path,*a,**kw):
        fd=original_open(path,*a,**kw)
        if Path(path)==source:
            opened.append(fd)
            if swapped:
                source.rename(source.with_suffix('.frozen'));source.write_text("raise RuntimeError('PATH_REOPEN_ATTACK_EXECUTED')\n")
        return fd
    return tools,source,run,digest,local,nv,read,opened_source,forwarded,opened

for name, options in [('full frozen STOP/cancel interception',{}),('same FD frozen guard after path replacement',dict(swapped=True)),('changed guard bytes refused',dict(altered=True)),('symlink guard refused',dict(symlink=True))]:
    with TemporaryDirectory() as tmp:
        root=Path(tmp);tools,source,run,digest,local,nv,read,opener,forwarded,opened=guard_fixture(root,**options)
        old_argv,old_subprocess=sys.argv,subprocess.run
        env=dict(ENV,TREE_INSTRUCT_PACKAGE_SHA256=digest,TREE_INSTRUCT_REVIEW_SHA256='b'*64)
        runner,queries=client()
        with patch.object(c,'ROOT',root),patch.object(c,'__file__',str(tools/'calibration_contract.py')),patch.object(c.socket,'gethostname',lambda:'h100-9'),patch.dict(os.environ,env,clear=True),patch.dict(sys.modules,{'pynvml':nv}),patch.object(Path,'read_text',read),patch.object(os,'open',opener),patch.object(subprocess,'run',runner):
            patched_run=subprocess.run;err=rejection(c.zero_guard_child);assert subprocess.run is patched_run and sys.argv is old_argv
        assert subprocess.run is old_subprocess and forwarded==[] and len(queries)==3
        rows=[json.loads(x) for x in (run/'resource-samples.jsonl').read_text().splitlines()] if (run/'resource-samples.jsonl').exists() else []
        refusal=json.loads((run/'guard-cancellation-refused.json').read_text()) if (run/'guard-cancellation-refused.json').exists() else None
        if options.get('altered') or options.get('symlink'):
            assert rows==[] and refusal is None
        else:
            assert len(opened)==1 and rows[-1]['event']=='STOP_OWN_JOB' and 'dual15%' in rows[-1]['error'] and refusal['scheduler_cancellation_executed'] is False
        record(name,dict(error=err,opened_source_fd_count=len(opened),records=rows,refusal=refusal,subprocess_and_argv_restored=True,actual_runtime_queries=queries))

# Original Z09 plus independently extended malformed/foreign/mixed cgroup cases.
child_cases=[]
for name,job,current in [('original Z09','888',CGROUP),('missing own job','777','0::/user.slice\n'),('mixed sibling','777','0::/slurm/job_777/step_batch\n1:cpu:/slurm/job_888/step_batch\n'),('numeric prefix','777','0::/slurm/job_7770/step_batch\n'),('missing current cgroup','777',''),('malformed current cgroup','777','unknown'),('own plus nested sibling','777','0::/slurm/job_777/job_888/step_batch\n')]:
    with TemporaryDirectory() as tmp:
        root=Path(tmp);tools,source_path,run,digest,local,nv,read,opener,forwarded,opened=guard_fixture(root,job=job)
        runner,queries=client(['ClusterName = wuhan-h100\n','',local['allocation_stdout']])
        env=dict(ENV,SLURM_JOB_ID=job,TREE_INSTRUCT_PACKAGE_SHA256=digest,TREE_INSTRUCT_REVIEW_SHA256='b'*64)
        def current_read(path,*a,**kw):
            return current if str(path)=='/proc/self/cgroup' else read(path,*a,**kw)
        with patch.object(c,'ROOT',root),patch.object(c,'__file__',str(tools/'calibration_contract.py')),patch.dict(os.environ,env,clear=True),patch.dict(sys.modules,{'pynvml':nv}),patch.object(Path,'read_text',current_read),patch.object(c.socket,'gethostname',lambda:'h100-9'),patch.object(subprocess,'run',runner):
            err=rejection(c.zero_guard_child)
        assert queries==[] and err['type']=='ValueError' and not (run/'guard-runtime-identity.json').exists()
        result=dict(name=name,current_cgroup=current,environment_job_id=job,queries=queries,error=err,guard_original_started=False,scheduler_commands_executed=0,real_gpu_calls=0,official_assets_loaded=False)
        child_cases.append(result);record('actual child before-query refusal: '+name,result)
regression=dict(status='V10_CHILD_CGROUP_BEFORE_QUERY_REGRESSIONS_PASS_CPU_ONLY',cases=child_cases,original_v9_counterexample_path=str(WORK/'tree-fresh200-zero-v9-context-free-review/CHILD_SIBLING_QUERY_COUNTEREXAMPLE.json'),original_v9_counterexample_sha256=c.sha(WORK/'tree-fresh200-zero-v9-context-free-review/CHILD_SIBLING_QUERY_COUNTEREXAMPLE.json'),actual_slurm_executed=False,actual_gpu_executed=False)
(HERE/'CHILD_BEFORE_QUERY_REGRESSION.json').write_text(json.dumps(regression,indent=2,sort_keys=True)+'\n')

# Execute actual cleanup algorithm with identity-safe fake kernel interfaces.
runtime_spec=importlib.util.spec_from_file_location('independent_owned_runtime',WORK/'tree_shared_owned_runtime_fixed.py')
owned_module=importlib.util.module_from_spec(runtime_spec);runtime_spec.loader.exec_module(owned_module)
relative='/slurm/job_777/step_batch'; signals=[]; sequence=iter([{1,42,43},{1},{1}]);real_read=Path.read_text
def member_read(path,*a,**kw):
    if str(path)=='/proc/42/cgroup':return '0::'+relative+'\n'
    if str(path)=='/proc/43/cgroup':return '0::/slurm/job_888/step_batch\n'
    return real_read(path,*a,**kw)
obj=owned_module.OwnedProcesses.__new__(owned_module.OwnedProcesses);obj.scope=Path('/fixture');obj.relative=relative;obj.baseline={1};obj.children=[];obj.streams=[]
with patch.object(owned_module,'cgroup_pids',lambda scope:next(sequence)),patch.object(Path,'read_text',member_read),patch.object(os,'pidfd_open',lambda pid:pid,create=True),patch.object(os,'close',lambda fd:None),patch.object(owned_module.signal,'pidfd_send_signal',lambda fd,sig:signals.append((fd,int(sig))),create=True),patch.object(owned_module.time,'sleep',lambda seconds:None):obj.cleanup()
assert signals==[(42,15)] and obj.cleanup_residual=={}
record('actual owned cleanup never signals foreign/moved PID or baseline',dict(signals=signals,residual=obj.cleanup_residual))

# Actual supervisor main AST: failed guard and failed termination record both
# reach owned cleanup. No source import, Slurm client, GPU, model or Ray runs.
tree=ast.parse((WRAPPER/'supervisor.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
for fail_record in (False,True):
    with TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'runs').mkdir();(root/'aliases').mkdir();candidate=root/'candidate';cleanup=[];spawned=[]
        model={'sha256':'d'*64};package=dict(scope='FROZEN_PENDING_INDEPENDENT_FRESH200_MU_PRE_DISPATCH_REVIEW',stage='fresh_mu_calibration',probability_calibration={'source_interface_reviewed':True,'environment':{'TREE_FRESH_INITIALIZATION_ONLY':'1','TREE_PPO_CALIBRATION_ONLY':'1'}},files={'source/repo/verl/trainer/ppo/ray_trainer.py':'c'*64,'source/repo/verl/trainer/main_ppo.py':'c'*64},external_files={},model_identity=model,official_model_path=str(root/'model'))
        search=ModuleType('search_r1');search.__file__=str(candidate/'repo/search_r1/__init__.py')
        preflight=ModuleType('search_r1.baselines.preflight');preflight.validate_inherited_gpu_leases=lambda n:None
        teacher=ModuleType('search_r1.baselines.select_teacher');teacher.model_manifest=lambda path:model
        class FakeOwned:
            def __init__(self,*a):self.baseline=set();self.cleanup_residual={}
            def spawn(self,*a,**kw):spawned.append(a);return SimpleNamespace(poll=lambda:1)
            def cleanup(self):cleanup.append('called')
        fake_subprocess=SimpleNamespace(check_output=lambda argv,**kw:'4d133efeb1f675139d2558640f1e21d50780f133\n' if argv[1]=='rev-parse' else '')
        import fcntl,signal,socket,time
        background=[1680782,1680783,1680784,1680785,1696443,1696444,1696445,1696446]
        def configuration(run,stage):
            (run/'resolved.yaml').write_text('tiny: fixture\n');return None,200
        ns=dict(SimpleNamespace=SimpleNamespace,PACKAGE=package,COMMIT='f7e762af365a2beb9429e4cb3b6d5f0d808d4d7d',CAND=candidate,ROOT=root,MODEL=root/'model',TOOLS=WRAPPER,os=os,Path=lambda value:root/'aliases' if value=='/tmp' else Path(value),sys=sys,time=time,socket=socket,fcntl=fcntl,signal=signal,json=json,subprocess=fake_subprocess,require=c.require,sha=lambda path:c.sha(path) if Path(path).is_file() else 'c'*64,forbid_recovery_environment=c.forbid_recovery_environment,cgroup_scope=lambda job:(Path('/fixture'),relative),CandidateOwnedProcesses=FakeOwned,available=lambda p:None,verify_zero_runtime=lambda:identity,local_scheduler_environment=c.local_scheduler_environment,write_owned_identity=c.write_owned_identity,configuration=configuration,protected_processes=lambda:None,gpu_sample=lambda:dict(processes=['GPU, '+str(pid)+', 1' for pid in background]),__file__=str(WRAPPER/'supervisor.py'))
        env=dict(ENV,CUDA_VISIBLE_DEVICES=','.join(map(str,range(8))),SLURM_JOB_GPUS='0,1,2,3,4,5,6,7',TREE_INSTRUCT_PACKAGE_SHA256='a'*64,TREE_INSTRUCT_REVIEW_SHA256='b'*64,PATH=os.environ.get('PATH',''),SEARCH_R1_GPU_LEASES='[]')
        original_write=Path.write_text
        ns['CONTENT']='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
        def failing_write(path,*a,**kw):
            if fail_record and path.name=='termination-cost.json':raise OSError('injected record failure')
            return original_write(path,*a,**kw)
        exec(compile(ast.Module(body=[main],type_ignores=[]),str(WRAPPER/'supervisor.py'),'exec'),ns)
        with patch.dict(os.environ,env,clear=True),patch.dict(sys.modules,{'search_r1':search,'search_r1.baselines.preflight':preflight,'search_r1.baselines.select_teacher':teacher}),patch.object(socket,'gethostname',lambda:'h100-9'),patch.object(signal,'signal',lambda *a:None),patch.object(Path,'write_text',failing_write):
            err=rejection(ns['main'])
        assert cleanup==['called'] and len(spawned)==1, repr(dict(error=err,cleanup=cleanup,spawned=spawned))
        result=json.loads((root/'runs/tree-fresh200-777/run-result.json').read_text());cost=json.loads((root/'runs/tree-fresh200-777/cost.json').read_text())
        assert result['status']=='FAILED' and cost['allocated_gpus']==8 and cost['requested_cpus']==32 and cost['allocated_cpus']==192 and cost['exit_code']==1
        record('parent actual main finally cleanup '+('after record failure' if fail_record else 'after guard failure'),dict(error=err,cleanup_calls=cleanup,cost=cost,result=result))

proof=dict(status='INDEPENDENT_V10_LOCAL_CPU_REGRESSIONS_PASS',cases=cases,distinct_case_groups=len(cases),actual_slurm_executed=False,actual_gpu_executed=False,actual_remote_executed=False,actual_official_assets_loaded=False,actual_optimizer_writes=0,actual_completed_outer=0,all_scheduler_calls_mocked=True)
(HERE/'INDEPENDENT_CPU_PROOF.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(status=proof['status'],distinct_case_groups=len(cases),resolved_finding='Z09_FIXED_ONLY_V10' ,proof_sha256=c.sha(HERE/'INDEPENDENT_CPU_PROOF.json'))))
