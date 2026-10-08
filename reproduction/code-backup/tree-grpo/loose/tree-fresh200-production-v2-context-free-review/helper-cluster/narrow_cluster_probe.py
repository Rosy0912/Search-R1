"""Independent source/CPU probes: no actual scheduler, /proc, NVML, PT or assets."""
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as N, ModuleType
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
SOURCE = WORK / 'tree-fresh200-train-runtime-v2'
sys.path.insert(0, str(SOURCE))
import production_contract as c
import slurm_identity as s
import submit_once as dispatcher
import render_dispatch as renderer
import guard_entry as guard
from test_runtime_contract_cpu import observation, write

CONFIG = 'ClusterName = wuhan-h100\n'
ALLOCATION = 'JobId=99 JobState=RUNNING NodeList=h100-9 BatchHost=h100-9 NumNodes=1 OverSubscribe=NO TresPerNode=gres:gpu:8 CPUs/Task=32 NumCPUs=96 Requeue=0 Restarts=0 TimeLimit=3-00:00:00 MinMemoryNode=256G'
BAD_ENV = dict(SLURM_JOB_ID='99',SLURM_JOB_NODELIST='h100-9',SLURM_CLUSTER_NAME='wuhan-h100',SLURM_CONF='/foreign',SLURM_CONF_SERVER='foreign',SLURM_CLUSTERS='foreign',SLURM_FEDERATION='foreign',SLURM_FEDERATION_CLUSTER='foreign',SBATCH_CLUSTERS='foreign',SBATCH_TIME='unlimited',SQUEUE_CLUSTERS='foreign',SACCT_CLUSTERS='foreign',SCONTROL_CLUSTER='foreign',SCANCEL_CLUSTERS='foreign',KEEP='preserved')
results=[]
all_calls=[]

def dump(name, value):
    (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def case(name, fn):
    fn()
    results.append(dict(name=name,status='PASS'))

def rejects(fn, types=(ValueError,)):
    try: fn()
    except types as error: return type(error).__name__+': '+str(error)
    raise AssertionError('Required rejection was missing')

def fake_scheduler(command, **kwargs):
    all_calls.append(dict(argv=list(command),env=dict(kwargs.get('env',{})),timeout=kwargs.get('timeout')))
    if command==['scontrol','--local','show','config']: return N(returncode=0,stdout=CONFIG,stderr='')
    if command==['scontrol','--local','show','federation']: return N(returncode=0,stdout='',stderr='')
    if command==['scontrol','--local','show','job','99']: return N(returncode=0,stdout=ALLOCATION,stderr='')
    raise AssertionError('Unexpected scheduler mock argv '+repr(command))

def freeze():
    inventory=SOURCE/'FILE_HASHES_V2.json'
    assert hashlib.sha256(inventory.read_bytes()).hexdigest()=='e21e85f7680caf2a1caf77e97adb214eebdaf724ef82b482891b776bd162ffc9'
    pins=json.loads(inventory.read_text())['wrapper_files']
    checked={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ('slurm_identity.py','submit_once.py','guard_entry.py','launch.sbatch','supervisor.py','render_dispatch.py','production_contract.py','SOURCE_CPU_RECEIPT_V2.json')}
    assert all(pins[name]==digest for name,digest in checked.items())
    assert checked['SOURCE_CPU_RECEIPT_V2.json']=='0cceff7d7753af51c84fb2daf8ca423980c07d419fbe8aedd5ddea4ffdbc84f8'
    dump('NARROW_FROZEN_HASHES.json',checked)

def exact_runtime():
    value=s.verify_runtime(BAD_ENV,'h100-9.example',fake_scheduler)
    assert value['cluster_name']==s.CLUSTER and value['job_id']==99 and value['allocated_cpus']==96
    assert value['wall_hours']==72 and value['no_requeue'] is True and value['restarts']==0
    assert len(all_calls)==3
    assert all('--local' in entry['argv'] and entry['timeout']==30 and entry['env']=={'SLURM_JOB_ID':'99','SLURM_JOB_NODELIST':'h100-9','KEEP':'preserved'} for entry in all_calls)
    dump('MOCK_EXACT_RUNTIME_IDENTITY.json',value)

def runtime_rejections():
    tokens=dict(token.split('=',1) for token in ALLOCATION.split())
    errors={}
    fields={'JobId':'999','JobState':'PENDING','NodeList':'h100-16','BatchHost':'h100-16','NumNodes':'2','OverSubscribe':'YES','TresPerNode':'gres:gpu:80','CPUs/Task':'320','NumCPUs':'31','Requeue':'1','Restarts':'1','TimeLimit':'UNLIMITED','MinMemoryNode':'128G'}
    for key,value in fields.items():
        text=ALLOCATION.replace(key+'='+tokens[key],key+'='+value)
        errors[key]=rejects(lambda text=text:s.allocation_identity(text,'99',s.CLUSTER))
    errors['duplicate']=rejects(lambda:s.allocation_identity(ALLOCATION+' JobId=99','99',s.CLUSTER))
    errors['explicit_foreign_cluster']=rejects(lambda:s.allocation_identity(ALLOCATION+' Cluster=foreign','99',s.CLUSTER))
    errors['wrong_host']=rejects(lambda:s.verify_runtime(BAD_ENV,'h100-16',fake_scheduler))
    errors['runtime_foreign_cluster']=rejects(lambda:s.verify_runtime(dict(BAD_ENV,SLURM_CLUSTER_NAME='foreign'),'h100-9',fake_scheduler))
    for label,config,federation,rc,stderr in [('foreign','ClusterName = foreign\n','',0,''),('duplicate',CONFIG+CONFIG,'',0,''),('federated',CONFIG,'FederationName=x',0,''),('query_failed',CONFIG,'',1,''),('query_stderr',CONFIG,'',0,'warning')]:
        def run(command,**kw):return N(returncode=rc,stdout=config if command[-1]=='config' else federation,stderr=stderr)
        errors['controller_'+label]=rejects(lambda:s.cluster_identity(run,BAD_ENV))
    dump('RUNTIME_REJECTIONS.json',errors)

def journal(path):
    return dict(status='SUBMITTED',cluster_name=s.CLUSTER,node_list=s.NODE,job_id=99,run_id='tree-fresh200-train-99',manifest_sha256='m',review_sha256='r',dispatch_intent_path=str(path),parsable_stdout='99;wuhan-h100',stdout='99;wuhan-h100\n',stderr='',returncode=0,local_cluster_identity=dict(cluster_name=s.CLUSTER,local_queries=True,configuration_stdout=CONFIG,federation_stdout=''))

def journal_lineage_rejections():
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        path=Path(tmp)/'intent.json';good=journal(path)
        kw=dict(cluster_name=s.CLUSTER,node_list=s.NODE,run_id='tree-fresh200-train-99',wait_seconds=0)
        write(path,good);assert c.submitted_intent(path,'m','r','99',**kw)['job_id']==99
        errors={}
        for key,changes in {'foreign_cluster':{'cluster_name':'foreign'},'foreign_suffix':{'parsable_stdout':'99;foreign','stdout':'99;foreign\n'},'bool_id':{'job_id':True},'unknown':{'status':'UNKNOWN'},'wrong_node':{'node_list':'h100-16'},'wrong_run':{'run_id':'foreign-99'},'wrong_manifest':{'manifest_sha256':'wrong'},'wrong_review':{'review_sha256':'wrong'},'wrong_path':{'dispatch_intent_path':str(path)+'x'},'proof_missing':{'local_cluster_identity':{}},'federation':{'local_cluster_identity':dict(good['local_cluster_identity'],federation_stdout='FederationName=x')}}.items():
            write(path,dict(good,**changes));errors[key]=rejects(lambda:c.submitted_intent(path,'m','r','99',**kw))
        legacy=dict(good);legacy.pop('cluster_name');write(path,legacy)
        errors['legacy']=rejects(lambda:c.submitted_intent(path,'m','r','99',**kw))
        dump('JOURNAL_LINEAGE_REJECTIONS.json',errors)

def prior_accounting():
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        folder=Path(tmp);prior=folder/'TREE_FRESH200_TRAIN_ONCE-old.json';target=folder/'TREE_FRESH200_TRAIN_ONCE-new.json';write(prior,journal(prior))
        calls=[]
        def run(command,**kwargs):
            calls.append(dict(argv=list(command),env=kwargs['env']))
            return N(returncode=0,stdout='99|wuhan-h100|COMPLETED|0:0\n',stderr='')
        assert len(dispatcher.prior_journals_clear(target,run,BAD_ENV))==1
        assert calls[0]['argv']==['sacct','--local','-X','-n','-P','-j','99','-o','JobIDRaw,Cluster,State,ExitCode']
        assert not any(k in s.EXCLUDED or k.startswith(s.PREFIXES) for k in calls[0]['env'])
        errors={}
        for label,output in [('foreign','99|foreign|COMPLETED|0:0\n'),('active','99|wuhan-h100|RUNNING|0:0\n'),('no_rows',''),('step_alias','99.batch|wuhan-h100|COMPLETED|0:0\n'),('same_id_duplicate','99|wuhan-h100|COMPLETED|0:0\n99|wuhan-h100|COMPLETED|0:0\n')]:
            errors[label]=rejects(lambda output=output:dispatcher.prior_journals_clear(target,lambda *a,**kw:N(returncode=0,stdout=output,stderr=''),BAD_ENV))
        dump('PRIOR_ACCOUNTING_LOCAL_TRACE.json',dict(calls=calls,rejections=errors))

def actual_submit(timeout,mutate_script=False):
    label='script-mutation' if mutate_script else 'timeout' if timeout else 'success'
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        root=Path(tmp).resolve();package=root/'instruct300/candidates/frozen';package.mkdir(parents=True)
        manifest,release_descriptor=observation(root/'observations')
        intent=root/'instruct300/TREE_FRESH200_TRAIN_ONCE-independent-cpu.json'
        manifest.update(scope='FROZEN_PENDING_INDEPENDENT_FRESH200_TRAIN200_PRE_DISPATCH_REVIEW',stage='fresh_mu_train200',bootstrap_thresholds_accepted=True,production_thresholds_accepted=False,dispatch={'once_intent':str(intent),'cluster_name':s.CLUSTER})
        for name in ('render_dispatch.py','submit_once.py'):(package/name).write_bytes((SOURCE/name).read_bytes())
        manifest['files']={name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in ('render_dispatch.py','submit_once.py')}
        m=write(package/'manifest.json',manifest)
        release,_,_=c.read_bound(release_descriptor['path']);release.update(manifest_sha256=m['sha256'],dispatch_intent_path=str(intent),dispatch_renderer_sha256=manifest['files']['render_dispatch.py'],dispatch_submitter_sha256=manifest['files']['submit_once.py'])
        r=write(root/'release.json',release);calls=[];actual_submit_bytes=[]
        def run(command,**kwargs):
            calls.append(dict(argv=list(command),env=kwargs['env'],timeout=kwargs['timeout']))
            if command[0]=='squeue':
                if mutate_script:
                    # Exact reviewed bytes have already been read and closed.
                    original_bytes=script.read_bytes()
                    assert hashlib.sha256(original_bytes).hexdigest()==script_sha
                    (OUT/'MOCK_REVIEWED_SCRIPT_BEFORE_MUTATION.sbatch').write_bytes(original_bytes)
                    script.write_bytes(original_bytes.replace(b'#SBATCH --time=72:00:00',b'#SBATCH --time=999:00:00'))
                    assert hashlib.sha256(script.read_bytes()).hexdigest()!=script_sha
                return N(returncode=0,stdout='',stderr='')
            if command[0]=='sbatch':
                # Intent must already exist durably before the only submission.
                assert json.loads(intent.read_text())['status']=='SUBMITTING'
                actual_submit_bytes.append(Path(command[-1]).read_bytes())
                if timeout:raise subprocess.TimeoutExpired(command,60,output=b'\x00\xffunknown\n',stderr=b'\xfe')
                return N(returncode=0,stdout='99;wuhan-h100\n',stderr='')
            return fake_scheduler(command,**kwargs)
        actual_inputs=c.production_inputs
        with patch.object(dispatcher,'ROOT',root),patch.object(renderer,'ROOT',root),patch.object(dispatcher,'__file__',str(package/'submit_once.py')),patch.object(dispatcher,'production_inputs',side_effect=lambda *a,**kw:actual_inputs(*a,**kw,trusted_root=root)),patch.dict(os.environ,BAD_ENV,clear=True):
            script=root/'dispatch.sbatch';script.write_text(renderer.render(package,m['sha256'],r['path'],r['sha256'],72));script_sha=hashlib.sha256(script.read_bytes()).hexdigest()
            args=(package,m['sha256'],r['path'],r['sha256'],script,script_sha)
            if timeout:rejects(lambda:dispatcher.submit(*args,run=run,hostname=s.NODE),(subprocess.TimeoutExpired,))
            else:assert dispatcher.submit(*args,run=run,hostname=s.NODE)['job_id']==99
            value=json.loads(intent.read_text());assert value['status']==('UNKNOWN' if timeout else 'SUBMITTED')
            if timeout:assert base64.b64decode(value['timeout_stdout']['bytes'])==b'\x00\xffunknown\n' and base64.b64decode(value['timeout_stderr']['bytes'])==b'\xfe'
            rejection=rejects(lambda:dispatcher.submit(*args,run=run,hostname=s.NODE))
            assert sum(entry['argv'][0]=='sbatch' for entry in calls)==1
            assert all(not any(k in s.EXCLUDED or k.startswith(s.PREFIXES) for k in entry['env']) for entry in calls)
            assert all('-M' not in entry['argv'] and entry['argv'][0]!='scancel' for entry in calls)
            (OUT/('MOCK_SUBMIT_'+label.upper()+'_JOURNAL.json')).write_bytes(intent.read_bytes())
            (OUT/('MOCK_RENDER_'+label.upper()+'.sbatch')).write_bytes(script.read_bytes())
            dump('MOCK_SUBMIT_'+label.upper()+'_TRACE.json',dict(calls=calls,reentry_rejection=rejection,sbatch_count=1))
            if mutate_script:
                actual_sha=hashlib.sha256(actual_submit_bytes[0]).hexdigest()
                assert value['status']=='SUBMITTED' and value['dispatch_script_sha256']==script_sha and actual_sha!=script_sha
                dump('ACTUAL_SUBMIT_SCRIPT_TOCTOU_COUNTEREXAMPLE.json',dict(status='CONFIRMED_REVIEWED_SCRIPT_BYTES_NOT_BOUND_TO_SUBMIT_PATH',source_submitter_sha256=hashlib.sha256((SOURCE/'submit_once.py').read_bytes()).hexdigest(),source_inventory_sha256=hashlib.sha256((SOURCE/'FILE_HASHES_V2.json').read_bytes()).hexdigest(),fixture_only=True,actual_slurm_calls=0,original_reviewed_script_sha256=script_sha,journal_dispatch_script_sha256=value['dispatch_script_sha256'],actual_bytes_read_by_mock_sbatch_sha256=actual_sha,mutated_only_temporary_script=True,mutation_stage='injected squeue response after script read/hash/exact renderer comparison, before sbatch argv consumption',mutation='SBATCH time72 changed to999',returned_journal_status=value['status'],mock_sbatch_count=1,call_sequence=[entry['argv'][0] for entry in calls],reviewed_bytes_verified_before_mutation=True,changed_bytes_accepted_as_submitted=True))

def actual_original_guard_execution():
    original=WORK/'resource_guard_identity.py';raw=original.read_bytes();source_sha=hashlib.sha256(raw).hexdigest()
    assert source_sha=='284c014cd6e273816c112ea8a7b515f3afa4e55bef5c74d23f319f93f146ff88'
    run=OUT/'tree-fresh200-train-99';run.mkdir(exist_ok=True)
    identity=run/'allocation-identity.json';out=run/'resource-samples.jsonl';refusal=out.with_suffix('.cancellation-refused.json')
    for path in (identity,out,refusal):path.unlink(missing_ok=True)
    current=s.verify_runtime(BAD_ENV,'h100-9',fake_scheduler)
    write(identity,dict(current,manifest_sha256='m',review_sha256='r',run_id=run.name))
    identity_sha=hashlib.sha256(identity.read_bytes()).hexdigest()
    mock_nvml=ModuleType('pynvml')
    def nvml_init():raise RuntimeError('CPU mock forces original guard STOP before any NVML/GPU read')
    mock_nvml.nvmlInit=nvml_init
    actual_read=Path.read_text
    def path_read(path,*args,**kwargs):
        if str(path)=='/proc/self/cgroup':return '0::/slurm/uid_0/job_99/step_batch\n'
        if str(path)=='/dataset/shaoshitong/search-r1-rl/tree-grpo/tools/expected-gpu-uuids.json':return '["CPU-MOCK-UUID"]'
        if str(path).startswith('/proc/'):raise AssertionError('Unexpected real /proc read')
        return actual_read(path,*args,**kwargs)
    forbidden=[]
    def deny_actual_subprocess(*args,**kwargs):forbidden.append((args,kwargs));raise AssertionError('No real subprocess allowed')
    argv=['guard_entry.py','--source',str(original),'--source-sha',source_sha,'--identity',str(identity),'--identity-sha',identity_sha,'--output',str(out)]
    old_argv=list(sys.argv)
    with patch.object(guard,'verify_runtime',return_value=current),patch.dict(os.environ,dict(BAD_ENV,TREE_INSTRUCT_PACKAGE_SHA256='m',TREE_INSTRUCT_REVIEW_SHA256='r'),clear=True),patch.object(sys,'argv',argv),patch.object(Path,'read_text',path_read),patch.dict(sys.modules,{'pynvml':mock_nvml}),patch.object(subprocess,'run',deny_actual_subprocess):
        error=rejects(guard.main,(RuntimeError,))
        assert sys.argv==[str(original),'--output',str(out)]
        assert subprocess.run is deny_actual_subprocess
    assert sys.argv==old_argv and forbidden==[]
    stop=json.loads(out.read_text());assert stop['job_id']=='99' and stop['event']=='STOP_OWN_JOB' and 'CPU mock forces original guard STOP' in stop['error']
    record=json.loads(refusal.read_text());assert record['command']==['scancel','99'] and record['scheduler_cancellation_executed'] is False
    assert record['foreign_processes_touched'] is False
    dump('ACTUAL_ORIGINAL_GUARD_EXECUTION_PROOF.json',dict(original_guard_sha256=source_sha,original_top_level_argv=[str(original),'--output',str(out)],source_hash_verified=True,source_exec_completed_to_stop=True,stop_record=stop,cancellation_refusal=record,actual_subprocess_count=0,exception=error,verify_runtime_in_main='injected preverified exact identity; scheduler parser separately exercised with actual functions',nvml='tiny module raises before GPU read'))

def wrapper_ast():
    tree=ast.parse((SOURCE/'guard_entry.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    argv_assignment=next(n for n in ast.walk(main) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='argv' for t in n.targets))
    execution=next(n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='exec')
    assert argv_assignment.lineno<execution.lineno
    direct_scheduler=[]
    for name in ('slurm_identity.py','submit_once.py','guard_entry.py','supervisor.py'):
        for node in ast.walk(ast.parse((SOURCE/name).read_text())):
            if isinstance(node,ast.List) and node.elts and isinstance(node.elts[0],ast.Constant) and node.elts[0].value in ('scancel','scontrol','sacct','squeue','sbatch'):
                command=[x.value for x in node.elts if isinstance(x,ast.Constant)]
                direct_scheduler.append(dict(file=name,line=node.lineno,constant_argv=command))
                assert command[0]!='scancel'
                if command[0] in ('scontrol','sacct','squeue'):assert '--local' in command
    dump('WRAPPER_SCHEDULER_AND_ARGV_AST.json',dict(guard_original_argv_line=argv_assignment.lineno,guard_original_exec_line=execution.lineno,direct_scheduler_literals=direct_scheduler,wrapper_scancel_literal_calls=0,original_external_guard_scancel='retained exact source but dynamically intercepted; actual top-level STOP probe confirms no subprocess'))

case('freeze exact narrow bytes',freeze)
case('actual runtime exact own cluster/allocation/environment',exact_runtime)
case('actual runtime rejects wrong fields/foreign controller/federation/errors',runtime_rejections)
case('actual journal consumer rejects foreign/UNKNOWN/legacy/same-number aliases',journal_lineage_rejections)
case('actual prior accounting local exact terminal lineage',prior_accounting)
case('actual submit success once plus irreversible reentry reject',lambda:actual_submit(False))
case('actual submit timeout exact UNKNOWN bytes plus irreversible reentry reject',lambda:actual_submit(True))
case('actual submit temporary script mutation counterexample confirmed',lambda:actual_submit(False,True))
case('actual frozen original top-level guard argv/STOP/scancel refusal',actual_original_guard_execution)
case('static wrapper argv ordering/no unbound scheduler calls',wrapper_ast)
case('freeze unchanged after probes',freeze)
dump('NARROW_CLUSTER_PROOF.json',dict(status='BLOCKED_ACTUAL_SUBMIT_SCRIPT_TOCTOU_CONFIRMED_SOURCE_CPU_ONLY',scope='original P01/P02 narrow regression plus exact submitted-byte binding',cases=results,case_count=len(results),blocking_findings=['P05 exact script path TOCTOU: reviewed original SHA journaled, mutated path bytes accepted by mock sbatch'],mock_scheduler_calls_only=True,actual_slurm_calls=0,actual_remote_calls=0,actual_gpu_calls=0,official_asset_loads=0,pt_loads=0,source132_reruns=0,owner_code_edits=0,gpu_ready=False,production_dispatch_authorized=False,limits=['All scheduler output is synthetic mock output; real Slurm formatting and default-controller behavior remain unexecuted.','Original guard bytes executed with mocked /proc/UUID JSON/NVML and preverified injected identity; actual cgroup/foreign GPU coexistence/owned cleanup remains unexecuted.','Fixture observation authority is synthetic CPU metadata; it authorizes no actual first-five or 6..200 training.']))
print(json.dumps(dict(status='PASS_SOURCE_CPU_ONLY',cases=len(results),actual_slurm_calls=0,actual_gpu_calls=0,gpu_ready=False),sort_keys=True))
