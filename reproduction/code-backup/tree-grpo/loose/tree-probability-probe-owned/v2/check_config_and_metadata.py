"""Independent tiny source/metadata cases. No production asset or runtime calls."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
OWNED = HERE.parent
V3 = WS / 'tree-probability-probe-runner-v3'
V2 = WS / 'tree-probability-probe-runner-v2'

def sha_bytes(value): return hashlib.sha256(value).hexdigest()
def sha(path): return sha_bytes(Path(path).read_bytes())
def require(value, message):
    if not value: raise ValueError(message)

spec = importlib.util.spec_from_file_location('independent_adapter_v2', OWNED/'resource_adapter_v2.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
candidate = V3/'search_r1/baselines/probability_probe_runner.py'
source = candidate.read_text()
tree = ast.parse(source)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='verified_bundle_config')
snippet = ast.get_source_segment(source, fn)
namespace = {'require': require}
exec(compile(ast.Module(body=[fn],type_ignores=[]), str(candidate), 'exec'), namespace)
consumer = namespace['verified_bundle_config']
cases=[]

def ok(name, call):
    call(); cases.append({'name':name,'result':'PASS'})

def refuses(name, call, message=None):
    try: call()
    except Exception as error:
        if message is not None: require(message in str(error), 'Unexpected refusal '+repr(error))
        cases.append({'name':name,'result':'REFUSED','exception':type(error).__name__,'message':str(error)})
    else: raise AssertionError(name+' was incorrectly accepted')

before=ast.parse((V2/'search_r1/baselines/probability_probe_runner.py').read_text())
old_functions={n.name:n for n in before.body if isinstance(n,ast.FunctionDef)}
new_functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
require(set(new_functions)-set(old_functions)=={'verified_bundle_config'},'Unexpected added functions')
for name, old in old_functions.items():
    now=copy.deepcopy(new_functions[name])
    if name=='run_probe':
        match=[n for n in ast.walk(now) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='verified_bundle_config']
        require(len(match)==1,'Expected exactly one new config consumer')
        match[0].func=ast.Attribute(value=ast.Name(id='OmegaConf',ctx=ast.Load()),attr='load',ctx=ast.Load())
        match[0].args=[ast.Subscript(value=ast.Name(id='bundle',ctx=ast.Load()),slice=ast.Constant(value='config_path'),ctx=ast.Load())]
    require(ast.dump(old,include_attributes=False)==ast.dump(now,include_attributes=False),'Non-config function changed '+name)
ok('All original runner functions AST-identical after the single config call substitution',lambda:None)
for p in [candidate, OWNED/'resource_adapter_v2.py', OWNED/'test_resource_adapter_v2.py']:
    compile(p.read_bytes(),str(p),'exec')
ok('Exact candidate/adapter/test bytes compile',lambda:None)

seen=[]
fake=ModuleType('omegaconf')
fake.OmegaConf=SimpleNamespace(create=lambda t:seen.append(t) or {'parsed':t},load=lambda *a:(_ for _ in ()).throw(AssertionError('Config path was reopened')))
text='mini: 384\nname: "原文本"\n'
bundle={'config_text':text,'config_path':'/must/not/open','identity':{'config_sha256':sha_bytes(text.encode('utf-8'))}}
with patch.dict(sys.modules,{'omegaconf':fake}):
    ok('Config consumer passes exact Unicode UTF8 text to create, never path load',lambda:require(consumer(bundle)=={'parsed':text} and seen==[text],'Wrong parser bytes'))
    for label, value in [('modified value',text.replace('384','192')),('newline removed',text.rstrip()),('bytes instead of str',text.encode()),('None',None)]:
        changed=copy.deepcopy(bundle);changed['config_text']=value
        count=len(seen)
        refuses('Changed config '+label+' refused before create',lambda b=changed:consumer(b))
        require(len(seen)==count,'Bad config was passed to parser')
    changed=copy.deepcopy(bundle);changed['identity']['config_sha256']='0'*64
    refuses('Wrong original SHA refuses',lambda:consumer(changed),'frozen original bytes')

for payload in [b'{"x":{"v":1,"v":2}}',b'{"x":[1e999]}',b'{"x":[NaN]}',b'{"x":[Infinity]}']:
    refuses('Strict metadata JSON refuses '+payload.decode(),lambda p=payload:A.strict_json_bytes(p))

root=A.ROOT
toy={'output':str(root/'runs/tree-probability-probe-99999/gpu-records'),'config_text':'mini: 384',
     'release':{'path':'unbound-entry','sha256':'unbound-sha'},'installed_versions':dict(A.INSTALLED_VERSIONS),
     'probe_source_files':[{'path':'toy','sha256':'a'*64}],'original_worker_sha256':'b'*64,
     'hf_index':{'weight_map':{'head':'x.safetensors'}}}
payload=A.bundle_payload_sha256(toy,'99999')
for field in ['config_text','installed_versions','probe_source_files','original_worker_sha256','hf_index']:
    altered=copy.deepcopy(toy);altered[field]='changed'
    ok('Whole normalized release binds '+field,lambda b=altered:require(A.bundle_payload_sha256(b,'99999')!=payload,'Unbound field'))
altered=copy.deepcopy(toy);altered['release']={'path':'new','sha256':'new'}
ok('Release entry excluded to avoid circular self-SHA; consumed separately by bound()',lambda:require(A.bundle_payload_sha256(altered,'99999')==payload,'Circular digest'))
altered=copy.deepcopy(toy);altered['output']=str(root/'runs/tree-probability-probe-88888/gpu-records')
refuses('Runtime output belongs to exact Slurm job',lambda:A.bundle_payload_sha256(altered,'99999'))

entries=[{'module':n,'path':str(root/'env/lib/python3.10/site-packages'/ (n.replace('.','/')+'.py')),'sha256':'a'*64} for n in sorted(A.INSTALLED_MODULES)]
modules={e['module']:SimpleNamespace(__file__=e['path']) for e in entries}
with patch.object(A.importlib.metadata,'version',side_effect=lambda n:A.INSTALLED_VERSIONS[n]),patch.object(A.importlib,'import_module',side_effect=lambda n:modules[n]),patch.object(A,'sha',return_value='a'*64):
    ok('All eight declared installed imports/version/path/SHA fixture passes',lambda:A.installed_bindings(entries,A.INSTALLED_VERSIONS))
    refuses('Missing installed source refuses',lambda:A.installed_bindings(entries[:-1],A.INSTALLED_VERSIONS))
    doubled=copy.deepcopy(entries);doubled[-1]=doubled[0]
    refuses('Duplicate instead of full installed source set refuses',lambda:A.installed_bindings(doubled,A.INSTALLED_VERSIONS))
    bad=copy.deepcopy(entries);bad[0]['path']='/outside/source.py'
    refuses('Installed source outside own env refuses',lambda:A.installed_bindings(bad,A.INSTALLED_VERSIONS))
    bad=copy.deepcopy(entries);bad[0]['sha256']='b'*64
    refuses('Imported implementation changed refuses',lambda:A.installed_bindings(bad,A.INSTALLED_VERSIONS))
    badversions=dict(A.INSTALLED_VERSIONS);badversions['torch']='2.8.0'
    refuses('Alternate local Torch version refuses',lambda:A.installed_bindings(entries,badversions))
with patch.object(A.importlib.metadata,'version',return_value='0'):
    refuses('Actual installed distribution mismatch refuses',lambda:A.installed_bindings(entries,A.INSTALLED_VERSIONS))

module_names=A.PROBE_MODULES+A.ORIGINAL_MODULES
source_entries=[{'path':str(root/'candidate/repo'/(n.replace('.','/')+'.py')),'sha256':'a'*64} for n in module_names]
source_entries += [{'path':str(root/'candidate/repo'/('dummy%d.py'%i)),'sha256':'a'*64} for i in range(700-len(source_entries))]
original=[{'path':e['path'].replace('/candidate/','/original/'),'sha256':e['sha256']} for e in source_entries]
worker=str(root/'original/repo/verl/workers/fsdp_workers.py');original.append({'path':worker,'sha256':'w'*64})
ib={'probe_source_files':source_entries,'identity':{'source_files':original},'original_worker_path':worker,'original_worker_sha256':'w'*64,'native_logprob_sha256':'a'*64}
module_map={n:SimpleNamespace(__file__=e['path']) for n,e in zip(module_names,source_entries)}
with patch.object(A.importlib,'import_module',side_effect=lambda n:module_map[n]),patch.object(A,'sha',return_value='a'*64):
    ok('Original/probe module imported paths couple to complete source map fixture',lambda:A.imported_sources(ib))
    bad=copy.deepcopy(ib);bad['original_worker_sha256']='x'*64
    refuses('Worker AST SHA is bound to original identity map',lambda:A.imported_sources(bad))
    bad=copy.deepcopy(ib);bad['native_logprob_sha256']='x'*64
    refuses('Native original logprob SHA bound to actual imported helper',lambda:A.imported_sources(bad))
    bad=copy.deepcopy(ib);bad['identity']['source_files'][len(A.PROBE_MODULES)]['sha256']='x'*64
    refuses('Changed original helper bytes refused',lambda:A.imported_sources(bad))
    bad=copy.deepcopy(ib);bad['probe_source_files'][-1]=copy.deepcopy(bad['probe_source_files'][0])
    refuses('Duplicate source mapping refuses',lambda:A.imported_sources(bad))

# Exact full prepare body executes against an explicitly synthetic dependency
# map. The parser and expensive source/asset contracts are stubs, never real.
fakecontracts=ModuleType('search_r1.baselines.probability_probe_contracts')
fakecontracts.bind_sources=lambda x:None
fakecontracts.bind_exact_files=lambda *x:None
fakecontracts.validate_selection=lambda *a,**k:None
fakecontracts.validate_fixed_config=lambda x:require(x=={'parsed':'mini: 384\n'},'Wrong consumed config fixture')
fake.OmegaConf.to_container=lambda x,resolve:x
job='99999';config_text='mini: 384\n'
base={'status':'GPU_PROBE_BUNDLE_INDEPENDENTLY_RELEASED','output':str(root/'runs/tree-probability-probe-99999/gpu-records'),
      'selection_file':{'path':'selection','sha256':'s'},'selection':{'inventory':{},'cpu_results':{},'monitor_companions':[]},'inventory':{},
      'identity':{'source_files':[],'config_sha256':sha_bytes(config_text.encode()),'parent_hf_sha256':'h','parent_state_manifest_sha256':'p'},
      'reviewed_resource_adapter':{'sha256':'a'},'release':{'path':'release','sha256':'r'},'runner_commit':'commit',
      'config_path':'config','config_text':config_text,'hf_identity':{'sha256':'h','files':{'model.safetensors.index.json':'index-sha'}},
      'hf_path':str(root/'hf'),'hf_index':{'weight_map':{'a':'tiny.safetensors'}},'parent_state_manifest':{'path':'parent'},
      'installed_sources':entries,'installed_versions':A.INSTALLED_VERSIONS,'probe_source_files':source_entries,'probe_mode':'eval_like_original_compute_log_prob'}
launch={'job_id':job,'bundle_sha256':'bundle','package_manifest_sha256':'package','allocation_text':'JobId=99999 JobState=RUNNING NodeList=h100-9 OverSubscribe=NO TresPerNode=gres:gpu:8 CPUs/Task=32 NumCPUs=32',
        'gpu_uuids':list(map(str,range(8))),'leases':[{'uuid':str(i)} for i in range(8)]}
def prep(b, release_change=None,index_change=None,bound_config=None):
    A.ACTIVE=None
    release={'status':'ZERO_UPDATE_GPU_PROBE_SOURCE_AND_DISPATCH_INDEPENDENTLY_RELEASED','selection_sha256':'s','runner_commit':'commit',
             'actual_optimizer_writes':0,'adapter_sha256':'a','bundle_payload_sha256':A.bundle_payload_sha256(b,job),'package_manifest_sha256':'package'}
    if release_change: release.update(release_change)
    def mock_bound(path, expected):
        key=str(path)
        if key=='launch':return launch
        if key=='release':return release
        if key=='selection':return b['selection']
        if key.endswith('/model.safetensors.index.json'):return index_change if index_change is not None else base['hf_index']
        if key=='parent':return {'identity':{'completed_step':110}}
        raise AssertionError('Unexpected fixture read '+key)
    with patch.dict(os.environ,{'TREE_PROBE_LAUNCH_PATH':'launch','TREE_PROBE_LAUNCH_SHA256':'l','SLURM_JOB_ID':job,'TREE_PROBE_BUNDLE_SHA256':'bundle','TREE_PROBE_BUNDLE_PATH':'bundle'}),\
         patch.dict(sys.modules,{'search_r1.baselines.probability_probe_contracts':fakecontracts,'omegaconf':fake}),\
         patch.object(A,'bound',side_effect=mock_bound),patch.object(A,'bound_bytes',return_value=config_text.encode() if bound_config is None else bound_config),\
         patch.object(A,'sha',side_effect=lambda path:'bundle' if str(path)=='bundle' else 'a'),\
         patch.object(A,'installed_bindings',return_value=None),patch.object(A,'imported_sources',return_value=None),patch.object(A,'check',return_value={'synthetic':True}):
        return A.prepare_and_validate_bundle(b)
ok('Full prepare source body accepts complete tiny mocked metadata interface',lambda:require(prep(copy.deepcopy(base)) is True,'Interface failed'))
for key,value in [('adapter_sha256','wrong'),('bundle_payload_sha256','wrong'),('package_manifest_sha256','wrong'),('runner_commit','wrong'),('selection_sha256','wrong'),('actual_optimizer_writes',False),('actual_optimizer_writes',1)]:
    refuses('Full prepare refuses wrong release '+key,lambda k=key,v=value:prep(copy.deepcopy(base),{k:v}))
refuses('Full prepare refuses HF index different from consumed bound JSON',lambda:prep(copy.deepcopy(base),index_change={'weight_map':{'a':'different.safetensors'}}))
refuses('Full prepare refuses config bytes different from bundled exact text',lambda:prep(copy.deepcopy(base),bound_config=b'mini: 192\n'))

actual=WS.parent/'outputs/tree-instruct-300/PPO_PROBE_ADAPTER_V2_FDINFO_CPU_ENV_RAW.json'
raw=json.loads(actual.read_bytes())
require(raw['adapter_sha256']==sha(OWNED/'resource_adapter_v2.py'),'Saved real toy used another adapter')
require(raw['runner_config_consumer_sha256']==sha_bytes(snippet.encode()),'Saved toy used another exact config consumer function')
require(raw['actual_installed_omegaconf_original_text_consumer_passed'] is True,'Saved actual parser toy failed')
require(raw['cuda_imported'] is False and raw['gpu_calls']==0 and raw['model_forwards']==0 and raw['optimizer_writes']==0,'Toy scope changed')
require((actual.parent/'PPO_PROBE_ADAPTER_V2_FDINFO_CPU_ENV.err').read_bytes()==b'','Saved actual toy stderr nonempty')
ok('Saved actual h1009 toy exact adapter/config-consumer snippet pins and CPU-only scope',lambda:None)

proof={'status':'INDEPENDENT_CONFIG_AND_METADATA_DELTA_CPU_CASES_PASSED','case_count':len(cases),'cases':cases,
       'adapter_sha256':sha(OWNED/'resource_adapter_v2.py'),'runner_sha256':sha(candidate),'config_consumer_snippet_sha256':sha_bytes(snippet.encode()),
       'actual_h1009_toy_saved_capture_sha256':sha(actual),'actual_h1009_toy_reexecuted_by_reviewer':False,
       'full_prepare_is_mocked_metadata_not_real_assets':True,'local_parser_is_stubbed':True,'actual_remote_calls':0,'actual_PT_HF_loaded':False,
       'actual_gpu_accepted':False,'actual_optimizer_writes':0,'production_source_modified':False}
out=HERE/'CONFIG_AND_METADATA_CPU_PROOF.json'
out.write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':proof['status'],'case_count':len(cases),'proof_sha256':sha(out)},sort_keys=True))
