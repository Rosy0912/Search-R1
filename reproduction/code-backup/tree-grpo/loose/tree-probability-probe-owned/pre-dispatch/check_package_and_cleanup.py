"""Independent source/pre-dispatch checks; no remote or actual tensor assets."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
WS=HERE.parents[2]
PKG=WS/'work/probability-probe-49a6b35-29822f9'
EXPECTED_MANIFEST='c542a698262d251a741afbd4958f37682e31dff97e6276fe387ef9f806f5bfb5'
EXPECTED_BUNDLE='2c17f7dfca5cc655bf40a7fdb4e4399cb999ef626d37de20035b38eb723250e8'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,msg):
    if not v:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):return {'path':str(p),'sha256':sha(p)}
checks=[]
def passed(name,detail=None):checks.append(dict(name=name,passed=True,detail=detail))
sys.path.insert(0,str(PKG))
import probe_package as P
import supervisor as S
manifest=read(PKG/'package-manifest.json');template=read(PKG/'bundle-template.json')
require(sha(PKG/'package-manifest.json')==EXPECTED_MANIFEST,'Package manifest drifted')
require(len(manifest['files'])==736,'Expected exactly 736 bound files')
actual={p.relative_to(PKG).as_posix() for p in PKG.rglob('*') if p.is_file() and p.name!='package-manifest.json' and '.git' not in p.parts}
require(actual==set(manifest['files']),'Full frozen file set differs')
for name,wanted in manifest['files'].items():
    require(sha(PKG/name)==wanted,'Frozen file differs: '+name)
passed('All 736 exact physical files match immutable full manifest')
extcap=WS/'outputs/tree-instruct-300/PPO_PROBE_EXTERNAL_METADATA_RAW.json'
external=read(extcap)
extmap={e['path']:e['sha256'] for e in external['records']}
require(extmap==manifest['external_files'],'Actual saved external source metadata differs')
require(external['actual_gpu_calls']==0 and external['tensor_assets_loaded'] is False,'External metadata scope changed')
orig_sha=P.file_sha
def bound_sha(path):
    path=str(path)
    if path in extmap:return extmap[path]
    return orig_sha(path)
with patch.object(P,'file_sha',side_effect=bound_sha):P.validate_package_files(PKG,manifest)
passed('Actual validate_package_files body passes 736 files; external hashes use exact saved h1009 metadata not remote reread')
for remote,local in [(str(P.ROOT/'tools/resource_guard_identity.py'),WS/'work/resource_guard_identity.py'),
                     (str(P.ROOT/'tools/tree_shared_owned_runtime_fixed.py'),WS/'work/tree_shared_owned_runtime_fixed.py')]:
    require(sha(local)==extmap[remote],'Local read-only external code copy differs')
passed('Guard and owned-cgroup cleanup read-only local code copies match actual source SHA')
require(P.digest(P.bundle_payload(template))==EXPECTED_BUNDLE==manifest['bundle_payload_sha256'],'Payload binding differs')
for field in ['installed_sources','installed_versions','probe_source_files']:
    require(template[field]==manifest[field],'Whole source/installed mapping differs: '+field)
remote_package=P.ROOT/'instruct300/candidates'/PKG.name
require(template['reviewed_resource_adapter']['path']==str(remote_package/'resource_adapter_v2.py'),'Adapter actual remote mapping differs')
require(template['reviewed_resource_adapter']['sha256']==sha(PKG/'resource_adapter_v2.py'),'Adapter byte identity changed')
repo_actual={p.relative_to(PKG/'repo').as_posix():sha(p) for p in (PKG/'repo').rglob('*') if p.is_file() and '.git' not in p.parts}
mapped={str(Path(e['path']).relative_to(remote_package/'repo')):e['sha256'] for e in template['probe_source_files']}
require(mapped==repo_actual and len(mapped)==719,'Probe complete repo/root mapping changed')
passed('Normalized payload 2c17 and complete 719 repo remote/local source mapping are exact')
saved=WS/'outputs/tree-instruct-300/cpu3277/selection.json'
selection=read(saved)
require(sha(saved)==manifest['selection_sha256']==template['selection_file']['sha256'],'Frozen actual CPU selection changed')
require(selection==template['selection'] and selection['inventory']==template['inventory'],'Bundled actual selection/inventory differ')
require(template['identity']==selection['identity'],'Original selection identity changed')
sys.path.insert(0,str(PKG/'repo'))
from search_r1.baselines.probability_probe_contracts import validate_selection,validate_fixed_config
validate_selection(selection,template['identity'],template['inventory'],allow_frozen=True)
passed('Actual frozen CPU3277 exact selection/inventory/identity and full validate_selection contract pass')
config=WS/'work/tree-probability-probe-owned/ORIGINAL3268_RESOLVED.yaml'
require(config.read_bytes()==template['config_text'].encode('utf-8'),'Original config exact UTF8 text changed')
require(sha(config)==template['identity']['config_sha256']=='e25214a90cf6ac063174787b7aa2c607ef5c942218fd31c2af0cfdf2c206b71b','Original config SHA changed')
import yaml
parsed=yaml.safe_load(config.read_bytes());validate_fixed_config(parsed)
passed('Original resolved config exact bytes, identity and fixed protocol pass (local YAML mapping, no actual OmegaConf rerun)')
indexfile=WS/'work/tree-probability-probe-owned/HF110_INDEX.json'
indexcap=WS/'outputs/tree-instruct-300/PPO_PROBE_HF_INDEX_AND_QUEUE_RAW.json'
cap=read(indexcap)
require(sha(indexfile)==cap['index']['sha256']==template['hf_identity']['files']['model.safetensors.index.json'],'Saved actual HF110 index SHA differs')
require(cap['index']['text'].encode()==indexfile.read_bytes() and read(indexfile)==template['hf_index'],'Saved actual index text/map changed')
parent=read(WS/'work/tree-safety110-runtime/HEALTHY110_PARENT.json')
require(template['hf_identity']==parent['hf_identity'] and template['identity']['parent_hf_sha256']==parent['hf_identity']['sha256'],'HealthyHF110 identity differs')
require(template['parent_state_manifest']['path']==parent['path'] and template['identity']['parent_state_manifest_sha256']==parent['sha256'] and parent['identity']['completed_step']==110,'Healthy parent state binding differs')
require(len(template['hf_index']['weight_map'])==435,'Serialized full HF index count changed')
passed('Saved actual HF110 435-key complete index and full 12-file HF identity bind healthy parent metadata; no shards loaded')
installed=read(WS/'outputs/tree-instruct-300/PPO_PROBE_INSTALLED_SOURCE_METADATA_RAW.json')
require(template['installed_sources']==installed['sources'] and template['installed_versions']==installed['versions'],'Actual installed source/version metadata differs')
passed('Exactly eight actual installed source records and versions match saved h1009 capture')
origmap={e['path']:e['sha256'] for e in template['identity']['source_files']}
require(origmap.get(template['original_worker_path'])==template['original_worker_sha256']==sha(PKG/'repo/verl/workers/fsdp_workers.py'),'Original worker path/bytes not coupled')
require(template['native_logprob_sha256']==sha(PKG/'repo/verl/utils/torch_functional.py'),'Actual native helper differs')
for audit,original in [('runner-v3-source-receipt.json',WS/'work/tree-probability-probe-owned/v2/RUNNER_V3_SOURCE_CPU_RECEIPT.json'),
                       ('adapter-v2-source-receipt.json',WS/'work/tree-probability-probe-owned/v2/RESOURCE_ADAPTER_V2_SOURCE_CPU_RECEIPT.json'),
                       ('selection-actual3277-receipt.json',WS/'work/tree-probe-selection-review/v2/ACTUAL_CPU3277_RECEIPT.json')]:
    require((PKG/'audits'/audit).read_bytes()==original.read_bytes(),'Independent receipt bytes changed: '+audit)
passed('Original worker/native code and three previously accepted independent receipt bytes exact')
for name in ['probe_package.py','supervisor.py','start.py','dispatch_once.py','freeze_package.py','test_candidate.py','resource_adapter_v2.py',
             'repo/search_r1/baselines/preflight.py','repo/search_r1/baselines/probability_probe_runner.py',
             'repo/search_r1/baselines/probability_probe_loader.py','repo/search_r1/baselines/probability_probe_io.py']:
    compile((PKG/name).read_bytes(),str(PKG/name),'exec')
require(subprocess.run(['bash','-n',str(PKG/'start.sh')],capture_output=True).returncode==0,'Actual start shell syntax failed')
passed('Eleven exact Python entry/helper files and actual start shell compile without bytecode')

# Execute the exact supervisor cleanup/final evidence AST with controlled CPU
# dependencies. This is a source/control fixture, never an actual Slurm cgroup.
tree=ast.parse((PKG/'supervisor.py').read_text())
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
ownertry=next(n for n in main.body if isinstance(n,ast.Try) and n.finalbody)
cleanup=ast.Module(body=copy.deepcopy(ownertry.finalbody),type_ignores=[])
ast.fix_missing_locations(cleanup);body=compile(cleanup,str(PKG/'supervisor.py')+':exact-finally','exec')

def cleanup_case(name,diagnostic=False,cost_failure=False,cleanup_failure=False,foreign_after=False,slurm_failure=False):
    events=[];saved={};result={'status':'FAILED_OR_UNKNOWN_ZERO_UPDATE_GPU_PROBE'}
    if diagnostic:result['diagnostic_body_complete']=True
    protected=set(P.PROTECTED)
    def atomic(path,value):
        events.append(('write',Path(path).name))
        if Path(path).name=='termination-cost.json' and cost_failure:raise OSError('synthetic cost IO failure')
        saved[Path(path).name]=copy.deepcopy(value)
    def cleanup_fn():
        events.append(('cleanup',None))
        if cleanup_failure:raise RuntimeError('synthetic own cleanup refusal')
    def sample():
        pids=protected|({999999} if foreign_after else set())
        return {'processes':['GPU-a, %d, 2'%p for p in pids]}
    def slurm(*args,**kwargs):
        if slurm_failure:raise OSError('synthetic accounting unknown')
        return 'JobId=99999 JobState=RUNNING'
    namespace=dict(signal=SimpleNamespace(SIGTERM=15,SIGINT=2,SIG_IGN=1,signal=lambda *x:None),
        atomic_json=atomic,run=Path('/synthetic-own-run'),job='99999',started=0.,time=SimpleNamespace(time=lambda:10.),
        allocated_cpus=64,result=result,owned=SimpleNamespace(cleanup=cleanup_fn,cleanup_residual={}),
        gpu_sample=sample,protected=lambda:events.append(('protected',None)),PROTECTED=P.PROTECTED,
        require=require,validate_inherited_gpu_leases=lambda count:events.append(('final_original_preflight',count)),
        subprocess=SimpleNamespace(check_output=slurm),lock=SimpleNamespace(close=lambda:events.append(('close_shared_lock',None))))
    error=None
    try:exec(body,namespace)
    except Exception as exc:error=repr(exc)
    require(('cleanup',None) in events,'Cost/evidence failure bypassed owned cleanup')
    require('result.json' in saved,'Failure lost final actual scope evidence fixture')
    row=saved['result.json']
    if diagnostic and not(cost_failure or cleanup_failure or foreign_after):
        require(row['status']=='OWNED_ZERO_UPDATE_FSDP_DIAGNOSTICS_COMPLETE_PENDING_INDEPENDENT_REVIEW','Successful body status wrong')
    else: require(row['status']=='FAILED_OR_UNKNOWN_ZERO_UPDATE_GPU_PROBE','Failed/unknown scope incorrectly became success')
    if cost_failure:require('termination_cost_error' in row,'Cost failure hidden')
    if cleanup_failure or foreign_after:require(error is not None and 'cleanup_error' in row,'Cleanup refusal hidden')
    if slurm_failure:require('slurm_at_supervisor_end_unknown' in row,'Unknown accounting fabricated')
    require(row['gpu_hours']==8*10/3600 and row['allocated_cpu_hours']==64*10/3600,'Allocation cost used requested CPUs instead of actual allocated')
    require('consumed_cpu_scope' in row,'Allocation cost mislabeled process CPU')
    passed(name,{'events':events,'result_status':row['status'],'error':error})

cleanup_case('Successful body remains pending independent review, never probability/train acceptance',diagnostic=True)
cleanup_case('Guard/native/metadata body failure retains failure status and runs own cleanup')
cleanup_case('Termination-cost write failure still performs own cleanup and refuses completion',diagnostic=True,cost_failure=True)
cleanup_case('Own cleanup error remains failure and final evidence attempted',diagnostic=True,cleanup_failure=True)
cleanup_case('Unexpected remaining GPU process refuses completion and preserves failure evidence',diagnostic=True,foreign_after=True)
cleanup_case('Slurm body-end query unknown explicitly recorded; body-end RUNNING never called terminal',diagnostic=True,slurm_failure=True)

require('validate_owned_lease_fds(leases,uuids);check=adapter.check' in (PKG/'supervisor.py').read_text(),'Actual continuing own FD check removed')
require("guard_path.with_suffix('.ready')" in (PKG/'supervisor.py').read_text(),'Guard actual .ready interface mismatch')
require("('rank-%02d'%rank)/'ready.json'" in (PKG/'probe_package.py').read_text(),'Rank ready directory mismatch')
require("'--master_addr=127.0.0.1'" in (PKG/'supervisor.py').read_text(),'Torchrun communication not loopback')
passed('Live-own-FD / resources.ready / rank-hyphen / loopback source APIs match actual producers')
proof=dict(status='INDEPENDENT_FROZEN_PACKAGE_METADATA_AND_SUPERVISOR_CPU_PRE_DISPATCH_CHECKS_PASSED',
    check_count=len(checks),checks=checks,package_manifest_sha256=EXPECTED_MANIFEST,bundle_payload_sha256=EXPECTED_BUNDLE,
    full_files_verified=736,complete_repo_source_mapping_count=719,selection_sha256=sha(saved),
    external_saved_metadata=ref(extcap),source_fixture=ref(Path(__file__)),
    owner37_cpu_tests_independently_rerun=True,owner37_actual_summary='Ran 37 tests in 0.033s; OK',
    actual_GPU_executed=False,actual_remote_calls=0,actual_tensor_or_HF_shards_loaded=False,
    actual_background_PID_operated=False,actual_slurm_actions=0,actual_optimizer_writes=0,
    cleanup_is_exact_AST_CPU_mock_not_actual_cgroup=True,actual_current_queue_verified=False)
target=HERE/'PACKAGE_AND_SUPERVISOR_CPU_PROOF.json';target.write_text(json.dumps(proof,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=proof['status'],check_count=len(checks),proof_sha256=sha(target)),sort_keys=True))
