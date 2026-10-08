"""Only R2 queue delta and full package bindings; no runtime assets/submission."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
WS=HERE.parents[2]
OLD=WS/'work/probability-probe-49a6b35-29822f9'
NEW=Path(str(OLD)+'-r2')
SHA='0a1a39ec5d9f86498d8e66a9f9291aefffba02f88776e2fa1aaeb200085e7dc8'
PAYLOAD='37d5debdb09beaac3346fd2912724b84413ed54cb673bb8015990c3eaa467f5f'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def require(value,message):
    if not value:raise ValueError(message)
def ref(path):return {'path':str(path),'sha256':sha(path)}
checks=[]
def ok(name):checks.append({'name':name,'passed':True})
old=read(OLD/'package-manifest.json');new=read(NEW/'package-manifest.json')
require(sha(OLD/'package-manifest.json')=='c542a698262d251a741afbd4958f37682e31dff97e6276fe387ef9f806f5bfb5','Old frozen package changed')
require(sha(NEW/'package-manifest.json')==SHA,'New manifest differs')
require(len(new['files'])==741,'Expected complete 741 file package')
actual={p.relative_to(NEW).as_posix() for p in NEW.rglob('*') if p.is_file() and p.name!='package-manifest.json' and '.git' not in p.parts}
require(actual==set(new['files']),'New full physical file set differs')
for name,value in new['files'].items():require(sha(NEW/name)==value,'New frozen file changed '+name)
ok('All 741 physical package files hash match; old c542 preserved')
changed={p for p in old['files'] if old['files'][p]!=new['files'].get(p)}
require(changed=={'bundle-template.json','dispatch_once.py','test_candidate.py'},'Unexpected common-file changes')
added=set(new['files'])-set(old['files'])
require(added=={'CPU_TEST_R2.stderr','CPU_TEST_R2.stdout','OWNER_SOURCE_CPU_PROOF_R2.json','QUEUE_R2_CHANGE.md','audits/prior-c542-refusal.json'},'Unexpected new package files')
for name in set(old['files'])-changed:require((OLD/name).read_bytes()==(NEW/name).read_bytes(),'Carried source bytes differ '+name)
ok('All other original runtime/runner/adapter/start/supervisor/protocol sources byte-identical')
prior=WS/'work/tree-probability-probe-owned/pre-dispatch/RECEIPT.json'
require((NEW/'audits/prior-c542-refusal.json').read_bytes()==prior.read_bytes(),'Prior refusal audit changed')
ok('Original refusal receipt bound exact in successor without overwriting old evidence')
oldsrc=(OLD/'dispatch_once.py').read_text();newsrc=(NEW/'dispatch_once.py').read_text()
original='                node_sets.append(set(expand(match.group(1))))'
replacement="""                expanded=expand(match.group(1))
                require(type(expanded) is list and expanded and all(type(node) is str and node.strip() and node==node.strip() for node in expanded),
                        'Unknown/empty/blank expanded node identity; refuse duplicate GPU dispatch')
                node_sets.append(set(expanded))"""
require(oldsrc.count(original)==1 and newsrc.count(replacement)==1,'Expected exactly one repaired branch')
require(newsrc==oldsrc.replace(original,replacement),'Delta exceeds exact queue repair')
ok('Dispatcher exact bytes differ only by fail-closed expanded hostname guard')
sys.path.insert(0,str(NEW));import probe_package as P
external=read(WS/'outputs/tree-instruct-300/PPO_PROBE_EXTERNAL_METADATA_RAW.json')
ext={e['path']:e['sha256'] for e in external['records']}
require(new['external_files']==old['external_files']==ext,'Bound external runtime sources changed')
original_sha=P.file_sha
with patch.object(P,'file_sha',side_effect=lambda p:ext[str(p)] if str(p) in ext else original_sha(p)):
    P.validate_package_files(NEW,new)
ok('Actual complete package validator passes new 741 bytes; externals use saved actual same-node metadata')
ot=read(OLD/'bundle-template.json');nt=read(NEW/'bundle-template.json')
oldremote=str(P.ROOT/'instruct300/candidates'/OLD.name);newremote=str(P.ROOT/'instruct300/candidates'/NEW.name)
def normalize(value):
    if type(value) is str:return value.replace(newremote,oldremote)
    if type(value) is dict:return {k:normalize(v) for k,v in value.items()}
    if type(value) is list:return [normalize(v) for v in value]
    return value
require(normalize(nt)==ot,'Unexpected bundle metadata beyond new candidate source paths')
require(P.digest(P.bundle_payload(nt))==PAYLOAD==new['bundle_payload_sha256'],'New normalized payload changed')
require(new['protocol']==old['protocol']==P.PROTOCOL,'Resources/protocol changed')
for key in ['installed_sources','installed_versions','external_files','selection_sha256','runner_commit','torchrun_port']:
    require(new[key]==old[key],'Package identity changed '+key)
ok('Bundle changes only new candidate paths; full payload37d5 and exact prior protocol retained')
repo_actual={p.relative_to(NEW/'repo').as_posix():sha(p) for p in (NEW/'repo').rglob('*') if p.is_file() and '.git' not in p.parts}
mapping={str(Path(e['path']).relative_to(Path(newremote)/'repo')):e['sha256'] for e in nt['probe_source_files']}
require(mapping==repo_actual and len(mapping)==719,'New full repo source mapping wrong')
require(nt['probe_source_files']==new['probe_source_files'],'Manifest/bundle complete mapping differs')
require(nt['reviewed_resource_adapter']['path']==newremote+'/resource_adapter_v2.py' and nt['reviewed_resource_adapter']==new['reviewed_resource_adapter'],'New adapter source path differs')
require(nt['selection']==read(WS/'outputs/tree-instruct-300/cpu3277/selection.json'),'Actual selection changed')
ok('New 719 repo remote/local mapping, adapter path and actual selection metadata exact')
owner=read(NEW/'OWNER_SOURCE_CPU_PROOF_R2.json')
for name,value in owner['files'].items():require(sha(NEW/name)==value,'Owner R2 declared file SHA changed')
require(owner['owner_tests']==39 and owner['new_regression_tests']==2 and owner['parent_refusal_receipt_sha256']==sha(prior),'Owner proof scope changed')
require('Ran 39 tests' in (NEW/'CPU_TEST_R2.stderr').read_text() and '\nOK\n' in (NEW/'CPU_TEST_R2.stderr').read_text(),'Owner actual 39 log missing')
ok('Actual owner39 R2 log and current dispatcher proof bound separately from unchanged historical37')
release=dict(status=P.RELEASE_STATUS,package_manifest_sha256=SHA,bundle_payload_sha256=PAYLOAD,
 selection_sha256=new['selection_sha256'],runner_commit=new['runner_commit'],adapter_sha256=nt['reviewed_resource_adapter']['sha256'],
 protocol=new['protocol'],dispatch_intent_path=P.DISPATCH_INTENT,actual_optimizer_writes=0,scope='Source/pre-dispatch only',actual_gpu_probe_accepted=False)
require(P.validate_release(new,SHA,nt,release) is True,'Proposed exact release schema does not match actual validator')
ok('Actual release validator accepts exact current envelope in memory; no file release or scheduler action from fixture')
for name in ['dispatch_once.py','test_candidate.py']:
    compile((NEW/name).read_bytes(),str(NEW/name),'exec')
ok('Exact changed source/test bytes compile without bytecode')
proof=dict(status='R2_EXACT_DELTA_AND_FROZEN_PACKAGE_BINDINGS_CPU_REVIEW_PASSED',checks=checks,check_count=len(checks),
 package_manifest_sha256=SHA,bundle_payload_sha256=PAYLOAD,package_file_count=741,repo_mapping_count=719,
 prior_review_receipt=ref(prior),prior_supervisor_metadata_cpu_proof=ref(WS/'work/tree-probability-probe-owned/pre-dispatch/PACKAGE_AND_SUPERVISOR_CPU_PROOF.json'),
 source_fixture=ref(Path(__file__)),historical37_and_cleanup17_repeated=False,
 owner_two_new_regressions_independently_executed=True,owner39_log_verified_not_reexecuted=True,
 actual_remote_calls=0,actual_scheduler_calls=0,actual_gpu_calls=0,actual_optimizer_writes=0,
 real_PT_HF_shards_loaded=False,actual_current_queue_verified=False,frozen_packages_modified=False)
target=HERE/'R2_PACKAGE_DELTA_CPU_PROOF.json';target.write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':proof['status'],'checks':len(checks),'sha256':sha(target)}))
