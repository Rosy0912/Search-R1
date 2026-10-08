"""Tiny local CPU intent delta checks; no real dispatcher/tool side effect."""
import ast
import contextlib
import copy
import difflib
import fcntl
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
OLD=WORK/'probability-probe-49a6b35-29822f9-r2'
NEW=WORK/'tree-probability-probe-owned/supervisor-candidate-r3'
FROZEN=WORK/'probability-probe-49a6b35-29822f9-r3'
sys.path.insert(0,str(NEW))
import probe_package as contract
import dispatch_once as dispatch

checks=[]
def check(name,condition,detail=None):
    assert condition,name
    checks.append(dict(name=name,passed=True,detail=detail))

def refused(name,operation):
    try:operation()
    except (ValueError,OSError,subprocess.TimeoutExpired) as error:check(name,True,type(error).__name__+': '+str(error))
    else:raise AssertionError('Expected rejection '+name)

frozen_manifest_sha='014751600fbc6336dca1fa2d78fcef2a1dcb69730e4f6e7dcbbf17fb38035660'
frozen_payload_sha='0b9cb4f2bfab39c813e71cba0809331ca7d94b64e70ab107de031107771ce57f'
frozen_manifest=json.loads((FROZEN/'package-manifest.json').read_text())
frozen_template=json.loads((FROZEN/'bundle-template.json').read_text())
check('exact complete new package manifest bytes',contract.file_sha(FROZEN/'package-manifest.json')==frozen_manifest_sha)
check('exact complete bundle payload bytes',contract.digest(contract.bundle_payload(frozen_template))==frozen_payload_sha==frozen_manifest['bundle_payload_sha256'])
for name in ['probe_package.py','dispatch_once.py','start.py','start.sh','supervisor.py']:
    check('candidate tested bytes identical to frozen '+name,(NEW/name).read_bytes()==(FROZEN/name).read_bytes() and contract.file_sha(FROZEN/name)==frozen_manifest['files'][name])

old_source=(OLD/'probe_package.py').read_text();new_source=(NEW/'probe_package.py').read_text()
old_literal='PROBABILITY_PROBE_ONCE-3268.json';new_literal='PROBABILITY_PROBE_HEADER_REPAIR_ONCE-3280.json'
check('exact probe package delta only new literal and refusal message',old_source.replace(old_literal,new_literal).replace(
    'This probe must keep its original fixed single-attempt intent path',
    'Header repair must use its new fixed intent; old or arbitrary intent paths are refused')==new_source)
compile(new_source,str(NEW/'probe_package.py'),'exec')
(HERE/'EXACT_PACKAGE_DIFF.txt').write_text(''.join(difflib.unified_diff(old_source.splitlines(True),new_source.splitlines(True),fromfile='r2/probe_package.py',tofile='r3/probe_package.py')))
check('new source literal exact',contract.DISPATCH_INTENT==str(contract.ROOT/'instruct300'/new_literal))
for name in ['dispatch_once.py','start.py','start.sh']:
    check('unchanged release/entry/once source '+name,(OLD/name).read_bytes()==(NEW/name).read_bytes())
old_ast=ast.parse((OLD/'dispatch_once.py').read_text());new_ast=ast.parse((NEW/'dispatch_once.py').read_text())
check('all dispatch AST unchanged including UNKNOWN/lock/queue/main',ast.dump(old_ast,include_attributes=False)==ast.dump(new_ast,include_attributes=False))

# Small schema-complete metadata fixture, not a frozen package or release issue.
template=dict(output=contract.OUTPUT_TOKEN,status='GPU_PROBE_BUNDLE_INDEPENDENTLY_RELEASED',runner_commit='1'*40,
    selection_file=dict(path=str(contract.ROOT/'fixture/selection.json'),sha256=contract.SELECTION_SHA),
    reviewed_resource_adapter=dict(path=str(contract.ROOT/'fixture/adapter.py'),sha256='2'*64),
    probe_source_files=[dict(path=str(contract.ROOT/'fixture/source.py'),sha256='3'*64)],
    installed_sources=[dict(path='/fixture/site/module.py',sha256='4'*64)],installed_versions=dict(torch='2.4.0'))
manifest=dict(schema_version=1,protocol=copy.deepcopy(contract.PROTOCOL),runner_commit='1'*40,
    bundle_payload_sha256=contract.digest(contract.bundle_payload(template)),selection_sha256=contract.SELECTION_SHA,
    reviewed_resource_adapter=copy.deepcopy(template['reviewed_resource_adapter']),
    probe_source_files=copy.deepcopy(template['probe_source_files']),installed_sources=copy.deepcopy(template['installed_sources']),
    installed_versions=copy.deepcopy(template['installed_versions']),files={'bundle-template.json':'7'*64})
release=dict(status=contract.RELEASE_STATUS,protocol=copy.deepcopy(contract.PROTOCOL),package_manifest_sha256='5'*64,
    bundle_payload_sha256=manifest['bundle_payload_sha256'],selection_sha256=contract.SELECTION_SHA,runner_commit='1'*40,
    actual_optimizer_writes=0,adapter_sha256='2'*64,dispatch_intent_path=contract.DISPATCH_INTENT)
check('new exact intent accepted by real release validator',contract.validate_release(manifest,'5'*64,template,release))
frozen_release=dict(release,package_manifest_sha256=frozen_manifest_sha,bundle_payload_sha256=frozen_payload_sha,
    selection_sha256=frozen_manifest['selection_sha256'],runner_commit=frozen_manifest['runner_commit'],
    adapter_sha256=frozen_manifest['reviewed_resource_adapter']['sha256'])
check('new concrete package accepts synthetic exact envelope only',contract.validate_release(frozen_manifest,frozen_manifest_sha,frozen_template,frozen_release))
for value in [str(contract.ROOT/'instruct300'/old_literal),str(contract.ROOT/'instruct300/arbitrary.json'),
              '/tmp/'+new_literal,'instruct300/'+new_literal]:
    bad=copy.deepcopy(release);bad['dispatch_intent_path']=value
    refused('old/alternate/escaped/relative intent rejected '+value,lambda bad=bad:contract.validate_release(manifest,'5'*64,template,bad))

for field,value in [('package_manifest_sha256','9'*64),('bundle_payload_sha256','9'*64),('runner_commit','9'*40),
                    ('selection_sha256','9'*64),('adapter_sha256','9'*64),('actual_optimizer_writes',False),('status','SOURCE_ONLY')]:
    bad=copy.deepcopy(release);bad[field]=value
    refused('unchanged release identity still rejects '+field,lambda bad=bad:contract.validate_release(manifest,'5'*64,template,bad))

# Exercise actual main and actual submit_once; metadata/path/tool boundaries mocked.
with tempfile.TemporaryDirectory(dir=HERE,prefix='main-fixture-') as directory:
    root=Path(directory); intent=root/new_literal; old_intent=root/old_literal
    old_intent.write_text('preserved old failed attempt\n'); old_bytes=old_intent.read_bytes()
    ownpackage=contract.ROOT/'fixture/package'; ownrelease=contract.ROOT/'fixture/release.json'
    submissions=[];queues=[];reads=[]
    def path(value):
        result=Path(value)
        if result==Path(contract.DISPATCH_INTENT):return intent
        if result==contract.ROOT/'instruct300'/old_literal:return old_intent
        return result
    def bound(value,sha):
        reads.append(str(value))
        if Path(value).name=='package-manifest.json':return copy.deepcopy(manifest)
        if Path(value).name=='bundle-template.json':return copy.deepcopy(template)
        if Path(value)==ownrelease:return copy.deepcopy(release)
        raise AssertionError('Unexpected metadata')
    def fake_submission(argv,**kwargs):
        submissions.append(argv);check('real main only attempts sbatch argv in fake callable',argv[0]=='sbatch')
        return SimpleNamespace(returncode=0,stdout='999;local-fixture',stderr='')
    def gate(value):queues.append(str(value));return dict(path='synthetic-evidence',sha256='8'*64)
    def invoke(requested_intent):
        arguments=['dispatch_once.py','--package',str(ownpackage),'--manifest-sha256','5'*64,
            '--release',str(ownrelease),'--release-sha256','6'*64,'--intent',str(requested_intent)]
        with patch.object(sys,'argv',arguments),patch.object(dispatch.socket,'gethostname',return_value='h100-9'),\
             patch.object(dispatch,'same_node_path',side_effect=path),patch.object(dispatch,'read_bound',side_effect=bound),\
             patch.object(dispatch,'validate_package_files'),patch.object(dispatch,'refreshed_queue_gate',side_effect=gate),\
             patch.object(dispatch.subprocess,'run',side_effect=fake_submission),contextlib.redirect_stdout(io.StringIO()):
            dispatch.main()
    refused('main refuses old caller intent before any fake submission',lambda:invoke(contract.ROOT/'instruct300'/old_literal))
    check('old caller rejected before queue gate/side effect',not submissions and not queues and old_intent.read_bytes()==old_bytes)
    invoke(Path(contract.DISPATCH_INTENT))
    check('new caller maps once to dedicated local fixture intent',len(submissions)==1 and queues==[str(intent)] and intent.is_file())
    outcome=json.loads(intent.with_suffix('.json.result.json').read_text())
    check('confirmation is not GPU acceptance',outcome['status']=='SLURM_SUBMISSION_CONFIRMED_NO_EXECUTION_ACCEPTANCE' and outcome['actual_gpu_probe_accepted'] is False)
    refused('same new caller cannot submit second attempt',lambda:invoke(Path(contract.DISPATCH_INTENT)))
    check('no repeat queue/submission and old intent never changed',len(submissions)==1 and len(queues)==1 and old_intent.read_bytes()==old_bytes)

# Minimal new-intent UNKNOWN and lock negative cases; no scheduler call.
with tempfile.TemporaryDirectory(dir=HERE,prefix='unknown-fixture-') as directory:
    intent=Path(directory)/new_literal;calls=[]
    def timeout(argv):
        check('UNKNOWN attempt intent precedes fake side effect',intent.is_file())
        calls.append(argv);raise subprocess.TimeoutExpired(argv,30)
    refused('new intent timeout classified unknown',lambda:dispatch.submit_once(intent,['fake-never-executed'],{},timeout))
    refusal=json.loads(intent.with_suffix('.json.result.json').read_text())
    check('UNKNOWN receipt forbids retry',refusal['status']=='SUBMISSION_OUTCOME_UNKNOWN_DO_NOT_RETRY' and refusal['automatic_retry_allowed'] is False)
    refused('new UNKNOWN cannot retry',lambda:dispatch.submit_once(intent,['fake-never-executed'],{},timeout))
    check('UNKNOWN fake callable called once',len(calls)==1)
    locked=Path(directory)/('locked-'+new_literal)
    with locked.with_suffix('.json.lock').open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        refused('nonblocking exclusive lock rejects concurrent caller',lambda:dispatch.submit_once(locked,['fake'],{},timeout))
    check('exclusive lock refusal creates no attempt',not locked.exists() and len(calls)==1)

# Actual refreshed queue callback still precedes the dedicated new intent marker.
queue_cases=[]
for name,queue,job,hosts,expected in [
    ('empty-expanded-node','83|tree-other|PENDING|(null)|gpu:8\n','JobId=83 NodeList=(null) ReqNodeList=h100-9 ReqTRES=gres/gpu=8','',0),
    ('matched-h1009','83|tree-other|RUNNING|h100-9|gpu:8\n','JobId=83 NodeList=h100-9 ReqNodeList=h100-9 ReqTRES=gres/gpu=8','h100-9\n',0),
    ('empty-current-tree-queue','',None,None,1)]:
    with tempfile.TemporaryDirectory(dir=HERE,prefix='fresh-queue-fixture-') as directory:
        intent=Path(directory)/new_literal;tool_calls=[];submit_calls=[]
        def tool(argv,**kwargs):
            tool_calls.append(argv)
            if argv[0]=='squeue':return queue
            if argv[:3]==['scontrol','show','job']:return job
            if argv[:3]==['scontrol','show','hostnames']:return hosts
            raise AssertionError('Unexpected tool')
        def fake(argv):submit_calls.append(argv);return SimpleNamespace(returncode=0,stdout='999;fixture',stderr='')
        error=None
        with patch.object(dispatch.subprocess,'check_output',side_effect=tool):
            try:dispatch.submit_once(intent,['fake-sbatch-never-executed'],{'fixed_new_intent':contract.DISPATCH_INTENT},fake,
                pre_submit=lambda:dispatch.refreshed_queue_gate(intent))
            except ValueError as failure:error=str(failure)
        check('fresh queue follows unique R3 literal and submission count '+name,intent.name==new_literal and len(submit_calls)==expected)
        check('queue refuses before marker or publishes once '+name,(not intent.exists() and error is not None) if expected==0 else (intent.exists() and error is None))
        evidence=json.loads(next(Path(directory).glob('*.queue-*.json')).read_text())
        queue_cases.append(dict(name=name,fixed_new_intent=contract.DISPATCH_INTENT,local_marker=intent.name,
            fake_submit_calls=len(submit_calls),marker_created=intent.exists(),queue_unknown=evidence['unknown_error'],
            queue_conflicts=evidence['conflicts'],error=error,recorded_tool_argv=tool_calls))

proof=dict(status='R3_FIXED_INTENT_SOURCE_CPU_DELTA_REVIEWED_NO_FULL_PACKAGE_RELEASE',
    source_sha256={name:contract.file_sha(NEW/name) for name in ['probe_package.py','dispatch_once.py','start.py','start.sh']},
    old_probe_package_sha256=contract.file_sha(OLD/'probe_package.py'),checks=checks,blocking_findings=[],
    new_dispatch_intent=contract.DISPATCH_INTENT,old_dispatch_intent=str(contract.ROOT/'instruct300'/old_literal),
    root_main_control='release validator pins new canonical literal; main caller path must equal pinned release; submit_once operates only after this check, durable marker/lock then side effect; UNKNOWN cannot retry.',
    mock_scope='Main metadata/file identity boundaries mocked; actual validate_release and main/submit_once control execute, scheduler and queue callables intercepted. Does not validate frozen new package/env/header/GPU.',
    remote_calls=0,real_scheduler_calls=0,real_gpu_calls=0,real_pt_hf_model_reads=0,actual_gpu_probe_accepted=False,
    release_issued=False,original_once_intent_modified=False,candidate_or_r2_modified=False,
    frozen_package_manifest_sha256=frozen_manifest_sha,frozen_bundle_payload_sha256=frozen_payload_sha,
    frozen_direct_source_sha256={name:contract.file_sha(FROZEN/name) for name in ['probe_package.py','dispatch_once.py','start.py','start.sh','supervisor.py']},
    queue_cases=queue_cases,
    owner_targeted_test=dict(path=str(NEW/'test_header_environment.py'),sha256=contract.file_sha(NEW/'test_header_environment.py'),
        scope='HeaderIntentTests two source cases read; parent independently runs full14 owner tests, not claimed as child execution'))
(HERE/'INDEPENDENT_INTENT_PROOF.json').write_text(json.dumps(proof,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=proof['status'],checks=len(checks),source_sha=proof['source_sha256']['probe_package.py'])))
