"""Read resources, or explicitly continue one proven prebind-no-execute launch.

The original failed launch claim and receipts remain immutable. Continuation
has a new exclusive claim and intent, then enters the exact published launch
gate once. Busy GPUs are refused; no polling, signals, runtime edits or retries.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

sys.dont_write_bytecode=True
ROOT=Path('/Users/shaoshitong/Documents/Search-G1')
OPS=Path(__file__).resolve().parents[2]
RUNTIME=ROOT/'qprobe-a800-micro4'
HANDLE=ROOT/'preparation/qprobe-a800-micro4-deployment.json'
SOURCE='f7d9f1488d3c1373098b9401463812b24fe94b63'
RUN='34fffb9fe8ad4d75b54b9c9747c256fb'
EXECUTION='6bf23d2d28c94ab095629b795d26dc30'
BASE='/data/shaoshitong/Search-G1-a800-fresh200-d85691245df24b628b309d50b7e5472a'
HELPER='reproduction/operations/a800_qprobe_micro4_switch_stop.py'
HELPER_SHA='7c9219979be8ecb42d027fc258157b6e4fe38f575a5e2bee4d300238de1d49eb'
FAILED=ROOT/'preparation/qprobe-micro4-launch-command.json'
FAILED_SHA='80347e4f9f1d447c72ff3b4590565f0589bad007872c9ac5d6f8ad5b832daa2a'
FAILED_STDERR_SHA='9f03c76ffb4b0f3a46c804aa3cff335b1dadbbe3f4c370e6741b064c3cbb9dde'
ORIGINAL_CLAIM=ROOT/'preparation/qprobe-a800-micro4-deployment.launch-once.json'
ORIGINAL_CLAIM_SHA='b8f31a5682a2ecba557fbbd700ed983cdcd6c711a3d1a08414a5968cc552f973'
FAILED_INSPECT=ROOT/'preparation/qprobe-micro4-failed-launch-inspect-command.json'
FAILED_INSPECT_SHA='ef6e336a404ffa15190550d65582751561daf181ea56898a94cc6cac6e955445'

REMOTE_PREFIX=r'''
import hashlib,importlib.util,io,json,os,re,socket,stat,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True;os.environ['CUDA_VISIBLE_DEVICES']=''
payload=json.load(sys.stdin);intent=payload['intent'];base=Path(intent['base']);repo=base/'repo'
'''

REMOTE_BODY=r'''
assert sys.executable=='/data/search-g1/env/bin/python' and intent['host']=='baai-a800'
assert str(base)==payload['base'] and intent['source_commit']==payload['source_commit']
assert document(base/'preparation/qprobe-intent.json')==intent
run=document(base/'logs/run-manifest.json')
assert (run['run_id'],run['execution_id'],run['source_commit'])==(payload['run_id'],payload['execution_id'],payload['source_commit'])
assert run['source_sha256']==intent['source_sha256'] and run['execution_host']==socket.gethostname() and run['execution_backend']=='ssh_local'
assert run['slurm_job_id'] is None and run['training_world_size']==4 and run['completed_updates']==0 and run['target_updates']==200
assert run['gpu_placement_bound'] is False and run['gpu_placement_policy']=='launch_time_four_idle_excluding_auxiliary_v1'
assert run['actor_micro_batch_per_rank']==4 and (run['budget']['per_rank_actor_micro_batch'],run['budget']['ppo_micro_batch_size'],run['budget']['ppo_mini_batch_size'])==(4,16,128)
assert (128//4)//4==8 and run['initial_probe_version'] is None and not run.get('full_state_restore')
assert run['auxiliary_physical_indices']==run['retriever_physical_indices']==[5,6] and run['probe_physical_index']==6
verify_source(repo,intent['source_commit'],intent['source_sha256'],470)
source=Path(intent['source_base']);assert document(source/'preparation/qprobe-intent.json')==intent
verify_source(source/'repo',intent['source_commit'],intent['source_sha256'],470)
prepared=document(base/'logs/qprobe-prepared.json');bound=document(base/'logs/qprobe-bound.json')
assert prepared['run_id']==bound['run_id']==run['run_id'] and prepared['execution_id']==run['execution_id']
assert prepared['source_commit']==bound['source_commit']==run['source_commit'] and prepared['trainer_started'] is False and bound['trainer_started'] is False
q=run['question_probe'];assert bound['question_probe']==q
assert (q['candidate_questions'],q['selected_questions'],q['responses_per_question'],q['exploration_questions'],q['bootstrap_max_groups_per_outer'])==(5120,512,5,64,5120)
absence=('logs/local-gpu-placement-binding.json','preparation/qprobe-independent-review.json','logs/qprobe-launch-claim.json',
 'preparation/micro4-known-prebind-noexecute-continuation-intent.json',
 'logs/prepared-launch-claim.json','logs/preclaim-retry-claim.json','logs/deployment-launch-receipt.json','logs/deployment-retry-receipt.json',
 'logs/native-allocation-admission.json','logs/native-allocation-observation.json','logs/pipeline-launch-claim.json','logs/training-launch-claim.json','logs/process-ledger.json',
 'logs/pipeline-status.json','logs/deployment-pipeline.log','logs/actual-updates.jsonl','logs/actual-attempts.jsonl',
 'logs/resolved-training-config.json','logs/policy-safety-calibration.json','logs/search-g1-common8-training.log')
for relative in absence:assert not (base/relative).exists() and not (base/relative).is_symlink(), 'Execution artifact already exists'
assert not any((base/'logs').glob('*rank*.jsonl')) and not any((base/'logs').glob('gpu-samples-*.jsonl'))
assert not any(path.is_file() or path.is_symlink() for path in (base/'checkpoints').rglob('*'))
assert not (base/'artifacts/live').exists() and not (base/'logs/initial-probe-adoption-claim.json').exists()
assert not any(path.is_file() or path.is_symlink() for path in (base/'artifacts/question-probe').rglob('*'))
for proc in Path('/proc').iterdir():
 if not proc.name.isdecimal():continue
 try:
  cwd=(proc/'cwd').resolve()
  assert not (proc.stat().st_uid==os.getuid() and (cwd==base or base in cwd.parents)), 'Existing live process in target namespace'
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
sys.path.insert(0,str(repo))
from reproduction.a800_common.deploy_qprobe import validate_review,validate_prepared_features
review=validate_review(payload['review'],intent)
ready=validate_prepared_features(document(base/'preparation/features-ready.json'),run)
assert ready['kind']=='copied_verified_existing_encoding' and document(base/'logs/qprobe-feature-adoption.json')==ready
assert ready['manifest_sha256']==hashlib.sha256(sealed(base/'question-features/manifest.json')).hexdigest()
from reproduction.common8 import native_allocation as allocation
rows,applications=allocation._query_local_gpu_state()
physical=allocation._local_inventory(rows);occupied=allocation._local_application_indices(physical,applications)
idle={index for index,row in physical.items() if index not in occupied and row['used_mib']<=allocation.LOCAL_IDLE_MAX_USED_MIB and row['util_percent']==0}
def birth(pid):
 item=dict(pid=pid,alive=False)
 try:
  proc=Path('/proc')/str(pid);text=(proc/'stat').read_text();fields=text[text.rfind(')')+2:].split()
  item.update(start_ticks=int(fields[19]),sid=int(fields[3]),uid=proc.stat().st_uid,state=fields[0],alive=fields[0] not in ('Z','X'))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return item
by_uuid={row['uuid']:index for index,row in physical.items()}
apps=[dict(gpu_uuid=uuid,physical_index=by_uuid[uuid],process=birth(int(pid))) for uuid,pid in applications]
training=excluded=None;selection_error=None
try:training,excluded=allocation.select_idle_local_placement(rows,applications)
except AssertionError:selection_error='AssertionError'
result=dict(role='micro4_prebind_noexecute_actual_resources',run_id=run['run_id'],execution_id=run['execution_id'],source_commit=run['source_commit'],base=str(base),
 host=socket.gethostname(),observed_unix=time.time(),no_execution_artifacts=True,gpu_placement_bound=False,
 physical_gpu_inventory=[dict(physical_index=index,gpu_uuid=row['uuid'],total_mib=row['total_mib'],used_mib=row['used_mib'],util_percent=row['util_percent']) for index,row in sorted(physical.items())],
 compute_applications=apps,idle_training_candidates=sorted(idle-{5,6}),blocked_auxiliary_indices=sorted({5,6}-idle),
 selected_idle_training_indices=training,prospective_excluded_indices=excluded,ready_for_one_continuation=(training is not None),selection_failure_class=selection_error,
 source_map_sha256=payload['source_map_sha256'],failed_launch_sha256=payload['failed_launch_sha256'],metadata_only=True,no_model_or_cuda_context=True,no_remote_mutation=True)
assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
print('MICRO4_RESOURCES_JSON:'+json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
if payload['mode']=='continue-once':
 prior=payload.get('prior_resources')
 if prior is not None:
  assert 0<=time.time()-prior['observed_unix']<=120 and prior['ready_for_one_continuation'] is True
  assert (prior['run_id'],prior['execution_id'],prior['source_commit'],prior['base'])==(run['run_id'],run['execution_id'],run['source_commit'],str(base))
  assert prior['source_map_sha256']==result['source_map_sha256'] and prior['failed_launch_sha256']==result['failed_launch_sha256']
 assert result['ready_for_one_continuation'] is True, 'Published selector refuses current occupied resources'
 from reproduction.a800_common import deploy_qprobe as published
 assert hashlib.sha256(sealed(Path(published.__file__))).hexdigest()==intent['source_sha256']['reproduction/a800_common/deploy_qprobe.py']
 gate=published.REMOTE_LAUNCH_GATE;assert hashlib.sha256(gate.encode()).hexdigest()==payload['launch_gate_sha256']
 assert payload['launch_code']==published.REMOTE_LAUNCH_PREPARED and hashlib.sha256(payload['launch_code'].encode()).hexdigest()==payload['embedded_launcher_sha256']
 name=base/'preparation/micro4-known-prebind-noexecute-continuation-intent.json'
 record=dict(role='explicit_known_prebind_noexecute_continuation_intent',run_id=run['run_id'],execution_id=run['execution_id'],source_commit=run['source_commit'],
  failed_launch_sha256=payload['failed_launch_sha256'],failed_stderr_sha256=payload['failed_stderr_sha256'],original_local_claim_sha256=payload['original_local_claim_sha256'],
  local_continuation_claim_sha256=payload['local_continuation_claim_sha256'],prior_resource_receipt_sha256=payload['prior_resource_receipt_sha256'],
  launch_gate_sha256=payload['launch_gate_sha256'],observed_idle_selection=result['selected_idle_training_indices'],gpu_binding_pending=True,no_automatic_retry=True,created_unix=time.time())
 raw=(json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode();fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 fd=os.open(name.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
 sys.stdin=io.StringIO(json.dumps(payload,sort_keys=True,allow_nan=False));exec(gate,{'__name__':'exact_published_micro4_known_noexecute_continuation'})
else:assert payload['mode']=='observe'
'''


def authority(helper):
    handoff=helper.document(helper.AUTHORITY);current=handoff['current_authoritative']
    assert handoff['goal_active'] is True
    assert tuple(current[k] for k in ('run_id','execution_id','source_commit','base','handle_path'))==(RUN,EXECUTION,SOURCE,BASE,str(HANDLE))
    correction=helper.document(helper.HUMAN)['latest_micro_batch_correction']
    assert tuple(correction[k] for k in ('actor_micro_batch_per_rank','actor_micro_batch_global','gradient_accumulation','ppo_mini_batch','world_size'))==(4,16,8,128,4)
    return helper.sha(helper.AUTHORITY),helper.sha(helper.HUMAN)


def known_failure(helper):
    assert helper.sha(FAILED)==FAILED_SHA and helper.sha(ORIGINAL_CLAIM)==ORIGINAL_CLAIM_SHA and helper.sha(FAILED_INSPECT)==FAILED_INSPECT_SHA
    failed=helper.document(FAILED);claim=helper.document(ORIGINAL_CLAIM)
    assert failed['action']=='launch' and failed['deployment_id']=='d85691245df24b628b309d50b7e5472a' and failed['exit_code']==1 and failed['failure_type'] is None and failed['observations']==[]
    assert claim['deployment_id']==failed['deployment_id'] and claim['no_automatic_retry'] is True
    original=helper.document(Path(failed['receipt']));assert all(original[k]==failed[k] for k in ('action','deployment_id','exit_code','failure_type','observations','transport_stderr','observed_unix'))
    error=Path(failed['transport_stderr']);assert helper.sha(error)==FAILED_STDERR_SHA
    text=helper.sealed(error,64*1024).decode()
    assert re.search(r'^AssertionError: Authorized local GPUs are occupied: \[6\]$',text,re.MULTILINE)
    frames=re.findall(r'File "([^"\n]+)", line ([0-9]+), in ([A-Za-z0-9_<>]+)',text)
    source_repo='/data/shaoshitong/Search-G1-qprobe-source-d85691245df24b628b309d50b7e5472a/repo'
    assert frames==[('<string>','120','<module>'),(source_repo+'/reproduction/common8/native_allocation.py','355','bind_idle_local_placement'),(source_repo+'/reproduction/common8/native_allocation.py','235','select_idle_local_placement')]
    inspection=helper.document(FAILED_INSPECT);assert inspection['exit_code']==0 and inspection['action']=='inspect'
    unique={json.dumps(row,sort_keys=True,allow_nan=False):row for row in inspection['observations']};assert len(unique)==1
    row=next(iter(unique.values()));assert row['base']==BASE and row['pipeline_claimed'] is False and row['trainer_claimed'] is False
    assert row['actual_updates_ledger']['exists'] is False and row['opportunity_ledgers']==[] and row['training_log']['exists'] is False and row['gpu_samples']['exists'] is False
    assert tuple(row['run_manifest'][k] for k in ('run_id','execution_id','source_commit'))==(RUN,EXECUTION,SOURCE)
    return dict(failed_launch=FAILED_SHA,failed_stderr=FAILED_STDERR_SHA,original_local_claim=ORIGINAL_CLAIM_SHA,failed_inspect=FAILED_INSPECT_SHA)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--observe',action='store_true');mode.add_argument('--continue-once',action='store_true')
    parser.add_argument('--resource-receipt',type=Path);parser.add_argument('--receipt',type=Path);args=parser.parse_args()
    assert args.continue_once or args.resource_receipt is None
    commit=subprocess.check_output(['git','-C',str(OPS),'rev-parse','HEAD'],text=True,timeout=10).strip()
    own=Path(__file__).resolve();relative=str(own.relative_to(OPS))
    for name in (relative,HELPER):assert subprocess.check_output(['git','-C',str(OPS),'show',commit+':'+name],timeout=10)==(OPS/name).read_bytes(), 'Publish these exact operators first'
    assert hashlib.sha256((OPS/HELPER).read_bytes()).hexdigest()==HELPER_SHA
    spec=importlib.util.spec_from_file_location('published_micro4_cpu_evidence_helpers',OPS/HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    auth=authority(helper);failure=known_failure(helper)
    deployment=helper.document(ROOT/'preparation/qprobe-micro4-actual-prelaunch-review.json')
    def receipt_for(action):
        paths=[Path(row['path']) for row in deployment['evidence_files'].values() if re.fullmatch(r'qprobe-a800-micro4-deployment\.'+action+r'-[0-9]+\.json',Path(row['path']).name)]
        assert len(paths)==1;return paths[0]
    intent,newrun,newexecution,evidence=helper.prepared_micro4(SimpleNamespace(new_handle=HANDLE,code_review=ROOT/'preparation/qprobe-micro4-independent-source-review.json',
        source_publication=ROOT/'preparation/qprobe-micro4-source-publication.json',deployment_review=ROOT/'preparation/qprobe-micro4-actual-prelaunch-review.json',prepare_receipt=receipt_for('prepare'),bind_receipt=receipt_for('bind')))
    assert (newrun,newexecution,intent['source_commit'],intent['base'])==(RUN,EXECUTION,SOURCE,BASE)
    feature=helper.document(receipt_for('feature-adopt'));assert feature['exit_code']==0 and feature['action']=='feature-adopt'
    assert feature['observations'][0]['run_id']==RUN and feature['observations'][0]['count']==169028 and feature['observations'][0]['fresh_gpu_encoding'] is False
    assert helper.sha(receipt_for('feature-adopt'))==next(row['sha256'] for row in deployment['evidence_files'].values() if row['path']==str(receipt_for('feature-adopt')))
    sys.path.insert(0,str(RUNTIME))
    spec=importlib.util.spec_from_file_location('verified_micro4_noexecute_published_gate',RUNTIME/'reproduction/a800_common/deploy_qprobe.py');published=importlib.util.module_from_spec(spec);spec.loader.exec_module(published)
    source,launcher,launch_code=published.verified_launcher(RUNTIME,intent);assert source==SOURCE
    payload=dict(intent=intent,review=published.validate_review(deployment,intent),mode='continue-once' if args.continue_once else 'observe',run_id=RUN,execution_id=EXECUTION,source_commit=SOURCE,base=BASE,
        expected_host='baai-a800',source_map_sha256=helper.canonical_sha(intent['source_sha256']),failed_launch_sha256=FAILED_SHA,failed_stderr_sha256=FAILED_STDERR_SHA,original_local_claim_sha256=ORIGINAL_CLAIM_SHA,
        launcher_source_commit=SOURCE,launcher_source_sha256=hashlib.sha256(launcher).hexdigest(),launch_code=launch_code,embedded_launcher_sha256=hashlib.sha256(launch_code.encode()).hexdigest(),
        launch_gate_sha256=hashlib.sha256(published.REMOTE_LAUNCH_GATE.encode()).hexdigest())
    remote=REMOTE_PREFIX+helper.SOURCE_CHECK+REMOTE_BODY;ast.parse(remote)
    result=dict(role='micro4_known_prebind_noexecute_operator',mode=payload['mode'],run_id=RUN,execution_id=EXECUTION,source_commit=SOURCE,base=BASE,operator_commit=commit,operator_sha256=helper.sha(own),
        authority_sha256=auth[0],human_intent_sha256=auth[1],handle_sha256=helper.sha(HANDLE),failed_launch_evidence_sha256=failure,reviewed_prepared_evidence_sha256=evidence,
        remote_code_sha256=hashlib.sha256(remote.encode()).hexdigest(),launch_gate_sha256=payload['launch_gate_sha256'],no_automatic_retry=True)
    if args.continue_once:
        proof=None;prior_sha=None
        if args.resource_receipt is not None:
            prior=helper.document(args.resource_receipt);proof=prior['resources'];prior_sha=helper.sha(args.resource_receipt)
            assert prior['mode']=='observe' and prior['exit_code']==0 and prior['operator_sha256']==result['operator_sha256'] and prior['handle_sha256']==result['handle_sha256']
            assert proof['ready_for_one_continuation'] is True and 0<=time.time()-proof['observed_unix']<=120
            assert tuple(proof[k] for k in ('run_id','execution_id','source_commit','base'))==(RUN,EXECUTION,SOURCE,BASE)
        claim=ROOT/'preparation/qprobe-micro4-known-prebind-noexecute-continuation-once.json'
        helper.once(claim,dict(result,resource_receipt_sha256=prior_sha,claimed_unix=time.time()))
        payload.update(prior_resources=proof,prior_resource_receipt_sha256=prior_sha,local_continuation_claim_sha256=helper.sha(claim))
    receipt=args.receipt or ROOT/'preparation'/('qprobe-micro4-noexec-'+payload['mode']+'-'+str(time.time_ns())+'.json');receipt.relative_to(ROOT/'preparation');assert not receipt.exists()
    assert authority(helper)==auth and known_failure(helper)==failure
    route=published.ssh_command(remote);assert route[-1].startswith('python3 -c ')
    route[-1]=shlex.quote('/data/search-g1/env/bin/python')+' -c '+shlex.quote(remote)
    with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        try:
            completed=subprocess.run(route,input=json.dumps(payload,sort_keys=True,allow_nan=False).encode(),stdout=out,stderr=err,timeout=180 if args.continue_once else 60)
            result.update(exit_code=completed.returncode,status='CONTINUATION_RETURNED' if args.continue_once else 'READ_RETURNED')
        except (OSError,subprocess.TimeoutExpired) as error:result.update(exit_code=None,status='TRANSPORT_UNKNOWN_NO_RETRY',failure_type=type(error).__name__)
        result.update(stdout_bytes=out.tell(),stderr_bytes=err.tell());out.seek(0);raw=out.read(128*1024+1);err.seek(0);error=err.read(64*1024)
    result['bounded_stderr_sha256']=hashlib.sha256(error).hexdigest();diagnostic=error.decode(errors='replace')
    result['remote_exception_frames']=[dict(file=Path(m.group(1)).name[:160],line=int(m.group(2)),function=m.group(3)[:96]) for m in re.finditer(r'^[ \t]*File "([^"\n]+)", line ([0-9]+), in ([A-Za-z0-9_<>]+)[ \t]*$',diagnostic,re.MULTILINE)][-8:]
    types=re.findall(r'^([A-Za-z][A-Za-z0-9_]*(?:Error|Exception))(?::|$)',diagnostic,re.MULTILINE);result['remote_exception_type']=types[-1][:96] if types else None
    result.update(resources=None,launch_receipt=None)
    try:
        assert len(raw)<=128*1024
        for line in raw.decode().splitlines():
            if not line.startswith(('MICRO4_RESOURCES_JSON:','DEPLOYMENT_JSON:')):continue
            value=json.loads(line.split(':',1)[1]);assert tuple(value[k] for k in ('run_id','execution_id','source_commit'))==(RUN,EXECUTION,SOURCE)
            key='resources' if line.startswith('MICRO4_') else 'launch_receipt';assert result[key] is None;result[key]=value
    except (AssertionError,ValueError,UnicodeDecodeError,KeyError,TypeError) as error:result.update(status='OUTPUT_REJECTED_STATE_UNKNOWN',failure_type=type(error).__name__,resources=None,launch_receipt=None)
    result['observed_unix']=time.time();helper.once(receipt,result)
    print(json.dumps(dict(receipt=str(receipt),status=result['status'],exit_code=result['exit_code'],resources=result['resources'],launch_receipt=result['launch_receipt']),allow_nan=False))
    return 0 if result['exit_code']==0 and (result['launch_receipt'] if args.continue_once else result['resources']) else 1


if __name__=='__main__':raise SystemExit(main())
