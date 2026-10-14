"""Inspect c52b's actual preclaim failure, or explicitly consume its single retry.

Busy assigned GPUs mean wait. No polling, foreign signals, admission exceptions,
preparation/binding repeats, model loading, or deletion of prior evidence.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode=True
ROOT=Path('/Users/shaoshitong/Documents/Search-G1')
OPS=Path(__file__).resolve().parents[2]
RUNTIME=ROOT/'qprobe-teacher-warmup-a800-snapshot-repair'
HANDLE=ROOT/'preparation/qprobe-teacher-warmup-a800-snapshot-repair-deployment.json'
SOURCE='3bd1c173664c581c0eb6fd9e6a872a718bf0d5b3'
BASE='/data/shaoshitong/Search-G1-a800-fresh200-c52b44701a0544bd9e219122e53d3c77'
RUN='afa30a5dab184b32a6c89c155e550235'
EXECUTION='79fba41ac07a414aaab7f85b87d55c7f'
HELPER='reproduction/operations/a800_qprobe_repair_observer.py'
HELPER_SHA='af3bd439b53c32e194dc2141ff3c7fa3eb67023e71bda171da0bdec744fd8d80'
LIBRARY_SHA='c1512efd74268662f94b4a446df0cd6c3cdcaabd83d2ef2f65df698d35f3cd52'
LAUNCHER_SHA='143565b2e32c78035e57e78ccc32bd98f3ff6c447e94094e9e3acc649e7e4bf1'
PREFIX_SHA='c4f92b2a90544d09154317eccac44c50c81ff795b40b820c2316a429d546f642'

REMOTE=r'''
import ast,csv,hashlib,importlib.util,io,json,os,re,socket,stat,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True;os.environ['CUDA_VISIBLE_DEVICES']=''
payload=json.load(sys.stdin);qintent=payload['intent'];base=Path(qintent['base']);repo=base/'repo'
assert str(base)=='/data/shaoshitong/Search-G1-a800-fresh200-c52b44701a0544bd9e219122e53d3c77'
def owned(path):
 path=Path(path);assert path.resolve()==path and path.stat().st_uid==os.getuid()
 return path
def read(path):
 path=owned(path);assert stat.S_ISREG(path.stat().st_mode) and 0<path.stat().st_size<=1024**2
 return json.loads(path.read_bytes())
def sha(path):
 h=hashlib.sha256()
 with owned(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
 return h.hexdigest()
def birth(pid):
 result=dict(pid=pid,alive=False,birth_matches=None)
 try:
  proc=Path('/proc')/str(pid);text=(proc/'stat').read_text();parts=text[text.rfind(')')+2:].split()
  result.update(start_ticks=int(parts[19]),sid=int(parts[3]),uid=proc.stat().st_uid,state=parts[0],alive=parts[0] not in ('Z','X'))
  cwd=(proc/'cwd').resolve();namespace=next((part for part in cwd.parts if re.fullmatch('Search-G1-a800-fresh200-[0-9a-f]{32}',part)),None)
  result['cwd_namespace']=namespace
  result['cwd_basename']=None if namespace else re.sub('[^A-Za-z0-9._-]','_',cwd.name)[:96]
  result['cwd_classification']='current_prepared_namespace' if cwd==base or base in cwd.parents else 'other_project_namespace' if namespace else 'outside_project_namespace'
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return result
assert read(base/'preparation/qprobe-intent.json')==qintent
assert qintent['role']=='a800_qprobe_prospective_intent' and qintent['host']=='baai-a800'
assert qintent['source_commit']=='3bd1c173664c581c0eb6fd9e6a872a718bf0d5b3'
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True,timeout=10).strip()==qintent['source_commit']
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],timeout=10).strip()
files=[p for p in repo.rglob('*') if p.is_file() and (p.suffix in ('.py','.sh','.yaml','.yml','.json') or str(p.relative_to(repo)) in ('verl/version/version','verl/single_controller/version/version')) and not {'.git','__pycache__','evidence'}.intersection(p.relative_to(repo).parts)]
assert {str(p.relative_to(repo)):sha(p) for p in files}==qintent['source_sha256']
run=read(base/'logs/run-manifest.json')
assert (run['run_id'],run['execution_id'],run['source_commit'])==('afa30a5dab184b32a6c89c155e550235','79fba41ac07a414aaab7f85b87d55c7f',qintent['source_commit'])
assert run['source_sha256']==qintent['source_sha256'] and run['completed_updates']==0 and run['target_updates']==200
library=owned(repo/'reproduction/a800_common/deploy.py')
assert sha(library)==payload['library_sha256']==qintent['source_sha256']['reproduction/a800_common/deploy.py']
spec=importlib.util.spec_from_file_location('verified_c52b_original_launcher',library)
published=importlib.util.module_from_spec(spec);spec.loader.exec_module(published)
launcher=published.REMOTE_LAUNCH_PREPARED;assert hashlib.sha256(launcher.encode()).hexdigest()==payload['launcher_sha256']
nodes=ast.parse(launcher).body
claim=[node for node in nodes if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='claim' for t in node.targets)]
assert len(claim)==1
prefix=''.join(launcher.splitlines(keepends=True)[:claim[0].lineno-1]);assert hashlib.sha256(prefix.encode()).hexdigest()==payload['prefix_sha256']
generic_path=base/'preparation/deployment-intent.json';generic=read(generic_path)
assert generic['role']=='fresh_a800_common_deployment_intent' and generic['base']==str(base)
assert generic['source_commit']==qintent['source_commit'] and generic['source_sha256']==qintent['source_sha256']
native_payload=dict(intent=generic,preclaim_retry=True,launcher_source_commit=qintent['source_commit'],
 launcher_source_sha256=sha(library),embedded_launcher_sha256=payload['launcher_sha256'])
saved_stdin=sys.stdin;namespace={}
try:
 sys.stdin=io.StringIO(json.dumps(native_payload));exec(prefix,namespace)
finally:sys.stdin=saved_stdin
assert namespace['retry_provenance']['original_pid']==2957011 and namespace['retry_provenance']['original_start_ticks']==604602873
assert namespace['retry_provenance']['original_stdout_log_sha256']=='029bc515daedd08b82c8a7bea26a4cd154fd7afdfeb641cc98b975aa6e68ea22'
for name in ('actual-attempts.jsonl','resolved-training-config.json','policy-safety-calibration.json'):
 assert not (base/'logs'/name).exists() and not (base/'logs'/name).is_symlink()
sys.path.insert(0,str(repo))
from reproduction.a800_common.deploy_qprobe import validate_review,validate_prepared_features
review_path=base/'preparation/qprobe-independent-review.json';review=validate_review(read(review_path),qintent)
launch_claim=read(base/'logs/qprobe-launch-claim.json')
assert launch_claim['deployment_id']==qintent['deployment_id'] and launch_claim['run_id']==run['run_id'] and launch_claim['review_sha256']==sha(review_path)
ready_path=base/'preparation/features-ready.json';ready=validate_prepared_features(read(ready_path),run)
assert ready['kind']=='copied_verified_existing_encoding' and ready['cache_dir']==str(base/'question-features')
assert read(base/'logs/qprobe-feature-adoption.json')==ready
bound=read(base/'logs/qprobe-bound.json');assert bound['run_id']==run['run_id'] and bound['source_commit']==run['source_commit']
q=run['question_probe'];assert bound['question_probe']==q
assert (q['candidate_questions'],q['selected_questions'],q['responses_per_question'],q['exploration_questions'],q['bootstrap_max_groups_per_outer'])==(5120,512,5,64,5120)
assert q['cache_dir']==ready['cache_dir'] and q['teacher_warmup_cache']==qintent['teacher_warmup_cache']
teacher=Path(q['teacher_warmup_cache']);assert teacher.is_absolute() and teacher.resolve()==teacher and str(teacher).startswith('/data/')
if teacher.exists():assert owned(teacher).is_dir()
assert not q.get('initial_head_checkpoint') and not run.get('full_state_restore') and run['mixed_first_selection'] is False
assert run['initial_model']==run['reference_model']=='/data/search-g1/assets/qwen2.5-3b-instruct'
assert run['precision_profile']=='bf16_policy_fp32_master' and run['training_world_size']==4 and run.get('actor_micro_batch_per_rank',2)==2
budget=run['budget'];assert (budget['per_rank_actor_micro_batch'],budget['ppo_micro_batch_size'],budget['ppo_mini_batch_size'],budget['per_rank_log_prob_micro_batch'],budget['lr_warmup_steps'])==(2,8,128,8,57)
from reproduction.common8.stage_retrieval import ASSETS
canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',', ':'),ensure_ascii=False).encode()
assert run['retriever_version']==hashlib.sha256(canonical(ASSETS)).hexdigest()
cache=owned(base/'question-features');manifest_path=cache/'manifest.json';manifest=read(manifest_path)
assert sha(manifest_path)==ready['manifest_sha256'] and manifest['spec_sha']==ready['spec_sha']
assert hashlib.sha256(canonical(manifest['spec'])).hexdigest()==ready['spec_sha']
assert manifest['count']==169028 and manifest['feature_dim']==2048 and manifest['spec']['train_parquet_sha256']==run['input_sha256']['train.parquet']
assert set(p.name for p in cache.iterdir())=={'manifest.json','index.jsonl','features.npy'}
for name in ('index.jsonl','features.npy'):
 path=owned(cache/name);row=manifest['files'][name]
 assert stat.S_ISREG(path.stat().st_mode) and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
assert manifest['files']['features.npy']['sha256']==ready['feature_file_sha256']
import numpy as np
vectors=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
assert vectors.dtype==np.float16 and vectors.shape==(169028,2048)
assert all(np.isfinite(vectors[offset:offset+4096]).all() for offset in range(0,169028,4096));del vectors
evidence_paths=[generic_path,base/'logs/run-manifest.json',review_path,ready_path,base/'logs/qprobe-feature-adoption.json',
 base/'logs/qprobe-bound.json',base/'logs/qprobe-launch-claim.json',base/'logs/deployment-launch-receipt.json',base/'logs/deployment-pipeline.log',manifest_path]
evidence={str(path.relative_to(base)):sha(path) for path in evidence_paths}
rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20))))
physical={int(row[0]):row[1].strip() for row in rows};assigned=[0,1,2,3,5,6]
assert run['training_physical_indices']==[0,1,2,3] and run['auxiliary_physical_indices']==[5,6] and run['excluded_physical_indices']==[4,7]
assert set(assigned)<=set(physical) and len({physical[i] for i in assigned})==6
inventory=[dict(physical_index=int(row[0]),gpu_uuid=row[1].strip(),total_mib=int(row[2]),used_mib=int(row[3]),util_percent=int(row[4])) for row in rows if int(row[0]) in assigned]
apps=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=20))))
by_uuid={physical[i]:i for i in assigned};current=[]
for row in apps:
 if row and row[0].strip() in by_uuid:
  pid=int(row[1]);item=birth(pid);item.update(physical_index=by_uuid[row[0].strip()],gpu_uuid=row[0].strip(),used_memory_mib=int(row[2]) if row[2].strip().isdigit() else None);current.append(item)
idle=not current and all(row['used_mib']<=1024 and row['util_percent']==0 for row in inventory)
copy_process={key:ready['copy_process'][key] for key in ('pid','start_ticks','sid','uid')}
original=birth(2952301);original['same_birth_as_copy_process']=all(original.get(key)==copy_process[key] for key in ('pid','start_ticks','sid','uid'))
gpu3=birth(2993553);gpu3['birth_matches_last_audit']=gpu3.get('start_ticks')==604812863 and gpu3.get('sid')==2993553 and gpu3.get('uid')==0
assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
result=dict(role='qprobe_c52b_preclaim_readonly_inspection',base=str(base),run_id=run['run_id'],execution_id=run['execution_id'],source_commit=run['source_commit'],
 observed_unix=time.time(),preclaim_terminal_zero_verified=True,original_pipeline_pid=2957011,original_pipeline_start_ticks=604602873,
 original_pipeline_alive=False,original_pipeline_birth_matches=namespace['retry_provenance']['original_pipeline_birth_matches'],
 native_preclaim_validation_prefix_sha256=payload['prefix_sha256'],
 absence_checks={name:True for name in ('preclaim-retry-claim.json','deployment-retry-receipt.json','deployment-preclaim-retry.log','pipeline-launch-claim.json','training-launch-claim.json','process-ledger.json','pipeline-status.json','native-allocation-admission.json','actual-updates.jsonl','actual-attempts.jsonl','resolved-training-config.json','policy-safety-calibration.json','optimizer_rank_ledgers','scheduler_rank_ledgers','checkpoints','trainer_log')},
 generic_intent_sha256=sha(generic_path),evidence_sha256=evidence,features_count=169028,feature_dim=2048,features_sha256=ready['feature_file_sha256'],
 features_finite_cpu_verified=True,feature_copy_process=copy_process,original_occupied_gpu6_pid_observation=original,
 original_failure_pid_equals_copy_pid=copy_process['pid']==2952301,original_failure_log_does_not_bind_gpu6_process_birth=True,gpu_context_cause='not_inferred_from_pid_equality',
 original_failure_gpu6_uuid='GPU-92249a14-a344-f77f-e18c-d1cea5c638a7',
 last_audited_gpu3_pid_observation=gpu3,current_assigned_compute_apps=current,assigned_gpu_inventory=inventory,assigned_idle=idle,ready_for_one_retry=idle,
 metadata_only_output=True,no_model_or_cuda_context=True,scientific_launcher_source_commit=qintent['source_commit'],scientific_launcher_library_sha256=sha(library),embedded_launcher_sha256=payload['launcher_sha256'])
print('QPROBE_C52B_INSPECTION_JSON:'+json.dumps(result,allow_nan=False),flush=True)
if payload['mode']=='retry':
 prior=payload['prior_inspection'];assert 0<=time.time()-prior['observed_unix']<=120
 assert prior['assigned_idle'] is True and prior['ready_for_one_retry'] is True and prior['current_assigned_compute_apps']==[]
 assert prior['generic_intent_sha256']==result['generic_intent_sha256'] and prior['evidence_sha256']==evidence
 assert idle, 'Assigned GPUs are busy; wait without stopping other processes'
 assert sha(library)==payload['library_sha256'] and launcher==published.REMOTE_LAUNCH_PREPARED
 try:
  sys.stdin=io.StringIO(json.dumps(native_payload));exec(launcher,{'__name__':'verified_unchanged_c52b_preclaim_retry'})
 finally:sys.stdin=saved_stdin
else:assert payload['mode']=='inspect'
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--inspect',action='store_true');actions.add_argument('--retry',action='store_true')
    parser.add_argument('--inspect-receipt',type=Path);parser.add_argument('--receipt',type=Path)
    args=parser.parse_args();mode='retry' if args.retry else 'inspect'
    commit=subprocess.check_output(['git','-C',str(OPS),'rev-parse','HEAD'],text=True).strip()
    own=Path(__file__).resolve();relative=str(own.relative_to(OPS));own_sha=hashlib.sha256(own.read_bytes()).hexdigest()
    for name in (relative,HELPER):
        assert subprocess.check_output(['git','-C',str(OPS),'show',commit+':'+name])==(OPS/name).read_bytes(), 'Publish the exact operator before execution'
    assert hashlib.sha256((OPS/HELPER).read_bytes()).hexdigest()==HELPER_SHA
    spec=importlib.util.spec_from_file_location('verified_qprobe_observer_helpers',OPS/HELPER)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    intent=helper.read(HANDLE);assert intent['source_commit']==SOURCE and intent['base']==BASE
    helper.authority(HANDLE,intent,RUN,EXECUTION);module=helper.runtime_module(RUNTIME,intent)
    assert intent['source_sha256']['reproduction/a800_common/deploy.py']==LIBRARY_SHA
    assert hashlib.sha256(module.REMOTE_LAUNCH_PREPARED.encode()).hexdigest()==LAUNCHER_SHA
    payload=dict(intent=intent,mode=mode,library_sha256=LIBRARY_SHA,launcher_sha256=LAUNCHER_SHA,prefix_sha256=PREFIX_SHA)
    if args.retry:
        assert args.inspect_receipt is not None
        args.inspect_receipt.relative_to(ROOT/'preparation');prior=helper.read(args.inspect_receipt)
        assert prior['mode']=='inspect' and prior['exit_code']==0 and prior['status']=='READ_RETURNED' and prior['operator_sha256']==own_sha
        assert prior['handle_sha256']==hashlib.sha256(HANDLE.read_bytes()).hexdigest()
        proof=prior['inspection'];assert (proof['run_id'],proof['execution_id'],proof['source_commit'],proof['base'])==(RUN,EXECUTION,SOURCE,BASE)
        assert 0<=time.time()-proof['observed_unix']<=120 and proof['assigned_idle'] is True and proof['ready_for_one_retry'] is True and proof['current_assigned_compute_apps']==[]
        payload['prior_inspection']=proof
    else:assert args.inspect_receipt is None
    route=module.ssh_command(REMOTE);assert route[-1].startswith('python3 -c ')
    route[-1]='/data/search-g1/env/bin/python -c '+route[-1][len('python3 -c '):]
    receipt=args.receipt or ROOT/'preparation'/('qprobe-c52b-preclaim-'+mode+'-'+str(time.time_ns())+'.json')
    assert receipt.is_absolute() and receipt.resolve()==receipt and receipt.suffix=='.json' and receipt.parent.is_dir()
    receipt.relative_to(ROOT/'preparation');assert not receipt.exists()
    authority_sha=helper.authority(HANDLE,intent,RUN,EXECUTION)
    result=dict(role='qprobe_c52b_preclaim_ops_entry',mode=mode,operator_commit=commit,operator_sha256=own_sha,
        source_commit=SOURCE,base=BASE,run_id=RUN,execution_id=EXECUTION,authority_sha256=authority_sha,
        handle_sha256=hashlib.sha256(HANDLE.read_bytes()).hexdigest(),scientific_library_sha256=LIBRARY_SHA,
        embedded_launcher_sha256=LAUNCHER_SHA,no_automatic_retry=True)
    if args.retry:
        helper.once(ROOT/'preparation/qprobe-c52b-preclaim-explicit-retry-once.json',dict(result,inspect_receipt=str(args.inspect_receipt),claimed_unix=time.time()))
    with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        try:
            actual=subprocess.run(route,input=json.dumps(payload,allow_nan=False).encode(),stdout=out,stderr=err,timeout=180)
            result.update(exit_code=actual.returncode,status='READ_RETURNED' if mode=='inspect' else 'RETRY_COMMAND_RETURNED')
        except (OSError,subprocess.TimeoutExpired) as failure:result.update(exit_code=None,status='TRANSPORT_OR_TIMEOUT_UNKNOWN',failure_type=type(failure).__name__)
        result.update(stdout_bytes=out.tell(),stderr_bytes=err.tell());out.seek(0);raw=out.read(128*1024+1)
        err.seek(0);result['bounded_stderr_sha256']=hashlib.sha256(err.read(64*1024)).hexdigest()
    result['inspection']=None;result['retry_launch_receipt']=None
    if len(raw)<=128*1024:
        try:
            for line in raw.decode(errors='replace').splitlines():
                if line.startswith(('QPROBE_C52B_INSPECTION_JSON:','DEPLOYMENT_JSON:')):
                    value=json.loads(line.split(':',1)[1]);assert (value['run_id'],value['execution_id'],value['source_commit'])==(RUN,EXECUTION,SOURCE)
                    result['inspection' if line.startswith('QPROBE_') else 'retry_launch_receipt']=value
        except (ValueError,KeyError,TypeError,AssertionError) as failure:
            result.update(status='OUTPUT_REJECTED_STATE_UNKNOWN',failure_type=type(failure).__name__)
            result['inspection']=None;result['retry_launch_receipt']=None
    else:result['status']='OUTPUT_LIMIT_REJECTED'
    result['checked_unix']=time.time();helper.once(receipt,result)
    print(json.dumps(dict(receipt=str(receipt),status=result['status'],exit_code=result['exit_code'],inspection=result['inspection'],retry_launch_receipt=result['retry_launch_receipt']),allow_nan=False))
    return 0 if result['exit_code']==0 and (result['inspection'] if mode=='inspect' else result['retry_launch_receipt']) else 1


if __name__=='__main__':
    raise SystemExit(main())
