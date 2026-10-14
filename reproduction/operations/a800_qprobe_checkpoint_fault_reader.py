"""One bounded CPU-only fault read of the pinned failed micro4 run.

Reuse the published identity/source/binding and SSH route. Read only small
metadata, whole durable ledgers, bounded traceback text and /proc identities;
checkpoint tensors are only statted. No recovery, signals, retries or scoring.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
OPS = Path(__file__).resolve().parents[2]
PRIMARY = OPS/'reproduction/operations/a800_qprobe_milestone_observer.py'
PRIMARY_SHA = 'bd58a2e67f7740643fc19287a58ff50f0cb3b0b81c9d8c04bbbbd43aa5877182'
PRIMARY_COMMIT = 'ef1945b6b3ac1ed6cc5f2463edd9e639e7f8444c'
RUN = '34fffb9fe8ad4d75b54b9c9747c256fb'
EXECUTION = '6bf23d2d28c94ab095629b795d26dc30'
SOURCE = 'f7d9f1488d3c1373098b9401463812b24fe94b63'
BASE = '/data/shaoshitong/Search-G1-a800-fresh200-d85691245df24b628b309d50b7e5472a'

REMOTE_FAULT = r'''
def safe_keys(value):
 need(type(value) is dict and len(value)<=128,'metadata_keys')
 return sorted(key for key in value if type(key) is str and re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,95}',key))
def fault_trace():
 raw,info=snapshot(base/'logs/trainer.log',1024*1024,append=True)
 need(raw is not None,'trainer_log_missing')
 text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',raw.decode(errors='replace'))
 text=re.sub(r'^\((?:TaskRunner|ActorRolloutRefWorker|WorkerDict|RayWorker) pid=[0-9]+\)\s*','',text,flags=re.MULTILINE)
 frames=[dict(file=Path(m.group(1)).name[:160],line=int(m.group(2)),function=m.group(3)[:96])
  for m in re.finditer(r'^[ \t]*File "([^"\n]+)", line ([0-9]+), in ([A-Za-z0-9_<>]+)[ \t]*$',text,re.MULTILINE)][-8:]
 classes=re.findall(r'^([A-Za-z][A-Za-z0-9_]*(?:Error|Exception))(?::|$)',text,re.MULTILINE)
 keys=re.findall(r'^KeyError: [\'\"]([A-Za-z_][A-Za-z0-9_]{0,63})[\'\"]\s*$',text,re.MULTILINE)
 return dict(status='observed',full_log_scanned=True,log_sha256=info['scanned_sha256'],
  exception_type=classes[-1][:96] if classes else None,safe_key=keys[-1] if keys else None,frames=frames,
  scope='class, identifier-only KeyError key and frame locations; no message or source text')
def ledger_counts():
 result=[];signatures=[]
 for rank in range(4):
  values,info=rows('logs/optimizer-rank-'+str(rank)+'.jsonl')
  need(values is not None and info['scan_from_start'] and not info['partial_final_bytes'],'whole_optimizer_prefix')
  groups=[];signature=[];writes=0;write_rows=0;rates=set()
  for row in values:
   need(row.get('run_id')==run['run_id'] and row.get('rank')==rank,'optimizer_identity')
   if row.get('kind')=='OPTIMIZER_WRITE':
    write_rows+=1
    for rate in row.get('learning_rates',[]):need(number(rate),'write_lr');rates.add(rate)
   if row.get('kind')!='OPPORTUNITY':continue
   context=row['context'];aid=context['attempt_id'];ordinal=context['attempt_index'];outer=context['logical_update']
   need(type(aid) is str and re.fullmatch('[0-9a-f]{32}',aid) and integer(ordinal,1) and integer(outer,1),'opportunity_context')
   if not groups or groups[-1]['attempt_id']!=aid:
    need(ordinal==len(groups)+1 and aid not in {g['attempt_id'] for g in groups},'attempt_order')
    groups.append(dict(attempt_id=aid,attempt_index=ordinal,logical_update=outer,processed=0,successful_adam=0,
     skip=0,optimizer_writes_before=writes,optimizer_writes_after=writes))
   group=groups[-1];index=row['opportunity_index'];before=row['optimizer_writes_before'];after=row['optimizer_writes_after'];wrote=row['optimizer_write_succeeded']
   need(group['attempt_index']==ordinal and group['logical_update']==outer and index==group['processed']<20 and context['minibatch_index']==index and context['prior_optimizer_writes']==before and
    row['processed'] is True and type(wrote) is bool and before==writes and after-before==int(wrote),'opportunity_counters')
   need(row['status'] in (('written','written_with_post_guard') if wrote else ('skipped_guard',)),'opportunity_status')
   group['processed']+=1;group['successful_adam']+=int(wrote);group['skip']+=int(not wrote);group['optimizer_writes_after']=after;writes=after
   signature.append([aid,ordinal,outer,index,before,after,wrote,row['status'],hash_value(context['behavior_probability_gate_sha256'])])
  need(write_rows==writes and len(groups)<=200 and len(rates)<=200,'write_receipt_count')
  signatures.append(digest(canonical(signature)))
  result.append(dict(rank=rank,processed=sum(g['processed'] for g in groups),successful_adam=writes,
   skip=sum(g['skip'] for g in groups),optimizer_write_receipt_count=write_rows,consumed_learning_rates=sorted(rates),
   attempts=groups,prefix_sha256=info['complete_prefix_sha256'],row_count=info['complete_line_count']))
 agreement=len(set(signatures))==1
 return dict(status='observed' if agreement else 'unknown',four_rank_receipt_agreement=agreement,per_rank=result,
  logical_processed=result[0]['processed'] if agreement else None,logical_successful_adam=result[0]['successful_adam'] if agreement else None,
  logical_skip=result[0]['skip'] if agreement else None,inflight=None,
  scope='whole durable opportunity prefixes independent of driver-completion gate; four replicas counted once')
def driver_and_scheduler():
 output={}
 fields=('step','attempt_index','logical_update','completed_outer','complete_outer','all_ranks_agreed','real_rank_count',
  'scheduler_advances','opportunities_processed','optimizer_writes_before','optimizer_writes_after','optimizer_writes_this_attempt','actual_optimizer_writes')
 for relative,label in (('logs/actual-attempts.jsonl','attempts'),('logs/actual-updates.jsonl','updates')):
  values,info=rows(relative);need(values is not None and info['scan_from_start'],'driver_ledger')
  need(len(values)<=200,'driver_count');out=[]
  for value in values:
   need(value['run_id']==run['run_id'],'driver_run')
   if label=='attempts':same_execution(value)
   item=select(value,fields);item['attempt_id']=value['attempt_id'];need(re.fullmatch('[0-9a-f]{32}',item['attempt_id']),'driver_attempt_id')
   if label=='updates':item['reward_mean']=scalar(value.get('metrics',{}).get('critic/rewards/mean'))
   else:need(value['status'] in ('complete','rejected'),'attempt_status');item['status']=value['status']
   out.append(item)
  output[label]=dict(rows=out,prefix_sha256=info['complete_prefix_sha256'],partial_final_status=info['partial_final_status'])
 scheduler=[]
 numeric=('logical_update','attempt_index','completed_outer_before','completed_outer_after','scheduler_last_epoch_before',
  'scheduler_last_epoch_after','scheduler_step_count_before','scheduler_step_count_after','writes_before','writes_after',
  'actual_optimizer_writes','opportunities_processed','scheduler_advances','process_pid','process_start_ticks','observed_unix')
 for rank in range(4):
  values,info=rows('logs/scheduler-rank-'+str(rank)+'.jsonl');need(values is not None and info['scan_from_start'] and len(values)<=1000,'scheduler_prefix')
  out=[]
  for value in values:
   same_execution(value);need(value['rank']==rank and value['source_map_sha256']==source_map_sha,'scheduler_identity')
   need(value['kind'] in ('BEGIN','COMPLETE','REJECTED'),'scheduler_kind');item=select(value,numeric);item['kind']=value['kind']
   for name in ('consumed_learning_rates','next_learning_rates'):
    rates=value[name];need(type(rates) is list and len(rates)<=16 and all(number(x) for x in rates),'scheduler_rates');item[name]=rates
   out.append(item)
  scheduler.append(dict(rank=rank,rows=out,prefix_sha256=info['complete_prefix_sha256'],partial_final_status=info['partial_final_status']))
 output['scheduler']=scheduler;return dict(status='observed',**output)
def checkpoint_details():
 prefix='checkpoints/search-g1-common8/actor/global_step_5';directory=base/prefix
 complete,ci=document(prefix+'/complete.json');manifest,mi=document(prefix+'/training_state/manifest.json')
 need(complete is not None and manifest is not None and complete['run_id']==run['run_id'] and complete['step']==manifest['completed_outer']==5,'cp5_identity')
 def proof(value):
  path=Path(value['path']);need(path.resolve()==path and path.is_relative_to(directory/'training_state') and not path.is_symlink(),'proof_path')
  stat_value=path.stat();need(stat_value.st_uid==os.getuid() and stat.S_ISREG(stat_value.st_mode),'proof_owner')
  return dict(path=str(path.relative_to(base)),declared_bytes=scalar(value.get('bytes')),actual_bytes=stat_value.st_size,
   declared_sha256=hash_value(value['sha256']),keys=safe_keys(value))
 q=manifest['question_probe'];need(q['completed_outer']==5 and q['actor_snapshot_id']==run['run_id']+':5','cp5_qprobe_identity')
 head=q['head'];head_path=Path(head['path'])
 need(head_path.resolve()==head_path and head_path.is_relative_to(directory/'training_state') and not head_path.is_symlink(),'head_path')
 head_stat=head_path.stat();need(head_stat.st_uid==os.getuid() and stat.S_ISREG(head_stat.st_mode),'head_owner')
 head_details=dict(keys=safe_keys(head),path=str(head_path.relative_to(base)),actual_bytes=head_stat.st_size,
  checkpoint_sha=hash_value(head['checkpoint_sha']) if 'checkpoint_sha' in head else None,
  sha256=hash_value(head['sha256']) if 'sha256' in head else None,bytes=scalar(head.get('bytes')))
 descriptor=None
 if Path(head['path']).suffix=='.json':
  relative=str(Path(head['path']).relative_to(base));descriptor,di=document(relative)
  need(descriptor is not None,'head_descriptor_missing');head_details['descriptor_sha256']=di['scanned_sha256'];head_details['descriptor_keys']=safe_keys(descriptor)
 files=[]
 for path in directory.rglob('*'):
  check();need(len(files)<128,'checkpoint_file_count')
  need(path.resolve()==path and not path.is_symlink(),'checkpoint_symlink');value=path.stat()
  need(value.st_uid==os.getuid(),'checkpoint_owner')
  if stat.S_ISREG(value.st_mode):files.append(dict(path=str(path.relative_to(base)),bytes=value.st_size))
 workers=manifest['worker_files'];need(sorted(w['rank'] for w in workers)==[0,1,2,3],'cp5_worker_ranks')
 return dict(status='observed',complete_sha256=ci['scanned_sha256'],complete_keys=safe_keys(complete),
  complete_scalars=select(complete,('step','finite_all_saved_tensors','validated_tensor_count','actual_optimizer_writes_per_rank')),
  manifest_sha256=mi['scanned_sha256'],manifest_keys=safe_keys(manifest),worker_proofs=[dict(rank=w['rank'],**proof(w)) for w in workers],
  driver_proof=proof(manifest['driver_file']),question_proof=proof(q),head=head_details,files=files,
  scope='JSON producer metadata and declared hashes; tensor files only statted, never opened or hashed')
def owned_and_project_births():
 ledger,_=document('logs/process-ledger.json',8*1024*1024);need(ledger is not None and ledger['run_id']==run['run_id'],'owned_run')
 records={};jobs=[]
 for name,job in ledger['jobs'].items():
  need(re.fullmatch('[A-Za-z0-9_.-]{1,64}',name),'job_name');live=[]
  for record in list(job['ledger'].values())+[job['leader']]:
   key=(record['pid'],record['start_ticks'],record['sid'],record['uid']);records[key]=record
  jobs.append(dict(name=name,state=job['state'],exit_code=scalar(job.get('exit_code'))))
 need(len(records)<=8192,'owned_birth_count');states=[process(value) for value in records.values()];alive=[s for s in states if s['alive']]
 project=[];scanned=0
 for path in Path('/proc').iterdir():
  check()
  if not path.name.isdigit():continue
  scanned+=1;need(scanned<=65536,'proc_scan_limit')
  try:
   if path.stat().st_uid!=os.getuid():continue
   text=(path/'stat').read_text();fields=text[text.rfind(')')+2:].split()
   if fields[0] in ('Z','X'):continue
   cwd=(path/'cwd').resolve()
   namespace=next((part for part in cwd.parents if re.fullmatch('/data/shaoshitong/Search-G1-a800-fresh200-[0-9a-f]{32}',str(part))),None)
   if namespace is None and re.fullmatch('/data/shaoshitong/Search-G1-a800-fresh200-[0-9a-f]{32}',str(cwd)):namespace=cwd
   if namespace is None:continue
   with (path/'cmdline').open('rb') as stream:cmd=stream.read(8193)
   need(len(cmd)<=8192,'proc_command_limit')
   labels=[]
   if b'main_ppo' in cmd or b'TaskRunner' in cmd or b'train.sh' in cmd:labels.append('trainer_or_training_worker')
   if b'run_pipeline' in cmd:labels.append('pipeline')
   if not labels:continue
   identity=dict(pid=int(path.name),start_ticks=int(fields[19]),sid=int(fields[3]),uid=os.getuid())
   current=process(identity)
   if not current['alive']:continue
   need(len(project)<128,'project_trainers_limit');project.append(dict(**current,namespace=str(namespace),roles=labels,
    owned_exact_birth=tuple(identity[k] for k in ('pid','start_ticks','sid','uid')) in records))
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 return dict(status='observed',recorded_unique_births=len(states),recorded_alive_births=alive,
  all_recorded_births_gone=not any(s['birth_matches'] for s in states),jobs=jobs,project_trainers=project,
  scope='owned PID/start/session/UID proof and bounded current project trainer classification; command/environment text never returned')
def current_nvml():
 try:
  actual=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],
   stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
 except (OSError,subprocess.TimeoutExpired):raise Unknown('nvml_transport_unknown')
 need(actual.returncode==0 and len(actual.stdout)<=65536,'nvml_inventory');devices=[]
 for line in actual.stdout.decode().splitlines():
  fields=[v.strip() for v in line.split(',')];need(len(fields)==5 and re.fullmatch('GPU-[A-Za-z0-9-]{8,80}',fields[1]),'nvml_schema')
  devices.append(dict(index=int(fields[0]),uuid=fields[1],total_mib=int(fields[2]),used_mib=int(fields[3]),util_percent=int(fields[4])))
 need(1<=len(devices)<=64,'nvml_device_count')
 return dict(status='observed',observed_unix=time.time(),devices=devices,scope='current NVML inventory only; no contexts or allocations created')
raw_payload=sys.stdin.read(1024*1024+1);need(len(raw_payload.encode())<=1024*1024,'payload_limit')
request=json.loads(raw_payload);sys.stdin=io.StringIO(raw_payload)
result=dict(role='qprobe_checkpoint_failure_readonly',run_id=request['expected_run'],execution_id=request['expected_execution'],
 source_commit=request['intent']['source_commit'],base=request['intent']['base'],status='unknown',metadata_only=True,read_once=True,
 remote_mutation=False,no_model_or_tensor_reads=True,no_qa_or_token_files=True,no_native_verifier=True)
try:
 exec(common_code,globals());run=pinned_run()
 need(run['run_id']=='34fffb9fe8ad4d75b54b9c9747c256fb' and run['execution_id']=='6bf23d2d28c94ab095629b795d26dc30' and
  run['source_commit']=='f7d9f1488d3c1373098b9401463812b24fe94b63' and len(run['source_sha256'])==470,'fixed_fault_identity')
 source_map_sha=digest(canonical(run['source_sha256']));run_manifest_sha=digest(budget_bytes(base/'logs/run-manifest.json',1024*1024))
 result['births']=births();result['runtime_status']=section(runtime_status);result['traceback']=section(fault_trace)
 result['driver_scheduler_ledgers']=section(driver_and_scheduler);result['four_rank_optimizer_ledgers']=section(ledger_counts)
 result['checkpoint5']=section(checkpoint_details);result['owned_and_project_births']=section(owned_and_project_births)
 result['current_nvml_inventory']=section(current_nvml)
 need('torch' not in sys.modules,'unexpected_torch_import');result['status']='observed';result['metadata_files']=metadata_files
except (Unknown,AssertionError,KeyError,IndexError,TypeError,ValueError,OverflowError,OSError) as failure:
 result.update(status='unknown',reason=str(failure) if isinstance(failure,Unknown) else type(failure).__name__)
finally:
 signal.alarm(0);result.update(observed_unix=time.time(),aggregate_bytes_read=spent,
  limits=dict(aggregate_read_bytes=TOTAL_LIMIT,ledger_bytes=LEDGER_LIMIT,output_bytes=128*1024,deadline_seconds=65))
encoded=json.dumps(result,allow_nan=False)
if len(encoded.encode())>128*1024:encoded=json.dumps(dict(role=result['role'],run_id=result['run_id'],execution_id=result['execution_id'],
 source_commit=result['source_commit'],base=result['base'],status='unknown',reason='output_limit'),allow_nan=False)
print('CHECKPOINT_FAULT_JSON:'+encoded,flush=True)
'''


def load_reader():
    raw = PRIMARY.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PRIMARY_SHA
    assert subprocess.check_output(['git','-C',str(OPS),'show',PRIMARY_COMMIT+':'+str(PRIMARY.relative_to(OPS))],timeout=30) == raw
    spec = importlib.util.spec_from_file_location('published_checkpoint_fault_primary',PRIMARY)
    primary = importlib.util.module_from_spec(spec);spec.loader.exec_module(primary)
    flex,commit,_,common_sha = primary.load_flex()
    common = flex.REMOTE_COMMON
    for before,after in {'return json.loads(path.read_text())':'return json.loads(budget_bytes(path,1024**2))',
        'owned(path).read_bytes()':'budget_bytes(owned(path),4*1024*1024)',
        'binding_path.read_bytes()':'budget_bytes(binding_path,1024**2)'}.items():
        assert common.count(before) == 1
        common = common.replace(before,after)
    marker = 'raw_payload=sys.stdin.read(1024*1024+1)'
    assert primary.REMOTE_READ.count(marker) == 1
    definitions = primary.REMOTE_READ.split(marker,1)[0]
    remote = primary.REMOTE_SUPPORT+'\ncommon_code='+repr(common)+'\n'+definitions+'\n'+REMOTE_FAULT
    ast.parse(remote)
    return flex,commit,remote,common_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,required=True)
    args = parser.parse_args()
    handle = ROOT/'preparation/qprobe-a800-micro4-deployment.json';runtime = ROOT/'qprobe-a800-micro4'
    args.receipt.relative_to(ROOT/'preparation')
    assert args.receipt.is_absolute() and args.receipt.resolve()==args.receipt and not args.receipt.exists()
    flex,commit,remote,common_sha = load_reader();intent = flex.read(handle)
    assert (intent['source_commit'],intent['base']) == (SOURCE,BASE) and len(intent['source_sha256'])==470
    authority_sha = flex.authority(handle,intent,RUN,EXECUTION)
    module = flex.runtime_module(runtime,intent);route = module.ssh_command(remote)
    assert route[-1].startswith('python3 -c ')
    route[-1] = '/data/search-g1/env/bin/python -c '+route[-1][len('python3 -c '):]
    flex.authority(handle,intent,RUN,EXECUTION)
    payload = json.dumps(dict(intent=intent,expected_run=RUN,expected_execution=EXECUTION),allow_nan=False).encode()
    result = dict(role='single_qprobe_checkpoint_fault_read_receipt',run_id=RUN,execution_id=EXECUTION,source_commit=SOURCE,base=BASE,
        operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),ops_commit=commit,primary_sha256=PRIMARY_SHA,
        remote_reader_sha256=hashlib.sha256(remote.encode()).hexdigest(),reused_common_sha256=common_sha,
        authority_sha256=authority_sha,no_automatic_retry=True,observation=None)
    flex.once(args.receipt.with_suffix('.claim.json'),dict(role='single_readonly_checkpoint_fault_claim',
        run_id=RUN,execution_id=EXECUTION,source_commit=SOURCE,operator_sha256=result['operator_sha256'],
        remote_reader_sha256=result['remote_reader_sha256'],created_unix=time.time(),remote_mutation=False))
    with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        try:
            actual = subprocess.run(route,input=payload,stdout=out,stderr=err,timeout=90)
            result.update(exit_code=actual.returncode,status='READ_RETURNED')
        except (OSError,subprocess.TimeoutExpired) as failure:
            result.update(exit_code=None,status='TIMEOUT_OR_TRANSPORT_UNKNOWN',failure_type=type(failure).__name__)
        result.update(stdout_bytes=out.tell(),stderr_bytes=err.tell());out.seek(0);raw=out.read(128*1024+1)
        err.seek(0);stderr=err.read(64*1024);result['bounded_stderr_sha256']=hashlib.sha256(stderr).hexdigest()
        text=stderr.decode(errors='replace')
        result['remote_exception_frames']=[dict(file=Path(m.group(1)).name[:160],line=int(m.group(2)),function=m.group(3)[:96])
            for m in re.finditer(r'^[ \t]*File "([^"\n]+)", line ([0-9]+), in ([A-Za-z0-9_<>]+)[ \t]*$',text,re.MULTILINE)][-8:]
    if result['exit_code']==0 and len(raw)<=128*1024:
        try:
            values=[json.loads(line.split(':',1)[1]) for line in raw.decode().splitlines() if line.startswith('CHECKPOINT_FAULT_JSON:')]
            assert len(values)==1
            value=values[0]
            assert (value['run_id'],value['execution_id'],value['source_commit'],value['base'])==(RUN,EXECUTION,SOURCE,BASE)
            assert value['role']=='qprobe_checkpoint_failure_readonly'
            result['observation']=value
        except (AssertionError,KeyError,TypeError,ValueError,UnicodeError) as failure:
            result.update(status='SANITIZED_OUTPUT_REJECTED',failure_type=type(failure).__name__)
    elif len(raw)>128*1024:result['status']='OUTPUT_LIMIT_EXCEEDED'
    result['returned_unix']=time.time();flex.once(args.receipt,result)
    print(json.dumps(dict(receipt=str(args.receipt),exit_code=result['exit_code'],status=result['status'],
        observation_status=None if result['observation'] is None else result['observation']['status'],no_automatic_retry=True),allow_nan=False))
    return 0 if result['observation'] is not None and result['observation']['status']=='observed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
