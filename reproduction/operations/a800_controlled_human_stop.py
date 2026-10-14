"""Stop the exact existing A800 pipeline after explicit human scope resolution.

Uses the already published run's PID-fd ownership primitive. This sends one
SIGTERM to the verified pipeline; the pipeline retains its native cleanup path.
It never launches, restores, retries, deletes checkpoints or signals a GPU user
by device index or process name. A later read must verify actual cleanup.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
RUNTIME = ROOT / 'actor-probes-ten'
SOURCE = 'c4d156de124de7a94652d75858bcbabc8c56083c'
RUN = '6a26247f538b4145a24b156e79fb04d1'
EXECUTION = '140e43d2243d49a99d88f77399f263b8'
PID, TICKS = 2635844, 602743013

REMOTE = r'''
import hashlib,json,os,signal,socket,sys,time
from pathlib import Path
p=json.load(sys.stdin);intent=p['intent'];base=Path(intent['base']);repo=base/'repo'
assert base==base.resolve() and base.stat().st_uid==os.getuid()
assert p['human_reply']=='停止当前实验，切换新算法'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
 return h.hexdigest()
def read(name):
 f=base/name;assert f==f.resolve() and not f.is_symlink() and f.stat().st_uid==os.getuid()
 assert f.stat().st_size<=16*1024*1024
 return json.loads(f.read_text())
def once(name,value):
 path=base/name
 with path.open('x') as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
run=read('logs/run-manifest.json');launch=read('logs/deployment-launch-receipt.json')
claim=read('logs/pipeline-launch-claim.json')
assert run['run_id']==p['run_id']==claim['run_id']==launch['run_id']
assert run['execution_id']==p['execution_id']==claim['execution_id']==launch['execution_id']
assert run['source_commit']==p['source_commit']==intent['source_commit']==launch['source_commit']
assert run['execution_backend']=='ssh_local' and run['execution_host']==socket.gethostname()
assert run['training_physical_indices']==[0,1,2,3] and run['excluded_physical_indices']==[4,7]
assert run['source_sha256']==intent['source_sha256']
for name,digest in run['source_sha256'].items():assert sha(repo/name)==digest
sys.path.insert(0,str(repo))
from reproduction.a800.owned_processes import identity,same,signal_owned
producer=claim['pipeline_identity']
assert producer==run['pipeline_identity']
assert (producer['pid'],producer['start_ticks'],producer['sid'],producer['uid'])==(p['pid'],p['ticks'],p['pid'],os.getuid())
assert (launch['pid'],launch['start_ticks'],launch['sid'])==(p['pid'],p['ticks'],p['pid'])
ledger=read('logs/process-ledger.json');assert ledger['run_id']==run['run_id']
births={name:dict(leader=job['leader'],ledger=job['ledger']) for name,job in ledger['jobs'].items()}
updates=[];path=base/'logs/actual-updates.jsonl';partial_update_tail=False
if path.exists():
 assert path.stat().st_size<=16*1024*1024
 raw=path.read_bytes();partial_update_tail=bool(raw and not raw.endswith(b'\n'))
 lines=raw.splitlines()
 if partial_update_tail:lines=lines[:-1]
 for line in lines:
  row=json.loads(line)
  assert row['run_id']==run['run_id'] and row['execution_id']==run['execution_id']
  updates.append({k:row[k] for k in ('step','opportunities_processed','actual_optimizer_writes')})
checkpoints=[str(x.relative_to(base)) for x in (base/'checkpoints').rglob('global_step_*') if x.is_dir()]
snapshot=dict(role='explicit_human_current_pipeline_stop_intent',run_id=run['run_id'],execution_id=run['execution_id'],
 source_commit=run['source_commit'],human_reply=p['human_reply'],human_instruction_sha256=p['human_instruction_sha256'],
 human_reply_recorded_unix=p['human_reply_recorded_unix'],pipeline_identity=producer,owned_births=births,
 completed_update_rows=updates,partial_update_tail=partial_update_tail,checkpoint_directories=checkpoints,checkpoints_preserved=True,
 scope='exact current pipeline only; original pipeline owns component cleanup; new algorithm fresh200',checked_unix=time.time())
once('preparation/human-qprobe-switch-stop-intent.json',snapshot)
sent=False;actual=None
if same(producer):
 actual=identity(producer['pid'])
 assert 'reproduction.a800_common.run_pipeline' in actual['command'] and str(base) in actual['command']
 if actual['state'] not in ('Z','X'):signal_owned(producer,signal.SIGTERM);sent=True
receipt=dict(role='human_pipeline_stop_signal_result',run_id=run['run_id'],execution_id=run['execution_id'],
 source_commit=run['source_commit'],pipeline_identity=producer,signal_sent=sent,signal='SIGTERM' if sent else None,
 actual_state_before=actual['state'] if actual else None,checkpoints_preserved=True,
 cleaned_not_yet_verified=True,completed_update_rows=updates,checked_unix=time.time())
once('logs/human-qprobe-switch-stop-signal.json',receipt)
print('HUMAN_STOP_JSON:'+json.dumps(receipt),flush=True)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human-intent', type=Path, required=True)
    parser.add_argument('--code-ready-receipt', type=Path, required=True)
    args = parser.parse_args()
    instruction = json.loads(args.human_intent.read_text())
    assert instruction['existing_run_id'] == RUN
    assert instruction['stop_scope_confirmation_pending'] is False
    assert instruction['explicit_human_reply'] == '停止当前实验，切换新算法'
    assert instruction['stop_after_code_review_and_publication_ready'] is True
    ready = json.loads(args.code_ready_receipt.read_text())
    assert ready['role'] == 'qprobe_new_algorithm_code_ready_verified'
    assert ready['GitHub_actual_bytes_verified'] is True and ready['independent_code_review_approved'] is True
    assert ready['actual_remote_ref_sha'] == ready['source_commit']
    new_repo = ROOT/'qprobe-teacher-warmup-a800'
    assert subprocess.check_output(['git','-C',str(new_repo),'rev-parse','HEAD'],text=True).strip() == ready['source_commit']
    assert not subprocess.check_output(['git','-C',str(new_repo),'status','--porcelain']).strip()
    source_spec = importlib.util.spec_from_file_location('new_qprobe_source_files',new_repo/'reproduction/common8/source_files.py')
    source_module = importlib.util.module_from_spec(source_spec);source_spec.loader.exec_module(source_module)
    actual_map = {str(p.relative_to(new_repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in source_module.runtime_files(new_repo)}
    assert actual_map and actual_map == ready['source_sha256']
    for name, expected in ready['source_sha256'].items():
        assert hashlib.sha256((new_repo/name).read_bytes()).hexdigest() == expected
    review = Path(ready['independent_review_path'])
    assert hashlib.sha256(review.read_bytes()).hexdigest() == ready['independent_review_sha256']
    reviewed = json.loads(review.read_text())
    review_spec = importlib.util.spec_from_file_location('new_qprobe_review_contract',new_repo/'reproduction/a800_common/deploy_qprobe.py')
    review_module = importlib.util.module_from_spec(review_spec);review_spec.loader.exec_module(review_module)
    review_module.validate_review(reviewed,dict(source_commit=ready['source_commit'],source_sha256=actual_map))
    assert subprocess.check_output(['git', '-C', str(RUNTIME), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE
    assert not subprocess.check_output(['git', '-C', str(RUNTIME), 'status', '--porcelain']).strip()
    handle = ROOT/'preparation/actor-probes-ten-production-after-3ecc-preclaim-exit-deployment.json'
    intent = json.loads(handle.read_text())
    assert intent['source_commit'] == SOURCE
    payload = dict(intent=intent,run_id=RUN,execution_id=EXECUTION,source_commit=SOURCE,pid=PID,ticks=TICKS,
        human_reply=instruction['explicit_human_reply'],human_reply_recorded_unix=instruction['reply_recorded_unix'],
        human_instruction_sha256=hashlib.sha256(args.human_intent.read_bytes()).hexdigest())
    claim = ROOT/'preparation/qprobe-switch-6a262-local-stop-once.json'
    with claim.open('x') as f:json.dump(dict(payload=payload,claimed_unix=time.time(),no_automatic_retry=True),f);f.flush();os.fsync(f.fileno())
    spec = importlib.util.spec_from_file_location('immutable_a800_deploy',RUNTIME/'reproduction/a800_common/deploy.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    error = claim.with_suffix('.stderr')
    try:
        with error.open('xb') as stream:
            result = subprocess.run(module.ssh_command(REMOTE),input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=stream,timeout=60)
        rows = [json.loads(x[len('HUMAN_STOP_JSON:'):]) for x in result.stdout.decode().splitlines() if x.startswith('HUMAN_STOP_JSON:')]
        receipt = dict(exit_code=result.returncode,results=rows,observed_unix=time.time(),no_automatic_retry=True)
    except subprocess.TimeoutExpired:
        receipt = dict(exit_code=None,failure='transport_timeout_unknown_stop_state',observed_unix=time.time(),no_automatic_retry=True)
    output=claim.with_name(claim.stem+'.result.json');output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(receipt=str(output),**receipt)))


if __name__ == '__main__':
    main()
