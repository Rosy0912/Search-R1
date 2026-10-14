"""One read of existing scalar ledgers; no training, evaluation, or repair."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REMOTE = r'''
import hashlib,json,os,stat,sys,time
from pathlib import Path
p=json.load(sys.stdin);base=Path(p['base'])
assert str(base)=='/data/shaoshitong/Search-G1-a800-fresh200-'+p['deployment_id']
def read(relative):
    path=base/relative
    assert path.parent==base/'logs'
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        st=os.fstat(fd);assert stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid() and st.st_size<=2*1024*1024
        with os.fdopen(fd,'rb',closefd=False) as f:raw=f.read(2*1024*1024+1)
        assert len(raw)<=2*1024*1024
    finally:os.close(fd)
    return raw,dict(relative=relative,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
raw,_=read('logs/run-manifest.json');run=json.loads(raw)
assert all(run[k]==p[k] for k in ('run_id','execution_id','source_commit'))
assert run['training_world_size']==4
def rows(relative):
    raw,meta=read(relative);assert raw.endswith(b'\n')
    values=[json.loads(line) for line in raw.splitlines()]
    assert all(v['run_id']==run['run_id'] for v in values)
    return values,meta
updates,umeta=rows('logs/actual-updates.jsonl')
first=[v for v in updates if v['step']==1];assert len(first)==1
update=first[0]
attempts,ameta=rows('logs/actual-attempts.jsonl')
attempt=[v for v in attempts if v['logical_update']==1];assert len(attempt)==1
attempt=attempt[0]
assert attempt['execution_id']==run['execution_id'] and attempt['status']=='complete'
assert attempt['all_ranks_agreed'] is True and attempt['real_rank_count']==4
assert attempt['complete_outer'] is True and attempt['completed_outer']==1 and attempt['scheduler_advances']==1
assert attempt['attempt_id']==update['attempt_id'] and attempt['opportunities_processed']==20
assert 1<=attempt['optimizer_writes_this_attempt']<=20
out=dict(role='existing_first_outer_scalar_ledgers_readonly',checked_unix=time.time(),**{k:run[k] for k in ('run_id','execution_id','source_commit')},
         complete_outer=1,actual_update=update,actual_update_file=umeta,attempt=attempt,attempt_file=ameta,schedulers=[])
keys=('kind','rank','world_size','run_id','execution_id','process_pid','process_start_ticks','logical_update','attempt_id',
      'writes_before','writes_after','actual_optimizer_writes','completed_outer_before','completed_outer_after',
      'consumed_learning_rates','next_learning_rates','scheduler_last_epoch_before','scheduler_last_epoch_after',
      'scheduler_step_count_before','scheduler_step_count_after','scheduler_advances','opportunities_processed','warmup_steps','base_learning_rate')
for rank in range(4):
    values,meta=rows('logs/scheduler-rank-%d.jsonl'%rank)
    values=[v for v in values if v.get('logical_update')==1]
    assert [v['kind'] for v in values]==['BEGIN','COMMIT_PREPARED','COMPLETE']
    assert all(v['rank']==rank and v['execution_id']==run['execution_id'] and v['attempt_id']==attempt['attempt_id'] for v in values)
    final=values[-1]
    assert final['completed_outer_after']==1 and final['scheduler_advances']==1 and final['opportunities_processed']==20
    assert final['writes_after']-final['writes_before']==attempt['optimizer_writes_this_attempt']
    assert final['warmup_steps']==57 and final['base_learning_rate']==1e-6
    assert final['consumed_learning_rates']==[0.0]
    assert final['scheduler_last_epoch_after']-final['scheduler_last_epoch_before']==1
    assert final['scheduler_step_count_after']-final['scheduler_step_count_before']==1
    out['schedulers'].append(dict(rank=rank,file=meta,rows=[{k:v[k] for k in keys if k in v} for v in values]))
print(json.dumps(out,allow_nan=False))
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--handle', type=Path, required=True)
    parser.add_argument('--observation', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / 'actor-probes-ten'))
    from reproduction.a800_common.deploy import ssh_command
    intent = json.loads(args.handle.read_text())
    assert intent['host'] == 'baai-a800'
    observations = json.loads(args.observation.read_text())['observations']
    assert len(observations) == 1
    observed = observations[0]
    assert all(observed[k] == intent[k] for k in ('base','deployment_id','source_commit'))
    payload = {k: observed[k] for k in ('base','deployment_id','run_id','execution_id','source_commit')}
    result = subprocess.run(ssh_command(REMOTE), input=json.dumps(payload).encode(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert result.returncode == 0, 'Existing metadata observation failed; no automatic retry or mutation'
    assert len(result.stdout) <= 2*1024*1024
    value = json.loads(result.stdout)
    path = args.root / 'preparation' / ('actor-probes-ten-A9-first-outer-ledgers-%d.json' % time.time_ns())
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(receipt=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         complete_outer=1,actual_optimizer_writes=value['attempt']['optimizer_writes_this_attempt'],
                         four_rank_scheduler_once=True,first_consumed_lr=0.0,no_mutation=True)))

if __name__ == '__main__':
    main()
