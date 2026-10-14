"""Read one completed final batch's timing without importing model libraries."""
import json
import time
from pathlib import Path

B=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013')
F=B/'final51713-worker-recovery-v1'


def main():
    plan=json.loads((F/'execution-plan.json').read_text())
    path=F/'evaluation.log'
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-16777216))
        lines=stream.read().decode(errors='replace').splitlines()
    events=[]
    for line in lines:
        for marker in ('FINAL51713_PROGRESS=','TREE_TIMING=','TREE_ACTION_TIMING=','RECURSIVE_WEIGHT_CACHE_PHASE='):
            if marker in line:
                try:
                    row=json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0]
                    events.append((marker,row))
                except ValueError:pass
                break
    batches=[r for marker,r in events if marker=='FINAL51713_PROGRESS=']
    if len(batches)<2:
        print(json.dumps(dict(time=time.time(),status='WAIT_TWO_COMPLETED_BATCHES',batches=batches[-2:])))
        return
    before,after=batches[-2:]
    actions=[];compose=[];rpc=[];cache=[]
    for marker,row in events:
        stamp=row.get('unix_time',row.get('time'))
        if stamp is not None and not before['time']<=stamp<=after['time']:continue
        if marker=='TREE_ACTION_TIMING=':
            # This marker has no timestamp; include only if its following/preceding
            # driver event places it within the completed interval below.
            continue
        if marker=='TREE_TIMING=' and row.get('event')=='compose_complete':compose.append(row['wall_seconds'])
        if marker=='TREE_TIMING=' and row.get('event')=='generation_rpc_complete':rpc.append(row['shared_batch_wall_seconds'])
        if marker=='RECURSIVE_WEIGHT_CACHE_PHASE=':cache.append(row)
    # Associate timestamp-less action records with the preceding timed driver event.
    last_time=None
    for marker,row in events:
        if row.get('unix_time') is not None:last_time=row['unix_time']
        if marker=='TREE_ACTION_TIMING=' and last_time is not None and before['time']<=last_time<=after['time']:
            actions.append(row)
    result=dict(time=time.time(),run_id=plan['run_id'],status='COMPLETE_BATCH_TIMING',
        interval=[before,after],action_records=len(actions),
        action_wall_seconds=sum(r['wall_seconds'] for r in actions),
        action_driver_cpu_seconds=sum(r['driver_cpu_seconds'] for r in actions),
        generation_rpc_seconds=sum(rpc),compose_seconds=sum(compose),compose_calls=len(compose),
        cache_phase_records=cache,
        scope='Recorded driver timing for the last completed batch; residual includes uninstrumented scoring and data transfer')
    for key in ('retrieval_http_json_wall_seconds','semantic_prefix_scan_wall_seconds','generated_tokens'):
        result[key]=sum(r.get('cost_delta',{}).get(key,0) for r in actions)
    result['residual_seconds']=after['batch_seconds']-result['action_wall_seconds']-result['compose_seconds']
    print(json.dumps(result))


if __name__=='__main__':main()
