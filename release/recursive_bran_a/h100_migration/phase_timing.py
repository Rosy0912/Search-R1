"""Read recorded phase timings and sampled memory, without recomputing models."""
import collections,json,pathlib,time
b=pathlib.Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
def summarize(path):
    sums=collections.Counter();counts=collections.Counter();stages=[];cache=[];rpcs=[]
    for line in path.read_text(errors='replace').splitlines():
        for marker,destination in [('TREE_DRIVER_STAGE=',stages),('RECURSIVE_WEIGHT_CACHE_PHASE=',cache)]:
            if marker in line:
                try:destination.append(json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0])
                except ValueError:pass
        if 'TREE_ACTION_TIMING=' in line:
            d=json.JSONDecoder().raw_decode(line.split('TREE_ACTION_TIMING=',1)[1])[0];counts['advance_batches']+=1
            sums['advance_wall_seconds']+=d['wall_seconds'];sums['driver_cpu_seconds']+=d['driver_cpu_seconds']
            for key,value in d['cost_delta'].items():sums[key]+=value
        if 'TREE_TIMING=' in line:
            d=json.JSONDecoder().raw_decode(line.split('TREE_TIMING=',1)[1])[0];counts[d['event']]+=1
            if d['event']=='generation_rpc_complete':rpcs.append(d)
    return {'path':str(path),'counts':dict(counts),'sums':dict(sums),'latest_driver_stages':stages[-16:],
        'policy_cache_release_records':cache,'generation_window_seconds':
        max((d['unix_time'] for d in rpcs),default=0)-min((d['unix_time']-d['shared_batch_wall_seconds'] for d in rpcs),default=0)}
old=b/'preserved/mu-settings-zero-update-3542960/training.log'
m=json.loads((b/'control/memory-live.json').read_text());m['age_seconds']=time.time()-m['time']
print(json.dumps({'time':time.time(),'entry':json.loads((b/'receipts/train-start.json').read_text()),
    'current':summarize(b/'logs/training.log'),'before_H100_profile_cache':summarize(old),'memory':m,
    'scope':'recorded scalar timing; reference/current work and GPU contention may differ; sampled memory is not continuous maximum'}))
