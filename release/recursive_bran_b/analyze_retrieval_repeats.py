"""Reconstruct ordered retrieval batches from completed native B evidence.

Read full trajectories only on HKUST; export hashes/counts, never query text.
Inherited prefixes are deduplicated by physical sampling ID. Each batch must
match its recorded HTTP query count and generation/action timing boundaries.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser();p.add_argument('--events',required=True);p.add_argument('--log',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    records={};physical={};files=list(Path(a.events).rglob('*.json'))
    for path in files:
        value=json.loads(path.read_text());state=value.get('state',value)
        for item in state.get('raw_ledger',[]):
            action=item['action'];emission=action['native_emission']
            match=re.search(r'<search>(.*?)</search>',action['legacy_reward_action_text'],re.DOTALL)
            searched=bool(match and '<information>' in item['observation_text'])
            digest=hashlib.sha256(match.group(1).strip().encode()).hexdigest() if searched else None
            key=(state['trajectory'],item['turn'])
            assert key not in records or records[key]==digest, 'Conflicting native action identity'
            records[key]=digest
            if searched:
                sid=emission['sample_id']
                assert sid not in physical or physical[sid]==digest
                physical[sid]=digest
    batches=[];current=None;run_ids=set()
    log=Path(a.log).read_text(errors='replace')
    for line in log.splitlines():
        found=re.search(r'(TREE_TIMING|COST80)=(\{.*\})',line)
        if not found:continue
        event=json.loads(found.group(2))
        if event.get('run_id'):run_ids.add(event['run_id'])
        kind=event.get('event',event.get('stage'))
        if kind=='generation_rpc_complete':
            if current is not None:batches.append(current)
            current=dict(expected_actions=event['active_requests'],action_count=0,query_hashes=[],http_shapes=[])
        elif kind=='retrieval_request_shape':
            assert current is not None
            current['http_shapes'].append(event['queries'])
        elif kind=='trajectory_action':
            assert current is not None
            current['action_count']+=1
            key=(event['trajectory_id'],event['turn_before'])
            assert key in records, 'Missing recorded physical trajectory action'
            digest=records[key]
            assert event['retrieved']==(digest is not None), 'Actual search action mapping changed'
            if digest is not None:current['query_hashes'].append(digest)
    if current is not None:batches.append(current)
    for batch in batches:
        assert batch['action_count']==batch['expected_actions'], 'Missing timing actions'
        assert batch['http_shapes']==[len(batch['query_hashes'])], 'HTTP batch shape reconstruction failed'
    assert sum(len(x['query_hashes']) for x in batches)==len(physical), 'Physical search coverage differs'
    nonempty=[tuple(x['query_hashes']) for x in batches if x['query_hashes']]
    batch_counts=Counter(nonempty);query_counts=Counter(physical.values())
    result=dict(scope='actual_ordered_http_batches_with_native_sampling_identity',run_ids=sorted(run_ids),
        source_json_files=len(files),physical_search_queries=len(physical),unique_query_hashes=len(query_counts),
        repeated_query_occurrences=sum(v-1 for v in query_counts.values()),
        generation_batches=len(batches),nonempty_http_batches=len(nonempty),
        empty_http_batches=len(batches)-len(nonempty),
        exact_repeated_nonempty_batches=sum(v-1 for v in batch_counts.values()),
        repeated_batch_multiplicities=sorted([v for v in batch_counts.values() if v>1],reverse=True),
        ordered_batch_hashes=[hashlib.sha256(json.dumps(x).encode()).hexdigest() for x in nonempty],
        query_count_per_batch=[len(x['query_hashes']) for x in batches],
        actual_log_sha256=hashlib.sha256(Path(a.log).read_bytes()).hexdigest(),raw_queries_exported=False)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
