"""Read-only actual emitted-token and allocated-GPU-time accounting.

The action ledger counts new emissions, including discarded/padded work. It is
not the final training-token total, which includes reused tree prefixes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


def action_costs(path):
    totals = {}; actions = 0; failed = 0; malformed = 0; incomplete = 0; examples = []
    if path is None: return None
    path = Path(path)
    # Multiprocess stdout can interleave or truncate JSON. Account only for
    # complete decodable actions and make every omitted record visible.
    with path.open(errors='replace') as stream:
        for number,line in enumerate(stream,1):
            if 'TREE_ACTION_TIMING=' not in line: continue
            if not line.endswith('\n'):
                incomplete += 1
                continue
            try:
                value = json.JSONDecoder().raw_decode(line.split('TREE_ACTION_TIMING=',1)[1])[0]
            except json.JSONDecodeError:
                malformed += 1
                if len(examples) < 20:
                    examples.append(dict(line=number,sha256=hashlib.sha256(line.encode()).hexdigest()))
                continue
            actions += 1; failed += int(not value['successful'])
            for key, cost in value['cost_delta'].items():
                if not isinstance(cost,(int,float)) or not math.isfinite(cost) or cost < 0:
                    raise ValueError('Invalid per-action compute ledger')
                totals[key] = totals.get(key,0)+cost
    return dict(log=str(path), recorded_actions=actions, failed_actions=failed, totals=totals,
        scope='Actual recorded action deltas; includes screening and evaluation when present in this log. No inherited-prefix double counting.',
        malformed_action_records=malformed,incomplete_action_records=incomplete,
        malformed_record_examples=examples,ledger_gaps_observed=bool(malformed or incomplete),
        interrupted_or_inflight_action_may_be_unrecorded=True)


def scheduler_costs(text):
    result = {}
    for line in text.splitlines():
        columns = line.split('|')
        if not columns[0].isdigit(): continue
        job,state,seconds,resources = columns[:4]
        if job in result: raise ValueError('Duplicate scheduler job rows')
        resources = dict(item.split('=',1) for item in resources.split(',') if '=' in item)
        gpus = int(resources.get('gres/gpu',0)); seconds = int(seconds)
        result[job] = dict(state=state, elapsed_seconds=seconds, allocated_gpus=gpus,
            allocated_gpu_hours=gpus*seconds/3600, allocated_resources=resources,
            scope='Scheduler allocation wall time, including initialization, waiting and failures; not utilization-adjusted FLOPs')
    return result


def collect(entries):
    ids=[str(row['job']) for row in entries]
    if len(set(ids)) != len(ids): raise ValueError('Each allocated job must be counted exactly once')
    text=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),
        '--format=JobIDRaw,State,ElapsedRaw,AllocTRES%200'],text=True)
    jobs=scheduler_costs(text)
    if set(jobs) != set(ids): raise ValueError('Scheduler evidence does not cover all declared jobs')
    rows=[]
    for entry in entries:
        scheduler=jobs[str(entry['job'])]
        state=scheduler['state'];log=entry.get('log')
        # A dependent job has no stdout file until Slurm starts its batch
        # process. This is verified zero work only for an unallocated,
        # zero-runtime pending or cancelled job. A cancelled job that received
        # any resources still needs its log; cancellation alone proves nothing.
        unstarted=((state=='PENDING' or state.split(' by ',1)[0]=='CANCELLED')
            and scheduler['elapsed_seconds']==0 and not scheduler['allocated_resources']
            and log is not None and not Path(log).exists())
        tokens=(dict(log=log,recorded_actions=0,failed_actions=0,totals={},
            scope='Scheduler confirms pending or cancelled before allocation, zero elapsed time and no allocated resources; no stdout was created.',
            interrupted_or_inflight_action_may_be_unrecorded=False) if unstarted else action_costs(log))
        row=dict(entry, scheduler=scheduler, tokens=tokens)
        if row['tokens'] is not None:
            # Exit0 is needed for an exhaustive process ledger. A live or
            # cancelled process may have emitted an unfinished action's tokens.
            row['tokens']['interrupted_or_inflight_action_may_be_unrecorded']=(
                (state != 'COMPLETED' and not unstarted) or row['tokens'].get('ledger_gaps_observed',False))
        rows.append(row)
    return dict(at=time.time(), entries=rows,
        allocated_gpu_hours=sum(row['scheduler']['allocated_gpu_hours'] for row in rows),
        recorded_raw_emitted_tokens=sum((row['tokens'] or {}).get('totals',{}).get('raw_emitted_tokens',0) for row in rows),
        token_total_is_lower_bound=any(row['tokens'] is not None and
            row['tokens']['interrupted_or_inflight_action_may_be_unrecorded'] for row in rows),
        accounting_notes=['One row per scheduler allocation; shared retrieval is charged once and shown separately.',
            'Aborted/retried launches must be included explicitly; historical E is a separate retained experiment.',
            'Tokens from an interrupted action may be unavailable. Do not label this ledger exhaustive until all such gaps are resolved.',
            'Screening tokens are included in the curriculum process total; do not add the separate screening ledger again.'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--entries',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=collect(json.loads(Path(args.entries).read_text()))
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(path),allocated_gpu_hours=result['allocated_gpu_hours'],
        recorded_raw_emitted_tokens=result['recorded_raw_emitted_tokens'],token_total_is_lower_bound=result['token_total_is_lower_bound'])))


if __name__ == '__main__': main()
