"""Summarize emitted timing/scalars only; no model, GPU, HTTP, or node writes."""
import json
import pathlib
import re
import time

base = pathlib.Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
entry = json.loads((base / 'receipts/train-start.json').read_text())
rows = {}
metrics = {}
for line in (base / 'logs/training.log').read_text(errors='replace').splitlines():
    clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
    if 'TREE_TIMING=' in clean:
        try:
            d = json.JSONDecoder().raw_decode(clean.split('TREE_TIMING=', 1)[1])[0]
        except ValueError:
            continue
        if d.get('run_id') != entry['run_id']:
            continue
        row = rows.setdefault(str(d['outer']), {'generation_rpc_count': 0, 'generation_rpc_seconds_sum': 0})
        if d['event'] == 'generation_rpc_complete':
            start = d['unix_time'] - d['shared_batch_wall_seconds']
            row['first_rpc_start'] = min(start, row.get('first_rpc_start', start))
            row['last_rpc_end'] = max(d['unix_time'], row.get('last_rpc_end', d['unix_time']))
            row['generation_rpc_count'] += 1
            row['generation_rpc_seconds_sum'] += d['shared_batch_wall_seconds']
        elif d['event'] in ('compose_start', 'compose_complete'):
            row[d['event']] = d['unix_time']
            if d['event'] == 'compose_complete':
                row['compose_seconds'] = d['wall_seconds']
                row['retained_training_rows'] = d['rows']
    match = re.search(r'\bstep:(\d+) - ', clean)
    if match:
        values = {}
        for item in clean[match.end():].split(' - '):
            key, separator, value = item.rpartition(':')
            if separator:
                try:
                    values[key] = float(value)
                except ValueError:
                    pass
        metrics[str(int(match.group(1)))] = values
for row in rows.values():
    if 'first_rpc_start' in row and 'compose_start' in row:
        row['generation_to_compose_start_seconds'] = row['compose_start'] - row['first_rpc_start']
print(json.dumps(dict(time=time.time(), entry=entry, recorded_per_outer=rows,
    emitted_outer_metrics=metrics, scope='Recorded native timings/scalars; shared batch latency; no causal speed attribution or fixed-heldout inference')))
