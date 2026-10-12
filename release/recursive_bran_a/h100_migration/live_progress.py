"""Small read-only H100 progress record; no Torch, GPU allocation, or signals."""
import json, pathlib, time, collections, subprocess
b = pathlib.Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
if not (b/'receipts/train-start.json').exists():
    submission = json.loads((b/'receipts/accounting-restart-new-allocation.json').read_text())
    job = submission['stdout'].strip()
    assert submission['returncode'] == 0 and job.isdigit()
    query = subprocess.run(['squeue','-h','-j',job,'-o','%i %T %R'],capture_output=True,text=True,timeout=10)
    log = b/'logs'/('slurm-'+job+'.log')
    print(json.dumps(dict(time=time.time(),waiting_entry=True,job=job,
        queue=query.stdout.strip(),job_query_returncode=query.returncode,
        query_error=query.stderr,ready=(b/'control/launch-ready.sh').exists(),
        setup_tail=log.read_text(errors='replace')[-1500:] if log.exists() else None)))
    raise SystemExit(0)
entry = json.loads((b/'receipts/train-start.json').read_text())
proc = pathlib.Path('/proc')/str(entry['pid'])
text = (b/'logs/training.log').read_text(errors='replace')
counts = collections.Counter(); stages = []
for line in text.splitlines():
    if 'TREE_TIMING=' in line:
        try: event = json.JSONDecoder().raw_decode(line.split('TREE_TIMING=',1)[1])[0]
        except ValueError: continue
        counts[(event.get('outer'),event.get('event'))] += 1
        if event.get('event') not in ('trajectory_action','trajectory_complete'): stages.append(event)
receipts = []
for path in sorted((b/'output/attempt-results').glob('attempt-*.json')):
    d = json.loads(path.read_text())
    receipts.append({'outer':d['accepted_outer'],'attempt':d['attempt_id'],
        'writes':d['actual_writes_this_round'],'status':d['status'],
        'rank_writes':d['actual_completed_optimizer_calls_per_rank']})
identity = None
if proc.exists():
    stat = (proc/'stat').read_text().rsplit(')',1)[1].split()
    identity = {'uid':proc.stat().st_uid,'state':stat[0],'starttick':int(stat[19])}
errors = [line for line in text.splitlines() if any(mark in line for mark in
    ('Traceback (most recent call last)', 'Error executing job', 'ray.exceptions.RayTaskError', 'ray.exceptions.RaySystemError'))]
print(json.dumps({'time':time.time(),'entry':entry,'process':identity,
    'log_bytes':(b/'logs/training.log').stat().st_size,
    'completed_outer_receipts':receipts,'latest_stage':stages[-1] if stages else None,
    'counts':[{'outer':k[0],'event':k[1],'count':v} for k,v in counts.items()],
    'error':bool(errors),'tail':text[-2500:] if errors or identity is None else text[-300:]}))
