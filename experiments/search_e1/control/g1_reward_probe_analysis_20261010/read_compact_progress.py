"""Read-only compact current G1 counters, fixed-dev counts and fresh log tail."""
from pathlib import Path
from collections import Counter
import datetime, hashlib, json, re, socket

BASE = Path('/data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512')
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
out = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), node=socket.gethostname(), readonly=True)
raw = (BASE / 'logs/run-manifest.json').read_bytes()
run = json.loads(raw)
out.update(run_id=run['run_id'], source_commit=run['source_commit'], run_manifest_sha256=sha(raw))
upath = BASE / 'logs/actual-updates.jsonl'; ub = upath.read_bytes()
updates = []
for line in ub.splitlines():
    try:
        row = json.loads(line)
    except ValueError:
        continue
    if row.get('run_id') == run['run_id']:
        updates.append(row)
out['native_progress'] = dict(path=str(upath), sha256=sha(ub), bytes=len(ub), complete_rows=len(updates),
                              actual_optimizer_writes=sum(r.get('actual_optimizer_writes', 0) for r in updates),
                              last_record=updates[-1], sequential_steps=[r['step'] for r in updates] == list(range(1, len(updates) + 1)))
log_path = BASE / 'logs/search-g1-common8-training.log'
hb = hashlib.sha256(); scalar_rows = []; errors = []
for ln, b in enumerate(log_path.open('rb'), 1):
    hb.update(b); text = b.decode(errors='replace')
    match = re.search(r'\bstep:(\d+)\s*-', text)
    if match:
        vals = {k: float(v) for k, v in re.findall(r'([\w/]+):(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)', text)}
        keys = ('train/perf/em', 'critic/rewards/mean', 'actor/entropy_loss', 'actor/pg_loss', 'actor/grad_norm',
                'actor/kl_loss', 'actor/kl_coef', 'train/cost/search_per_query', 'train/cost/nosearch_rate',
                'train/cost/oversearch_rate', 'env/actual_retrieval_calls/total', 'env/actual_retrieval_calls/mean',
                'env/number_of_valid_search')
        scalar_rows.append(dict(step=int(match[1]), line=ln, metrics={k: vals[k] for k in keys if k in vals}))
    if any(t in text for t in ('Traceback (most recent call last)', 'RuntimeError:', 'ValueError:', 'ERROR ')):
        errors.append(dict(line=ln, text=text.strip()[:1000]))
out['training_log'] = dict(path=str(log_path), sha256_read_prefix=hb.hexdigest(), bytes=log_path.stat().st_size,
                           scalar_rows=scalar_rows, error_lines=errors[-10:])
vpath = BASE / 'logs/validation-identities.jsonl'
if vpath.is_file():
    vb = vpath.read_bytes(); vr = []
    for ln, line in enumerate(vb.splitlines(), 1):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get('run_id') != run['run_id'] or r.get('split') != 'validation':
            continue
        sources = r['data_sources']; em = r['em']
        by_source = {name: dict(count=sources.count(name), correct=sum(e for s, e in zip(sources, em) if s == name))
                     for name in sorted(set(sources))}
        vr.append(dict(line=ln, step=r['step'], row_sha256=sha(line), question_count=len(em),
                       correct=sum(em), weighted_em=sum(em) / len(em), weighted_f1=r.get('weighted_f1'),
                       by_source=by_source, question_order_sha256=sha(json.dumps(r['question_ids']).encode()),
                       validation_sha256=r['validation_sha256'], checked_unix=r['checked_unix'],
                       ordered_unique_question_count=len(set(r['question_ids']))))
    out['fixed_development_validation'] = dict(path=str(vpath), file_sha256=sha(vb), rows=vr)
out['processes'] = []
for pid in (1327340, 1336445, 1338978):
    p = Path('/proc') / str(pid)
    try:
        stat = (p / 'stat').read_text(); tail = stat[stat.rfind(')') + 2:].split()
        out['processes'].append(dict(pid=pid, state=tail[0], start_ticks=int(tail[19]),
                                      cwd=str((p / 'cwd').resolve()),
                                      command=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode()[:1000]))
    except OSError:
        out['processes'].append(dict(pid=pid, absent=True))
traces = sorted((BASE / 'artifacts/behavior-traces').glob('attempt-*.json'), key=lambda p: p.stat().st_mtime_ns)
out['latest_behavior_trace_metadata'] = [dict(path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in traces[-2:]]
out['observed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(out, ensure_ascii=False))
