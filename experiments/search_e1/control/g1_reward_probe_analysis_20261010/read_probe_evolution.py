"""Read-only remote reducer: emits compact G1 scalar/probe provenance only."""
from pathlib import Path
from collections import Counter
import datetime, hashlib, json, os, re, socket, subprocess

BASE = Path('/data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512')
OUT = dict(read_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           node=socket.gethostname(), base=str(BASE), readonly=True)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def small_json(path, cap=262144):
    if not path.is_file():
        return None
    b = path.read_bytes() if path.stat().st_size <= cap else None
    out = dict(path=str(path), bytes=path.stat().st_size,
               mtime_ns=path.stat().st_mtime_ns)
    if b is None:
        out['too_large_for_summary'] = True
        return out
    out['sha256'] = hashlib.sha256(b).hexdigest()
    try:
        out['content'] = json.loads(b)
    except ValueError:
        out['parse_error'] = True
    return out

manifest = small_json(BASE / 'logs/run-manifest.json')
run = manifest['content'] if manifest else {}
OUT['identity'] = dict(run_id=run.get('run_id'), source_commit=run.get('source_commit'),
                       precision_profile=run.get('precision_profile'),
                       budget=run.get('budget'), manifest_path=manifest['path'] if manifest else None,
                       manifest_sha256=manifest['sha256'] if manifest else None)
OUT['status'] = {}
for name in ('pipeline-status.json', 'deployment-status.json', 'process-ledger.json'):
    row = small_json(BASE / 'logs' / name)
    if row:
        if name == 'process-ledger.json':
            c = row.pop('content', {})
            row['top_level_keys'] = list(c)
            row['job_count'] = len(c.get('jobs', [])) if isinstance(c, dict) else None
        OUT['status'][name] = row

update_path = BASE / 'logs/actual-updates.jsonl'
updates = []
partial = 0
if update_path.is_file():
    with update_path.open('rb') as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except ValueError:
                partial += 1
                continue
            if row.get('run_id') != run.get('run_id'):
                continue
            updates.append(dict(line=line_number, step=row.get('step'),
                                probe_version=row.get('probe_version'),
                                metrics=row.get('metrics', {}),
                                optimizer_writes_after=row.get('optimizer_writes_after'),
                                actual_optimizer_writes=row.get('actual_optimizer_writes')))
OUT['actual_updates'] = dict(path=str(update_path), bytes=update_path.stat().st_size if update_path.is_file() else None,
                             sha256=digest(update_path) if update_path.is_file() else None,
                             complete_rows=updates, malformed_or_partial_rows=partial)
log_path = BASE / 'logs/search-g1-common8-training.log'
scalars = []
metric_keys = Counter()
interesting = []
exceptions = []
if log_path.is_file():
    for ln, raw in enumerate(log_path.open(errors='replace'), 1):
        step_match = re.search(r'\bstep:(\d+)\s*-', raw)
        if step_match:
            vals = {k: float(v) for k, v in re.findall(
                r'([\w/]+):(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)', raw)}
            metric_keys.update(vals.keys())
            selected = {k: v for k, v in vals.items() if
                        any(t in k.lower() for t in ('probe', 'boundary', 'flip', 'ground', 'trust', 'cost/', 'train/perf/', 'critic/reward', 'val/test_score', 'actor/entropy', 'actor/pg_loss', 'actor/grad_norm', 'actor/kl_coef', 'actor/kl_loss', 'env/actual_retrieval', 'env/number_of_valid_search'))}
            scalars.append(dict(line=ln, step=int(step_match[1]), metrics=selected))
        if '[ECO-INIT]' in raw or 'SEARCH_G1_PROBE_VERSION=' in raw:
            interesting.append(dict(line=ln, text=raw.strip()[:2500]))
        if any(t in raw for t in ('Traceback (most recent call last)', 'RuntimeError:', 'ValueError:', 'ERROR ')):
            exceptions.append(dict(line=ln, text=raw.strip()[:1000]))
OUT['training_log'] = dict(path=str(log_path), bytes=log_path.stat().st_size if log_path.is_file() else None,
                           sha256=digest(log_path) if log_path.is_file() else None,
                           scalar_rows=scalars, metric_key_counts=dict(metric_keys),
                           probe_and_reward_identity_lines=interesting,
                           error_lines=exceptions[-20:])

OUT['probe_artifact_inventory'] = []
for root in (BASE / 'artifacts', BASE / 'logs'):
    if not root.is_dir():
        continue
    for path in root.rglob('*'):
        if not path.is_file() or path.suffix not in ('.json', '.jsonl', '.npz', '.pkl'):
            continue
        if root.name == 'logs' and not any(t in path.name.lower() for t in ('probe', 'calibration', 'refit', 'reward', 'behavior')):
            continue
        st = path.stat()
        row = dict(path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns)
        if path.suffix == '.json' and any(t in path.name for t in ('manifest', 'calibration-report', 'label-proof', 'accepted', 'refresh')):
            parsed = small_json(path)
            if parsed:
                row = parsed
        OUT['probe_artifact_inventory'].append(row)

OUT['runtime_source'] = []
for relative in ('verl/trainer/main_ppo_eco.py', 'probe_server.py', 'ts_probe_dump.py',
                 'ts_probe_train.py', 'reproduction/a800_common/train.sh', 'reproduction/common8/refit.py',
                 'reproduction/common8/behavior_trace.py', 'reproduction/common8/behavior_runtime.py'):
    path = BASE / 'repo' / relative
    if not path.is_file():
        continue
    raw = path.read_bytes()
    lines = raw.decode().splitlines()
    excerpts = []
    if relative.endswith('main_ppo_eco.py'):
        for function in ('get_metrics', '_score_probe'):
            start = next((i for i, line in enumerate(lines) if line.startswith('    def ' + function + '(')), None)
            if start is not None:
                end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith('    def ')), len(lines))
                excerpts.append(dict(function=function, first_line=start + 1,
                                     source='\n'.join(lines[start:end])))
        for i, line in enumerate(lines):
            if any(t in line for t in ('last_metrics', 'ECO_BAL_', 'ECO_WRONG_', 'probe_base =', 'probe_alpha_cost =')):
                excerpts.append(dict(first_line=i + 1, source=line))
    elif relative.endswith('train.sh'):
        excerpts = [dict(first_line=1, source='\n'.join(line for line in lines if 'ECO_' in line or 'PROBE_' in line))]
    elif relative.endswith('behavior_trace.py'):
        for function in ('make_trace', 'make_actor_row', 'write_private_evidence'):
            start = next((i for i, line in enumerate(lines) if line.startswith('def ' + function + '(')), None)
            if start is not None:
                end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith('def ')), len(lines))
                excerpts.append(dict(function=function, first_line=start + 1, source='\n'.join(lines[start:end])))
    OUT['runtime_source'].append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                      bound_sha256=run.get('source_sha256', {}).get(relative),
                                      excerpts=excerpts))

OUT['owned_processes'] = []
for directory in Path('/proc').iterdir():
    if not directory.name.isdigit():
        continue
    try:
        cwd = str((directory / 'cwd').resolve())
        if not cwd.startswith(str(BASE)):
            continue
        args = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        if not any(t in args for t in ('main_ppo', 'run_pipeline', 'fresh200', 'probe_server', 'refit')):
            continue
        stat = (directory / 'stat').read_text()
        tail = stat[stat.rfind(')') + 2:].split()
        OUT['owned_processes'].append(dict(pid=int(directory.name), cwd=cwd,
                                           start_ticks=int(tail[19]), state=tail[0], command=args[:1200]))
    except (OSError, ValueError, IndexError):
        continue
OUT['observed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(OUT, ensure_ascii=False))
