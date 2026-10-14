"""Read-only evidence audit of actual rank writes, loss statistics and native5."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import time


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2), b''): value.update(chunk)
    return value.hexdigest()


def read_jsonl(path):
    if not path.is_file(): return []
    raw = path.read_bytes()
    # A live writer may not have completed its last line; never credit that line.
    if not raw.endswith(b'\n'): raw = raw.rsplit(b'\n',1)[0] if b'\n' in raw else b''
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def verify_rank(record, run, identity, outer, output=None):
    output = run/'output' if output is None else output
    rank = record['rank']
    direct = record['direct_rank_record']
    path = Path(direct['path'])
    if path.parent != output/'fresh-state' or path.is_symlink() or sha(path) != direct['sha256']:
        raise ValueError('Direct rank evidence is missing, modified or outside the run')
    actual = json.loads(path.read_text())
    if any(actual[key] != value for key,value in identity.items()):
        raise ValueError('Rank source/run/config identity changed')
    writes = record['optimizer_calls']
    ledger = actual['optimizer_opportunity_ledger']
    if (actual['status'] != 'DIRECT_RANK_COMPLETE_OUTER_OBSERVED'
            or actual['rank'] != rank or actual['world_size'] != 8
            or actual['actual_optimizer_calls'] != writes
            or actual['scheduler_step_before'] != outer-1 or actual['scheduler_step_after'] != outer
            or record['scheduler_before'] != outer-1 or record['scheduler_after'] != outer
            or len(ledger) != 20 or not 0 <= writes <= 20
            or sum(entry['actual_writes'] for entry in ledger) != writes):
        raise ValueError('Rank optimizer/scheduler/opportunity evidence disagrees')
    for index, entry in enumerate(ledger,1):
        if (entry['index'] != index or entry['actual_writes'] not in (0,1)
                or entry['rows'] != 16 or entry['local_row_indices'] != list(range((index-1)*16,index*16))
                or entry['status'].startswith('SUCCESS') != bool(entry['actual_writes'])):
            raise ValueError('Rank opportunity ledger does not cover the actual320rows')
    norms = [e['gradient_preclip'] for e in ledger if e['actual_writes']]
    if any(value is None or not math.isfinite(value) for value in norms):
        raise ValueError('A committed update lacks a finite observed gradient norm')
    audit_path = output/'rank-audit'/f"step-{outer:06d}-attempt-{actual['attempt_id']:06d}-rank-{rank:02d}.json"
    losses = []
    if outer in (5,50,100,150,200):
        audit = json.loads(audit_path.read_text())
        if (any(audit[key] != value for key,value in identity.items())
                or audit['successful_optimizer_calls'] != writes or audit['existing_grad_norms'] != norms):
            raise ValueError('Sparse Tree rank audit differs from actual updates')
        for call in range(writes):
            tail_path = output/'rank-audit/ratio-tail'/f"step-{outer:06d}-attempt-{actual['attempt_id']:06d}-rank-{rank:02d}-optimizer-{call:02d}.json"
            tail = json.loads(tail_path.read_text())
            if tail['optimizer_call'] != call+1 or tail['outer_step'] != outer or tail['rank'] != rank:
                raise ValueError('Successful-write loss record belongs to another update')
            losses.extend(value for micro in tail['microbatches'] for value in micro['loss_parts'].values())
    return dict(rank=rank, actual_adam_writes=writes, total_adam_writes=record['total_optimizer_calls'],
        observed_loss_scalars=len(losses), finite_loss_scalars=sum(math.isfinite(value) for value in losses),
        all_observed_loss_finite=(all(math.isfinite(value) for value in losses) if losses else None),
        loss_scope='Sparse Tree successful-write microbatch losses at5/50/100/150/200; other rounds use console aggregates',
        observed_grad_norms=len(norms), finite_grad_norms=len(norms),
        direct_record_sha256=direct['sha256'])



def checkpoint(run, identity, hash_native):
    root = run/'checkpoints/training_state/global_step_5'
    result = dict(pre_evaluation_saved=False, post_evaluation_saved=False,
                  same_world_reload_verified=False, all_native_files_hash_verified=False)
    for label,path in [('pre',root/'manifest.json'),('post',root/'post-evaluation/manifest.json')]:
        if not path.is_file(): continue
        manifest = json.loads(path.read_text())
        expected = {key:value for key,value in identity.items() if key != 'run_id'}
        expected['completed_step'] = 5
        if (manifest['status'] != 'COMPLETE_SAME_WORLD_STATE_SAVED'
                or any(manifest['identity'][key] != value for key,value in expected.items())
                or len(manifest['worker_states']) != 8
                or {r['identity']['rank'] for r in manifest['worker_states']} != set(range(8))):
            raise ValueError('Native5 boundary source or rank coverage differs')
        if label == 'post' and manifest.get('boundary_phase') != 'post-evaluation':
            raise ValueError('Post-evaluation RNG boundary absent')
        for record in manifest['worker_states']+[manifest['driver']]:
            member = Path(record['path'])
            if member.parent != path.parent or member.is_symlink() or not member.is_file():
                raise ValueError('Native5 file missing or outside the checkpoint')
            if 'bytes' in record and member.stat().st_size != record['bytes']:
                raise ValueError('Native5 file size changed')
            if hash_native and sha(member) != record['sha256']:
                raise ValueError('Native5 actual file SHA differs')
        if any(r['full_worker_state_finite'] is not True for r in manifest['worker_states']):
            raise ValueError('Native5 finite-state evidence absent')
        result[('pre' if label == 'pre' else 'post')+'_evaluation_saved'] = True
        result[label+'_manifest_sha256'] = sha(path)
        if label == 'pre':
            result['same_world_reload_verified'] = (manifest['driver']['exact_reload_verified'] is True
                and all(r['exact_same_world_reload_verified'] is True for r in manifest['worker_states']))
    result['all_native_files_hash_verified'] = bool(hash_native and result['pre_evaluation_saved'] and result['post_evaluation_saved'])
    return result


def evidence_namespace(run, config):
    if config.get('protocol') != 'recursive_mechanism_ablation150_v1':
        return run/'output', json.loads((run/'receipts/train-start.json').read_text())['job']
    entry = json.loads((run/'receipts/current-entry.json').read_text())
    output = Path(entry['output'])
    if (output.parent != run/'output' or output.resolve() != output or output.is_symlink()
            or re.fullmatch(r'attempt-[0-9a-f]{32}',output.name) is None
            or any(entry[k] != config[k] for k in ('run_id','source_commit','config_sha256','group'))):
        raise ValueError('Ablation evidence namespace does not match its current entry')
    if entry.get('resume_manifest'):
        raise ValueError('Resumed evidence needs linked ancestry; this audit verifies fresh runs only')
    return output, entry['allocation']


def inspect(run, log_path, hash_native=False):
    run = Path(run).resolve()
    config = json.loads((run/'receipts/configuration.json').read_text())
    output, job = evidence_namespace(run,config)
    fresh = json.loads((output/'fresh-initialization.json').read_text())
    identity = dict(job_id=str(job), run_id=config['run_id'], code_commit=config['source_commit'],
                    config_sha256=config['config_sha256'],data_identity=config['data_identity'])
    if any(fresh[key] != value for key,value in identity.items()):
        raise ValueError('Fresh receipt and run configuration disagree')
    log_text = Path(log_path).read_text(errors='replace')
    round_metrics = {}
    for line in log_text.splitlines(keepends=True):
        if not line.endswith('\n') or 'training/actual_outer_updates:' not in line: continue
        match = re.search(r'(?:^|\s)step:(\d+) - ',line)
        if match is None: continue
        values = {}
        for key,value in re.findall(r'([A-Za-z0-9_/.-]+):([-+0-9.eE]+|nan|inf)',line[match.start():]):
            values[key] = float(value)
        if values.get('training/actual_outer_updates') == int(match.group(1)):
            round_metrics[int(match.group(1))] = values
    attempts = []
    for path in sorted((output/'attempt-results').glob('attempt-*.json')):
        value = json.loads(path.read_text());outer = value['accepted_outer']
        ranks = value['all_rank_result']['ranks']
        if len(ranks) != 8 or {r['rank'] for r in ranks} != set(range(8)):
            raise ValueError('Completed attempt lacks eight physical ranks')
        verified = [verify_rank(r,run,identity,outer,output) for r in ranks]
        if len({r['actual_adam_writes'] for r in verified}) != 1:
            raise ValueError('Actual Adam writes differ across ranks')
        attempts.append(dict(outer=outer,attempt=value['attempt_id'], ranks=verified,
            actual_adam_writes_per_rank=verified[0]['actual_adam_writes'],
            cumulative_adam_writes_per_rank={str(r['rank']):r['total_adam_writes'] for r in verified},
            normal_guard=value.get('normal_guard'),attempt_file_sha256=sha(path),
            console_metrics=round_metrics.get(outer),
            reported_pg_loss_finite=(math.isfinite(round_metrics[outer]['actor/pg_loss'])
                if outer in round_metrics and 'actor/pg_loss' in round_metrics[outer] else None)))
    if [r['outer'] for r in attempts] != list(range(1,len(attempts)+1)):
        raise ValueError('Completed rounds do not form this fresh run’s continuous prefix')
    boundary = checkpoint(run,identity,hash_native)
    sixth = []
    for line in log_text.splitlines(keepends=True):
        if not line.endswith('\n') or 'TREE_TIMING=' not in line: continue
        event = json.JSONDecoder().raw_decode(line.split('TREE_TIMING=',1)[1])[0]
        if (event.get('run_id') == config['run_id'] and event.get('outer') == 6
                and event.get('event') in ('generation_rpc_complete','trajectory_complete')):
            sixth.append(event)
    first5 = attempts[:5]
    five_writes = len(first5) == 5 and all(r['actual_adam_writes_per_rank'] > 0 for r in first5)
    finite = len(first5) == 5 and all(r['reported_pg_loss_finite'] is True
        and all(rank['all_observed_loss_finite'] is not False for rank in r['ranks']) for r in first5)
    return dict(at=time.time(), run=str(run), output=str(output), identity=identity, completed_rounds=len(attempts),
        zero_write_rounds=sum(r['actual_adam_writes_per_rank'] == 0 for r in attempts),attempts=attempts,
        native5=boundary,sixth_real_generation_observed=bool(sixth),sixth_first_evidence=sixth[0] if sixth else None,
        first5_each_has_real_adam_writes=five_writes,first5_observed_loss_all_finite=finite,
        initial_acceptance_verified=bool(five_writes and finite and boundary['same_world_reload_verified']
            and boundary['all_native_files_hash_verified'] and sixth),
        scope='Read-only physical file and runtime receipt audit; scheduler liveness must be checked separately; no arbitrary cold/vLLM restart proof')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True);parser.add_argument('--log',required=True)
    parser.add_argument('--hash-native',action='store_true')
    args=parser.parse_args(); print(json.dumps(inspect(args.run,args.log,args.hash_native),allow_nan=False))


if __name__ == '__main__': main()
