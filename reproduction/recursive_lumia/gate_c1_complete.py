"""Release C2 only after actual C1 training, native files, evaluations and exit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .compare import load_evaluation
from .inspect_run import inspect, sha
from .c1_evidence import validate_rounds, verify_events
from .prepare_ablation import condition_for


def validate_exit(record, entry, complete, terminal, wait, expected_job, scheduler_text):
    rows=[line.split('|') for line in scheduler_text.splitlines() if line.strip()]
    matching=[r for r in rows if r[0]==str(expected_job)]
    if len(matching)!=1 or matching[0][1:4]!=['jmzhang','COMPLETED','0:0']:
        raise ValueError('The exact C1 scheduler allocation has not completed successfully')
    if (record['group']!='C1-root-only' or record['condition']!=condition_for('C1-root-only')
            or str(entry['allocation'])!=str(expected_job)
            or complete['status']!='COMPLETE_NATIVE_150_TRAINING_AND_FOUR_SCHEDULED_DEV'
            or complete['completed_outer']!=150 or complete['evaluation_steps']!=[5,50,100,150]
            or terminal['rc']!=0 or wait['rc']!=0 or terminal.get('error') is not None
            or wait['pid']!=entry['pid'] or terminal['pid']!=entry['pid']
            or terminal['tick']!=entry['tick'] or terminal['uid']!=entry['uid']
            or str(wait['allocation'])!=str(expected_job) or wait['group']!=record['group']
            or wait['source']!=record['source_commit']):
        raise ValueError('C1 completion/native return/actual parent wait disagree')
    for key in ('run_id','group','config_sha256','source_commit'):
        if complete[key]!=record[key] or entry[key]!=record[key]:
            raise ValueError('C1 completion belongs to a different run/configuration')
    if terminal['run_id']!=record['run_id'] or terminal['group']!=record['group']:
        raise ValueError('Native exit record belongs to another run')


def verify_final_native(run, record, entry, complete, total):
    path=run/'checkpoints/training_state/global_step_150/post-evaluation/manifest.json'
    if complete['checkpoint']['path']!=str(path) or complete['checkpoint']['sha256']!=sha(path):
        raise ValueError('Completion does not identify the actual post-evaluation150 checkpoint')
    value=json.loads(path.read_text());identity=value['identity']
    expected=dict(job_id=str(entry['allocation']),code_commit=record['source_commit'],
        config_sha256=record['config_sha256'],data_identity=record['data_identity'],
        completed_step=150,actual_update_limit=150)
    workers=value['worker_states']
    if (value['status']!='COMPLETE_SAME_WORLD_STATE_SAVED' or value.get('boundary_phase')!='post-evaluation'
            or any(identity[k]!=v for k,v in expected.items())
            or len(workers)!=8 or {r['identity']['rank'] for r in workers}!=set(range(8))
            or value['driver']['next_step']!=151):
        raise ValueError('Final native checkpoint lacks the complete150 boundary')
    for worker in workers:
        if (any(worker['identity'][k]!=v for k,v in expected.items())
                or worker['identity']['world_size']!=8 or not worker['full_worker_state_finite']
                or worker['scheduler_last_epoch']!=150 or worker['actual_total_optimizer_calls']!=total
                or worker['observed_adam_steps']!=([total] if total else [])):
            raise ValueError('Native worker optimizer/scheduler evidence differs')
    for member in workers+[value['driver']]:
        file=Path(member['path'])
        if (file.parent!=path.parent or file.is_symlink() or not file.is_file()
                or ('bytes' in member and file.stat().st_size!=member['bytes'])
                or sha(file)!=member['sha256']):
            raise ValueError('Final native state bytes are missing or modified')
    return dict(manifest=str(path),manifest_sha256=sha(path),all_nine_native_files_hash_verified=True)


def verify(run, log, expected_job):
    run=Path(run).resolve()
    record=json.loads((run/'receipts/configuration.json').read_text())
    if sha(record['config'])!=record['config_sha256']:
        raise ValueError('Frozen C1 configuration changed')
    entry=json.loads((run/'receipts/current-entry.json').read_text())
    output=Path(entry['output'])
    if output.parent!=run/'output' or output.resolve()!=output:
        raise ValueError('C1 output namespace is not inside this exact run')
    complete=json.loads((output/'training-complete.json').read_text())
    terminal=json.loads((output/'entry-exit.json').read_text())
    wait=json.loads((run/'receipts/actual-parent-wait.json').read_text())
    scheduler=subprocess.check_output(['sacct','-X','-n','-P','-j',str(expected_job),
        '--format=JobIDRaw,User,State,ExitCode'],text=True)
    validate_exit(record,entry,complete,terminal,wait,expected_job,scheduler)
    audit=inspect(run,log,hash_native=True)
    rounds=validate_rounds(audit,complete['actual_optimizer_calls_per_rank'])
    mechanism=verify_events(output,record,outer=5)
    manifest_path=Path(record['data'])/'manifest.json'
    if sha(manifest_path)!=record['data_identity']:
        raise ValueError('Frozen data manifest changed')
    expected_ids=json.loads(manifest_path.read_text())['splits']['validation']['question_ids_sha256']
    evaluations=[]
    for outer in (5,50,100,150):
        metadata,rows,files=load_evaluation(output/'evaluations'/f'dev512-outer-{outer:06d}.jsonl')
        expected=dict(run_id=record['run_id'],job=str(expected_job),code_commit=record['source_commit'],
            config_sha256=record['config_sha256'],data_identity=record['data_identity'],completed_outer=outer,split='dev512')
        ids_sha=hashlib.sha256(json.dumps(list(rows),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        if any(metadata[k]!=v for k,v in expected.items()) or ids_sha!=expected_ids:
            raise ValueError('Evaluation identity or fixed512 question coverage differs')
        evaluations.append(dict(outer=outer,em=metadata['em'],**files))
    if complete['final_dev_metrics']['val/em/all']!=evaluations[-1]['em']:
        raise ValueError('Terminal dev metric differs from actual paired150 rows')
    final_native=verify_final_native(run,record,entry,complete,rounds['actual_adam_per_rank'])
    return dict(at=time.time(),status='C1_COMPLETE150_VERIFIED_RELEASE_C2',job=str(expected_job),
        run_id=record['run_id'],source_commit=record['source_commit'],config_sha256=record['config_sha256'],
        rounds=rounds,mechanism=mechanism,evaluations=evaluations,final_native=final_native,
        native5=audit['native5'],sixth_real_generation_observed=True,
        limits='Fresh-run ancestry only; full native bytes and same-world verification, not arbitrary cold/vLLM bitwise restart proof.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--log',required=True)
    p.add_argument('--job',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=verify(a.run,a.log,a.job)
    with Path(a.output).open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':main()
