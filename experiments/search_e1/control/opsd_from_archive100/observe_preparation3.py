"""External CPU-only progress evidence, bound to one real controller lifetime.

Reads new preparation-3 outputs by incremental bytes. Never starts a model,
posts a retrieval query, changes routing, edits a runtime or stops a process.
Controller identity loss/change is recorded and terminates only this observer.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


class IdentityMismatch(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def read_json(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(.05)


def process_identity(pid):
    process = Path('/proc') / str(pid)
    try:
        stat = (process / 'stat').read_text().rsplit(')', 1)[1].split()
        command = (process / 'cmdline').read_bytes().split(b'\0')
        argv = [value.decode(errors='replace') for value in command if value]
        return dict(pid=pid, uid=process.stat().st_uid, state=stat[0], ppid=int(stat[1]),
                    pgid=int(stat[2]), starttick=int(stat[19]), argv=argv)
    except (FileNotFoundError, ProcessLookupError):
        return None


def env_markers(pid):
    raw = (Path('/proc') / str(pid) / 'environ').read_bytes()
    values = dict(item.split(b'=', 1) for item in raw.split(b'\0') if b'=' in item)
    return {key: values.get(key.encode(), b'').decode() for key in ['E1_BASE', 'E1_RUN_ID']}


class IncrementalLines:
    def __init__(self, path):
        self.path = path
        self.inode = None
        self.offset = 0
        self.rows = 0
        self.generation = 0
        self.anchor = b''

    def observe(self):
        events = []
        try:
            stream = self.path.open('rb')
        except FileNotFoundError:
            return dict(path=str(self.path), exists=False, rows=0, bytes=0,
                        previous_observed_rows=self.rows, events=['FILE_ABSENT'], generation=self.generation)
        with stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            snapshot_size = stat.st_size
            reason = None
            if self.inode is None:
                reason = 'FIRST_OBSERVATION'
            elif identity != self.inode:
                reason = 'INODE_CHANGED'
            elif snapshot_size < self.offset:
                reason = 'FILE_TRUNCATED'
            elif self.anchor:
                stream.seek(self.offset - len(self.anchor))
                if stream.read(len(self.anchor)) != self.anchor:
                    reason = 'CONTENT_REWRITTEN_BEFORE_OFFSET'
            if reason is not None:
                events.append(dict(event=reason, old_inode=self.inode, new_inode=identity,
                                   old_rows=self.rows, old_offset=self.offset, current_size=snapshot_size))
                self.rows, self.offset, self.anchor = 0, 0, b''
                self.inode = identity
                self.generation += 1
            read_bytes = 0
            stream.seek(self.offset)
            # Bound this pass to the observed file size; ongoing appends are
            # counted next time. Only complete '\n'-terminated lines count.
            while self.offset < snapshot_size:
                chunk = stream.read(min(1024 * 1024, snapshot_size - self.offset))
                if not chunk:
                    events.append(dict(event='FILE_SHRANK_DURING_READ', expected_size=snapshot_size,
                                       reached_offset=self.offset))
                    break
                self.rows += chunk.count(b'\n')
                self.offset += len(chunk)
                read_bytes += len(chunk)
            stream.seek(max(0, self.offset - 4096))
            self.anchor = stream.read(min(4096, self.offset))
        return dict(path=str(self.path), exists=True, rows=self.rows, bytes=snapshot_size,
                    bytes_read_this_sample=read_bytes, offset=self.offset,
                    device=identity[0], inode=identity[1], generation=self.generation, events=events)


def memory_summary(path, now, candidate=False):
    value = read_json(path)
    result = dict(path=str(path), updated_at_unix=value.get('updated_at_unix'),
        sample_start_unix=value.get('sample_start_unix'), sample_end_unix=value.get('sample_end_unix'),
        samples=value.get('samples'), free_measurement=value.get('free_measurement'),
        age_seconds=now-float(value['updated_at_unix']), fresh=now-float(value['updated_at_unix'])<=30)
    if value.get('free_measurement') != 'nvidia-smi memory.free':
        raise IdentityMismatch('Memory source is computed reserve instead of actual memory.free')
    if candidate:
        result.update(candidate_pid=value['candidate_pid'], source_identity_sha256=value['source_identity_sha256'],
            max_process_mib=value['max_process_mib'], min_free_mib=value['min_free_mib'], current=value['current'])
    else:
        result.update(observer=value['observer'], phase=value['phase'],
            minimum_actual_free_mib=value['minimum_actual_free_mib'],
            actual_free_minima_by_gpu_mib=value['actual_free_minima_by_gpu_mib'],
            current_actual_free_by_gpu_mib={str(row['gpu']):row['actual_free_mib'] for row in value['current']},
            peaks_mib=value['peaks_mib'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--candidate-base', type=Path, required=True)
    parser.add_argument('--controller-pid', type=int, required=True)
    parser.add_argument('--controller-starttick', type=int, required=True)
    parser.add_argument('--controller-path', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--tag', default='preparation-3-progress-20261007')
    args = parser.parse_args()
    if args.interval != 60 or '/' in args.tag or '..' in args.tag:
        raise ValueError('Reviewed observer uses a 60-second interval and a simple new tag')
    base, candidate = args.base, args.candidate_base
    receipt_dir = base/'receipts'
    sample_path = receipt_dir/(args.tag+'-samples.jsonl')
    summary_path = receipt_dir/(args.tag+'-summary.json')
    startup_path = receipt_dir/(args.tag+'-startup.json')
    observer_lock = (receipt_dir/(args.tag+'.lock')).open('a')
    fcntl.flock(observer_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if any(path.exists() for path in [sample_path,summary_path,startup_path]):
        raise RuntimeError('Observer evidence already exists; preserve it and use a reviewed new tag')
    run_path = receipt_dir/'preparation-3-run.json'
    run = read_json(run_path)
    if run.get('controller_pid') != args.controller_pid or run.get('run_id') != args.run_id or run.get('process_uid') != os.getuid():
        raise IdentityMismatch('Initial run receipt/controller UID does not match')
    config_path = Path(run['config_path'])
    if sha(config_path) != run['config_sha256']:
        raise IdentityMismatch('Initial controller configuration SHA differs')
    config = read_json(config_path)
    control_sha = config['controls'].get(str(args.controller_path))
    if not control_sha or sha(args.controller_path) != control_sha:
        raise IdentityMismatch('Initial controller source path/SHA differs from its configuration')
    current = process_identity(args.controller_pid)
    expected_argv = [str(base/'env/bin/python'),str(args.controller_path)]
    if current is None or current['uid'] != os.getuid() or current['starttick'] != args.controller_starttick or current['argv'] != expected_argv or current['state']=='Z':
        raise IdentityMismatch('Initial real controller UID/path/starttick/argv mismatch')
    expected_rollout = base/'runs/preparation-3-rollouts.jsonl'
    if config['rollout_output'] != str(expected_rollout) or config['opsd_output'] != str(base/'runs/opsd'):
        raise IdentityMismatch('Configuration is not the new preparation-3 output namespace')
    candidate_pid=run['candidate']['candidate_pid']
    candidate_digest=run['candidate']['source_identity_sha256']
    question_proofs=[]
    for rank in range(8):
        question_path=Path(str(expected_rollout)+'.parts')/f'questions-{rank}.jsonl'
        question_stat=question_path.stat()
        question_count=IncrementalLines(question_path).observe()['rows']
        question_proofs.append(dict(path=str(question_path),rank=rank,question_rows=question_count,
            expected_trajectory_rows=question_count*5,device=question_stat.st_dev,
            inode=question_stat.st_ino,size=question_stat.st_size,mtime_ns=question_stat.st_mtime_ns))
    if sum(item['question_rows'] for item in question_proofs)!=169103:
        raise IdentityMismatch('Actual preparation-3 question shards do not contain the original 169103 questions')
    started=time.time()
    startup=dict(time=started,observer_pid=os.getpid(),observer_uid=os.getuid(),
        observer_source=str(Path(__file__).resolve()),observer_source_sha256=sha(Path(__file__).resolve()),
        controller=current,run_id=args.run_id,run_receipt_path=str(run_path),run_receipt_sha256=sha(run_path),
        config_path=str(config_path),config_sha256=sha(config_path),controller_source_sha256=control_sha,
        interval_seconds=60,cpu_only=True,retrieval_requests=False,question_shards=question_proofs,
        scope='Only preparation-3 actual output bytes; no preparation-1 rows; no training mutations')
    with startup_path.open('x') as stream:json.dump(startup,stream,indent=2)
    shards=[IncrementalLines(Path(str(expected_rollout)+'.parts')/f'rollouts-{rank}.jsonl') for rank in range(8)]
    merged=IncrementalLines(expected_rollout)
    active_identity_cache={}
    last_total,last_time,last_updates=None,None,0
    with sample_path.open('x') as samples:
        def record(value):
            samples.write(json.dumps(value)+'\n');samples.flush();atomic_json(summary_path,value)
        while True:
            now=time.time()
            snapshot=dict(time=now,observer_pid=os.getpid(),controller_pid=args.controller_pid,
                          run_id=args.run_id,elapsed_seconds=now-float(run['time']))
            try:
                latest_run=read_json(run_path)
                if latest_run.get('controller_pid')!=args.controller_pid or latest_run.get('run_id')!=args.run_id:
                    raise IdentityMismatch('Controller run receipt was replaced by another PID/run')
                current=process_identity(args.controller_pid)
                if current is None or current['state']=='Z':
                    snapshot.update(event='CONTROLLER_TERMINAL',status='OBSERVER_EXITED',controller=current,
                                    reason='Original controller missing or terminal; no automatic restart')
                    record(snapshot);return
                if current['uid']!=os.getuid() or current['starttick']!=args.controller_starttick or current['argv']!=expected_argv:
                    raise IdentityMismatch('Controller UID/path/starttick/argv changed')
                if sha(config_path)!=run['config_sha256'] or sha(args.controller_path)!=control_sha:
                    raise IdentityMismatch('Active controller source or config changed')
                snapshot['controller']=current
                phase=read_json(receipt_dir/'pipeline-stage.json')
                if phase.get('controller_pid')!=args.controller_pid or phase.get('run_id')!=args.run_id:
                    raise IdentityMismatch('Stable phase pointer switched to a different controller/run')
                snapshot['phase']=phase
                child_receipt=read_json(receipt_dir/'preparation-3-active-child.json')
                if child_receipt['controller_pid']!=args.controller_pid or child_receipt['run_id']!=args.run_id or child_receipt['uid']!=os.getuid():
                    raise IdentityMismatch('Active child ownership receipt differs')
                child=process_identity(child_receipt['child_pid'])
                if child is not None and child['state']!='Z':
                    markers=env_markers(child['pid'])
                    if child['uid']!=os.getuid() or child['argv']!=child_receipt['argv'] or child['ppid']!=args.controller_pid or child['pgid']!=child_receipt['child_pgid'] or markers!=dict(E1_BASE=str(base),E1_RUN_ID=args.run_id):
                        raise IdentityMismatch('Active child actual UID/argv/parent/session/environment differs')
                    previous=active_identity_cache.get(child['pid'])
                    if previous is not None and previous!=child['starttick']:
                        raise IdentityMismatch('Active child PID was reused with a different starttick')
                    active_identity_cache[child['pid']]=child['starttick']
                snapshot['active_child']=dict(receipt=child_receipt,actual=child,
                    status='LIVE' if child is not None and child['state']!='Z' else 'TERMINAL_OR_STAGE_TRANSITION')
                rows=[]
                for rank,counter in enumerate(shards):
                    proof=question_proofs[rank]
                    stat=Path(proof['path']).stat()
                    if (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)!=(proof['device'],proof['inode'],proof['size'],proof['mtime_ns']):
                        raise IdentityMismatch('Original preparation-3 question shard identity changed')
                    row=counter.observe()
                    row.update(rank=rank,expected_rows=proof['expected_trajectory_rows'],
                        question_rows=proof['question_rows'],elapsed_seconds=snapshot['elapsed_seconds'],
                        elapsed_scope='Controller run lifetime including initialization; not generation-only time')
                    rows.append(row)
                total=sum(value['rows'] for value in rows if value['exists'])
                reset=any(any(isinstance(event,dict) and event['event'] in ['INODE_CHANGED','FILE_TRUNCATED','CONTENT_REWRITTEN_BEFORE_OFFSET','FILE_SHRANK_DURING_READ'] for event in value['events']) for value in rows)
                snapshot['rollouts']=dict(shards=rows,total_rows=total,expected_rows=845515,
                    merged=merged.observe(),source='New preparation-3 shard complete newline counts only',
                    mean_rows_per_elapsed_second=total/snapshot['elapsed_seconds'] if snapshot['elapsed_seconds']>0 else None,
                    interval_rows_per_second=(total-last_total)/(now-last_time) if last_total is not None and not reset and total>=last_total and now>last_time else None,
                    interval_reset_or_replacement=reset)
                last_total,last_time=total,now
                live_path=base/'runs/opsd/live-update.json'
                if live_path.exists():
                    update=read_json(live_path)
                    if update['source_grpo_step']!=100 or update['world_size']!=8 or update['source_model']!=config['source_model']:
                        raise IdentityMismatch('OPSD counter source/world/model does not match this experiment')
                    count=int(update['opsd_updates'])
                    if count<last_updates or update['display_step']!=100+count:
                        raise IdentityMismatch('Real OPSD counter regressed or display step is inconsistent')
                    snapshot['opsd']=dict(actual_updates=count,display_step=100+count,
                        evidence_path=str(live_path),mtime_unix=live_path.stat().st_mtime,
                        ranks=[{key:rank.get(key) for key in ['rank','updates','finite','changed_from_initial','adam_steps']} for rank in update['ranks']],
                        scope='Observed real optimizer update receipt; observer does not certify five-update acceptance')
                    last_updates=count
                else:
                    if last_updates>0:raise IdentityMismatch('Previously observed real OPSD update receipt disappeared')
                    snapshot['opsd']=dict(actual_updates=0,evidence_path=str(live_path),receipt_absent=True,
                        opsd_output_exists=(base/'runs/opsd').exists(),
                        scope='No optimizer update receipt yet; rollout/preparation or baseline may still be running')
                try:
                    with urllib.request.urlopen('http://10.120.17.101:18004/health',timeout=5) as response:health=json.load(response)
                    if health.get('pid')!=candidate_pid or health.get('source_identity_sha256')!=candidate_digest:
                        raise IdentityMismatch('Candidate health switched PID/source identity')
                    snapshot['candidate_health']=health
                except (OSError,ValueError) as error:
                    snapshot['candidate_health']=dict(observation_error=repr(error),healthy_unproven=True)
                snapshot['eight_gpu_memory_v2']=memory_summary(receipt_dir/'memory-continuous-v2.json',now)
                snapshot['candidate_gpu7_memory_v2']=memory_summary(candidate/'receipts/gpu7-memory-v2-continuous.json',now,candidate=True)
                cm=snapshot['candidate_gpu7_memory_v2']
                if cm['candidate_pid']!=candidate_pid or cm['source_identity_sha256']!=candidate_digest:
                    raise IdentityMismatch('Candidate memory evidence switched PID/source identity')
                snapshot.update(event='SAMPLE',status='OBSERVING',sample_completed_unix=time.time())
                record(snapshot)
            except IdentityMismatch as error:
                snapshot.update(event='IDENTITY_MISMATCH',status='OBSERVER_EXITED',reason=str(error))
                record(snapshot);raise
            except (OSError,ValueError,KeyError,TypeError) as error:
                snapshot.update(event='OBSERVATION_ERROR',status='CONTROLLER_STILL_UNPROVEN_THIS_SAMPLE',reason=repr(error))
                record(snapshot)
            time.sleep(max(0,60-(time.time()-now)))


if __name__=='__main__':
    main()
