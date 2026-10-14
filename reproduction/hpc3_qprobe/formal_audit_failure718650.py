"""Read-only failure closure for the saved native4 outer5; never a pair receipt.

The actor/head checkpoint predates the completed dev512 evaluation. A CPU
receipt cannot recreate its lost post-evaluation CUDA RNG boundary. Preserve
both facts and require a real native restore before any continuation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

ROOT = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
BASE = ROOT.parent / 'qprobe-teacher-warmup200-4gpu-jxu957-recovery718509-20261014'
JOB = '718650'
OWNER = 14090
RUN = 'c3856315d3f345d7920ebf5725a083b4'
SOURCE = '5816a20f331e10aaff9bbb95c313611ead289fd7'
TERMINAL = [JOB, 'FAILED', '1:0', 'jxu957', '14090', 'jxu957', 'formal-user',
            'emergency_acd', '2026-10-14T20:13:05', '2026-10-14T22:20:47', 'ACD1-1']
CHECKPOINT = BASE / 'checkpoints/search-g1-common8/actor/global_step_5'
ARCHIVE = BASE / ('failed-job-' + JOB)
CAPS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
CONTROL = ROOT / 'formal-failure-control' / JOB


def need(ok, message):
    if not ok:
        raise ValueError('FORMAL_AUDIT_FAILURE718650_' + message)


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    fd = os.open(str(path.parent), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def progress(value):
    target = ARCHIVE / 'progress.json'; temp = ARCHIVE / 'progress.partial'
    with temp.open('w') as stream:
        json.dump(dict(value, checked_unix=time.time()), stream, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    temp.replace(target)


def digest(path, callback=None):
    h = hashlib.sha256(); count = 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(block); count += len(block)
            if callback and count % (256 * 2**20) == 0: callback(count)
    if callback: callback(count)
    return h.hexdigest()


def terminal(raw):
    need([line.split('|') for line in raw.splitlines() if line.strip()] == [TERMINAL],
         'EXACT_FAILED_OWNER_ALLOCATION_REQUIRED')


def metadata():
    need(os.getuid() == OWNER and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
         all(os.environ.get(k) == '1' for k in CAPS), 'CPU_OWNER_AND_THREADCAPS_REQUIRED')
    raw = subprocess.check_output(['sacct', '-nP', '-X', '-j', JOB,
        '--format=JobID,State,ExitCode,User,UID,Account,QOS,Partition,Start,End,NodeList'],
        text=True, timeout=30)
    terminal(raw)
    run = read(BASE / 'logs/run-manifest.json')
    prior = read(BASE / 'preparation/training-submission.json')
    claim = read(BASE / 'preparation/training-submit-claim.json')
    need(run['run_id'] == RUN and run['source_commit'] == SOURCE and
         run['training_world_size'] == 4 and run['actor_micro_batch_per_rank'] == 4 and
         prior['job'] == JOB and prior['returncode'] == 0 and
         prior['commit'] == claim['commit'] == SOURCE, 'FAILED_SOURCE_SUBMISSION_DIFFERS')
    entry = ROOT / 'source' / SOURCE / 'reproduction/hpc3_qprobe/train.sh'
    need(claim['command'][-3:] == [str(entry), str(BASE), SOURCE], 'ORIGINAL_ENTRY_DIFFERS')
    for arg in ('--account=jxu957', '--qos=formal-user', '--partition=emergency_acd',
                '--gres=gpu:4', '--cpus-per-task=48', '--mem=512G', '--chdir=' + str(BASE)):
        need(arg in claim['command'], 'ORIGINAL_RESOURCE_BINDING_DIFFERS')
    stage = read(BASE / 'logs/pipeline-status.json')
    need(stage['stage'] == 'FAILED' and stage['run_id'] == RUN and
         stage['error'] == "AssertionError('trainer failed; see project log')", 'NEW_FAILURE_DIAGNOSE_FIRST')
    log = (BASE / 'logs/search-g1-common8-training.log').read_text()
    need('formal_attempt_audit.py", line 264' in log and
         'Inconsistent report arithmetic: actual global masked-mean weight mass' in log,
         'ACTUAL_AUDIT_FAILURE_MISSING')
    need(not any(x in log for x in ('torch.OutOfMemoryError', 'CUDA error:',
                                   'QPROBE_POLICY_SNAPSHOT_CHANGED')), 'ADDITIONAL_TRAINING_FAILURE')
    clean = read(BASE / 'logs/cleanup-status.json')
    ledger = read(BASE / 'logs/process-ledger.json')
    need(clean['run_id'] == RUN and clean['cleaned'] is True and ledger['run_id'] == RUN and
         all(row['state'] == 'STOPPED' for row in ledger['jobs'].values()), 'OWNED_CLEANUP_INCOMPLETE')
    for name in ('corpus-cache-cleanup.json', 'retrieval-staging-cleanup.json'):
        row = read(BASE / 'logs' / name)
        need(row['run_id'] == RUN and row['slurm_job_id'] == JOB and row['cleaned'] is True,
             'PRIVATE_TMPFS_CLEANUP_INCOMPLETE')
    updates = [json.loads(line) for line in (BASE / 'logs/actual-updates.jsonl').read_text().splitlines()]
    attempts = [json.loads(line) for line in (BASE / 'logs/actual-attempts.jsonl').read_text().splitlines()]
    need([row['step'] for row in updates] == list(range(1, 6)) and
         [row['attempt_index'] for row in attempts] == list(range(1, 6)) and
         len({row['attempt_id'] for row in attempts}) == 5, 'SAVED_PREFIX_CHANGED')
    need(not (BASE / 'artifacts/question-probe/completed-pairs/global_step_5').exists(),
         'PAIR_PUBLICATION_ALREADY_ATTEMPTED_INSPECT')
    complete = read(CHECKPOINT / 'complete.json')
    need(complete['run_id'] == RUN and complete['step'] == 5 and
         complete['validated_tensor_count'] == 435 and complete['finite_all_saved_tensors'] is True and
         complete['actual_optimizer_writes_per_rank'] == 100, 'CHECKPOINT5_INCOMPLETE')
    actual_monitor = []
    for rank in range(4):
        rows = [json.loads(line) for line in (BASE / 'logs' / ('optimizer-rank-%d.jsonl' % rank)).read_text().splitlines()]
        budgets = [row for row in rows if row['kind'] == 'OUTER_BUDGET']
        monitors = [row for row in rows if row['kind'] == 'POLICY_MONITOR']
        need(len(budgets) == 5 and len(monitors) == 200 and budgets[-1]['cumulative_writes'] == 100 and
             all(row['local_micro'] == 4 and row['local_mini'] == 32 and
                 row['local_trajectories'] == 640 for row in budgets), 'ACTUAL_NATIVE4_BUDGET_DIFFERS')
        actual_monitor.append(dict(rank=rank,monitor_count=len(monitors),
            observed_weight_masses=sorted({row['effective_weight_mass'] for row in monitors}),
            last_budget=budgets[-1]))
    return dict(job=JOB,run_id=RUN,source_commit=SOURCE,actual_accounting=raw,
        base=str(BASE),checkpoint=str(CHECKPOINT),updates=updates,attempts=attempts,
        monitors=actual_monitor,actual_checkpoint_head_version=complete['question_probe']['head_version'],
        actor_head_completed_outers=5,first_five_acceptance_complete=False,
        completed_pair_available=False,checkpoint_rng_boundary='pre-dev512-after5',
        post_evaluation_rng_boundary_available=False,actual_cuda_restore_accepted=False,
        owned_components_stopped=True,private_tmpfs_cleaned=True,checked_unix=time.time())


def verify_saved_prefix():
    """CPU readback of real saved states and the repaired original reader chain."""
    import torch
    import numpy as np
    from search_g1.full_state_restore import equal_tree
    from search_g1.qprobe_checkpoint import validate_rank_state, _validate_qstate
    from search_g1.qprobe_model import OnlineQProbe
    from reproduction.common8.behavior_runtime import read_gate_receipt
    from reproduction.common8.runtime import verify_scheduler_receipts
    from reproduction.common8.formal_inline_safety import verify_first_five
    from reproduction.common8.formal_inline_evaluation import verify_pair
    from reproduction.hpc3_qprobe.verify_goal import verify_selection
    run=read(BASE/'logs/run-manifest.json')
    rows=[json.loads(line) for line in (BASE/'logs/actual-updates.jsonl').read_text().splitlines()]
    gates={str(row['step']):read_gate_receipt(BASE,run,row['step'],update_record=row) for row in rows}
    schedulers=verify_scheduler_receipts(BASE,run,1,5,gates)
    safety=verify_first_five(BASE,run,gates,schedulers)
    need(safety['structural_accepted'] is True and safety['actual_complete_outers']==5 and
         safety['actual_optimizer_writes_per_rank']==100,'ACTUAL_STRUCTURAL_PREFIX_REJECTED')
    development=verify_pair(BASE,run)
    publish(ARCHIVE/'corrected-reader-prefix.json',dict(safety=safety,development=development,
        original_source_commit=SOURCE,helper_source=str(Path(__file__).resolve().parents[2]),
        original_raw_evidence_modified=False,actual_cuda_restore_accepted=False,checked_unix=time.time()))
    rank_proofs=[]
    for rank in range(4):
        progress(dict(stage='CPU_SAVED_RANK_STATE',rank=rank,ranks_done=len(rank_proofs)))
        saved=torch.load(CHECKPOINT/'training_state'/('rank-%d.pt'%rank),map_location='cpu',weights_only=False)
        need(saved['run_id']==RUN and saved['source_commit']==SOURCE,'SAVED_WORKER_SOURCE_DIFFERS')
        proof=validate_rank_state(saved,rank,4,5,expected_writes=100)
        need(set(saved['used_attempt_ids'])=={row['attempt_id'] for row in rows},'SAVED_ATTEMPT_PREFIX_DIFFERS')
        need(saved['rng']['vllm_version']=='0.6.3' and
             saved['rng']['boundary']=='idle synchronous generation engine','SAVED_WORKER_RNG_BOUNDARY_DIFFERS')
        rank_proofs.append(proof);del saved
    progress(dict(stage='CPU_HEAD_DRIVER_SELECTOR_READBACK',ranks_done=4))
    qstate,qpath,head_path=_validate_qstate(CHECKPOINT,5,RUN+':5')
    head=OnlineQProbe.load(head_path,expected_actor_snapshot_id=RUN+':5',
        expected_feature_spec_sha=qstate['feature_spec_sha'],expected_checkpoint_sha=qstate['head_sha256'])
    raw_head=torch.load(head_path,map_location='cpu',weights_only=True)
    equal_tree(head.state_dict(),raw_head)
    need(head.completed_outer==head.update_count==5 and len(head.replay_labels)==7680,
         'HEAD_FIVE_COMMITS_OR_ALL_WARMUP_REPLAY_DIFFERS')
    driver=torch.load(CHECKPOINT/'training_state/driver.pt',map_location='cpu',weights_only=False)
    need(driver['completed_outer']==5 and driver['probe_version']==qstate['reward_probe_version'],
         'SAVED_DRIVER_ACTOR_PROBE_DIFFERS')
    equal_tree(driver['candidate_stream'],qstate['candidate_state'])
    need(driver['attempt_progress']==dict(completed_outer=5,cumulative_writes=100,
        attempt_index=5,aborted_attempts=0),'SAVED_DRIVER_ATTEMPT_PROGRESS_DIFFERS')
    rng=np.random.default_rng();rng.bit_generator.state=qstate['selection_rng']
    equal_tree(rng.bit_generator.state,qstate['selection_rng'])
    selections=[]
    for row in rows:
        directory=BASE/'artifacts/question-probe/attempts'/row['attempt_id']
        selection=read(directory/'selection.json');rollout=read(directory/'rollout-audit.json')
        event=read(BASE/'artifacts/question-probe/completed'/(row['attempt_id']+'.json'))
        need(selection['step']==rollout['step']==event['outer']==row['step'] and
             selection['head_version']==event['prediction_version'] and
             event['head_update']['completed_outer']==event['head_update']['update_count']==row['step'],
             'ACTUAL_PREQUENTIAL_HEAD_ORDER_DIFFERS')
        measured=verify_selection(selection,rollout,run['question_probe']['seed'])
        selections.append(dict(step=row['step'],attempt_id=row['attempt_id'],head_version=event['head_version'],
            selection_before_labels_verified=True,prequential=measured))
    need(selections[-1]['head_version']==qstate['head_version'],'SAVED_HEAD_LAST_COMMIT_DIFFERS')
    result=dict(status='ACTUAL_SAVED_NATIVE4_CHECKPOINT5_AND_CORRECTED_STRUCTURAL_PREFIX_CPU_PASS',
        run_id=RUN,source_commit=SOURCE,ranks=rank_proofs,head_version=head.version,
        head_replay_groups=len(head.replay_labels),head_private_rng_cpu_readback_equal=True,
        head_adam_model_replay_cpu_readback_equal=True,selector_rng_cpu_readback_equal=True,
        driver_candidate_saved_state_equal=True,candidate_source_cursor=driver['candidate_stream']['consumed_source_rows'],
        corrected_first_five_structural_reader_accepted=True,selection_audits=selections,
        actual_constructed_stream_restored=False,actual_cuda_restore_accepted=False,
        completed_pair_available=False,post_evaluation_rng_boundary_available=False,
        actual_checkpoint_rng_boundary='pre-dev512-after5',checked_unix=time.time())
    publish(ARCHIVE/'saved-prefix-cpu-validation.json',result)
    return result


def archive():
    value = metadata()
    need(not ARCHIVE.exists(), 'ALREADY_ATTEMPTED_INSPECT_DO_NOT_REPLAY')
    ARCHIVE.mkdir(mode=0o770)
    publish(ARCHIVE / 'terminal.json', value)
    inventory = []
    for folder in ('logs', 'preparation', 'artifacts', 'teacher-warmup-cache', 'checkpoints'):
        for path in sorted((BASE / folder).rglob('*')):
            s = path.lstat()
            need(not stat.S_ISLNK(s.st_mode), 'SYMLINK_EVIDENCE_REQUIRES_SEPARATE_AUDIT')
            if not stat.S_ISREG(s.st_mode): continue
            need(s.st_uid == OWNER, 'NONOWNED_EVIDENCE')
            inventory.append(dict(path=str(path),relative_path=str(path.relative_to(BASE)),
                inode=s.st_ino,device=s.st_dev,bytes=s.st_size,mtime_ns=s.st_mtime_ns))
    publish(ARCHIVE / 'inventory.json',dict(files=inventory,bytes=sum(row['bytes'] for row in inventory)))
    files = []; verified = 0
    for index,row in enumerate(inventory):
        path = Path(row['path'])
        def report(count):
            progress(dict(stage='FULL_SHA_READ',files_done=index,total_files=len(inventory),
                completed_file_bytes=verified,current_file=row['relative_path'],current_file_bytes_read=count,
                full_read_bytes=verified+count))
        sha = digest(path, report)
        s = path.stat()
        need((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns) ==
             (row['device'],row['inode'],row['bytes'],row['mtime_ns']), 'EVIDENCE_CHANGED_DURING_READ')
        if row['bytes'] <= 16*2**20:
            target = ARCHIVE / 'small-evidence' / row['relative_path']
            target.parent.mkdir(parents=True,exist_ok=True)
            with path.open('rb') as source,target.open('xb') as dest:
                shutil.copyfileobj(source,dest,8*2**20);dest.flush();os.fsync(dest.fileno())
            need(digest(target) == sha, 'INDEPENDENT_SMALL_COPY_SHA_DIFFERS')
        files.append(dict(row,full_read_sha256=sha)); verified += row['bytes']
    complete = read(CHECKPOINT / 'complete.json')
    by_name = {str(Path(row['path']).relative_to(CHECKPOINT)): row for row in files
               if Path(row['path']).is_relative_to(CHECKPOINT)}
    for name,sha in complete['files_sha256'].items():
        need(by_name[name]['full_read_sha256'] == sha, 'ORIGINAL_CHECKPOINT_SHA_DIFFERS')
    need(set(by_name) == {str(p.relative_to(CHECKPOINT)) for p in CHECKPOINT.rglob('*') if p.is_file()},
         'CHECKPOINT_FILE_SET_CHANGED')
    saved_prefix=verify_saved_prefix()
    publish(ARCHIVE / 'terminal-and-files.json',dict(value,files=files,full_read_bytes=verified,
        checkpoint_file_count=len(by_name),large_files_retained_in_place=True,
        status='ACTUAL_FAILED_NATIVE4_CHECKPOINT5_AND_RAW_EVIDENCE_FULL_SHA_PRESERVED',
        cpu_state_validation_pending=False,saved_prefix_cpu_validation=saved_prefix,checked_unix=time.time()))
    progress(dict(stage='COMPLETE',files_done=len(files),full_read_bytes=verified,
                  checkpoint_file_count=len(by_name),actual_cuda_restore_accepted=False))
    return dict(status='FAILED_NATIVE4_EVIDENCE_PRESERVED',files=len(files),full_read_bytes=verified,
                checkpoint_file_count=len(by_name),actual_cuda_restore_accepted=False)


def start(commit):
    metadata()
    need(isinstance(commit,str) and len(commit)==40 and all(c in '0123456789abcdef' for c in commit),
         'PUBLISHED_HELPER_COMMIT_REQUIRED')
    source = ROOT / 'source' / commit
    proof = read(ROOT / 'preparation' / ('source-' + commit + '-verification.json'))
    need(proof['commit']==commit and proof['published_remote_sha']==commit and
         proof['status']=='PUBLISHED_SOURCE_FULL_SHA_PASS', 'HELPER_PUBLICATION_NOT_VERIFIED')
    for name,sha in proof['files'].items():
        need(digest(source/name)==sha, 'PUBLISHED_HELPER_SOURCE_CHANGED')
    need(not ARCHIVE.exists() and not CONTROL.exists(), 'EXISTING_OUTCOME_INSPECT_DO_NOT_REPLAY')
    CONTROL.mkdir(mode=0o770,parents=True)
    command=[str(BASE/'env/bin/python'),'-m','reproduction.hpc3_qprobe.formal_audit_failure718650','supervise']
    publish(CONTROL/'claim.json',dict(job=JOB,failed_source=SOURCE,helper_commit=commit,
        command=command,cwd=str(source),verified_uid=os.getuid(),checked_unix=time.time(),
        no_training=True,no_gpu_allocation=True,no_model_reload=True,no_submission=True))
    env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES='',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             PYTHONUNBUFFERED='1',QPROBE_SOURCE=str(source),SEARCH_G1_BASE=str(BASE),
             SEARCH_G1_PROFILE='common8')
    with (CONTROL/'output.log').open('xb') as stream:
        child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    from reproduction.a800.owned_processes import identity
    birth=identity(child.pid)
    need(birth['uid']==OWNER and birth['pid']==birth['sid'], 'DETACHED_OWNER_BIRTH_DIFFERS')
    publish(CONTROL/'process.json',dict(pid=child.pid,birth=birth,helper_commit=commit,
        checked_unix=time.time(),scope='actual detached CPU process; full read completion remains pending'))
    return dict(pid=child.pid,birth=birth,control=str(CONTROL),helper_commit=commit)


def supervise():
    from reproduction.a800.owned_processes import identity
    claim=read(CONTROL/'claim.json')
    need(claim['job']==JOB and Path(claim['cwd']).resolve()==Path.cwd().resolve(), 'SUPERVISOR_SOURCE_DIFFERS')
    birth=identity(os.getpid());code=0;result=None;error=None
    publish(CONTROL/'started.json',dict(birth=birth,checked_unix=time.time()))
    try: result=archive()
    except Exception as exc:
        code=1;error=repr(exc)
        if ARCHIVE.is_dir():progress(dict(stage='FAILED',error=error))
        import traceback
        traceback.print_exc()
    publish(CONTROL/'result.json',dict(exit_code=code,result=result,error=error,
        birth=birth,checked_unix=time.time(),scope='actual CPU archive return; no CUDA restore acceptance'))
    return code


def inspect():
    need(os.getuid()==OWNER,'ACTUAL_OWNER_REQUIRED')
    claim=read(CONTROL/'claim.json');record=read(CONTROL/'process.json')
    need(record['helper_commit']==claim['helper_commit'] and claim['job']==JOB,'CONTROLLER_BINDING_DIFFERS')
    from reproduction.a800.owned_processes import identity
    try: actual=identity(record['pid'])
    except (FileNotFoundError,ProcessLookupError): actual=None
    same=actual is not None and all(actual[k]==record['birth'][k] for k in ('pid','uid','sid','start_ticks'))
    result=read(CONTROL/'result.json') if (CONTROL/'result.json').is_file() else None
    if result is not None:
        need(all(result['birth'][k]==record['birth'][k] for k in ('pid','uid','sid','start_ticks')),
             'ACTUAL_RETURN_BIRTH_DIFFERS')
    return dict(checked_unix=time.time(),pid=record['pid'],same_live_birth=same and actual['state']!='Z',
        actual_identity=actual,helper_commit=claim['helper_commit'],result=result,
        progress=read(ARCHIVE/'progress.json') if (ARCHIVE/'progress.json').is_file() else None,
        actual_cuda_restore_accepted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('mode',choices=('archive','metadata','start','supervise','inspect'))
    p.add_argument('--commit')
    a = p.parse_args()
    try:
        if a.mode=='supervise':sys.exit(supervise())
        result={'archive':archive,'metadata':metadata,'inspect':inspect}.get(a.mode)
        print(json.dumps(result() if result else start(a.commit)),flush=True)
    except Exception as error:
        if a.mode=='archive' and ARCHIVE.is_dir():progress(dict(stage='FAILED',error=repr(error)))
        raise
