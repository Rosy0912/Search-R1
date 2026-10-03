"""Preserve a complete nonfinite update300 snapshot, then stop only its owned runtime.

This never turns failed tensor validation into acceptance, restarts training, or
operates before the original stop monitor has suspended the update300 coordinator.
"""
import argparse,fcntl,hashlib,json,os,signal,time
from pathlib import Path
from reproduction.a800.stop_at_checkpoint import identity,matches,send,logged_updates,owned_session_records


def failed_snapshot(directory, canonical, started):
    import torch
    from safetensors import safe_open
    saved=json.loads((directory/'model.safetensors.index.json').read_text())['weight_map']
    expected=json.loads((canonical/'model.safetensors.index.json').read_text())['weight_map']
    extra=set(saved)-set(expected)
    if extra:
        configs=[json.loads((p/'config.json').read_text()) for p in (directory,canonical)]
        assert extra=={'lm_head.weight'} and 'model.embed_tokens.weight' in expected
        assert all(c.get('model_type')=='qwen2' and c.get('tie_word_embeddings') is True for c in configs)
    assert expected and set(saved)-extra==set(expected), 'Incomplete snapshot keys'
    shapes={}
    for file in set(expected.values()):
        with safe_open(str(canonical/file),framework='pt',device='cpu') as reader:
            for key in (k for k,v in expected.items() if v==file):shapes[key]=tuple(reader.get_slice(key).get_shape())
    if extra:shapes['lm_head.weight']=shapes['model.embed_tokens.weight']
    files=set(saved.values())|{'config.json','model.safetensors.index.json'}
    for file in files:
        assert not Path(file).is_absolute() and '..' not in Path(file).parts
        p=directory/file;assert p.is_file() and p.stat().st_size>0 and p.stat().st_mtime>=started
    bad=[];count=0;hashes={}
    for file in set(saved.values()):
        with safe_open(str(directory/file),framework='pt',device='cpu') as reader:
            keys={k for k,v in saved.items() if v==file};assert set(reader.keys())==keys
            for key in keys:
                assert tuple(reader.get_slice(key).get_shape())==shapes[key]
                tensor=reader.get_tensor(key)
                if not torch.isfinite(tensor).all():bad.append(key)
                count+=1;del tensor
    assert bad, 'This cleanup path requires actual nonfinite tensors, not merely bad metrics'
    for file in sorted(files):
        h=hashlib.sha256()
        with (directory/file).open('rb') as reader:
            for chunk in iter(lambda:reader.read(8*1024*1024),b''):h.update(chunk)
        hashes[file]=h.hexdigest()
    return dict(checkpoint=str(directory),saved_tensor_count=count,serialized_keys_shapes_complete=True,
                finite_all_saved_tensors=False,nonfinite_tensor_keys=sorted(bad),file_sha256=hashes,
                accepted=False,usable_for_resume=False)


def atomic(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(data, stream, indent=2); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def verify_files(proof):
    for name, expected in proof['file_sha256'].items():
        h=hashlib.sha256()
        with (Path(proof['checkpoint'])/name).open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
        assert h.hexdigest()==expected, 'Preserved checkpoint changed'


def cleanup(result, logs, log):
    task=result['task_identity'];driver=result['driver_identity'];leader=result['session_leader_identity']
    ledger={r['pid']:r for r in result['owned_member_ledger']}
    def remaining():
        live=owned_session_records(driver['sid'],driver['uid'])
        # Never acquire new identities after the bound session leader disappears.
        if matches(leader):
            now=identity(leader['pid'])
            assert now['command']==leader['command'], 'Session leader exec changed'
            for r in live:
                old=ledger.get(r['pid'])
                if old:
                    assert all(r[k]==old[k] for k in ('start_ticks','sid','uid','command')), 'Recorded member identity changed'
                else: ledger[r['pid']]=r
            result['owned_member_ledger']=list(ledger.values())
            atomic(logs/'r1-update300-failed-snapshot.json',result)
        for r in live:
            old=ledger.get(r['pid'])
            assert old and all(r[k]==old[k] for k in ('start_ticks','sid','uid','command')), 'Unbound session member requires inspection'
        return live
    # A partially completed cleanup is resumable using the recorded identities.
    if matches(task):
        assert identity(task['pid'])['command']==task['command']
        send(task,signal.SIGKILL)
    if matches(driver):
        assert identity(driver['pid'])['command']==driver['command']
        send(driver,signal.SIGINT)
    end=time.monotonic()+20
    while time.monotonic()<end and remaining(): time.sleep(.5)
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for record in remaining():
            try: send(record,sig)
            except ProcessLookupError: pass
        time.sleep(2)
    assert not remaining()
    assert logged_updates(log.read_text(errors='replace'))==result['completed_updates']
    result.update(state='STOPPED_AFTER_REJECTED_NONFINITE_CHECKPOINT',remaining_owned_processes=[],stopped_unix=time.time())
    atomic(logs/'r1-update300-failed-snapshot.json',result)
    atomic(logs/'r1-stop300-status.json',result)
    (logs/'a800-pipeline-stage.txt').write_text('STOPPED_UPDATE300_NONFINITE_CHECKPOINT_REJECTED\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,default=Path('/data/search-r1'));parser.add_argument('--run-id',required=True)
    args=parser.parse_args();logs=args.base/'logs'
    lock=(logs/'stop-failed-boundary.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    proof_path=logs/'r1-update300-failed-snapshot.json'
    while True:
        state=json.loads((logs/'r1-stop300-status.json').read_text())
        assert state['run_id']==args.run_id
        if state['state'] in ('STOPPED_AFTER_VERIFIED_CHECKPOINT','STOPPED_AFTER_REJECTED_NONFINITE_CHECKPOINT'):return
        if proof_path.exists():
            saved=json.loads(proof_path.read_text());assert saved['run_id']==args.run_id
            if saved['state'] in ('CLEANUP_PENDING_NONFINITE_CHECKPOINT','CLEANUP_FAILED_NONFINITE_CHECKPOINT','STOPPED_AFTER_REJECTED_NONFINITE_CHECKPOINT'):
                result=saved; break
        if state['state']=='CHECKPOINT_REJECTED_COORDINATOR_LEFT_SUSPENDED':
            result=None;break
        if state['state']=='MONITOR_FAILED_REQUIRES_INSPECTION':raise RuntimeError('Original monitor failed outside nonfinite gate')
        time.sleep(2)
    manifest=json.loads((logs/'a800-continuation-manifest.json').read_text());assert manifest['run_id']==args.run_id
    log=Path(manifest['train_log'])
    assert log.read_text(errors='replace').splitlines()[0]=='SEARCH_R1_SEGMENT '+args.run_id
    if result is None:
        assert 'Nonfinite checkpoint tensor' in state['error']
        for _ in range(100):
            if not matches(state['monitor_identity']):break
            time.sleep(.1)
        assert not matches(state['monitor_identity'])
        records=[state[k] for k in ('task_identity','driver_identity','session_leader_identity')]
        for r in records:
            assert matches(r) and identity(r['pid'])['command']==r['command']
        task,driver,leader=records
        assert task['sid']==driver['sid']==leader['pid'] and task['uid']==driver['uid']==leader['uid']==os.getuid()
        assert task['sid']!=os.getsid(0) and identity(task['pid'])['state']=='T'
        assert task['command'].startswith('ray::main_task')
        assert '-m verl.trainer.main_ppo ' in driver['command'] and str(args.base/'checkpoints') in driver['command']
        assert str(args.base/'repo/reproduction/a800/run_continuation.sh') in leader['command']
        steps=logged_updates(log.read_text(errors='replace'));assert steps==list(range(manifest['completed_updates']+1,301))
        checkpoint=Path(manifest['checkpoint_dir'])/'actor/global_step_300'
        proof=failed_snapshot(checkpoint,Path(manifest['reference_model']),manifest['run_started'])
        result=dict(run_id=args.run_id,target_update=300,state='CLEANUP_PENDING_NONFINITE_CHECKPOINT',
                    completed_updates=steps,checkpoint_evidence=proof,original_error=state['error'],
                    task_identity=task,driver_identity=driver,session_leader_identity=leader,
                    owned_member_ledger=owned_session_records(driver['sid'],driver['uid']),
                    accepted=False,preserved_unix=time.time(),next_rollout_may_have_started=True)
        atomic(proof_path,result)
    assert result['accepted'] is False and result['checkpoint_evidence']['finite_all_saved_tensors'] is False
    assert result['completed_updates']==list(range(manifest['completed_updates']+1,301))
    assert logged_updates(log.read_text(errors='replace'))==result['completed_updates']
    verify_files(result['checkpoint_evidence'])
    try: cleanup(result,logs,log)
    except Exception as error:
        result.update(state='CLEANUP_FAILED_NONFINITE_CHECKPOINT',cleanup_error=repr(error))
        atomic(proof_path,result);atomic(logs/'r1-stop300-status.json',result)
        raise
    print(json.dumps(dict(state=result['state'],accepted=False)),flush=True)

if __name__=='__main__':main()
