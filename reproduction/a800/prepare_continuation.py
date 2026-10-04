"""Bind a new weights-only segment to an immutable completed G1 parent run.

The original manifest, logs, probes and checkpoints remain in their original
base. No Adam/RNG/sampler state is restored. This descriptor is not launch proof.
"""
import argparse
import json
import os
import time
from pathlib import Path
from reproduction.a800.prepare_run import prepare, digest
from reproduction.a800.owned_processes import publish


def verified_parent(parent, completed):
    assert completed == 200, 'Only the reviewed round200 recovery is supported'
    run_path=parent/'logs/run-manifest.json'
    run=json.loads(run_path.read_text())
    state=json.loads((parent/'logs/pipeline-status.json').read_text())
    assert state['run_id']==run['run_id'] and state['stage']=='FAILED'
    ledger=json.loads((parent/'logs/process-ledger.json').read_text())
    assert ledger['run_id']==run['run_id']
    assert all(job['state']=='STOPPED' for job in ledger['jobs'].values())
    # State files alone do not establish absence: audit every recorded PID birth.
    for job in ledger['jobs'].values():
        for record in job['ledger'].values():
            stat=Path('/proc')/str(record['pid'])/'stat'
            try:fields=stat.read_text().rsplit(')',1)[1].split()
            except FileNotFoundError:continue
            assert fields[0]=='Z' or int(fields[19])!=record['start_ticks'], 'Parent owned process remains live'
    for name,value in run['source_sha256'].items():
        assert digest(parent/'repo'/name)==value, 'Parent immutable source changed'
    for name,value in run['input_sha256'].items():assert digest(parent/'data'/name)==value
    from reproduction.a800.verify_continuation import segment_rows,bind_segment_versions,verify_five
    import re
    text=Path(run['train_log']).read_text(errors='replace')
    rows=segment_rows(text,run,1,completed)
    adopted=re.findall(r'SEARCH_G1_REFIT_ADOPTED=([^\s]+)',text)
    bind_segment_versions(rows,run,0,adopted,[50,100,150]);verify_five(parent,run,1)
    checkpoint=Path(run['checkpoint_dir'])/'actor'/f'global_step_{completed}'
    marker_path=checkpoint/'complete.json';marker=json.loads(marker_path.read_text())
    assert marker['run_id']==run['run_id'] and marker['step']==completed
    assert marker['finite_all_saved_tensors'] is True
    for name,value in marker['files_sha256'].items():assert digest(checkpoint/name)==value
    from verl.utils.checkpoint_integrity import validate_saved_checkpoint
    count=validate_saved_checkpoint(checkpoint,Path(run['initial_model']),run['started_unix'])
    assert count==marker['validated_tensor_count']
    names=set(marker['files_sha256'])|{'tokenizer.json','tokenizer_config.json'}
    encoder_hashes={name:digest(checkpoint/name) for name in sorted(names)}
    from reproduction.a800.tokenizer_identity import validate_tokenizer_identity
    tokenizer_identity=validate_tokenizer_identity(Path(run['initial_model']),checkpoint)
    return run,checkpoint,dict(parent_base=str(parent),parent_run_id=run['run_id'],
        parent_manifest_sha256=digest(run_path),parent_train_log_sha256=digest(Path(run['train_log'])),
        parent_checkpoint_marker_sha256=digest(marker_path),parent_checkpoint_sha256=encoder_hashes,
        parent_source_commit=run['source_commit'],tokenizer_identity=tokenizer_identity,checked_unix=time.time())


def prepare_segment(parent, source_commit):
    base=Path(os.environ['SEARCH_G1_BASE'])
    assert base.resolve()!=parent.resolve(), 'Continuation must preserve original base'
    assert not (base/'logs/run-manifest.json').exists(), 'Inspect existing segment instead of re-preparing'
    original,checkpoint,binding=verified_parent(parent,200)
    run=prepare(source_commit,publish_manifest=False)
    assert run['input_sha256']==original['input_sha256']
    assert run['initial_model_sha256']==original['initial_model_sha256']
    run.update(completed_updates=200,acceptance_update=205,actor_checkpoint=str(checkpoint),
        actor_checkpoint_sha256=binding['parent_checkpoint_sha256'],parent=binding,
        continuation_semantics='weights-only; fresh Adam; no RNG/sampler state; original KL base and LR position200; logical201..300')
    target=base/'logs/run-manifest.json';temporary=target.with_suffix('.json.new')
    with temporary.open('x') as f:
        json.dump(run,f,indent=2);f.flush();os.fsync(f.fileno())
    # Atomic exclusive publication: a failed preparation cannot resemble a fresh run.
    os.link(temporary,target);temporary.unlink()
    return run


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--source-commit',required=True)
    args=ap.parse_args();run=prepare_segment(args.parent,args.source_commit)
    print('SEARCH_G1_CONTINUATION_BOUND='+run['run_id'],flush=True)
