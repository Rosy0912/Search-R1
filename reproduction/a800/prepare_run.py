"""Bind verified assets and immutable calibration/train/validation inputs before GPU work."""
import argparse,hashlib,json,os,time,uuid
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare(source_commit):
    base=Path(os.environ.get('SEARCH_G1_BASE','/data/search-g1'))
    target=base/'logs/run-manifest.json'
    if target.exists():
        raise RuntimeError('Run manifest already exists; inspect existing run instead of duplicating it')
    status=json.loads((base/'logs/assets-status.json').read_text())
    assert status['complete'] is True and not status.get('failed'), 'Official model hashes not verified'
    audit=json.loads((base/'data/audit.json').read_text())
    assert audit['question_disjoint'] and audit['calibration_count']==300 and audit['validation_count']==64
    assert len(set(audit['calibration_question_ids']))==300
    assert len(set(audit['validation_question_ids']))==64
    assert not set(audit['calibration_question_ids'])&set(audit['validation_question_ids'])
    model=base/'assets/qwen2.5-3b-instruct'
    index=json.loads((model/'model.safetensors.index.json').read_text())
    names=set(index['weight_map'].values())|{'model.safetensors.index.json','config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','generation_config.json'}
    model_hashes={name:digest(model/name) for name in sorted(names)}
    official=json.loads((base/'assets.json').read_text())
    assert len(official)==len(names)==9
    for item in official:
        name=Path(item['path']).name
        assert name in model_hashes and (model/name).stat().st_size==item['size']
        if item.get('sha256'):
            assert model_hashes[name]==item['sha256'], 'Official model LFS SHA mismatch'
        else:
            content=(model/name).read_bytes()
            assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==item['git_blob_sha1'], 'Official Git blob mismatch'

    files=['train.parquet','calibration-300.parquet','validation-64.parquet','audit.json']
    data_hashes={name:digest(base/'data'/name) for name in files}
    source_hashes={str(p.relative_to(base/'repo')):digest(p) for p in sorted((base/'repo').rglob('*')) if p.is_file() and p.suffix in ('.py','.sh','.yaml','.yml') and '__pycache__' not in p.parts and '.git' not in p.parts}
    result=dict(run_id=uuid.uuid4().hex,started_unix=time.time(),initial_model=str(model),initial_model_sha256=model_hashes,
        source_commit=source_commit,source_sha256=source_hashes,source_snapshot_semantics='disk fingerprint before launch; not proof of code already loaded',
        input_sha256=data_hashes,validation_question_ids=audit['validation_question_ids'],validation_source_counts=audit['validation_source_counts'],
        training_physical_gpus=[0,1,2,3],retrieval_physical_gpus=[5,6],evaluator_physical_gpus=[6],
        target_updates=300,save_interval=50,acceptance_update=5,initial_probe_version=None,
        checkpoint_dir=str(base/'checkpoints/search-g1-a800'),train_log=str(base/'logs/search-g1-training.log'),
        calibration_protocol='real closed-book EM plus evidence-deletion answer-change; both heads before GRPO')
    # Publish only a complete descriptor; exclusive creation prevents duplicate run IDs.
    with target.open('x') as f:json.dump(result,f,indent=2)
    print('SEARCH_G1_INPUTS_BOUND='+result['run_id'],flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit',required=True)
    prepare(parser.parse_args().source_commit)
