"""Bind one fresh continuation segment to verified source/reference/recipe."""
import hashlib
import json
from pathlib import Path
import time
import uuid
from transformers import AutoTokenizer
from verl.utils.checkpoint_integrity import validate_saved_checkpoint

BASE=Path('/data/search-r1')
SHA='f5b9e0d987a7fcb352b1614e9ce4c6b5df1c9c78f3fc83a4fbe247b9f81f7f6a'
def digest(file):
    h=hashlib.sha256()
    with file.open('rb') as reader:
        for block in iter(lambda:reader.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

if __name__=='__main__':
    source=BASE/'checkpoints/migration-source/global_step_100'
    reference=BASE/'assets/qwen2.5-3b'
    assert digest(BASE/'transfer/checkpoint-100.tar.zst')==SHA, 'Transferred archive hash mismatch'
    count=validate_saved_checkpoint(source,reference,0)
    a,b=[AutoTokenizer.from_pretrained(str(p),local_files_only=True) for p in (source,reference)]
    assert a.get_vocab()==b.get_vocab() and a.special_tokens_map==b.special_tokens_map
    config_a,config_b=[json.loads((p/'config.json').read_text()) for p in (source,reference)]
    for key in ('model_type','hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads','vocab_size','tie_word_embeddings'):
        assert config_a[key]==config_b[key], 'Checkpoint architecture differs from base'
    hashes=lambda p:{f.name:digest(f) for f in p.iterdir() if f.is_file()}
    run_id=uuid.uuid4().hex
    log=BASE/'logs'/('a800-continuation-'+run_id+'.log')
    with log.open('x') as writer: writer.write('SEARCH_R1_SEGMENT '+run_id+'\n')
    manifest=dict(run_id=run_id,run_started=time.time(),completed_updates=100,acceptance_step=105,
                  optimizer_reset=True,rng_restored=False,sampler_restored=False,
                  source_completed_updates=149,unsaved_updates_lost=49,
                  actor_checkpoint=str(source),reference_model=str(reference),
                  source_archive_sha256=SHA,provenance_verified=True,validated_source_tensors=count,
                  actor_file_hashes=hashes(source),reference_file_hashes=hashes(reference),
                  train_log=str(log),checkpoint_dir=str(BASE/'checkpoints/paper-grpo-qwen2.5-3b-a800-continuation'),
                  training_physical_gpus=[0,1,2,3],retrieval_physical_gpus=[5,6],
                  effective_batch=512,mini_batch=256,grpo_responses=5,warmup_steps=142,
                  terminal_counter=501,reward='generated answer EM only',
                  train_script_sha256=digest(BASE/'repo/reproduction/a800/train_continuation.sh'),
                  source_snapshot={str(f.relative_to(BASE/'repo')):digest(f) for f in (BASE/'repo/verl').rglob('*.py')},
                  snapshot_semantics='Disk hashes before launch; not proof of loaded modules')
    path=BASE/'logs/a800-continuation-manifest.json'
    assert not path.exists(), 'Existing segment manifest: inspect instead of duplicate launch'
    with path.open('x') as writer: json.dump(manifest,writer,indent=2)
    print('SOURCE_CANONICAL_FINITE_TOKENIZER_AND_ARCHIVE_VERIFIED',count,run_id,flush=True)
