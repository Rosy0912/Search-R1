"""One-time CPU file identity audit of the exact final model view; no tensors loaded."""
import hashlib
import json
import math
import struct
import time
from pathlib import Path

B=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013')
F=B/'final51713-batch1024-v1'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def meta(path):
    s=path.stat()
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


plan=json.loads((F/'execution-plan.json').read_text())
manifest_path=Path(plan['checkpoint_manifest'])
assert digest(manifest_path)==plan['checkpoint_manifest_sha256']
manifest=json.loads(manifest_path.read_text())
derived=manifest['derived_model_view'];source_manifest=Path(derived['original_manifest_path'])
assert digest(source_manifest)==derived['original_manifest_sha256']
original=json.loads(source_manifest.read_text());assert original['identity']['completed_step']==200
assert original['identity']['guard_continue_accounting']['completed_outer_updates']==200
model=Path(manifest['hf_checkpoint']);source=Path(original['hf_checkpoint'])
canonical=Path(derived['canonical_tokenizer'])
tokenizer_files={'tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','added_tokens.json'}
files=[];dtypes={}
for path in sorted(model.rglob('*')):
    if not path.is_file():continue
    relative=path.relative_to(model).as_posix();before=meta(path)
    if relative in tokenizer_files:
        counterpart=canonical/relative;assert digest(path)==digest(counterpart)
        binding='original_training_tokenizer'
    else:
        counterpart=source/relative;assert before['device']==counterpart.stat().st_dev and before['inode']==counterpart.stat().st_ino
        binding='original200_weight_or_configuration_same_inode'
    checksum=digest(path);after=meta(path);assert before==after
    if path.suffix=='.safetensors':
        with path.open('rb') as stream:
            length=struct.unpack('<Q',stream.read(8))[0];assert 0<length<32*1024*1024
            header=json.loads(stream.read(length))
        for name,row in header.items():
            if name=='__metadata__':continue
            dtypes[row['dtype']]=dtypes.get(row['dtype'],0)+math.prod(row['shape'])
    files.append(dict(relative_path=relative,sha256=checksum,metadata=before,binding=binding))
assert any(row['relative_path'].endswith('.safetensors') for row in files)
print(json.dumps(dict(time=time.time(),run_id=plan['run_id'],checkpoint_manifest_sha256=plan['checkpoint_manifest_sha256'],
    original200_manifest_sha256=derived['original_manifest_sha256'],model_view=str(model),original_hf_checkpoint=str(source),
    files=files,safetensors_numel_by_dtype=dtypes,accepted=True,
    scope='Fresh exact input file hashes and same-inode weight binding; does not independently hash live GPU tensors')))
