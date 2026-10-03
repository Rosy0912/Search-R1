"""Verify base against immutable source8e assets, not a newly self-labelled receipt.
The original manifest is exact git-show source8e:reproduction/h20/assets.json.
"""
import hashlib,json,re
from pathlib import Path
TRAINING_ASSETS_SHA='a033b9d6e5b4453374c25d8d9a33d70254c1a79ea0d71677f21afb083d94b3fd'
MODEL_REVISION='3aab1f1954e9cc14eb9509a215f9e5ca08227a9b'
MODEL_FILES={'config.json','generation_config.json','merges.txt','model-00001-of-00002.safetensors','model-00002-of-00002.safetensors','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','vocab.json'}

def original_base_records():
 path=Path(__file__).parent/'training-assets-8e.json';raw=path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=TRAINING_ASSETS_SHA:raise ValueError('Original source8e asset manifest bytes changed')
 records=[r for r in json.loads(raw) if r['repo']=='Qwen/Qwen2.5-3B']
 if len(records)!=9 or {r['file'] for r in records}!=MODEL_FILES:raise ValueError('Original base asset set differs')
 for r in records:
  if r['kind']!='model' or r['revision']!=MODEL_REVISION or r['path']!='qwen2.5-3b/'+r['file']:raise ValueError('Original base asset source differs')
 return records

def verify_original_record(path,record):
 """Stream full SHA256 and original Git blob content identity, as original valid()."""
 if path.stat().st_size!=record['size']:raise ValueError('Base asset differs from original size')
 sha=hashlib.sha256();blob=hashlib.sha1(('blob '+str(record['size'])+'\0').encode())
 with path.open('rb') as handle:
  for data in iter(lambda:handle.read(8*1024*1024),b''):sha.update(data);blob.update(data)
 actual=sha.hexdigest()
 if record['sha256'] is not None:
  if not re.fullmatch('[0-9a-f]{64}',record['sha256']) or actual!=record['sha256']:raise ValueError('Base asset differs from original full SHA')
 elif not re.fullmatch('[0-9a-f]{40}',record['git_blob_sha1']) or blob.hexdigest()!=record['git_blob_sha1']:raise ValueError('Base asset differs from original Git blob')
 return actual

def verify_base_ancestry(role):
 directory=Path(role['directory']);records=original_base_records()
 if set(role['files'])!=MODEL_FILES:raise ValueError('Base role must have exact original nine assets')
 for record in records:
  path=directory/record['file']
  if not path.resolve().is_relative_to(directory.resolve()):raise ValueError('Base original asset escapes role directory')
  actual=verify_original_record(path,record)
  if role['files'][record['file']]!=actual:raise ValueError('Base receipt differs from original content')
 return {'training_assets_sha256':TRAINING_ASSETS_SHA,'model_revision':MODEL_REVISION,'original_files':9}
