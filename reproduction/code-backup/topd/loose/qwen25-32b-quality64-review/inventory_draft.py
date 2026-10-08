"""Canonical shape draft from real official binding; never a verified-HF receipt."""
import argparse,json,hashlib
from pathlib import Path

def build(binding,metadata):
    if metadata['id']!='Qwen/Qwen2.5-32B' or metadata['sha']!='1818d35814b8319459f4bd55ed1ac8709630f003':raise RuntimeError('Official source mismatch')
    official={r['rfilename']:r for r in metadata['siblings']};files={}
    for name,digest in binding['model_files'].items():
        source=official[name]
        if 'lfs' in source and digest!=source['lfs']['sha256']:raise RuntimeError('Official LFS identity differs')
        files[name]=dict(size=source['size'],sha256=digest)
    shards=sorted(n for n in files if n.endswith('.safetensors'))
    required={'config.json','generation_config.json','tokenizer_config.json','tokenizer.json','merges.txt','vocab.json','model.safetensors.index.json'}
    if len(shards)!=17 or set(files)!=set(shards)|required:raise RuntimeError('Incomplete canonical17+7 draft')
    return dict(status='DRAFT_NOT_FRESH_HF_VERIFIED_NOT_TEACHER_READY',model_directory=binding['model_directory'],files=files,shards=shards,
                scope='SHAPE_ONLY_REQUIRES_ACTUAL_FULL_SHA_INDEX_HEADER_GATE')

def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    b=json.loads(Path(a.binding).read_text());m=Path(b['official_metadata_path']);expected='e546ba207f643dbe1a098323e28d45ff07942a6390bc73b8391c0acfc20d84d1'
    if hashlib.sha256(m.read_bytes()).hexdigest()!=expected or b['official_metadata_sha256']!=expected:raise RuntimeError('Official captured metadata mismatch')
    draft=build(b,json.loads(m.read_text()));draft['binding_sha256']=hashlib.sha256(Path(a.binding).read_bytes()).hexdigest()
    with Path(a.output).open('x') as f:json.dump(draft,f,indent=2)
if __name__=='__main__':main()
