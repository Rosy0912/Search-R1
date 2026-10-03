"""Download public source assets once in the selected cluster; pin revisions."""
import os,json,hashlib,gzip,shutil
from pathlib import Path
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','60')
from huggingface_hub import HfApi,snapshot_download,hf_hub_download
base=Path(os.environ['E1_BASE']);assets=base/'assets';assets.mkdir(exist_ok=True)
api=HfApi();manifest={}
requests=[('model','Qwen/Qwen2.5-3B-Instruct','model',None),('encoder','intfloat/e5-base-v2','model',None),('datasets','RUC-NLPIR/FlashRAG_datasets','dataset',[f'{s}/*.jsonl' for s in ['nq','hotpotqa','triviaqa','popqa','2wikimultihopqa','musique','bamboogle']]),('index','PeterJinGo/wiki-18-e5-index','dataset',['part_aa','part_ab']),('corpus','PeterJinGo/wiki-18-corpus','dataset',['wiki-18.jsonl.gz'])]
for name,repo,kind,patterns in requests:
 info=api.repo_info(repo,repo_type=kind,files_metadata=True);revision=info.sha
 print('DOWNLOAD',name,repo,revision,flush=True)
 dest=assets/name
 snapshot_download(repo,repo_type=kind,revision=revision,local_dir=str(dest),allow_patterns=patterns,ignore_patterns=['*.bin','*.h5','*.onnx','*.msgpack'],max_workers=4)
 records=[]
 for entry in info.siblings:
  path=dest/entry.rfilename
  if not path.is_file():continue
  h=hashlib.sha256()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  digest=h.hexdigest();expected=entry.lfs.sha256 if getattr(entry,'lfs',None) else None
  if expected and digest!=expected:raise RuntimeError('LFS hash mismatch: '+entry.rfilename)
  records.append(dict(path=entry.rfilename,bytes=path.stat().st_size,sha256=digest,upstream_sha256=expected))
 manifest[name]=dict(repository=repo,revision=revision,endpoint=os.environ['HF_ENDPOINT'],files=records)
 (assets/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print('VERIFIED',name,len(records),flush=True)
index=assets/'index/e5_Flat.index'
if not index.exists():
 with index.with_suffix('.part').open('wb') as out:
  for name in ['part_aa','part_ab']:
   with (assets/'index'/name).open('rb') as f:shutil.copyfileobj(f,out,8*1024*1024)
 index.with_suffix('.part').replace(index)
corpus=assets/'corpus/wiki-18.jsonl'
if not corpus.exists():
 with gzip.open(assets/'corpus/wiki-18.jsonl.gz','rb') as f,corpus.with_suffix('.part').open('wb') as out:shutil.copyfileobj(f,out,8*1024*1024)
 corpus.with_suffix('.part').replace(corpus)
(base/'receipts/assets.ready').touch();print('ALL_ASSETS_READY',flush=True)
