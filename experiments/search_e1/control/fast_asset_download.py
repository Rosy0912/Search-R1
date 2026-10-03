"""Pinned public HF assets, resumed prefixes and verified parallel HTTP byte ranges."""
import os,json,hashlib,subprocess,time,re,shutil,itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
base=Path(os.environ['E1_BASE']);assets=base/'assets';workers=int(os.environ.get('E1_DOWNLOAD_WORKERS','16'));chunk=128*1024**2
specs=[dict(kind='index',repo='wiki-18-e5-index',revision='a4d31160a035f30764604f4827cd8f1d0315eb86',name='part_aa',size=42949672960,sha='a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023'),dict(kind='index',repo='wiki-18-e5-index',revision='a4d31160a035f30764604f4827cd8f1d0315eb86',name='part_ab',size=21609402413,sha='b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11'),dict(kind='corpus',repo='wiki-18-corpus',revision='69c1c00ffe7c5554c68d8548355cb22e46aabc51',name='wiki-18.jsonl.gz',size=5123307260,sha='7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db')]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
only_ranges=os.environ.get('E1_ONLY_FETCH_RANGES')=='1'
seed=json.loads(Path(os.environ['E1_RANGE_SEED']).read_text()) if os.environ.get('E1_RANGE_SEED') else None
if seed and not only_ranges:raise RuntimeError('Remote resume metadata is only valid for range-only assistance')
groups=[]
for spec in specs:
 dest=assets/spec['kind']/spec['name'];dest.parent.mkdir(exist_ok=True,parents=True)
 if dest.exists() and dest.stat().st_size==spec['size'] and digest(dest)==spec['sha']:
  spec.update(prefix=str(dest),prefix_bytes=spec['size'],ranges=[]);groups.append([]);continue
 prefixes=list((dest.parent/'.cache/huggingface/download').glob('*.'+spec['sha']+'.incomplete'))
 if len(prefixes)>1:raise RuntimeError('Ambiguous resume prefix')
 prefix=prefixes[0] if prefixes else None;start=prefix.stat().st_size if prefix else 0
 if seed:
  original=next(x for x in seed if x['kind']==spec['kind'] and x['name']==spec['name'])
  if any(original[k]!=spec[k] for k in ['repo','revision','size','sha']):raise RuntimeError('Resume metadata does not match pinned asset')
  start=original['prefix_bytes']
 if not 0<=start<=spec['size']:raise RuntimeError('Invalid prefix size')
 spec.update(prefix=str(prefix) if prefix else None,prefix_bytes=start,ranges=[])
 folder=assets/'segmented'/spec['name'];folder.mkdir(parents=True,exist_ok=True)
 for lo in range(start,spec['size'],chunk):
  hi=min(lo+chunk,spec['size'])-1;path=folder/f'{lo}-{hi}'
  spec['ranges'].append((lo,hi,str(path)))
 groups.append([(spec,*row) for row in spec['ranges']])
(base/'receipts/segmented-inputs.json').write_text(json.dumps(specs,indent=2))
jobs=[x for row in itertools.zip_longest(*groups) for x in row if x is not None]
def range_header(path):
 if not path.exists():return None
 values=re.findall(r'(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$',path.read_text())
 return tuple(map(int,values[-1])) if values else None
def download(job):
 spec,lo,hi,filename=job;path=Path(filename);count=hi-lo+1
 if path.exists() and path.stat().st_size==count:return count,0
 temp=path.with_suffix('.part');old_headers=path.with_suffix('.headers');prefix_headers=path.with_suffix('.prefix.headers')
 expected=(lo,hi,spec['size'])
 prior=range_header(prefix_headers) or range_header(old_headers)
 n=temp.stat().st_size if temp.exists() and prior==expected else 0
 if not 0<=n<=count:raise RuntimeError('Invalid partial range size')
 if n==count:temp.replace(path);return count,0
 if n and not prefix_headers.exists():shutil.copyfile(old_headers,prefix_headers)
 if not n and temp.exists():temp.unlink()
 start=lo+n;tail=path.with_suffix('.tail');headers=path.with_suffix('.tail.headers')
 url=f"https://hf-mirror.com/datasets/PeterJinGo/{spec['repo']}/resolve/{spec['revision']}/{spec['name']}?download=true&segment={start}"
 command=['curl','-4','-fLsS','--connect-timeout','15','--max-time','900','--speed-time','60','--speed-limit','16384','--retry','1','--range',f'{start}-{hi}','--max-filesize',str(count-n),'-D',str(headers),'-o',str(tail),url]
 result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 received=tail.stat().st_size if tail.exists() else 0
 valid=range_header(headers)==(start,hi,spec['size']) and 0<received<=count-n
 if valid:
  with temp.open('ab') as out,tail.open('rb') as src:shutil.copyfileobj(src,out,8*1024**2)
  if n==0:shutil.copyfile(headers,prefix_headers)
  tail.unlink()
 if result.returncode:raise RuntimeError(f"Range {spec['name']} {lo}-{hi}: curl exit {result.returncode}: {result.stderr[-200:]}")
 if not valid or received!=count-n or temp.stat().st_size!=count:raise RuntimeError('Unexpected range response')
 temp.replace(path);return count,received
started=time.time();done=0;network_bytes=0;errors=[]
with ThreadPoolExecutor(max_workers=workers) as pool:
 futures=[pool.submit(download,job) for job in jobs]
 for f in as_completed(futures):
  try:
   complete,downloaded=f.result();done+=complete;network_bytes+=downloaded
  except Exception as error:
   errors.append(repr(error));print('RANGE_FAILED',repr(error),flush=True)
  record=dict(completed_bytes=done,new_download_bytes=network_bytes,elapsed_seconds=time.time()-started,workers=workers,range_jobs=len(jobs),completed_jobs=sum(x.done() for x in futures),status='DOWNLOADING',failed_ranges=errors)
  (base/'receipts/segmented-progress.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
if errors:raise RuntimeError('Range download failed; complete chunks preserved for resume: '+repr(errors))
if only_ranges:
 shards=[]
 for spec in specs:
  for lo,hi,filename in spec['ranges']:
   path=Path(filename);shards.append(dict(path=str(path.relative_to(assets)),bytes=hi-lo+1,sha256=digest(path)))
 (base/'receipts/range-shards.json').write_text(json.dumps(shards,indent=2))
 (base/'receipts/ranges.ready').touch();print('RANGES_ONLY_COMPLETE',flush=True);raise SystemExit(0)
from assemble_assets import assemble
manifest=json.loads((assets/'manifest.json').read_text())
index_specs=[s for s in specs if s['kind']=='index']
index=assets/'index/e5_Flat.index'
existing=manifest.get('index',{}).get('files',[])
valid_index=next((f for f in existing if f['path']=='e5_Flat.index'),None)
if valid_index and index.exists() and index.stat().st_size==valid_index['bytes'] and digest(index)==valid_index['sha256']:
 index_file=valid_index
else:
 index_file=assemble(index_specs,index)
corpus_spec=next(s for s in specs if s['kind']=='corpus')
corpus_gz=assets/'corpus/wiki-18.jsonl.gz'
if not (corpus_gz.exists() and corpus_gz.stat().st_size==corpus_spec['size'] and digest(corpus_gz)==corpus_spec['sha']):
 assemble([corpus_spec],corpus_gz)
manifest['index']=dict(repository='PeterJinGo/wiki-18-e5-index',revision=index_specs[0]['revision'],endpoint='https://hf-mirror.com',files=[index_file],source_parts=[dict(path=s['name'],bytes=s['size'],upstream_sha256=s['sha'],materialized_separately=False) for s in index_specs],note='Merged file streamed from verified source parts; merged SHA256 is locally computed, not an upstream published hash')
manifest['corpus']=dict(repository='PeterJinGo/wiki-18-corpus',revision=corpus_spec['revision'],endpoint='https://hf-mirror.com',files=[dict(path=corpus_spec['name'],bytes=corpus_spec['size'],sha256=corpus_spec['sha'],upstream_sha256=corpus_spec['sha'])])
(assets/'manifest.json').write_text(json.dumps(manifest,indent=2))
import gzip
corpus=assets/'corpus/wiki-18.jsonl'
if not corpus.exists():
 with gzip.open(assets/'corpus/wiki-18.jsonl.gz','rb') as src,corpus.with_suffix('.assembling').open('wb') as out:shutil.copyfileobj(src,out,8*1024**2)
 corpus.with_suffix('.assembling').replace(corpus)
(base/'receipts/assets.ready').touch();print('ALL_PINNED_ASSETS_READY',flush=True)
