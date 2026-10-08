import concurrent.futures,hashlib,json,pathlib,time,shutil,requests
api=json.loads(pathlib.Path('work/e1-300/instruct-model-public-api.json').read_text());revision=api['sha'];assert revision=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
root=pathlib.Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/bapo-tree-instruct300-aa8e725');root.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(root).free>10*1024**3
(root/'PUBLIC-SOURCE-API.json').write_text(json.dumps(api,indent=2)+'\n')
def fetch(item):
 name=item['rfilename'];assert '/' not in name
 p=root/name;tmp=root/(name+'.part');h=hashlib.sha256();g=hashlib.sha1();g.update(('blob '+str(item['size'])+'\0').encode());n=0
 if p.exists():
  b=p.read_bytes();assert len(b)==item['size'];h.update(b);g.update(b);n=len(b)
 else:
  url='https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/'+revision+'/'+name
  offset=tmp.stat().st_size if tmp.exists() else 0
  headers={'Range':f'bytes={offset}-'} if offset else {}
  r=requests.get(url,params={'download':'true','nonce':str(time.time_ns())},headers=headers,timeout=(15,60),stream=True);r.raise_for_status()
  if offset and r.status_code==206:
   assert r.headers.get('Content-Range','').startswith(f'bytes {offset}-'),r.headers.get('Content-Range')
   with tmp.open('rb') as prior:
    for chunk in iter(lambda:prior.read(2*1024**2),b''):h.update(chunk);g.update(chunk);n+=len(chunk)
  else:offset=0
  started=last=time.monotonic()
  with tmp.open('ab' if offset else 'wb') as f:
   for chunk in r.iter_content(2*1024**2):
    if not chunk:continue
    f.write(chunk);h.update(chunk);g.update(chunk);n+=len(chunk)
    if time.monotonic()-last>=10:print(json.dumps(dict(file=name,bytes=n,elapsed=time.monotonic()-started)),flush=True);last=time.monotonic()
  assert n==item['size'],(name,n,item['size'])
 sha=h.hexdigest();expected=item.get('lfs',{}).get('sha256')
 if expected:assert sha==expected,(name,sha,expected)
 else:assert g.hexdigest()==item['blobId'],(name,g.hexdigest(),item['blobId'])
 if not p.exists():tmp.rename(p)
 rec=dict(file=name,bytes=n,sha256=sha,revision=revision,expected_lfs_sha256=expected,git_blob_sha1=g.hexdigest());print(json.dumps(rec),flush=True);return rec
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 records=list(ex.map(fetch,[x for x in api['siblings'] if x['rfilename']!='.gitattributes']))
(root/'VERIFIED-MODEL-MANIFEST.json').write_text(json.dumps(dict(repo='Qwen/Qwen2.5-3B-Instruct',revision=revision,files=records),indent=2)+'\n');print('OFFICIAL_INSTRUCT_ALL_FILES_VERIFIED',flush=True)
