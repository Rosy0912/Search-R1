import pathlib,json,base64,subprocess,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl')
out={};summaries={}
for root in r.glob('bapo-aligned-cohost-node11-v*-20261009'):
 if root.name=='bapo-aligned-cohost-node11-v11-20261009':continue
 for d in ['tools','inputs','scheduler-head']:
  for f in (root/d).rglob('*'):
   if f.is_file() and not f.is_symlink() and f.suffix in ['.py','.sh','.yaml','.yml','.json','.md','.txt'] and f.stat().st_size<1000000:
    out[str(f.relative_to(r))]=base64.b64encode(f.read_bytes()).decode()
 summaries[root.name]={'git_head':subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip(),'runs':{}}
 for run in (root/'runs').glob('*'):
  files={}
  for f in run.glob('*'):
   if f.is_file() and f.name in ['train.log','original-exit-code.txt','cleanup-exit-code.txt','resolved-config.yaml','source-commit.txt','formal-supervisor.log']:
    with f.open('rb') as stream:stream.seek(max(0,f.stat().st_size-16000));files[f.name]=stream.read().decode(errors='replace')
  summaries[root.name]['runs'][run.name]=files
print(json.dumps(dict(epoch=time.time(),files=out,summaries=summaries)))
