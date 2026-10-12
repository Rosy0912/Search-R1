import pathlib,json,os,time,shutil,subprocess
PLAN = None
HF_RECEIPT = None
GITHUB_SHA = None
assert HF_RECEIPT['status']=='ALL_SELECTED_HF_FILES_SHA_VERIFIED'
base=pathlib.Path('/dataset/shaoshitong/search-r1-rl')
paths=[pathlib.Path(x['path']) for x in PLAN['delete']]
def under(path,root):
 try:path.relative_to(root);return True
 except ValueError:return False
# Abort if any process has a BAPO command, cwd, or open file in candidate paths.
refs=[]
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'bapo-aligned-cohost-node11-' in cmd and ('python' in cmd or 'ray' in cmd or 'train' in cmd):
   if 'cleanup_execute' not in cmd and not 'required_hf_status' in cmd:refs.append(dict(pid=proc.name,kind='cmdline',detail=cmd[:300]))
  links=[proc/'cwd']+list((proc/'fd').glob('*'))
  for link in links:
   try:dest=pathlib.Path(os.readlink(link))
   except OSError:continue
   if any(under(dest,d) for d in paths):refs.append(dict(pid=proc.name,kind='open_path',detail=str(dest)))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not refs, json.dumps(refs)
receipt={'epoch_start':time.time(),'github_sha':GITHUB_SHA,'hf_receipt':HF_RECEIPT,'before':subprocess.check_output(['df','-B1','/dataset'],text=True),'deleted':[],'preserved_latest_200':True}
for d in paths:
 assert d.is_absolute() and under(d,base)
 assert d.parts[len(base.parts)].startswith('bapo-aligned-cohost-node11-v')
 assert 'topd' not in str(d) and 'global_step_200' not in d.parts and 'outer_200_COMPLETE' not in d.parts
 if d.exists():
  if d.is_symlink():d.unlink()
  else:shutil.rmtree(d)
  receipt['deleted'].append(str(d))
receipt['after']=subprocess.check_output(['df','-B1','/dataset'],text=True)
receipt['queue']=subprocess.check_output(['squeue','-h','-o','%i|%j|%T|%N'],text=True)
receipt['epoch_end']=time.time()
(base/'bapo-aligned-cohost-node11-v11-20261009/cleanup-receipt-20261012.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
