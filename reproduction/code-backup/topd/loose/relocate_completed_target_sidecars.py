import subprocess,sys
code=r'''import json,pathlib,os,fcntl,hashlib,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');m=r/'assets-root/assets/qwen2.5-32b'
s=json.loads((r/'logs/qwen25-32b-download-h16proxy-state.json').read_text());assert s.get('complete') is True and not s.get('errors') and not s.get('active')
i=json.loads((r/'logs/g1-initial-static-cpu-3099/recipe/base-hf-inventory.json').read_text());b=json.loads((r/'logs/qwen25-32b-terminal-fixed413-binding.json').read_text())
assert len(b['model_files'])==24 and i['tensor_count']==771 and all(i['files'][n]['sha256']==h for n,h in b['model_files'].items())
extra=set(i['files'])-set(b['model_files']);assert extra=={'.h16proxy-download.lock'}|{f'model-0000{x}-of-00017.safetensors.resume-v1.{e}' for x in (2,3,4) for e in ('lock','transfer')}
fds=[]
for n in extra:
 p=m/n;assert not p.is_symlink() and p.stat().st_size==i['files'][n]['size']
 if n.endswith('.lock'):
  fd=os.open(p,os.O_RDWR);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);fds.append(fd)
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:links=list((proc/'fd').iterdir())
  except OSError:continue
  for link in links:
   try:assert os.readlink(link)!=str(p),('ACTUAL_SIDECAR_OPEN_FD',proc.name,n)
   except FileNotFoundError:pass
out=r/'logs/qwen25-32b-completed-target-sidecars-3099';assert not out.exists();out.mkdir(mode=0o700)
for n in sorted(extra):os.rename(m/n,out/n)
for fd in fds:os.close(fd)
assert {p.name for p in m.iterdir() if p.is_file()}==set(b['model_files'])
d={'status':'OWN_COMPLETED_DOWNLOAD_SIDECARS_RELOCATED_NOT_DELETED','epoch':time.time(),'source':str(m),'destination':str(out),'official24_hashes_exact':True,'tensor_count':771,'moved':{n:i['files'][n] for n in sorted(extra)},'remaining_model_files':24,'actual_other_process_open_fds':[],'all_lock_flocks_acquired':True}
with (out/'receipt.json').open('x') as f:json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
for directory in (m,out):
 fd=os.open(directory,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(d))
'''
p=subprocess.run([sys.executable,'work/remote_topd.py'],input="python3 - <<'MOVE_OWN_SIDECARS'\n"+code+'\nMOVE_OWN_SIDECARS\n',text=True,capture_output=True)
from pathlib import Path
Path('outputs/qwen25-32b-completed-target-sidecar-move.log').write_text(p.stdout+p.stderr)
print(p.stdout);print(p.stderr);raise SystemExit(p.returncode)
