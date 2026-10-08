"""Fresh static official header identity; CPU C-API proof is not CUDA compile proof."""
import hashlib,json,os,pathlib,sys
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
BASE=ROOT/'assets/compiler-headers/python3.10-3.10.12-22.04.18-amd64'
INVENTORY='2fad02f3fdc7e74f7f734880022eb2e2b2edc94c24d597dd0b947d6a881a7201'
SMOKE='5286484a3b15a8a69344b7b1f9cc9a766af40ade951dc632a6ed0f1032af7871'
def binding():return {'base':str(BASE),'inventory_sha256':INVENTORY,'smoke_sha256':SMOKE,'cpath':str(BASE/'overlay/usr/include/python3.10')+':'+str(BASE/'overlay/usr/include')}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_tree(base,inventory):
 base=pathlib.Path(base);overlay=base/'overlay';files={};links={}
 for parent,dirs,names in os.walk(overlay,followlinks=False):
  for name in dirs+names:
   p=pathlib.Path(parent)/name;n=str(p.relative_to(overlay))
   if p.is_symlink():links[n]=os.readlink(p)
   elif p.is_file():files[n]={'size':p.stat().st_size,'sha256':sha(p)}
   elif not p.is_dir():raise ValueError('Nonregular header member')
 if files!=inventory['files'] or links!=inventory['symlinks']:raise ValueError('Actual compiler header exact inventory drift')
 return {'regular_files':len(files),'symlinks':len(links)}
def verify(config):
 if config!=binding() or BASE.is_symlink():raise ValueError('Static compiler header binding changed')
 inv=BASE/'inventory.json';smoke=BASE/'cpu-capi-smoke.json'
 if inv.is_symlink() or smoke.is_symlink() or sha(inv)!=INVENTORY or sha(smoke)!=SMOKE:raise ValueError('Compiler header proof bytes changed')
 inventory=json.loads(inv.read_text());receipt=json.loads(smoke.read_text())
 if inventory['overlay_directory']!=str(BASE/'overlay') or inventory['actual_python_deb_version']!='3.10.12-1~22.04.18' or inventory['actual_node']!='h100-11':raise ValueError('Compiler header official runtime identity drift')
 if receipt['status']!='CPU_CAPI_COMPILE_IMPORT_PASS_NO_GPU_PROOF' or receipt['inventory_sha256']!=INVENTORY or receipt['CPATH']!=config['cpath'] or receipt['actual_import']['actual_executable']!=str(ROOT/'env/bin/python') or sys.version.split()[0]!='3.10.12':raise ValueError('Compiler header native CPU ABI identity drift')
 result=verify_tree(BASE,inventory)
 return dict(result,scope='FRESH_STATIC_HEADERS_CPU_CAPI_ONLY_GPU_COMPILE_UNPROVEN',inventory_sha256=INVENTORY,smoke_sha256=SMOKE,cpath=config['cpath'])
