import pathlib,json,sys,time,fcntl,subprocess,shlex,hashlib,importlib.util
ROOT=pathlib.Path(__file__).resolve().parent
lock=(ROOT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
token=sys.stdin.readline().strip();assert token
from huggingface_hub import HfApi
import requests
api=HfApi(token=token);assert api.model_info('Sutoonq/TMP_MODEL').private
spec=importlib.util.spec_from_file_location('node10_source',ROOT.parent/'h100-cleanup-20261012-tree-grpo/uploader.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
scopes=['tree-grpo','opsd'];finished=[]
while len(finished)<len(scopes):
 for scope in scopes:
  if scope in finished:continue
  saved=ROOT/(scope+'-delete-receipt.json')
  if saved.exists() and json.loads(saved.read_text()).get('status')=='HF_FULL_SHA_VERIFIED_STALE_CHECKPOINTS_DELETED':finished.append(scope);continue
  base=ROOT.parent/('h100-cleanup-20261012-'+scope);done=base/'complete.json'
  if not done.exists():continue
  complete=json.loads(done.read_text());assert complete['status']=='ALL_FILES_HF_OFFICIAL_SHA_VERIFIED'
  plan=json.loads((base/'plan.json').read_text());revision=api.model_info(plan['repo']).sha
  records=list(plan['records'])
  if scope=='opsd':
   extra=ROOT.parent/'h100-cleanup-20261012-opsd-supplement'
   if not (extra/'complete.json').exists():continue
   assert json.loads((extra/'complete.json').read_text())['status']=='ALL_FILES_HF_OFFICIAL_SHA_VERIFIED'
   records.extend(json.loads((extra/'plan.json').read_text())['records'])
  verified=[]
  for rec in records:
   info=api.get_paths_info(plan['repo'],[rec['repo_path']],revision=revision)[0];assert info.size==rec['archive_bytes']
   if info.lfs:assert info.lfs.sha256==rec['archive_sha256']
   else:
    assert rec['archive_bytes']<16*1024**2
    r=requests.get('https://huggingface.co/'+plan['repo']+'/resolve/'+revision+'/'+rec['repo_path'],headers={'Authorization':'Bearer '+token},timeout=60);r.raise_for_status();assert hashlib.sha256(r.content).hexdigest()==rec['archive_sha256']
   verified.append({'path':rec['repo_path'],'size':info.size,'sha256':rec['archive_sha256']})
  logs=ROOT.parent/'h100-cleanup-20261012-logs/complete.json';assert logs.exists() and json.loads(logs.read_text())['status']=='ALL_FILES_HF_OFFICIAL_SHA_VERIFIED'
  gate={'status':'ALL_SELECTED_SOURCE_HF_SIZE_SHA_VERIFIED','repo':plan['repo'],'revision':revision,'scope':scope,'records':verified,'source_records':records,'epoch':time.time()};(ROOT/(scope+'-hf-gate.json')).write_text(json.dumps(gate,indent=2))
  script=(ROOT/'delete_success.py').read_text().replace('scope=SCOPE','scope='+repr(scope)).replace('gate=GATE','gate='+repr(gate)).replace("'hf_gate':GATE","'hf_gate':"+repr(gate))
  p=subprocess.run(mod.source_args('python3 -c '+shlex.quote(script)),capture_output=True,text=True,timeout=600)
  if p.returncode:
   (ROOT/(scope+'-delete-error.json')).write_text(json.dumps({'epoch':time.time(),'exit_code':p.returncode,'stderr':p.stderr[-2000:]}));raise RuntimeError('Exact scoped deletion preflight failed')
  receipt=json.loads(p.stdout);(ROOT/(scope+'-delete-receipt.json')).write_text(json.dumps(receipt,indent=2));finished.append(scope);print(json.dumps({'status':receipt['status'],'scope':scope,'deleted_bytes':sum(c['bytes'] for c in receipt['candidates'])}),flush=True)
 (ROOT/'state.json').write_text(json.dumps({'epoch':time.time(),'status':'WAITING_FOR_HF_VERIFIED_BACKUPS' if len(finished)<len(scopes) else 'ALL_AUTHORIZED_CLEANUP_COMPLETE','completed_scopes':finished}))
 if len(finished)<len(scopes):time.sleep(30)
(ROOT/'complete.json').write_text(json.dumps({'status':'ALL_AUTHORIZED_CLEANUP_COMPLETE','epoch':time.time(),'scopes':finished}))
