import pathlib,json,time,subprocess,shlex,fcntl,os
ROOT=pathlib.Path(__file__).resolve().parent
from uploader_parallel import source_args
lock=(ROOT/'cleanup-supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(phase,**extra):
 (ROOT/'cleanup-status.json').write_text(json.dumps(dict(pid=os.getpid(),epoch=time.time(),phase=phase,**extra)))
status('WAITING_FOR_HF_SHA_VERIFIED')
while not (ROOT/'complete.json').exists():
 # A failed uploader never unlocks cleanup.
 time.sleep(15)
receipt=json.loads((ROOT/'complete.json').read_text());assert receipt['status']=='ALL_SELECTED_HF_FILES_SHA_VERIFIED'
plan=json.loads((ROOT/'cleanup-plan.json').read_text());manifest=json.loads((ROOT/'plan.json').read_text());assert receipt['total_bytes']==manifest['total_bytes'];assert receipt['files']==len(manifest['records'])
src=(ROOT/'cleanup_execute_guarded_template.py').read_text().replace('PLAN = None','PLAN = '+repr(plan)).replace('HF_RECEIPT = None','HF_RECEIPT = '+repr(receipt)).replace('GITHUB_SHA = None',"GITHUB_SHA = 'e7a336043c01b1b560c5d4d8962e670701aa7ef1'").replace('SOURCE_MANIFEST = None','SOURCE_MANIFEST = '+repr(manifest))
status('CHECKING_SOURCE_AND_ACTIVE_REFERENCES')
p=subprocess.run(source_args('python3 -c '+shlex.quote(src)),capture_output=True,text=True,timeout=900)
if p.returncode:
 status('CLEANUP_GUARD_BLOCKED',returncode=p.returncode,stderr=p.stderr[-2000:]);raise SystemExit(p.returncode)
(ROOT/'cleanup-receipt.json').write_text(p.stdout);status('CLEANUP_DELETED_VERIFIED_FILES')
checks=[]
for attempt in range(2):
 if attempt:time.sleep(20)
 code='''import pathlib,json,subprocess,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009');receipt=json.loads((r/'cleanup-receipt-20261012.json').read_text());run=r/'runs/formal-aligned-em200-16c570e111f947e2b0026983df2f7938-3359'
j=dict(epoch=time.time(),df=subprocess.check_output(['df','-B1','/dataset'],text=True),queue=subprocess.check_output(['squeue','-h','-o','%i|%j|%T|%N'],text=True),deleted_missing=all(not pathlib.Path(x).exists() for x in receipt['deleted']),latest200_state_exists=(run/'checkpoints/fresh200_state/outer_200_COMPLETE/ranks/rank_0.pt').is_file(),latest200_model_exists=(run/'checkpoints/actor/global_step_200/model-00001-of-00003.safetensors').is_file(),source_preserved=(r/'repo/verl').is_dir())
assert j['deleted_missing'] and j['latest200_state_exists'] and j['latest200_model_exists'] and j['source_preserved'];print(json.dumps(j))
'''
 p=subprocess.run(source_args('python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=60)
 if p.returncode:status('FINAL_VERIFICATION_FAILED',stderr=p.stderr[-2000:]);raise SystemExit(p.returncode)
 checks.append(json.loads(p.stdout))
(ROOT/'cleanup-final-checks.json').write_text(json.dumps(checks));status('HF_BACKUP_AND_BAPO_CLEANUP_COMPLETE',hf_receipt=receipt)
