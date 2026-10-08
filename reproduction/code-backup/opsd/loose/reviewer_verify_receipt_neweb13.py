import subprocess,shlex,json
from pathlib import Path
code="""import os,sys,json,hashlib
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/opsd');d=root/'tools/formal-8e60976d364a-pilot-2994-hf-2996-reviewed-c85d38bd29f6';sys.path.insert(0,str(d))
os.environ['OPSD_STAGE_MANIFEST_SHA256']='eb13b32b6dbcd608c237805bc950023a627ea4bdd4e98596a1afd5c9417d7051'
from verify_formal_prerequisites import verify_deployed,verify_receipt,digest
r=root/'tools/acceptance-2994-2996/receipt.json';assert digest(r)=='384d1a3cabbb563311cc64ae1aaa2e4a4c1d0c54523ca89b06e2eb6fbf63ca5d'
m=verify_deployed();x=verify_receipt(r)
assert x['pilot_job']=='2994' and x['reload_job']=='2996' and x['picked_local_micro']==2
print(json.dumps({'independent_actual_receipt':'PASS','receipt_sha256':digest(r),'deployed_manifest_sha256':m,'bound_evidence':x['evidence'],'actual_fresh_outer_updates':5,'formal500_completed':False,'final51713_completed':False}))
"""
cmd='cd /mnt/data/shaoshitong/search-r1-rl/opsd/repo && env CUDA_VISIBLE_DEVICES= /mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python -c '+shlex.quote(code)
r=subprocess.run(['python3','work/remote.py',cmd],capture_output=True,text=True)
if r.returncode:print(r.stdout,r.stderr);raise SystemExit(r.returncode)
p=Path('work/opsd/reproduction/baselines/opsd-validation-20261003/evidence/actual_receipt_neweb13_independent_revalidation.json');p.write_text(r.stdout);print(r.stdout)
