import hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo')
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
receipt=json.loads((ROOT/'logs/hardware-acceptance.json').read_text())
assert receipt['status']=='PASS_INDEPENDENT_REVIEW'
assert receipt['source_commit']==subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()
assert receipt['environment_sha256']==digest(ROOT/'logs/environment-acceptance.json')
assert receipt['assets_manifest_sha256']==digest(ROOT/'assets-root/assets.json')
assert receipt['derived_assets_record_sha256']==digest(ROOT/'logs/derived-assets.sha256')
assert receipt['common_split_manifest_sha256']==digest(ROOT/'logs/common-split-manifest.json')
for record in receipt['evidence']:
 p=Path(record['path']).resolve()
 assert ROOT in p.parents,'Evidence outside own root'
 assert digest(p)==record['sha256'],'Evidence changed: '+str(p)
assert len(receipt['evidence'])>=4
assert any('review' in r['path'] for r in receipt['evidence']),'Independent review missing'
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
visible=__import__('os').environ['CUDA_VISIBLE_DEVICES'].split(',')
records=[line.split(',') for line in raw.splitlines()]
actual=[uid.strip() for idx,uid in records if idx.strip() in visible or uid.strip() in visible]
assert len(actual)==8 and set(actual)==set(receipt['gpu_uuids'])
assert receipt['common_split_content_identity']==json.loads((ROOT/'logs/common-split-manifest.json').read_text())['content_identity']['sha256']
print('HARDWARE_RECEIPT_CURRENT_SOURCE_ENV_ASSETS_SPLIT_UUID_EVIDENCE_OK',flush=True)
