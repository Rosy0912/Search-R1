"""Owner packaging only, after actualfive PASS. No scheduling."""
import hashlib,json,shutil,tarfile
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
review=Path('work/tree-fixed-five-review/ACTUAL3103_RECEIPT.json');r=json.loads(review.read_text())
assert r['status']=='PASS' and r['stage']=='five' and r['actual_outer_updates']==5
assert r['scientific_five_accepted'] and r['operational_five_accepted'] and r['hf5_full_cpu_reload']
p=Path('work/tree-fixed-formal-296217a');assert not p.exists(),'Never overwrite frozen formal package'
p.mkdir()
for f in Path('work/tree-fixed-formal-296217a-draft').iterdir():
 if f.is_file() and f.name!='DRAFT_NOT_DISPATCHABLE.json':shutil.copyfile(f,p/f.name)
shutil.copyfile(review,p/review.name)
for item in r['small_review_bundle_mapping']:
 f=Path(item['local_path']);assert sha(f)==item['sha256'];shutil.copyfile(f,p/f.name)
for a,b in [('work/check_fixed_formal_afterany_draft.py','work/check_fixed_formal_afterany_296217a.py'),('work/summarize_tree_fixed_formal_draft.py','work/summarize_tree_fixed_formal_296217a.py'),('work/dispatch_tree_fixed_formal_draft.py','work/dispatch_tree_fixed_formal.py')]:shutil.copyfile(a,b)
m=json.loads(Path('work/tree-fixed-five-296217a/manifest.json').read_text());m['files']={f.name:sha(f) for f in sorted(p.iterdir()) if f.is_file()};m['actual_five_receipt_sha256']=sha(review);m['external_helpers']['validate_tree_stage.py']=sha('work/validate_tree_stage.py')
for file in ('check_fixed_formal_afterany_296217a.py','summarize_tree_fixed_formal_296217a.py'):m['external_helpers'][file]=sha('work/'+file)
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
s=Path('work/start_tree_fixed_five_296217a.sh').read_text().replace('tree-fixed-five-296217a','tree-fixed-formal-296217a').replace('8f3fb72fc21037d4d629026be9db0cac00c24b00df21a45982d4a63a0fd6710b',sha(p/'manifest.json')).replace('--stage five','--stage formal')
s=s.rstrip()+f' --actual-five-review "$TOOLS/ACTUAL3103_RECEIPT.json" --actual-five-review-sha {sha(review)}\n'
Path('work/start_tree_fixed_formal_296217a.sh').write_text(s)
archive=Path('work/tree-fixed-formal-296217a.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
 for f in sorted(p.iterdir()):tar.add(f,arcname=p.name+'/'+f.name)
print(json.dumps({'manifest_sha256':sha(p/'manifest.json'),'wrapper_sha256':sha('work/start_tree_fixed_formal_296217a.sh'),'archive_sha256':sha(archive),'five_receipt_sha256':sha(review)}))
