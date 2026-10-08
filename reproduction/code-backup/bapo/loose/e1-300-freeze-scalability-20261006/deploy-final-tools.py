"""Deploy only reviewed task-local tools/receipts; never submit or initialize CUDA."""
import io,tarfile,json,hashlib,pathlib,sys,subprocess,os
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006')
SOURCE='b7119791b86951727cea36a1881e8a525c932494'
def sha(b):return hashlib.sha256(b).hexdigest()
blob=sys.stdin.buffer.read();assert sha(blob)==sys.argv[1]
assert ROOT.resolve()==ROOT and not ROOT.is_symlink()
assert subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE
assert not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True)
assert not list((ROOT/'runs').iterdir()) and not list((ROOT/'scheduler-head').iterdir()),'New job/head already present; refuse deployment'
assert sha((ROOT/'logs/cpu-gate.json').read_bytes())=='8a32af877fdc8393c8d13378756d606e2e3b89cb04b416c8025a7c011e300095'
with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as t:
 members=t.getmembers();assert len({m.name for m in members})==len(members)
 assert all(m.isfile()and not m.issym()and not m.islnk()for m in members)
 files={m.name:t.extractfile(m).read()for m in members}
manifest=json.loads(files['logs/tools-manifest.json']);approval=json.loads(files['logs/launch-approval.json'])
assert manifest['source_commit']==approval['source_commit']==SOURCE and approval['root']==str(ROOT)
assert approval['status']=='PASS_INDEPENDENT_E1_LAUNCH_REVIEW'
assert approval['actual_gpu']is False and approval['production_training_permission_signed']is False
assert sha(files['logs/tools-manifest.json'])==approval['tools_manifest_sha256']
report='logs/FINAL29-LAUNCH-TOOLS-INDEPENDENT-REVIEW.md'
assert approval['independent_review']['path']==str(ROOT/report)and approval['independent_review']['sha256']==sha(files[report])
expected={'tools/'+n for n in manifest['files']}|{'logs/tools-manifest.json','logs/launch-approval.json',report}|{'scheduler-head/'+n for n in ('submit.sh','hardware.sbatch','formal300.sbatch','approved-launch-hashes.json','watch-terminal.py')}
assert set(files)==expected and len(manifest['files'])==approval['tools_file_count']==29
for name,h in manifest['files'].items():assert sha(files['tools/'+name])==h
head=json.loads(files['scheduler-head/approved-launch-hashes.json'])
for n in ('submit.sh','hardware.sbatch','formal300.sbatch'):assert files['scheduler-head/'+n]==files['tools/'+n]and sha(files['tools/'+n])==head[n]
backup=ROOT/'deployment-archive/prior28-tools-before-final29';backup.mkdir(parents=True)
old={}
for p in (ROOT/'tools').iterdir():
 assert p.is_file()and not p.is_symlink()
 data=p.read_bytes();old[p.name]=sha(data);(backup/p.name).write_bytes(data)
assert len(old)==28
(backup/'old-tool-shas.json').write_text(json.dumps(old,sort_keys=True,indent=2)+'\n')
for name,data in files.items():
 p=ROOT/name;assert p.parent.resolve()==p.parent and not p.is_symlink()
 assert p.parent in (ROOT/'tools',ROOT/'logs',ROOT/'scheduler-head')
 temp=p.with_name(p.name+'.deploy-tmp');assert not temp.exists()
 with temp.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
 if p.suffix in ('.sh','.sbatch'):p.chmod(0o755)
assert {p.name:sha(p.read_bytes())for p in (ROOT/'tools').iterdir()if p.is_file()}==manifest['files']
print(json.dumps({'status':'DEPLOYED_REVIEWED_FINAL29_NO_JOB_SUBMITTED','source':SOURCE,'tools_manifest_sha256':approval['tools_manifest_sha256'],'launch_approval_sha256':sha(files['logs/launch-approval.json']),'tools_file_count':29,'old28_backup':str(backup)},sort_keys=True))
