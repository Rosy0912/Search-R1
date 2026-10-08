from pathlib import Path
import json,hashlib,tarfile,io,os
R=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004');O=Path('outputs/gigpo-e1-policy-safety').resolve();W=Path('work/gigpo-e1-quality-resume-runtime').resolve();proofdir=R/'logs/action-prefix-native-proof-v2'
h=lambda b:hashlib.sha256(b).hexdigest()
pinp=O/'ACTION-PREFIX-NATIVE-PROOF-INPUTS-v2.json';proof=json.loads(pinp.read_text());candidate=O/'ACTION-PREFIX-NATIVE-CANDIDATE-2262942-v2.json';cv=json.loads(candidate.read_text());assert h(candidate.read_bytes())=='14c97028111b9fa2b5e7ba784577067582bf3daf073be472684de24066939369'
files=[];payloads=[];reused=[]
def add(local,target):
 p=Path(local).resolve();b=p.read_bytes();assert p.is_file() and not p.is_symlink() and len(b)<16*1024*1024
 member=f'payload/{len(files):03d}-{p.name}';v={'local_path':str(p),'path':str(target),'member':member,'bytes':len(b),'sha256':h(b)};files.append(v);payloads.append((member,b))
for p in list(proof['roles'].values())+proof['additional_evidence']:
 b=Path(p['local_path']).read_bytes();assert len(b)==p['bytes'] and h(b)==p['sha256']
 if p['path'].startswith(str(proofdir)+'/'):add(p['local_path'],p['path'])
 else:reused.append({k:p[k] for k in ('path','bytes','sha256')})
for localdir,target,names in [(W/'action-prefix-native-v2',cv['helpers'],[Path(p['path']).name for p in cv['files'] if p['path'].startswith(cv['helpers']+'/')]),(W/'action-prefix-native-deployment-v2',cv['controls'],[Path(p['path']).name for p in cv['files'] if p['path'].startswith(cv['controls']+'/')])]:
 for n in sorted(names):add(localdir/n,Path(target)/n)
add(pinp,R/'logs/ACTION-PREFIX-NATIVE-PROOF-INPUTS-v2.json');add(candidate,R/'logs/ACTION-PREFIX-NATIVE-CANDIDATE-2262942-v2.json')
assert len(files)==38 and len(reused)==52 and len({p['path'] for p in files})==38
manifest=dict(scope='CODE_AND_BOUNDED_METADATA_ONLY_NO_PT_OR_MODEL_DATA_CACHE',source_commit=cv['code_commit'],candidate_sha256=h(candidate.read_bytes()),fresh_directories=[cv['helpers'],cv['controls'],str(proofdir)],files=files,reused_files=reused)
mb=(json.dumps(manifest,indent=2)+'\n').encode();mfile=O/'ACTION-PREFIX-NATIVE-INSTALL-PLAN-v2.json'
with mfile.open('xb') as f:f.write(mb)
a=W/'action-prefix-native-install-code-metadata-v2.tar.gz'
with tarfile.open(a,'x:gz') as t:
 for n,b in [('manifest.json',mb)]+payloads:
  inf=tarfile.TarInfo(n);inf.size=len(b);inf.mode=0o600;t.addfile(inf,io.BytesIO(b))
assert a.stat().st_size<16*1024*1024
print(json.dumps({'archive':str(a),'bytes':a.stat().st_size,'sha256':h(a.read_bytes()),'manifest_sha256':h(mb),'installed_new_count':len(files),'reused_hash_count':len(reused),'source':cv['code_commit'],'candidate':cv['code_commit']}))
