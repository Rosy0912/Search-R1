import pathlib,json,hashlib,os,datetime
P=pathlib.Path
source=P('/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012')
base=P('/mnt/data/shaoshitong/search-g1-final200-bs256-20261014-v1')
asset=P('/dev/shm/search-g1-final200-bs256-assets-20261014-v1')
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def once(p,x):
 with P(p).open('x') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
base.mkdir(exist_ok=True);(base/'preparation').mkdir(exist_ok=True)
intent=base/'preparation/asset-intent.json';result=base/'preparation/asset-receipt.json'
assert not intent.exists() and not result.exists(),'Asset intent exists; inspect receipt; do not replay'
assert not (source/'logs/final-identity.json').exists(),'Original evaluation already complete; do not stop it'
u=[json.loads(l) for l in (source/'logs/actual-updates.jsonl').read_text().splitlines()];assert u[-1]['step']==200 and u[-1]['optimizer_writes_after']==4000
cp=source/'checkpoints/search-g1-common8/actor/global_step_200';complete=json.loads((cp/'complete.json').read_text());assert complete['step']==200 and complete['actual_optimizer_writes_per_rank']==4000
once(intent,{'source':str(source),'checkpoint':str(cp),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Prepare final-only input bindings; no stopping or model mutation'})
verified=[]
for name,pinned in complete['files_sha256'].items():
 if name.startswith('training_state/'):continue
 actual=sha(cp/name);assert actual==pinned,name;verified.append({'relative':name,'full_read_sha256':actual,'bytes':(cp/name).stat().st_size})
for p in cp.iterdir():
 if p.is_file() and p.name not in [x['relative'] for x in verified] and p.name!='complete.json':
  verified.append({'relative':p.name,'full_read_sha256':sha(p),'bytes':p.stat().st_size})
claim=json.loads((source/'logs/corpus-cache-claim.json').read_text());corpus=P(claim['corpus_extraction']['target']);assert corpus.stat().st_ino==claim['corpus_extraction']['output_inode'] and corpus.stat().st_size==claim['corpus_extraction']['content_size']
stage=json.loads((source/'logs/retrieval-staging-claim.json').read_text());index=P(stage['root'])/'e5_Flat.index';assert index.is_file()
asset.mkdir(mode=0o700)
for orig,target in [(corpus,asset/'wiki_dump.jsonl'),(index,asset/'e5_Flat.index')]:
 assert not target.exists();os.link(orig,target);assert orig.stat().st_ino==target.stat().st_ino
for name in ['env','bootstrap-tools','data']:
 assert not (base/name).exists();(base/name).symlink_to(source/name,target_is_directory=True)
for name in ['logs','tmp','cache','journal']:(base/name).mkdir(exist_ok=True)
config=source/'logs/resolved-training-config.json';(base/'preparation/source-config.json').write_bytes(config.read_bytes())
r=json.loads((source/'logs/run-manifest.json').read_text())
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_base':str(source),'checkpoint':str(cp),'checkpoint_manifest':str(cp/'complete.json'),'checkpoint_manifest_sha256':sha(cp/'complete.json'),'generation_config_sha256':sha(cp/'generation_config.json'),'resolved_config_sha256':sha(base/'preparation/source-config.json'),'hf_files_full_sha_verified':verified,'training_state_independent_verification_performed':False,'run_id':r['run_id'],'validation_sha256':r['input_sha256']['final.parquet'],'retrieval_index_path':str(asset/'e5_Flat.index'),'retrieval_corpus_path':str(asset/'wiki_dump.jsonl'),'corpus_prior_verified_sha256':claim['corpus_extraction']['content_sha256'],'retained_asset_hardlinks':True,'no_weights_or_rollouts_transferred_to_mac':True}
once(result,receipt)
print(json.dumps({'base':str(base),'receipt':str(result),'hf_full_sha_files':len(verified),'hf_bytes':sum(x['bytes'] for x in verified),'checkpoint_manifest_sha256':receipt['checkpoint_manifest_sha256']}))
