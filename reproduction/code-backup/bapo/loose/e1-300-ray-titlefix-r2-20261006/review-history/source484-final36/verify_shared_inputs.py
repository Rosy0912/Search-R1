import pathlib,json,hashlib,os,subprocess,sys
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-r2-20261006')
OLD=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo')
REPO=ROOT/'repo'
sys.path.insert(0,str(REPO))
from omegaconf import OmegaConf
from search_r1.baselines.holdout import prepare_common_holdout,assert_prebuilt_unchanged

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
source=(ROOT/'source-expected.txt').read_text().strip()
assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==source
assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True)
for name in ('repo','tools','logs','cache','runs','assets','common-data','assets-root'):
 p=ROOT/name;assert p.is_dir()and not p.is_symlink()
for name in ('env','retriever-gpu-lib','lib'):
 assert (ROOT/name).is_symlink()and (ROOT/name).resolve()==(OLD/name).resolve()
assert (ROOT/'assets-root/assets').is_symlink()and (ROOT/'assets-root/assets').resolve()==(OLD/'assets-root/assets').resolve()
assert sha(ROOT/'assets-root/assets.json')==sha(OLD/'assets-root/assets.json')
model=ROOT/'assets/qwen2.5-3b-instruct-aa8e725';assert model.is_dir()and not model.is_symlink()
assert sha(model/'VERIFIED-MODEL-MANIFEST.json')=='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
m=json.loads((model/'VERIFIED-MODEL-MANIFEST.json').read_text());assert m['repo']=='Qwen/Qwen2.5-3B-Instruct'and m['revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
expected={'model-00001-of-00002.safetensors':'67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2','model-00002-of-00002.safetensors':'a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d'}
assert {r['file']:r['sha256']for r in m['files']if r['file'].endswith('.safetensors')}==expected
for r in m['files']:
 p=model/r['file'];assert p.parent==model and p.is_file()and not p.is_symlink();assert p.stat().st_size==r['bytes']and sha(p)==r['sha256']
data=ROOT/'common-data';config=OmegaConf.create({'data':{'train_files':'unused','val_files':'unused','final_val_files':'unused','prompt_key':'prompt','train_data_num':None,'val_data_num':None},'baselines':{'method':'bapo','holdout':{'enabled':True,'input_mode':'prebuilt_manifest','manifest_path':str(data/'manifest.json'),'manifest_sha256':'0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24','data_root':str(data),'seed':42},'bapo':{'stage_validation_size':64,'stage_validation_files':None}}})
record=prepare_common_holdout(config);assert_prebuilt_unchanged(config,record)
print('E1_SHARED_INPUTS_PASS='+json.dumps({'root':str(ROOT),'source_commit':source,'model_manifest_sha256':sha(model/'VERIFIED-MODEL-MANIFEST.json'),'data_manifest_sha256':record['manifest_sha256'],'train_rows':record['train_rows'],'official_model_revision':m['revision']}),flush=True)
