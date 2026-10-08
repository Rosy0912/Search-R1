"""Actual shared-data tokenizer contexts, CPU only; never changes a split."""
from pathlib import Path
import hashlib,json,os,sys,time,subprocess
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r2-20261006')
COMMON=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005')
DOC_SHA='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as handle:
  for block in iter(lambda:handle.read(1048576),b''):h.update(block)
 return h.hexdigest()
def audit():
 require(sys.platform=='linux' and os.uname().nodename=='h100-5','target Linux CPU audit required')
 require(sys.prefix==str(ROOT/'env'),'actual audit interpreter prefix mismatch')
 source=(ROOT/'source-expected.txt').read_text().strip()
 require(subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==source,'audit actual source HEAD mismatch')
 require(not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip(),'audit source is dirty')
 sys.path.insert(0,str(ROOT/'repo'))
 import search_r1.baselines.fresh200_contract as contract
 require(Path(contract.__file__).resolve()==ROOT/'repo/search_r1/baselines/fresh200_contract.py','audit contract source origin mismatch')
 from search_r1.baselines.fresh200_contract import verify_official_model_assets,critical_source_hashes
 from verl.utils import hf_tokenizer
 require(Path(__import__('verl.utils.tokenizer',fromlist=['x']).__file__).resolve()==ROOT/'repo/verl/utils/tokenizer.py','audit tokenizer helper source mismatch')
 from verl.utils.dataset.rl_dataset import RLHFDataset
 require(Path(__import__('verl.utils.dataset.rl_dataset',fromlist=['x']).__file__).resolve()==ROOT/'repo/verl/utils/dataset/rl_dataset.py','actual dataset source origin mismatch')
 model=ROOT/'assets/qwen2.5-3b-instruct-aa8e725';model_manifest=verify_official_model_assets(model)
 tokenizer=hf_tokenizer(str(model))
 data=COMMON/'common-data';manifest=data/'manifest.json'
 require(sha(manifest)=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24','common manifest changed')
 doc=json.loads(manifest.read_text());started=time.monotonic();records={}
 for name,count in [('train',169028),('validation',512),('stage',64),('final',51713)]:
  entry=doc['splits'][name];path=data/entry['path']
  require(path.resolve().parent==data.resolve() and sha(path)==entry['sha256'] and entry['rows']==count,'actual split identity mismatch: '+name)
  dataset=RLHFDataset(str(path),tokenizer,prompt_key=doc['prompt_key'],max_prompt_length=4096,
                      cache_dir=str(ROOT/'cache/rlhf'),filter_prompts=True,return_raw_chat=False,truncation='error')
  require(len(dataset)==count,'split dropped rows: '+name)
  lengths=[]
  for index in range(len(dataset)):
   row=dataset[index];mask=row['attention_mask']
   require(bool(((mask==0)|(mask==1)).all()),'invalid initial mask')
   length=int(mask.sum());require(length>0,'empty initial context')
   lengths.append(length)
   if index%8192==0:print('CONTEXT_BUDGET_PROGRESS '+name+' '+str(index)+'/'+str(count),file=sys.stderr,flush=True)
  max_length=max(lengths)
  require(max_length<=2048,'initial source would crop max_start_length: '+name)
  require(max_length+3*(500+500)<=4096,'four-turn worst-case rolling context would crop: '+name)
  records[name]=dict(path=str(path),rows=len(lengths),sha256=entry['sha256'],minimum_initial_tokens=min(lengths),
      maximum_initial_tokens=max_length,worst_before_fourth_request_tokens=max_length+3000,
      lengths_sha256=hashlib.sha256(json.dumps(lengths,separators=(',',':')).encode()).hexdigest())
 require(sha(manifest)==DOC_SHA and all(sha(r['path'])==r['sha256']for r in records.values()),'shared inputs changed during audit')
 return dict(status='ACTUAL_TARGET_CPU_ALL_SHARED_CONTEXT_BUDGET_OK_NOT_GPU_ACCEPTANCE',source_commit=(ROOT/'source-expected.txt').read_text().strip(),
   source_sha256=critical_source_hashes(),data_manifest_sha256=DOC_SHA,official_model_manifest_sha256=model_manifest,
   tokenizer_eos_token_id=tokenizer.eos_token_id,tokenizer_pad_token_id=tokenizer.pad_token_id,records=records,
   force_final_answer=False,max_turns=4,max_start_length=2048,max_rolling_context=4096,max_response_tokens=500,max_observation_tokens=500,
   elapsed_seconds=time.monotonic()-started)
if __name__=='__main__':
 require(not sys.flags.optimize,'optimized Python forbidden')
 print(json.dumps(audit(),allow_nan=False))
