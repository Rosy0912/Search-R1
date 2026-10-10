"""Separate CPU final acceptance wrapper; never imports or resumes training.

Run only after independent source review and trainer3360 terminal exit0.
Calls unchanged frozen full-HF verifier; all weights remain on retained18.
"""
import hashlib,json,os,socket,stat,subprocess,sys,time
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
SOURCE='8b6f99d1b3353145200eb3fa7c16a579a79a987c'
CONFIG='c9415c0096853f78d870e205fc956f20045cde985836de5225f609eaf091d4b3'
PACKAGE_SHA='e2503977c8746edbc83b722d8224548f060909df9fc25d91a9cb88a947a79a21'
VERIFIER_REL='source/repo/reproduction/baselines/tree-validation-20261002/evidence/runtime/check_real_checkpoint.py'
VERIFIER_SHA='b3075d7b9346bb7682e0ea05962de0624c117b354590b728ed8a18b7e8ba3c97'
RUN=ROOT/'runs/tree-branpo-fresh200-3360'
PACKAGE=ROOT/('instruct300/candidates/branpo-fresh200-'+SOURCE)
BASE=ROOT/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
CHECKPOINT=RUN/'checkpoints/actor/global_step_200'
NATIVE=RUN/'checkpoints/training_state/global_step_200/manifest.json'

def require(condition,message):
 if not condition:raise ValueError(message)
def file_sha(path):
 require(stat.S_ISREG(path.lstat().st_mode),'Evidence member must be ordinary file: '+str(path))
 digest=hashlib.sha256()
 with path.open('rb') as stream:
  before=os.fstat(stream.fileno())
  for data in iter(lambda:stream.read(8*1024*1024),b''):digest.update(data)
  after=os.fstat(stream.fileno())
 require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Evidence changed while hashing')
 return digest.hexdigest()
def read_json(path,expected=None):
 digest=file_sha(path);require(expected is None or expected==digest,'JSON SHA changed: '+str(path))
 value=json.loads(path.read_text());require(file_sha(path)==digest,'JSON changed while reading');return value,digest
def identity(path):
 files={str(p.relative_to(path)):file_sha(p) for p in sorted(path.rglob('*')) if p.is_file()}
 return dict(files=files,sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())
def write_once(path,value):
 with path.open('x') as out:json.dump(value,out,indent=2,allow_nan=False);out.flush();os.fsync(out.fileno())
def main():
 require(socket.gethostname().split('.')[0]=='h100-10','Only retained18 same-node verification')
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU acceptance must explicitly hide all GPUs')
 require(sys.flags.optimize==0,'Frozen verifier assertions require unoptimized Python')
 available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
 require(available>=64*1024**3,'At least64GiB actual free CPU memory required; no GPU fallback')
 package,package_sha=read_json(PACKAGE/'manifest.json',PACKAGE_SHA)
 require(package['code_commit']==SOURCE and package['files'][VERIFIER_REL]==VERIFIER_SHA,'Frozen package/verifier differs')
 require(file_sha(PACKAGE/VERIFIER_REL)==VERIFIER_SHA,'Frozen full-HF verifier bytes changed')
 terminal,terminal_sha=read_json(RUN/'run-result.json')
 require(terminal['status']=='ACTUAL_HUMAN_AUTHORIZED_DIRECT_FRESH200_TRAINING_COMPLETE' and terminal['training_body_exit_code']==0 and terminal['actual_completed_outer']==200 and terminal['code_commit']==SOURCE,'Wait for exact3360 terminal200exit0; never stop or modify trainer')
 require(file_sha(RUN/'resolved.yaml')==CONFIG,'Actual3360 resolved config differs')
 native,native_sha=read_json(NATIVE);ids=native['identity']
 require(native['status']=='COMPLETE_SAME_WORLD_STATE_SAVED' and ids['job_id']=='3360' and ids['code_commit']==SOURCE and ids['config_sha256']==CONFIG and ids['completed_step']==200 and ids['actual_update_limit']==200,'Native checkpoint200 source/config/counter mismatch')
 require(ids['data_identity']==package['common_data_identity'] and ids['model_identity']==package['model_identity']['sha256'],'Native split/official model differs')
 require(Path(native['hf_checkpoint']).resolve()==CHECKPOINT.resolve(),'Native HF path differs')
 require(CHECKPOINT.resolve().is_relative_to(RUN.resolve()),'HF weights must remain in own same-node run')
 base_before=identity(BASE);require(base_before==package['model_identity'],'Official immutable base files differ')
 before=identity(CHECKPOINT);require(before==native['hf_identity'],'HF file inventory differs from native published milestone')
 out=RUN/'output/final-hf-acceptance';require(not out.exists(),'Existing evidence: refuse overwrite/reload retry');out.mkdir()
 env={k:v for k,v in os.environ.items() if not k.startswith('TREE_')}
 env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONOPTIMIZE='0',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(out/'hf-cache'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
 started=time.time()
 with (out/'verifier.log').open('x') as log:
  result=subprocess.run([sys.executable,str(PACKAGE/VERIFIER_REL),'--base',str(BASE),'--checkpoint',str(CHECKPOINT),'--output',str(out/'full-HF-cold-reload.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
 require(result.returncode==0,'Actual full CPU cold reload failed; preserve log/evidence, no automatic retry')
 full,full_sha=read_json(out/'full-HF-cold-reload.json')
 require(full['status']=='HF_RELOAD_PARAMETER_DELTA_ACCEPTED' and full['all_parameters_finite'] is True and full['cpu_forward_logits_finite'] is True and full['changed_parameters']>0 and full['total_parameters']>1_000_000_000,'Complete finite/nonzero cold model proof missing')
 require(Path(full['checkpoint']).resolve()==CHECKPOINT.resolve() and Path(full['base']).resolve()==BASE.resolve(),'Cold loader used different weights')
 require(full['checkpoint_file_sha256']==before['files'],'Cold reload observed a different HF inventory')
 # Tokenizer-only extra checks: no second model load or generation batch.
 os.environ.update({k:v for k,v in env.items() if k in ('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HOME')})
 from transformers import AutoTokenizer
 base_tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
 tok=AutoTokenizer.from_pretrained(CHECKPOINT,local_files_only=True)
 special=lambda t:dict(bos=t.bos_token_id,eos=t.eos_token_id,pad=t.pad_token_id,unk=t.unk_token_id,all_special_tokens=t.all_special_tokens,all_special_ids=t.all_special_ids,chat_template=t.chat_template)
 require(base_tok.get_vocab()==tok.get_vocab() and special(base_tok)==special(tok),'Full tokenizer/special/PAD/chat identity differs')
 probes=['The capital of France is','你好，世界。','<|im_start|>user\nFind an answer.<|im_end|>']
 for text in probes:
  b=base_tok.encode(text,add_special_tokens=True);c=tok.encode(text,add_special_tokens=True)
  require(b==c and base_tok.decode(b)==tok.decode(c),'Actual cold tokenizer encode/decode differs')
 tokenizer=dict(status='ACTUAL_COLD_TOKENIZER_IDENTITY_VERIFIED',vocab_equal=True,special_pad_bos_eos_unk_equal=True,all_special_tokens_ids_equal=True,chat_template_equal=True,actual_encode_decode_probes=len(probes))
 write_once(out/'tokenizer-cold-reload.json',tokenizer)
 require(identity(BASE)==base_before and identity(CHECKPOINT)==before,'Base/checkpoint changed during cold verification')
 require(read_json(NATIVE)[1]==native_sha and read_json(RUN/'run-result.json')[1]==terminal_sha,'Native/terminal publication changed during verification')
 receipt=dict(status='JOB3360_FINAL_HF_COLD_RELOAD_IDENTITY_BOUND',job_id='3360',run_id=RUN.name,source_commit=SOURCE,config_sha256=CONFIG,package_manifest_sha256=package_sha,native_manifest_sha256=native_sha,terminal_receipt_sha256=terminal_sha,hf_identity_sha256=before['sha256'],full_hf_report_sha256=full_sha,tokenizer_report_sha256=file_sha(out/'tokenizer-cold-reload.json'),wrapper_sha256=file_sha(Path(__file__).resolve()),frozen_full_verifier_sha256=VERIFIER_SHA,all_parameters_finite=True,shape_keys_full_checked=True,actual_cpu_model_cold_reload=True,actual_tokenizer_cold_reload=True,total_parameters=full['total_parameters'],changed_parameters=full['changed_parameters'],max_abs_parameter_delta=full['max_abs_parameter_delta'],cpu_forward_logits_finite=True,wall_seconds=time.time()-started,weights_left_on_h100_10=True,training_or_optimizer_restored=False,independent_final_acceptance=False)
 write_once(out/'identity-bound-final-hf.json',receipt);print(json.dumps(receipt,allow_nan=False))
if __name__=='__main__':main()
