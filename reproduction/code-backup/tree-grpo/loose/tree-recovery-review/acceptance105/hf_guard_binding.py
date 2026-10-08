import json,hashlib,subprocess,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234');gate=root/'output/acceptance-step105'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()
ready=json.loads((gate/'ready.json').read_bytes());report=json.loads((gate/'hf105-full-reload.json').read_bytes());ck=Path(ready['hf_checkpoint']);evidence=[]
assert report['status']=='HF_RELOAD_PARAMETER_DELTA_ACCEPTED' and report['checkpoint']==str(ck) and report['base']=='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/assets/qwen2.5-3b-instruct-aa8e725'
assert all(report[k] is True for k in ['tokenizer_vocab_equal','bos_eos_equal','all_parameters_finite','cpu_forward_logits_finite'])
assert all(not v for v in report['hf_loading_info'].values()) and report['changed_parameters']>0 and report['changed_tensors']==434
for name,expected in ready['hf_identity']['files'].items():
 p=ck/name;actual=sha(p);assert actual==expected;assert report['checkpoint_file_sha256'][name]==actual;evidence.append(dict(path=str(p),sha256=actual,bytes=p.stat().st_size))
assert hashlib.sha256(json.dumps(ready['hf_identity']['files'],sort_keys=True).encode()).hexdigest()==ready['hf_identity']['sha256']
for name in ['ready.json','hf105-full-reload.json','state105-files-cpu.json']:
 p=gate/name;raw=p.read_bytes();evidence.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
state=json.loads((gate/'state105-files-cpu.json').read_bytes());assert state['status']=='COMPLETE_STEP105_STATE_FILES_CPU_VERIFIED' and len(state['workers'])==8 and state['driver_actual']==105 and state['sampler_cursor']==54400
assert state['ready_sha256']==sha(gate/'ready.json') and state['manifest_sha256']==ready['full_state_manifest_sha256']
assert not (gate/'approved.json').exists() and not (gate/'released.json').exists() and not list((root/'output/tree-rank-audit').glob('step-000106-rank-*.json'))
print(json.dumps(dict(status='PASS_HF105_FILE_BYTES_AND_CPU_REPORT_BINDING',ready=ready,state_report=state,hf_report=report,evidence=evidence,no_approval_or106=True)),flush=True)
