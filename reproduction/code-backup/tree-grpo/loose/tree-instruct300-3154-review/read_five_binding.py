from pathlib import Path
import json,hashlib,time,subprocess
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');R=ROOT/'runs/tree-instruct300-3154';C=ROOT/'instruct300/candidates/tree-f994af3-aa9e3161';gate=R/'output/acceptance-step5';proof=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def read(p):
 b=p.read_bytes();proof.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)));return json.loads(b)
m=read(C/'manifest.json');ready=read(gate/'ready.json');state=read(gate/'state5-files-cpu.json');hf=read(gate/'hf5-full-reload.json');manifest=read(Path(ready['full_state_manifest']))
assert sha(C/'check_complete_state.py')==m['files']['check_complete_state.py'] and sha(ROOT/'tools/check_real_checkpoint.py')==m['external_files'][str(ROOT/'tools/check_real_checkpoint.py')]
assert ready['job_id']=='3154' and ready['completed_outer_updates']==5 and ready['next_outer_update']==6 and ready['actual_update_limit']==300 and ready['full_state_reload_verified'] is True
identity=manifest['identity'];assert identity['code_commit']==m['code_commit']=='f994af33c6c93b6b9844863be912cab2a4020dc1' and identity['data_identity']==m['common_data_identity'] and identity['model_identity']==m['model_identity']['sha256'] and identity['completed_step']==5 and identity['actual_update_limit']==300
assert state['status']=='COMPLETE_STEP5_STATE_FILES_CPU_VERIFIED' and state['identity']==identity and state['ready_sha256']==sha(gate/'ready.json') and state['manifest_sha256']==ready['full_state_manifest_sha256']==sha(Path(ready['full_state_manifest'])) and state['driver_actual']==5 and state['sampler_cursor']==2560
assert len(manifest['worker_states'])==len(state['workers'])==8
for rank,rec in enumerate(sorted(manifest['worker_states'],key=lambda z:z['identity']['rank'])):
 assert rec['identity']['rank']==rank and rec['identity']['world_size']==8 and all(rec['identity'][k]==v for k,v in identity.items()) and rec['exact_same_world_reload_verified'] is True and rec['scheduler_last_epoch']==5
 assert sha(Path(rec['path']))==rec['sha256'];proof.append(dict(path=rec['path'],sha256=rec['sha256'],bytes=Path(rec['path']).stat().st_size))
 sr=next(z for z in state['workers'] if z['rank']==rank);assert sr['finite'] is True and sr['logical_sha256']==rec['logical_state_sha256'] and sr['optimizer_steps']==120 and sr['scheduler_step']==5 and sr['optimizer_state_entries']==37
 assert rec['optimizer_state_entries']==37 and all(abs(lr-5e-6/57)<1e-20 for lr in rec['optimizer_lrs'])
driver=manifest['driver'];assert driver['exact_reload_verified'] is True and driver['next_step']==6 and sha(Path(driver['path']))==driver['sha256'];proof.append(dict(path=driver['path'],sha256=driver['sha256'],bytes=Path(driver['path']).stat().st_size))
assert hf['status']=='HF_RELOAD_PARAMETER_DELTA_ACCEPTED' and hf['checkpoint']==ready['hf_checkpoint'] and hf['base']==str(ROOT/'instruct300/assets/qwen2.5-3b-instruct-aa8e725') and hf['all_parameters_finite'] is True and hf['cpu_forward_logits_finite'] is True and hf['tokenizer_vocab_equal'] is True and hf['bos_eos_equal'] is True and all(not v for v in hf['hf_loading_info'].values())
assert hf['changed_parameters']>0 and hf['changed_tensors']==434 and hf['total_parameters']==3085938688 and hf['max_abs_parameter_delta']>0
hroot=Path(ready['hf_checkpoint']);actual={str(p.relative_to(hroot)):sha(p) for p in hroot.rglob('*') if p.is_file()};assert actual==ready['hf_identity']['files']==hf['checkpoint_file_sha256'] and hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()==ready['hf_identity']['sha256']==manifest['hf_identity']['sha256']
for name,h in actual.items():proof.append(dict(path=str(hroot/name),sha256=h,bytes=(hroot/name).stat().st_size))
base=ROOT/'instruct300/assets/qwen2.5-3b-instruct-aa8e725';bt=read(base/'tokenizer_config.json');ht=read(hroot/'tokenizer_config.json');norm=lambda v:v.get('content') if isinstance(v,dict) else v
assert bt['chat_template']==ht['chat_template']
for k in ['bos_token','eos_token','pad_token']:assert norm(bt.get(k))==norm(ht.get(k)),k
slurm=subprocess.check_output(['scontrol','show','job','3154'],text=True);assert 'JobState=RUNNING' in slurm and 'Restarts=0' in slurm and 'TresPerNode=gres:gpu:8' in slurm
assert not (gate/'approved.json').exists() and not (gate/'released.json').exists() and not list((R/'output/tree-rank-audit').glob('step-000006-rank-*.json'))
print(json.dumps(dict(status='ACTUAL_READY_STATE_HF_BINDING_ALL_FILE_SHA_PASS',epoch=time.time(),identity=identity,ready=ready,ready_sha256=sha(gate/'ready.json'),manifest=manifest,state_cpu=state,hf_cpu=hf,tokenizer_chat_template_equal=True,bos_eos_pad_semantic_equal=True,slurm=slurm,proof=proof,source_statechecker_sha256=sha(C/'check_complete_state.py'),source_hfchecker_sha256=sha(ROOT/'tools/check_real_checkpoint.py'),scope='Independent immutable file hashes and pinned actual CPUchecker results; no new full3B reload, no approval')))
