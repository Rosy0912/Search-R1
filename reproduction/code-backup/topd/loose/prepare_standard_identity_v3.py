from pathlib import Path
import shutil
W=Path('work');names=['qwen25-32b-trained-eval64-r1','topd-32b-native-score-gate-r1','topd-32b-shared-owner-r1','topd-32b-pipeline-ipc-r1','topd-shared32b-supervisor-r1']
for n in names:
 d=W/(n+'-v3');assert not d.exists();shutil.copytree(W/(n+'-v2'),d,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
 for p in d.rglob('*'):
  if p.is_file() and p.suffix in ('.py','.json','.md','.sbatch'):
   s=p.read_text()
   for old in names:s=s.replace(old+'-v2',old+'-v3')
   p.write_text(s)
 doc=(d/'STANDARD_ROUTE.md').read_text().replace('isolated v2','isolated v3')
 doc+='\n## Source and receipt inputs\n\nOriginal standard main_ppo source_reference is 0d8a9214dc6e19f10e21549b56f6c01b45d67100, engineered manifest50fcaec86dcc0f7266f49cdf482a17d3b414b54a439f43a10617e49573320a25. Actual outer raw uses checkpoint_directory and records from original record_outer; result.export.directory/config.final_checkpoint must agree. Training review binds raw refs and actual update counts. Trained evaluation binding contains eight required runtime source files (including training_identity.py), original source/runtime assets, inventory, raw train-result, training_effective_config, and independently reviewed training receipt.\n\nSelected teacher is actual standard100, not G1 or official Base. Base below26 result remains the historical fallback trigger; trained64 has no second cutoff. All current WindowV4 runtime9e414/guardd340/full9835 manifest refs remain explicit. Numerical original6144, full all8 allocator0/release/ACK and coexistence approval must use this selected model actual FP32 hash, never old3B/Base proofs. Formal TOPD remains500 outer/total501; accept5 same64 once plus original periodic50, original alpha.1/twoepochs unchanged. Source CPU preparation cannot authorize model execution.\n'
 (d/'README.md').write_text(doc)
for old,new in [('refresh_standard_teacher_identity_v2_pins.py','refresh_standard_teacher_identity_v3_pins.py'),('freeze_standard_teacher_identity_v2.py','freeze_standard_teacher_identity_v3.py'),('test_standard_teacher_identity_v2_chain.py','test_standard_teacher_identity_v3_chain.py')]:
 s=(W/old).read_text()
 for n in names:s=s.replace(n+'-v2',n+'-v3')
 s=s.replace('identity-v2','identity-v3').replace('test_standard_teacher_identity_v2_chain.py','test_standard_teacher_identity_v3_chain.py')
 if old.startswith('refresh_'):
  s=s.replace('toolfiles={p.name:',"(score/'files.sha256').write_text(''.join(sha(p)+'  '+p.name+'\\n' for p in sorted(score.iterdir()) if p.is_file() and p.name not in ('files.sha256','tool-files.json','files.json')))\ntoolfiles={p.name:")
 if old.startswith('freeze_'):s=s.replace("'chain_and_eval':33","'chain_and_eval':34").replace("'passed':187","'passed':188")
 (W/new).write_text(s)
p=W/'test_standard_teacher_identity_v3_chain.py';s=p.read_text()+'''\ndef test_actual_native_entry_sha256sum_check():
 import subprocess
 r=subprocess.run(['shasum','-a','256','-c','files.sha256'],cwd=S,capture_output=True,text=True)
 assert r.returncode==0,r.stdout+r.stderr
 actual={line.split('  ',1)[1] for line in (S/'files.sha256').read_text().splitlines()}
 assert {'score_gate.py','supervisor.py','test_score_gate.py','native-cpu.sbatch','native-score.sbatch'}<=actual
''';p.write_text(s)
