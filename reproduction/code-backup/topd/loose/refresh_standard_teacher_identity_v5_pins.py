import pathlib,json,hashlib,re
W=pathlib.Path('work');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
evaldir=W/'qwen25-32b-trained-eval64-r1-v5';score=W/'topd-32b-native-score-gate-r1-v5';ipc=W/'topd-32b-pipeline-ipc-r1-v5';owner=W/'topd-32b-shared-owner-r1-v5';sup=W/'topd-shared32b-supervisor-r1-v5'
def manifest(folder,name='files.json'):
 d={'scope':'STANDARD_SEARCH_R1_TEACHER_ROUTE_SOURCE_PREPARATION_NOT_ACTUAL_APPROVAL','files':{str(p.relative_to(folder)):{'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in (name,'tool-files.json') and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}}
 (folder/name).write_text(json.dumps(d,indent=2)+'\n');return sha(folder/name)
old_gate_hash=re.search(r"gate_ref\['sha256'\]=='([0-9a-f]+)'",(ipc/'topd32b_shared_pipeline/readiness.py').read_text()).group(1)
old_score_manifest=sha(score/'tool-files.json')
evalhash=manifest(evaldir)
p=score/'score_gate.py';s=p.read_text();s=re.sub(r"TRAINED_EVAL_MANIFEST='[0-9a-f]+'",f"TRAINED_EVAL_MANIFEST='{evalhash}'",s)
start=s.index('TRAINED_HELPERS=');end=s.index('\nBASE_FP32_SHA=',start);old=s[start:end];print(old)
helpernames=['checkpoint_inventory.py','terminal.py','trained_assets.py','training_identity.py'];s=s[:start]+'TRAINED_HELPERS='+repr({n:sha(evaldir/n) for n in helpernames})+s[end:];p.write_text(s)
gatehash=sha(p)
# single canonical source gate and trained optional metadata kind
p=ipc/'topd32b_shared_pipeline/readiness.py';s=p.read_text().replace(old_gate_hash,gatehash).replace('ab36d235d8327cf70f81d20236228f2853d6436ec8f6340e164a9dcf8fec9a1b',gatehash).replace('ab6324524187ef70c213709fcde105e01444b6c4a5a15d7abf65005313382e84',gatehash);p.write_text(s)
for folder in (owner,sup):
 for p in folder.rglob('*'):
  if p.is_file() and p.suffix in ('.py','.json','.md'):
   s=p.read_text().replace(old_gate_hash,gatehash).replace('ab36d235d8327cf70f81d20236228f2853d6436ec8f6340e164a9dcf8fec9a1b',gatehash).replace('9cc2577a700c9a624b88111dc93cfe48557b41ba7696272c656427b5e420fbde',gatehash);p.write_text(s)
pins={p.name:sha(p) for p in sorted((ipc/'topd32b_shared_pipeline').glob('*.py'))};(owner/'ipc-source-pins.json').write_text(json.dumps(pins,indent=2)+'\n');p=owner/'import_gate.py';s=re.sub(r"PINS_SHA='[a-f0-9]+'",f"PINS_SHA='{sha(owner/'ipc-source-pins.json')}'",p.read_text());p.write_text(s)
# Native entry only points at its new namespace; never submits automatically.
for p in score.glob('*.sbatch'):
 s=p.read_text().replace('topd-32b-native-score-gate-v7','topd-32b-native-score-gate-r1-v5');p.write_text(s)
(score/'files.sha256').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in sorted(score.iterdir()) if p.is_file() and p.name not in ('files.sha256','tool-files.json','files.json')))
toolfiles={p.name:sha(p) for p in sorted(score.iterdir()) if p.is_file() and p.name not in ('tool-files.json','files.json')};(score/'tool-files.json').write_text(json.dumps(toolfiles,indent=2)+'\n')
scoremanifest=sha(score/'tool-files.json')
p=sup/'dependencies.py';s=p.read_text().replace(old_score_manifest,scoremanifest).replace('9802729dec47f2dc5bc47c9f9c9c41b20913feceae63456f152630ac7a671e40a8',scoremanifest).replace('9802729dec47f2dc5bc47c9f9c41b20913feceae63456f152630ac7a671e40a8',scoremanifest).replace('12121a63963e9b918324ddcfe0776dd65fa17316eed8de83061d6df2c54e0377',scoremanifest);p.write_text(s)
for folder in (ipc,owner,sup):manifest(folder)
print('gate',gatehash,'evalmanifest',evalhash,'scoremanifest',scoremanifest)
