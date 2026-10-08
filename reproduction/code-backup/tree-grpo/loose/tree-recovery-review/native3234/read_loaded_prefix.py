import pathlib,json,hashlib,time,subprocess
run=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234')
files=[]
for p in [run/'output/recovery-loaded.json',run/'output/entry-audit/entry-config.yaml',run/'resolved.yaml',run/'stage.log']:
 if not p.is_file():continue
 raw=p.read_bytes()
 if p.name=='stage.log':files.append({'path':str(p),'full_sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'tail':raw[-14000:].decode(errors='replace')})
 else:files.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'text':raw.decode()})
print(json.dumps({'snapshot_epoch':time.time(),'files':files,'scontrol':subprocess.check_output(['scontrol','show','job','3234'],text=True)}),flush=True)
