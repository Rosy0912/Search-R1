import pathlib,json,hashlib,time,re
run=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234');d=[]
for p in [run/'output/validation-cost/validation-358612-1.json',run/'output/validation-cost/validation-358612-1.jsonl',run/'output/entry-audit/entry.jsonl']:
 raw=p.read_bytes();d.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'text':raw.decode()})
raw=(run/'stage.log').read_bytes();lines=raw.decode(errors='replace').splitlines();scorelines=[l for l in lines if re.search(r'(?<![\w/])step:100 - ',l)]
print(json.dumps({'snapshot_epoch':time.time(),'files':d,'stage_log_sha256':hashlib.sha256(raw).hexdigest(),'stage_log_bytes':len(raw),'step100_original_lines':scorelines,'completed_outer_log_lines':[l for l in lines if 'training/actual_outer_updates:' in l]}),flush=True)
