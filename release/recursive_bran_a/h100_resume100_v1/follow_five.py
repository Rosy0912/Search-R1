"""Follow five complete restored outers and next actual stage; no model, GPU, or writes."""
import json,time
from pathlib import Path
b=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013')
owner=json.loads((b/'receipts/train-start.json').read_text());pid=owner['pid']
assert owner['job']=='3393' and owner['run_id']=='d97047b5902642c883d24aa908945cda'
p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split();tick=int(s[19]);assert pid==700089 and tick==667268101
log=b/'logs/training.log';offset=0;pending='';complete=set();next_growth=False;started=time.monotonic();previous=None
while time.monotonic()-started<1800:
 with log.open(errors='replace') as f:f.seek(offset);pending+=f.read();offset=f.tell()
 lines=pending.split('\n');pending=lines.pop()
 for line in lines:
  if 'training/actual_outer_updates:' in line:
   import re
   match=re.search(r'training/actual_outer_updates:\s*(\d+)',line)
   if match and 101<=int(match.group(1))<=105:
    complete.add(int(match.group(1)));print(json.dumps(dict(event='complete_outer',outer=int(match.group(1)),time=time.time(),line=line)),flush=True)
  if 'TREE_DRIVER_STAGE=' in line:
   try:d=json.JSONDecoder().raw_decode(line.split('TREE_DRIVER_STAGE=',1)[1])[0]
   except ValueError:continue
   key=(d.get('outer'),d.get('outer_step'),d.get('stage'),d.get('event'))
   if key!=previous:print(json.dumps(dict(event='native_stage',stage=d,time=time.time())),flush=True);previous=key
   outer=d.get('outer',d.get('outer_step'))
   if outer==106 and d.get('event') in ('end','complete','done'):next_growth=True
 if len(complete)==5 and next_growth:
  print(json.dumps(dict(accepted=True,completed=sorted(complete),next106_actual_growth=True,time=time.time())),flush=True);raise SystemExit(0)
 try:
  fields=(p/'stat').read_text().rsplit(')',1)[1].split();assert int(fields[19])==tick and fields[0]!='Z'
 except (FileNotFoundError,AssertionError):
  print(json.dumps(dict(accepted=False,owner_terminal=True,completed=sorted(complete),tail=log.read_text(errors='replace')[-8000:])),flush=True);raise SystemExit(1)
 time.sleep(15)
print(json.dumps(dict(accepted=False,observation_window_elapsed=True,owner_live=True,completed=sorted(complete),time=time.time())),flush=True)
