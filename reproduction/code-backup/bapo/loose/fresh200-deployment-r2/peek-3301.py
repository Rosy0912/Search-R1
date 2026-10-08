from pathlib import Path
import json,subprocess,os,datetime
def observed_json(path):
 try:return json.loads(path.read_text())
 except (OSError,json.JSONDecodeError) as error:return {"observation_error":str(error),"path":str(path),"not_terminal_evidence":True}
R=Path("/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r2-20261006")
assert os.uname().nodename=="h100-5"
runs=list((R/"runs").glob("diagnostic-discovery128-*-3301"));assert len(runs)<=1
record={"time":datetime.datetime.now(datetime.timezone.utc).isoformat(),"job_id":3301,"source_commit":(R/"source-expected.txt").read_text().strip(),"run_roots":[str(p) for p in runs],"formal_outer":0}
for label,args in [("scontrol",["scontrol","show","job","3301"]),("sacct",["sacct","-j","3301","--format=JobID,State,ExitCode,Elapsed,Start,End,NodeList","-P"])]:
 result=subprocess.run(args,text=True,capture_output=True);record[label]={"code":result.returncode,"stdout":result.stdout,"stderr":result.stderr}
if runs:
 run=runs[0];record["file_inventory"]=[str(p.relative_to(run)) for p in run.iterdir()]
 tails={}
 for p in sorted(run.glob("*.log")):
  with p.open("rb") as f:
   f.seek(0,2);n=f.tell();f.seek(max(0,n-12000));tails[p.name]=f.read().decode("utf-8",errors="replace")
 record["tails"]=tails
 native=run/"checkpoints/native-diagnostics"
 record["native_progress"]={"exists":native.is_dir(),"rank_ledgers":[]}
 if native.is_dir():
  record["native_progress"]["raw_tensor_files"]=sum(1 for _ in native.glob("rank_*.pt"))
  for ledger in sorted(native.glob("rank_*_diagnostic_ledger.json")):
   raw=observed_json(ledger);record["native_progress"]["rank_ledgers"].append({key:raw.get(key) for key in ["rank","status","optimizer_writes","expected_optimizer_steps","stop_reason","scheduler_outer_steps"]})
  status=native/"status.json"
  if status.is_file():record["native_progress"]["status"]=observed_json(status)

 watch=run/"resource-watch.jsonl"
 if watch.is_file():
  with watch.open("rb") as f:
   f.seek(0,2);n=f.tell();f.seek(max(0,n-4194304));lines=f.read().decode("utf-8").splitlines()
  for line in reversed(lines):
   try:record["latest_resource_watch"]=json.loads(line);break
   except json.JSONDecodeError:record["partial_watch_line_observed"]=True

 for p in run.glob("*.json"):
  if p.stat().st_size<100000:record.setdefault("small_json",{})[p.name]=observed_json(p)
print(json.dumps(record))
