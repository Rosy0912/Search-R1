import sys,subprocess,shlex,pathlib,json,hashlib
sys.path.insert(0,"/Users/shaoshitong/Documents/remote_codex_control")
import control
code=r'''import pathlib,subprocess,json,socket,datetime,hashlib
root=pathlib.Path("/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006")
state=root/"logs/fresh200-evidence-repair3325-once-v1"
if socket.gethostname().split(".")[0]!="h100-8":raise RuntimeError("Wrong node")
out={"hostname":socket.gethostname(),"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"state_exists":state.exists(),"state_entries":sorted(p.name for p in state.iterdir()) if state.exists() else [],"metadata":{},"file_identities":[]}
for n in ["SUBMIT-INTENT.json","SUBMIT-RECEIPT.json","BINDING.json"]:
 p=state/n
 if p.is_file():
  raw=p.read_bytes();out["file_identities"].append({"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
  if n!="BINDING.json":out["metadata"][n]=json.loads(raw)
receipt=out["metadata"].get("SUBMIT-RECEIPT.json",{});job=receipt.get("job_id")
if job is not None and not (isinstance(job,str) and job.isdecimal()):raise RuntimeError("Unknown receipt job identity")
commands=[("queue",["squeue","-h","-w","h100-8","-o","%i %j %T"])]
if job:commands.append(("scheduler",["scontrol","show","job",job,"-o"]))
for name,cmd in commands:
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=20);out[name]={"returncode":r.returncode,"stdout":r.stdout,"stderr":r.stderr}
out["job_id"]=job
out["run_matches"]=[str(p) for p in (root/"runs").glob("gigpo-e1-fresh200-live-job"+job+"-*")] if job else []
print(json.dumps(out))
'''
inner="env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /usr/bin/python3 -E -B -S -c "+shlex.quote(code)
mid="env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 "+shlex.quote(inner)
route=control.ssh("mgmt-host")+["env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 -o HostName=10.33.0.16 -o HostKeyAlias=100.108.173.4 -o StrictHostKeyChecking=yes h100-16 "+shlex.quote(mid)]
r=subprocess.run(route,capture_output=True,text=True,timeout=60)
if r.returncode:raise RuntimeError(r.stderr)
p=pathlib.Path("outputs/gigpo-e1-instruct200/EVIDENCE-REPAIR3325-ACTUAL-SUBMISSION-"+sys.argv[1]+".json")
with p.open("x") as f:f.write(r.stdout)
d=json.loads(r.stdout)
print(json.dumps(d,ensure_ascii=True))

