import sys,subprocess,shlex,pathlib,json,base64
sys.path.insert(0,"/Users/shaoshitong/Documents/remote_codex_control")
import control
code=r'''import pathlib,subprocess,json,socket,datetime,os,hashlib,base64
root=pathlib.Path("/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006")
run=root/"runs/gigpo-e1-fresh200-live-job3325-1791369939321796343"
if socket.gethostname().split(".")[0]!="h100-8":raise RuntimeError("Wrong node")
out={"hostname":socket.gethostname(),"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"job_id":"3325","run":str(run),"files":[],"raw_metadata":{}}
names=["run-result.json","baseline-evaluation.json","actual-official-zero-initialization.json","fresh200-initialization.json","actual-baseline-state-comparison.json","continuation-terminal.json","actual-native-fsdp-before-zero.json"]
for name in names:
 p=run/name;raw=p.read_bytes();out["files"].append({"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()});out["raw_metadata"][name]=base64.b64encode(raw).decode()
result=json.loads((run/"run-result.json").read_bytes())
initial=json.loads((run/"actual-official-zero-initialization.json").read_bytes())
fresh=json.loads((run/"fresh200-initialization.json").read_bytes())
mainlogs=[p for p in (run/"ray").glob("session_*/logs/worker-*-5933.out") if not p.is_symlink()]
if len(mainlogs)!=1:raise RuntimeError("Unknown fixed original mainworker log: "+repr(mainlogs))
for p in [root/"logs/fresh200-live-3325.log",run/"calibration.log",mainlogs[0],run/"training-prefix-turns.json",run/"actual-rollout-2560.pt.partial"]:
 st=p.stat();h=hashlib.sha256()
 with p.open("rb") as f:
  for block in iter(lambda:f.read(1048576),b""):h.update(block)
 st2=p.stat()
 if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)!=(st2.st_dev,st2.st_ino,st2.st_size,st2.st_mtime_ns):raise RuntimeError("Original file changed during terminal hash")
 out["files"].append({"path":str(p),"bytes":st.st_size,"sha256":h.hexdigest(),"device":st.st_dev,"inode":st.st_ino,"mtime_ns":st.st_mtime_ns})
 if p.suffix==".log" or p==mainlogs[0]:out["raw_metadata"][p.name]=base64.b64encode(p.read_bytes()).decode()
pids=[5933]+[row["pid"] for row in initial["workers"]]
out["owned_pid_exists"]={str(pid):pathlib.Path("/proc/"+str(pid)).exists() for pid in pids}
out["recorded_cgroup"]=result["cgroup"];out["recorded_cgroup_exists"]=pathlib.Path(result["cgroup"]).exists()
out["audit_files"]=[{"path":str(p),"bytes":p.stat().st_size} for p in (run/"audit").rglob("*") if p.is_file()]
out["live_progress_exists"]=(run/"live-progress.json").exists()
out["checkpoint_files"]=[str(p) for p in (run/"checkpoints").rglob("*") if p.is_file()]
for n,c in [("scheduler",["scontrol","show","job","3325","-o"]),("queue",["squeue","-h","-w","h100-8","-o","%i %j %T"])]:
 r=subprocess.run(c,capture_output=True,text=True,timeout=20);out[n]={"returncode":r.returncode,"stdout":r.stdout,"stderr":r.stderr}
print(json.dumps(out))
'''
inner="env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /usr/bin/python3 -E -B -S -c "+shlex.quote(code)
mid="env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 "+shlex.quote(inner)
route=control.ssh("mgmt-host")+["env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 -o HostName=10.33.0.16 -o HostKeyAlias=100.108.173.4 -o StrictHostKeyChecking=yes h100-16 "+shlex.quote(mid)]
a=subprocess.run(route,capture_output=True,text=True,timeout=60)
if a.returncode:raise RuntimeError(a.stderr)
d=json.loads(a.stdout);out=pathlib.Path("outputs/gigpo-e1-instruct200/job3325-terminal-originals-v1");out.mkdir()
for name,encoded in d.pop("raw_metadata").items():
 raw=base64.b64decode(encoded)
 with (out/name).open("xb") as f:f.write(raw)
with (out/"TERMINAL-ORIGINALS-RECEIPT.json").open("x") as f:json.dump(d,f,indent=2);f.write("\n")
print(json.dumps(d,indent=2))

