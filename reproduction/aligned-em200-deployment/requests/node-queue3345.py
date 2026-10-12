import subprocess,json,datetime
v={"scope":"READONLY_NODE_SCHEDULING_NO_ADMIN_MUTATION","utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
for key,cmd in [("node",["scontrol","show","node","h100-5","-o"]),("old3315",["scontrol","show","job","3315","-o"]),("accounting",["sacct","-j","3315,3345","--format=JobID,State,ExitCode,Start,End,Elapsed","-P"])]:
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=20);v[key]={"exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
print(json.dumps(v))
