import subprocess,json,datetime
v={"scope":"USER_AUTHORIZED_STOP_OLD3315_ONLY_NO_SERVICE_OR_FOREIGN_PROCESS_OPERATION","utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
p=subprocess.run(["scontrol","show","job","3315","-o"],capture_output=True,text=True,timeout=15);v["before"]={"exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
expected="BAPO_FRESH200:formal200:60f566b3ddabc88ce9de56ead5ad30a3d6795dd2:bab8981d9fd14afc98923eb4ebb89e33"
if p.returncode!=0 or expected not in p.stdout or "JobState=COMPLETING" not in p.stdout:raise RuntimeError("Own original3315 identity/completing required")
p=subprocess.run(["scancel","--signal=KILL","--full","3315"],capture_output=True,text=True,timeout=20);v["scoped_cancel"]={"exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
print(json.dumps(v))
