import subprocess,json,datetime,os
v={"scope":"READONLY_OWN_OLD3315_COMPLETION_ONLY_NO_STOP_NO_ADMIN","utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
for key,args in [("pids",["scontrol","listpids","3315"]),("steps",["scontrol","show","step","3315"]),("tasks",["ps","-eo","pid,ppid,stat,etimes,args"])]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=15)
 if key=="tasks":p.stdout="\n".join(x for x in p.stdout.splitlines() if "3315" in x or "slurmstepd" in x)
 v[key]={"exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
print(json.dumps(v))
