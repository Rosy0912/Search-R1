import subprocess,json,datetime
print(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':3315,'slurm':subprocess.check_output(['scontrol','show','job','3315'],text=True)}))
