import pathlib,subprocess,shlex,json,hashlib,datetime
H=pathlib.Path(__file__).resolve().parent
code="""import pathlib,json,subprocess,base64,hashlib,os
r=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3210')
v={'job_id':'3210','files':{},'slurm':subprocess.check_output(['scontrol','show','job','3210','-o'],text=True),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)}
for n in ['train.log','resolved-profile.json','formal-wrapper.log']:
 p=r/n
 if p.is_file():
  b=p.read_bytes();v['files'][n]={'base64':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
v['resource_history_bytes']=(r/'resources.ndjson').stat().st_size if (r/'resources.ndjson').exists() else 0;v['resource_history_retained_remote']=str(r/'resources.ndjson');v['paused_manifest_exists']=(r/'full_state/global_step_5/manifest.json').is_file();print(json.dumps(v))
"""
q=subprocess.run(['python3',str(H.parent/'remote.py'),'python3 -c '+shlex.quote(code)],capture_output=True,timeout=75)
t=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=H/('formal3210-observed-'+t);out.mkdir();(out/'transport.stderr').write_bytes(q.stderr)
if q.returncode!=0:(out/'transport.stdout').write_bytes(q.stdout);raise SystemExit(q.returncode)
v=json.loads(q.stdout);import base64
for n,r in v['files'].items():
 b=base64.b64decode(r['base64']);assert hashlib.sha256(b).hexdigest()==r['sha256'];(out/n).write_bytes(b)
(out/'observed-original.json').write_bytes(q.stdout)
rows=[]
for l in (out/'train.log').read_text().splitlines() if (out/'train.log').exists() else []:
 if 'OPSD_E1_OUTER_JSON=' in l:
  value,end=json.JSONDecoder().raw_decode(l.split('OPSD_E1_OUTER_JSON=',1)[1]);rows.append(value)
print(json.dumps({'snapshot':str(out),'job_state':v['slurm'],'gpu':v['gpu'],'logged_outer_count':len(rows),'latest_outer':({k:rows[-1][k] for k in ('actual_outer_updates','global_step','driver_pid','job_id','source_commit','config_sha256')} if rows else None), 'latest_metrics':({k:rows[-1]['metrics'].get(k) for k in ('critic/score/mean','actor/opsd_loss','actor/grad_norm','actor/lr','env/finish_ratio','env/ratio_of_valid_action','timing_s/step','timing_s/gen','timing_s/update_actor')} if rows else None),'paused_manifest_exists':v['paused_manifest_exists'],'scope':'read-only observation, transport ownership and strict first5 acceptance separate'},ensure_ascii=False))
