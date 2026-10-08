import json, pathlib, gzip, hashlib, math, datetime
P=pathlib.Path(__file__).parent/'formal3210-pause-resource-independent-20261004T171020Z'
def sha(b):return hashlib.sha256(b).hexdigest()
x=json.loads((P/'collection-original.json').read_bytes()); run=P/'run'; assert x['job_id']=='3210' and x['node']=='h100-10' and x['uid']==0
assert x['source_clean'] is True and x['source_commit']=='a0968c2d24beeaa0400f60e4e34677fe6deef9c9'
assert 'JobState=RUNNING' in x['slurm'] and 'NodeList=h100-10' in x['slurm'] and 'gres:gpu:h100:8' in x['slurm']
assert x['receipt_exists'] is False
manifest=run/'full_state/global_step_5/manifest.json';assert sha(manifest.read_bytes())==x['checkpoint_manifest_sha256']
service=json.loads((run/'evidence/service.json').read_bytes()); sp=service['owner']['pid']; assert sp==590120
assert service['owner']['start']=='1819168667';cmd=service['command'];assert '--faiss_gpu' in cmd and cmd[cmd.index('--device')+1]=='cuda' and cmd[cmd.index('--host')+1]=='127.0.0.1' and cmd[cmd.index('--port')+1]=='18084' and cmd[cmd.index('--topk')+1]=='3'
rank_reports=x['rank_reports']; assert sorted(r['rank'] for r in rank_reports)==list(range(8));assert len({r['pid'] for r in rank_reports})==8
identity=[]
for r in rank_reports:
 assert r['step']==5 and r['world_size']==8 and r['status']=='EXACT_FULL_STATE_RELOAD_OK'
 pp=run/f"checkpoints/actor/global_step_5/training_state/rank_{r['rank']}.json";assert json.loads(pp.read_bytes())==r
 proc=x['current_processes'][str(r['pid'])];assert proc['uids']==['0']*4 and proc['in_slurm_job'] and '/job_3210/step_0' in proc['cgroup'] and proc['state'] not in ('Z','X')
 uu=[]
 for s in x['samples']:
  assert s['job_id']=='3210' and len(s['gpus'])==8 and len(s['gpu_processes'])==16
  app=[a for a in s['gpu_processes'] if a['pid']==r['pid']]; assert len(app)==1
  a=app[0];assert a['process_start_ticks']==proc['process_start_ticks'] and a['uids']==['0']*4 and '/job_3210/step_0' in a['cgroup'];uu.append(a['uuid'])
 assert len(set(uu))==1
 identity.append(dict(rank=r['rank'],pid=r['pid'],start=proc['process_start_ticks'],uuid=uu[0],state=proc['state']))
assert len({r['uuid'] for r in identity})==8
expected={sp}|{r['pid'] for r in rank_reports}
for s in x['samples']:
 assert {a['pid'] for a in s['gpu_processes']}==expected
 sa=[a for a in s['gpu_processes'] if a['pid']==sp];assert len(sa)==8 and {a['uuid'] for a in sa}=={r['uuid'] for r in identity}
 for a in sa:assert a['process_start_ticks']==service['owner']['start'] and a['uids']==['0']*4 and '/job_3210/step_0' in a['cgroup']
driver=x['current_processes']['591546'];assert driver['process_start_ticks']=='1819179500' and driver['uids']==['0']*4 and '/job_3210/step_0' in driver['cgroup'] and driver['state'] not in ('Z','X')
times=[datetime.datetime.fromisoformat(s['time']) for s in x['samples']];assert times[1]>times[0] and all(t.utcoffset()==datetime.timedelta(0) for t in times)
http=json.loads((run/'evidence/http-top3-24.json').read_bytes());assert http['job_id']=='3210' and len(http['queries'])==24 and http['service']['pid']==sp and http['service']['start']==service['owner']['start'];res=http['response']['result'];assert len(res)==24
for rr in res:
 assert len(rr)==3
 for doc in rr:assert math.isfinite(doc['score']) and isinstance(doc['document']['contents'],str) and doc['document']['contents']
log=(run/'train.log').read_text(); outer=[json.JSONDecoder().raw_decode(l.split('OPSD_E1_OUTER_JSON=')[1])[0] for l in log.splitlines() if 'OPSD_E1_OUTER_JSON=' in l];assert [o['actual_outer_updates'] for o in outer]==list(range(1,6))
for o in outer:
 assert o['driver_pid']==591546 and o['job_id']=='3210' and o['source_commit']==x['source_commit'] and o['global_step']==o['actual_outer_updates']
 assert all(isinstance(v,(int,float)) and math.isfinite(v) for v in o['metrics'].values())
 assert o['metrics']['actor/optimizer_steps']==20
paused=[json.JSONDecoder().raw_decode(l.split('OPSD_E1_STEP5_PAUSED_JSON=')[1])[0] for l in log.splitlines() if 'OPSD_E1_STEP5_PAUSED_JSON=' in l];assert len(paused)==1 and paused[0]['driver_pid']==591546 and paused[0]['manifest_sha256']==x['checkpoint_manifest_sha256']
f=next((run/'evidence').rglob('resources-history.ndjson.gz')); raw=gzip.decompress(f.read_bytes()); lines=raw.splitlines(keepends=True);rows=[json.loads(l) for l in lines]; i=max(range(len(rows)),key=lambda i:max(g['used_mib'] for g in rows[i]['gpus']));peak=rows[i]
(P/'peak-resource-context-original.ndjson').write_bytes(b''.join(lines[i-3:i+4])); peaksummary=dict(index=i,time=peak['time'],gpus=peak['gpus'],gpu_processes=peak['gpu_processes'],phase='Initial validation/rollout window: snapshot16:31:59 still has no Initial validation metrics or outer record. Precise CUDA operation not instrumented.',neighbor_first=rows[i-3]['time'],neighbor_last=rows[i+3]['time'])
(P/'peak-resource-attribution.json').write_text(json.dumps(peaksummary,indent=2)+'\n')
result=dict(scope='ACTUAL identity/local retrieval/PAUSED5 resource review only; not native payload/HF/accuracy acceptance or authorization',identity_checks_passed=True,job_id='3210',source_commit=x['source_commit'],driver_pid=591546,driver_start=driver['process_start_ticks'],rank_identity=identity,service_pid=sp,service_start=service['owner']['start'],http_seconds=http['seconds'],pause_times=[s['time'] for s in x['samples']],pause_max_gpu_mib=max(g['used_mib'] for s in x['samples'] for g in s['gpus']),observed_outer_records=5,observed_per_rank_report_reload_status='EXACT_FULL_STATE_RELOAD_OK is report evidence, not independent PT semantic acceptance',history_raw_sha256=sha(raw),history_gzip_sha256=sha(f.read_bytes()),peak_time=peak['time'],peak_mib=79161,total_mib=81559,peak_used_fraction=79161/81559,continuous_upper_bound=False,runtime15percent_reserve_proven=False,release_authorized=False)
(P/'pause-resource-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
