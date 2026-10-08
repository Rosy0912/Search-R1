import json, pathlib, gzip, hashlib, math, datetime
P=pathlib.Path(__file__).parent/'formal3210-actual10-resource-recovered-20261004T184157Z'
def sha(b):return hashlib.sha256(b).hexdigest()
x=json.loads((P/'collection-original.json').read_bytes()); run=P/'run'; assert x['job_id']=='3210' and x['node']=='h100-10' and x['uid']==0
assert x['source_clean'] is True and x['source_commit']=='a0968c2d24beeaa0400f60e4e34677fe6deef9c9'
assert 'JobState=RUNNING' in x['slurm'] and 'NodeList=h100-10' in x['slurm'] and 'gres:gpu:h100:8' in x['slurm']
assert x['receipt_exists'] is True
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
http=json.loads(next((run/'evidence').rglob('fresh-http-top3-24.json')).read_bytes());assert http['job_id']=='3210' and len(http['queries'])==24 and http['url']=='http://127.0.0.1:18084/retrieve'
for pr in [http['service_before'],http['service_after']]:assert pr['pid']==sp and pr['process_start_ticks']==service['owner']['start'] and pr['uids']==['0']*4 and '/job_3210/step_0' in pr['cgroup']
res=http['response']['result'];assert len(res)==24
for rr in res:
 assert len(rr)==3
 for doc in rr:assert math.isfinite(doc['score']) and isinstance(doc['document']['contents'],str) and doc['document']['contents']
log=next((P/'direct').glob('*.out')).read_text(); outer=[json.JSONDecoder().raw_decode(l.split('OPSD_E1_OUTER_JSON=')[1])[0] for l in log.splitlines() if 'OPSD_E1_OUTER_JSON=' in l];assert [o['actual_outer_updates'] for o in outer]==list(range(1,len(outer)+1)) and len(outer)>=10
for o in outer:
 assert o['driver_pid']==591546 and o['job_id']=='3210' and o['source_commit']==x['source_commit'] and o['global_step']==o['actual_outer_updates']
 assert all(isinstance(v,(int,float)) and math.isfinite(v) for v in o['metrics'].values())
 assert o['metrics']['actor/optimizer_steps']==20
paused=[json.JSONDecoder().raw_decode(l.split('OPSD_E1_STEP5_PAUSED_JSON=')[1])[0] for l in log.splitlines() if 'OPSD_E1_STEP5_PAUSED_JSON=' in l];assert len(paused)==1 and paused[0]['driver_pid']==591546 and paused[0]['manifest_sha256']==x['checkpoint_manifest_sha256']
f=P/'resources-history.ndjson.gz'; raw=gzip.decompress(f.read_bytes()); lines=raw.splitlines(keepends=True);rows=[json.loads(l) for l in lines]; i=max(range(len(rows)),key=lambda i:max(g['used_mib'] for g in rows[i]['gpus']));peak=rows[i]
(P/'peak-resource-context-original.ndjson').write_bytes(b''.join(lines[i-3:i+4])); peaksummary=dict(index=i,time=peak['time'],gpus=peak['gpus'],gpu_processes=peak['gpu_processes'],phase='Later real training window; exact CUDA operation not instrumented. No claim of continuous peak or 15percent reserve.',neighbor_first=rows[i-3]['time'],neighbor_last=rows[i+3]['time'])
(P/'peak-resource-attribution.json').write_text(json.dumps(peaksummary,indent=2)+'\n')
previous=json.loads((P.parent/'formal3210-pause-resource-independent-20261004T171020Z/pause-resource-independent-audit.json').read_bytes())
assert [(r['rank'],r['pid'],r['start'],r['uuid']) for r in identity]==[(r['rank'],r['pid'],r['start'],r['uuid']) for r in previous['rank_identity']]
assert sha((run/'resolved-profile.json').read_bytes())==x['resolved_profile_raw_sha256']== '108dbee602e223b8121452f8dbe06685cd830e14ae91126e3f72cd0ed00a58b0'
assert sha(raw)==x['history']['raw_sha256'] and sha(f.read_bytes())==x['history']['gzip_sha256'] and len(raw)==x['history']['raw_bytes']
errors=[];gaps=[];maxima={};last=None
for n,r in enumerate(rows):
 assert str(r['job_id'])=='3210'
 t=datetime.datetime.fromisoformat(r['time']);assert t.utcoffset()==datetime.timedelta(0)
 if last is not None:assert t>last;gaps.append((t-last).total_seconds())
 last=t
 if 'query_error' in r:errors.append({'row':n,'time':r['time'],'error':r['query_error'],'message':r.get('query_error_message')})
 for g in r.get('gpus',[]):maxima[g['uuid']]=max(maxima.get(g['uuid'],0),g['used_mib'])
 for a in r.get('gpu_processes',[]):assert a['in_slurm_job'] and a['uid']=='0'
hist={'rows':len(rows),'first':rows[0]['time'],'last':rows[-1]['time'],'query_error_rows':errors,'max_gap_seconds':max(gaps),'gaps_over10seconds':sum(g>10 for g in gaps),'max_gpu_mib':maxima,'raw_bytes':len(raw),'raw_sha256':sha(raw),'gzip_bytes':len(f.read_bytes()),'gzip_sha256':sha(f.read_bytes()),'all_original_errors_and_lines_retained':True,'scope':'Complete captured growing prefix, not continuous peak bound or eventual full300run'}
(P/'actual10-history-independent-audit.json').write_text(json.dumps(hist,indent=2)+'\n')
result=dict(scope='ACTUAL10 live identity/source/local retrieval resource review; stored checkpoint5 is historical, not native200Adam/state10 proof',identity_checks_passed=True,job_id='3210',source_commit=x['source_commit'],driver_pid=591546,driver_start=driver['process_start_ticks'],rank_identity=identity,service_pid=sp,service_start=service['owner']['start'],http_seconds=http['seconds'],active_snapshot_times=[s['time'] for s in x['samples']],active_snapshot_max_gpu_mib=max(g['used_mib'] for s in x['samples'] for g in s['gpus']),observed_outer_records=len(outer),observed_per_rank_report_reload_status='EXACT_FULL_STATE_RELOAD_OK is report evidence, not independent PT semantic acceptance',history_raw_sha256=sha(raw),history_gzip_sha256=sha(f.read_bytes()),peak_time=peak['time'],peak_mib=max(g['used_mib'] for g in peak['gpus']),total_mib=81559,peak_used_fraction=max(g['used_mib'] for g in peak['gpus'])/81559,continuous_upper_bound=False,runtime15percent_reserve_proven=False,release_authorized=False)
(P/'actual10-resource-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
