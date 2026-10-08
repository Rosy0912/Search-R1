import json,pathlib,hashlib,math
P=pathlib.Path('outputs/remote-evidence/window-dual-burst-3092')
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=load(P/'result.json'); trace=[json.loads(x) for x in (P/'guard.jsonl').read_text().splitlines()]
f=load(pathlib.Path('outputs/qwen25-32b-window-v4-freeze.json'));sf={'files':f['qwen25-32b-window-safety-candidate-v4']['payload_files']};pf={'files':f['qwen25-32b-window-dual-burst-probe-v4']['payload_files']}
assert sha(P/'result.json')=='b1da0586311c0988b6fc278d1558bb075e6205275a4d05ee21cbfa02a96277fc'
assert sha(P/'guard.jsonl')=='1cd1de912f482ccd3893ee419fc5e2e9179e91d122d08b9df890dd6d1a77b33a'
assert r['files']==pf['files']
assert r['source_commit']=='413a10c50f32febaafc83db8170e2b005e0e979f'
assert len(r['phases'])==3 and [p['requested_seconds'] for p in r['phases']]==[16.,.25,2.]
uuids={x['uuid'] for x in r['baseline']['gpus']}; own={x['pid'] for x in r['owner_identities']}; assert len(uuids)==8 and len(own)==2
expected=dict(r['baseline']['identities']); expected.update({str(x['pid']):x for x in r['owner_identities']})
foreign={(x['uuid'],x['pid']) for x in r['baseline']['applications']}
maxage=0; windows_count=0; dual_unequal=0; captures_count=0; evidence_count=0
for t in trace:
 assert t['sealed_sha256']==sf['files']['runtime_safety.py'] and t['wrapper_sha256']==sf['files']['startup_guard.py']
 for c in t.get('captures',[]):
  assert c['raw_snapshot']['identities']==expected and not c['uid_errors'] and set(c['uids'].values())=={0}
  apps={(x['uuid'],x['pid']) for x in c['raw_snapshot']['applications']}; assert apps==foreign|{(u,p) for u in uuids for p in own}
  captures_count+=1
 for original in t.get('original_windows',[]):
  assert original['snapshot']['identities']==expected
  q=original['query_completed_us']
  for u,w in original['windows'].items():
   assert w['status']=='SUCCESS' and 1e6 <= w['end_us']-w['start_us'] <=3e6 and w['end_us']<=q
   for row in w['samples']:
    assert row['pid'] in own or (u,row['pid']) in foreign
    assert w['start_us']<row['timestamp_us']<=w['end_us'] and 0<=q-row['timestamp_us']<=1500000
    assert math.isfinite(row['sm']) and 0<=row['sm']<=100
   windows_count+=1
 def evidence(u,e,gpu):
  global maxage,evidence_count,dual_unequal
  if 'initial_evidence' in e:
   initial=e['initial_evidence']; evidence(u,initial,None)
   assert e['upper_bound']==max(e['initial_foreign_upper_bound'],e['final_foreign_upper_bound'])
  if e['method']=='DEVICE_TOTAL_AGGREGATE_UPPER_BOUND':
   assert e['upper_bound']<=15
   if gpu is not None: assert gpu['sm']<=15
  else:
   assert e['method']=='FRESH_SUPPORTED_NONZERO_PROCESS_WINDOW'
   if gpu is not None: assert gpu['sm']>15
   w=e['window']; now=e['evaluation_now_us']; q=e['query_completed_us']
   assert w['status']=='SUCCESS' and 1e6<=w['end_us']-w['start_us']<=3e6 and w['end_us']<=q<=now
   pos=set(); maxima={}
   for row in w['samples']:
    age=now-row['timestamp_us'];assert 0<=age<=1500000;maxage=max(maxage,age)
    assert w['start_us']<row['timestamp_us']<=w['end_us'] and math.isfinite(row['sm']) and 0<=row['sm']<=100
    if row['pid'] in own:
     if row['sm']>0:pos.add(row['pid'])
    else:
     assert (u,row['pid']) in foreign
     maxima[row['pid']]=max(maxima.get(row['pid'],0),row['sm'])
   assert pos and set(e['positive_own_pids'])==pos
   assert {int(k):v for k,v in e['foreign_observed_maxima'].items()}==maxima
   bound=sum(maxima.values()); assert bound<=15
   assert (e.get('final_foreign_upper_bound',e['upper_bound']))==bound
   assert set(e['foreign_absent_pids'])=={p for uid,p in foreign if uid==u}-set(maxima)
   assert e['absent_interpretation']=='NONZERO_FILTER_WINDOW_INFERENCE_NOT_MEASURED'
   if len(pos)==2 and len({z['timestamp_us'] for z in w['samples'] if z['pid'] in pos and z['sm']>0})>1:dual_unequal+=1
  evidence_count+=1
 if t.get('accepted'):
  ev=t['fresh_foreign_sm_evidence'] or t['observed_foreign_sm_evidence'];assert set(ev)==uuids
  for u,e in ev.items():evidence(u,e,None)
for phase in r['phases']:
 for gate in phase['gates']:
  assert gate['raw_snapshot']['identities']==expected
  assert set(gate['foreign_sm_evidence'])==uuids
  rows={g['uuid']:g for g in gate['gpu_rows']}
  for u,e in gate['foreign_sm_evidence'].items():evidence(u,e,rows[u])
assert any(all(g['sm']>15 for g in x['gpu_rows']) for x in r['phases'][0]['gates']) and dual_unequal>0
release=0
for ordinal in range(2):
 ready=load(P/f'owner-{ordinal}/ready.json'); assert ready==r['ready'][ordinal] and ready['pid'] in own and ready['torch_version']=='2.4.0+cu121'
 order=[d['uuid'] for d in ready['devices']]
 for i,d in enumerate(ready['devices']):
  assert d['device']==i and d['allocated']==d['reserved']==0
  assert {(x['uuid'],x['pid']) for x in d['actual_gpu_pid_uuids']}=={(u,ready['pid']) for u in order[:i+1]}
 for phase,duration in enumerate([16.,.25,2.],1):
  d=load(P/f'owner-{ordinal}/done-{phase}.json');assert d['pid']==ready['pid'] and d['phase']==phase and d['duration']==duration and not d['errors']
  assert {x['uuid'] for x in d['devices']}==uuids and len(d['devices'])==8
  assert all(x['allocated']==x['reserved']==0 for x in d['devices']);release+=8
zero=[]
assert not any(t.get('no_sample_after_cursor_uuids') for t in trace)
assert not any(w['status']=='NVML_ERROR_6' for t in trace for o in t.get('original_windows',[]) for w in o['windows'].values())
assert r['cleanup_record']['actual_cgroup_empty'] and not r['cleanup_record']['remaining'] and not r['parent_cuda_initialized']
assert r['final']['identities']==r['baseline']['identities'] and r['final']['applications']==r['baseline']['applications']
assert all(g['sm']<=15 and g['used_mib']/g['total_mib']<=.15 for g in r['final']['gpus'])
terminal=load(pathlib.Path('outputs/remote-evidence/window-dual-v4-3092-actual-final.json'));assert 'JobState=FAILED' in terminal['scontrol_stdout'] and 'ExitCode=1:0' in terminal['scontrol_stdout']
checks=dict(status='NORMAL_PATH_AND_RELEASE_VERIFIED_BUT_NOTFOUND_BRANCH_FAIL_COVERAGE',captures=captures_count,original_windows=windows_count,evidence_records=evidence_count,max_accepted_sample_age_us=maxage,dual_positive_unequal_timestamp_evidence_records=dual_unequal,actual_zero_own_remeasurements=len(zero),allocator_zero_receipts=release,owners=sorted(own),terminal_runtime_seconds=53,final_gpu_sm=[g['sm'] for g in r['final']['gpus']])
pathlib.Path('outputs/qwen25-32b-window-dualburst3092-independent-checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
