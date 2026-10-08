import pathlib,json,hashlib,importlib.util,collections,re
R=pathlib.Path('work/evidence/fresh200-behavior3312-terminal-raw');J=R/'resource-watch.jsonl';W=pathlib.Path('work/fresh200-deployment-behavior/tools/resource_watch.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((R/'terminal-full-file-hash-manifest.json').read_text());entries={x['path']:x for x in manifest['files']};assert len(entries)==442
verified=[]
for name,e in entries.items():
 p=R/name
 if p.is_file():assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'];verified.append(name)
assert sha(J)=='9d29f17e3e2640e586562d8c49b3c1d9f25ccf1860840d9f8ea612dc8bf81872'
s=importlib.util.spec_from_file_location('w',W);w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
rows=[json.loads(x)for x in J.read_text().splitlines()];run=manifest['run_root'];uuids=None;normal=0;recovery=0;bad=0;freezes={};resumes={}
def assess(r):
 global uuids
 allocated=w.inventory(r['raw_inventory'],list(r['allocated_gpus']))
 for entry in allocated.values():
  n=r['raw_nvml'].get(entry['uuid'],{})
  if 'whole_gpu_sm_percent'in n:entry.update(whole_gpu_sm_percent=w.number(n['whole_gpu_sm_percent'],'whole SM'),whole_gpu_sm_source='nvml_query',whole_gpu_sm_query_us=n.get('whole_gpu_sm_query_us'))
  else:entry['whole_gpu_sm_source']='nvidia_smi_inventory_query'
 assert allocated==r['allocated_gpus'];u={x['uuid']for x in allocated.values()}
 if uuids is None:uuids=u
 assert u==uuids and len(u)==8
 proofs=r['owned_before_query']
 for pid,p in proofs.items():assert p['run']==run and p['slurm_job_id']=='3312' and p['uid']==0 and set(p['lease_uuids'])==u and p['start_ticks']>0
 sm=w.parse_pmon(r['raw_pmon'],allocated);mem=w.parse_memory(r['raw_compute'],allocated);fresh,_=w.fresh_nvml(allocated,r['raw_nvml'],r['window_start_us'])
 for _,pid in set(sm)|set(mem)|set(fresh):
  t=r['pid_start_ticks'][str(pid)];assert t['before']>0 and t['before']==t['after']
  if str(pid)in proofs:assert proofs[str(pid)]['start_ticks']==t['before']
 return w.assess_with_bounds(allocated,sm,mem,lambda pid:str(pid)in proofs,r['raw_nvml'],r['window_start_us'])
prev=-1
for row in rows:
 assert row['run']==run and row['slurm_job_id']=='3312' and row['time']>=prev;prev=row['time'];ev=row.get('event')
 if ev in (None,'recovery_sample') and row.get('reason')!='monitor_unknown_fail_closed':
  out=assess(row)
  for k in ('external_processes','external_totals','pause','exceeded'):assert out[k]==row[k]
  assert out['pause']is False
  if ev:recovery+=1
  else:normal+=1
 elif row.get('reason')=='monitor_unknown_fail_closed':
  assert row['pause']is True
  for sample in row['diagnostic']['samples']:
   try:assess(sample)
   except ValueError as e:assert 'External SM unknown' in str(e);bad+=1
   else:raise AssertionError('Original failed query unexpectedly safe')
 elif ev=='owned_freeze_signal':
  c=row['certificate'];assert c['run']==run and c['slurm_job_id']=='3312' and c['uid']==0 and c['pid']==row['pid'] and c['start_ticks']==row['start_ticks'] and set(c['allocated_uuid_set'])==uuids and set(c['inherited_gpu_lease_uuids'])==uuids and c['gpu_leases_inherited']is True
  assert c['role']in ('phase','phase_descendant','ray_worker','retriever','trainer');freezes[row['pid']]=row['start_ticks']
 elif ev=='owned_resume_signal':assert row['signal']==18 and freezes[row['pid']]==row['start_ticks'];resumes[row['pid']]=row['start_ticks']
assert freezes==resumes and len(freezes)==301 and recovery==3
counts=collections.Counter(r.get('event','sample')for r in rows);assert counts['recovery_started']==counts['recovery_validated']==counts['recovery_finished']==1
finished=next(r for r in rows if r.get('event')=='recovery_finished');assert finished['elapsed_seconds']<60 and abs(finished['reserved_gpu_hours']-finished['elapsed_seconds']*8/3600)<1e-12
cleanup=json.loads((R/'own-process-cleanup.json').read_text());assert cleanup['own_marker']==run and cleanup['remaining']==[];assert (R/'original-exit-code.txt').read_text().strip()=='1' and (R/'cleanup-exit-code.txt').read_text().strip()=='0'
watch=pathlib.Path('work/evidence/fresh200-behavior-formal3312-terminal-watch.actual.log');terminal=json.loads(watch.read_text().splitlines()[-1]);fields=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:]*)=(\S+)',terminal['stdout']));assert fields['JobState']=='FAILED' and fields['ExitCode']=='1:0' and fields['RunTime']=='00:17:27' and fields['Restarts']=='0' and fields['Requeue']=='0';assert fields['Comment']=='BAPO_FRESH200:formal200:90755a561abd239fbb23882939410e643bb913ae:8c29851b54174f2083023a98cde9c19b'
rankpaths=sorted(R.glob('checkpoints/fresh200_state/attempt_1_candidate_outer_1_PARTIAL/ranks/rank_*.json'));assert len(rankpaths)==8
ranks=[]
for p in rankpaths:
 d=json.loads(p.read_text());assert d['optimizer_state_entries']==0 and d['scheduler_last_epoch']==0 and d['previous_complete_outer']==0;ranks.append(d)
out=dict(status='PASS_ACTUAL_3312_FAULT_RESOURCE_COST_AND_ZERO_WRITE_REVIEW_NOT_NATIVE',job_id=3312,source_commit='90755a561abd239fbb23882939410e643bb913ae',run_root=run,raw_manifest_sha256=sha(R/'terminal-full-file-hash-manifest.json'),local_actual_files_verified=len(verified),remote_total_hashed_files=442,large_remote_only_artifacts_not_cpu_replayed=[n for n in entries if n not in verified],resource_journal_sha256=sha(J),resource_journal_bytes=J.stat().st_size,watcher_sha256=sha(W),journal_entries=len(rows),event_counts=dict(counts),safe_normal_samples_replayed=normal,safe_recovery_samples_replayed=recovery,failed_sampling_attempts_replayed=bad,fail_closed_pause_count=1,owned_pid_epochs_frozen_resumed=len(freezes),recovery_seconds=finished['elapsed_seconds'],resource_protection='Actual unknown occupancy caused own certified-process freeze; three safe samples preceded own resume. Not an uninterrupted safe journal.',reserved_elapsed_seconds=1047,reserved_gpu_hours=1047*8/3600,cleanup_remaining=[],cleanup_stopped_pids=len(cleanup['stopped_pids']),optimizer_writes=0,accepted_outers=0,scheduler_steps=0,rank_current_state_metadata_verified=8,native_acceptance=False,first5_acceptance=False,whole_production_acceptance=False,mustfix=[],root_cause_report='outputs/fresh200/3312-ADMISSION-FAULT-INDEPENDENT-ROOT-CAUSE.md',repair_code_review='outputs/fresh200/3312-CONFIG-COUNTER-REPAIR-4FILE-INDEPENDENT-REVIEW.json')
P=pathlib.Path('outputs/fresh200/3312-ACTUAL-FAULT-RESOURCE-COST-ZERO-WRITE-INDEPENDENT-REVIEW.json');P.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));print('report_sha256',sha(P))
