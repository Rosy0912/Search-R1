"""STARTING_RETRIEVER: real partial residency, never a fabricated all8 service.
Keeps original V4 supported-process collector and fresh two-snapshot SM bounds.
"""
import math,os,pathlib,time
class KnownOwnAttachTransition(RuntimeError):
 def __init__(self,message,evidence=None):
  super().__init__(message);self.evidence=evidence

def membership(identity,relative):
 return any(x.split(':',2)[-1]==relative or x.split(':',2)[-1].startswith(relative+'/') for x in identity['cgroup'])
def descendant(pid,root):
 seen=set()
 while pid>1 and pid not in seen:
  if pid==root:return True
  seen.add(pid);pid=int(pathlib.Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[1])
 return False

def ownership(safety,snapshot,baseline,uuids,service,relative,descendant_check=descendant,uid_read=None):
 need=safety.require;uid_read=uid_read or (lambda p:os.stat('/proc/'+str(p)).st_uid)
 gpus={g['uuid']:g for g in snapshot['gpus'] if g['uuid'] in uuids}
 need(len(gpus)==4 and all(all(math.isfinite(g[k]) for k in ('used_mib','total_mib','sm')) and g['total_mib']>0 and 0<=g['used_mib']<=g['total_mib'] and 0<=g['sm']<=100 for g in gpus.values()),'Unknown allocated GPU measurements')
 need(all(math.isfinite(a['used_mib']) and a['used_mib']>=0 for a in snapshot['applications']),'Unknown process memory')
 foreign={(a['uuid'],a['pid']) for a in baseline['applications'] if a['uuid'] in uuids}
 current={(a['uuid'],a['pid']) for a in snapshot['applications'] if a['uuid'] in uuids};need(foreign<=current,'Foreign baseline PID/UUID disappeared')
 if service is not None:
  need(safety.pid_identity(service['pid'])==service and uid_read(service['pid'])==os.getuid() and membership(service,relative),'Starting retriever actual identity/UID/job changed')
 own=set()
 for a in snapshot['applications']:
  identity=snapshot['identities'][str(a['pid'])];key=(a['uuid'],a['pid'])
  if key in foreign:
   need(identity==baseline['identities'][str(a['pid'])],'Foreign PID reused or scope changed')
  else:
   need(service is not None and a['uuid'] in uuids and membership(identity,relative) and uid_read(a['pid'])==os.getuid() and descendant_check(a['pid'],service['pid']),'Unknown/foreign GPU process or unleased UUID during retriever startup')
   need(safety.pid_identity(a['pid'])==identity,'Starting owner identity changed');own.add(key)
 for uid,g in gpus.items():
  foreign_memory=sum(a['used_mib'] for a in snapshot['applications'] if a['uuid']==uid and (uid,a['pid']) in foreign)
  need(foreign_memory/g['total_mib']<=.15,'Foreign GPU memory exceeds15%')
 return foreign,own

def qualify_zero_own(safety,snapshot,foreign,own,windows,query):
 """Original V4 zero-own/NOT_FOUND eligibility; NEVER a numeric SM acceptance."""
 need=safety.require;targets=set();zero=set();unknown=set();evidence={}
 for gpu in snapshot['gpus']:
  uid=gpu['uuid']
  if gpu['sm']<=15 or not any(u==uid for u,_ in foreign):continue
  targets.add(uid);window=windows.get(uid)
  need(window is not None and window['status'] in ('SUCCESS','NVML_ERROR_6'),'Unsupported zero-own window')
  start,end=window['start_us'],window['end_us']
  need(end<=query and 0<=query-end<=2_000_000 and 1_000_000<=end-start<=3_000_000,'Invalid zero-own sampling window')
  rows=window['samples']
  if window['status']=='NVML_ERROR_6':
   need(rows==[] and any(u==uid for u,_ in own),'NOT_FOUND empty window lacks resident known-owned retriever')
   unknown.add(uid);zero.add(uid);evidence[uid]={'scope':'UNKNOWN_NOT_MEASURED_ZERO'};continue
  need(rows,'Empty zero-own sampling window');known={pid for u,pid in foreign|own if u==uid};maxima={};has_own=False;positive=False
  for row in rows:
   need(row['pid'] in known,'Unknown PID in zero-own samples')
   need(start<row['timestamp_us']<=end and 0<=query-row['timestamp_us']<=1_500_000,'Stale/future zero-own sample at actual query')
   need(math.isfinite(row['sm']) and 0<=row['sm'] and ((uid,row['pid']) in own or row['sm']<=100),'Invalid zero-own SM')
   key=(uid,row['pid'])
   if key in own:has_own=True;positive=positive or row['sm']>0
   if key in foreign:maxima[row['pid']]=max(maxima.get(row['pid'],0),row['sm'])
  need(sum(maxima.values())<=15,'Observed foreign SM exceeds15% in rejected window')
  need(has_own,'No actual known-own returned row for transition')
  if not positive:zero.add(uid)
  evidence[uid]={'observed_foreign_maxima':maxima,'returned_known_own':has_own,'positive_own':positive,'scope':'RETRY_QUALIFICATION_NOT_NUMERIC_ACCEPTANCE'}
 need(zero,'No actual high zero-own target')
 return {'targets':sorted(targets),'zero_own_targets':sorted(zero),'not_found_unknown_targets':sorted(unknown),'evidence':evidence}

def qualify_activity_transition(safety,snapshot,baseline,uuids,service,relative,foreign,own,descendant_check,uid_read,raw_writer=None,lease_reader=None):
 """UNKNOWN old window authorizes nothing; use complete new original measurements."""
 windows,query=safety.collect_process_windows(snapshot,foreign)
 raw=dict(initial=snapshot,windows=windows,query_completed_us=query)
 if raw_writer:raw_writer(dict(stage='activity-transition-windows-before-validation',epoch=time.time(),actual=raw))
 try:bounds,evidence=safety.sm_aggregate_bounds(snapshot,foreign,own,windows,query)
 except RuntimeError as error:
  error.measurement_evidence=raw
  if str(error) not in ('No positive known-own sample proves active process sampler','High-device-SM needs supported fresh process samples'):raise
  try:
   eligibility=qualify_zero_own(safety,snapshot,foreign,own,windows,query)
   final=safety.nvml_snapshot()
   if raw_writer:raw_writer(dict(stage='activity-transition-postfailure-before-validation',epoch=time.time(),actual=final))
   f,o=ownership(safety,final,baseline,uuids,service,relative,descendant_check,uid_read)
   now=int(time.time()*1_000_000)
   safety.require(f==foreign and final['identities']==snapshot['identities'],'Activity transition protected postfailure identity changed')
   safety.require(0<=now-query<=1_500_000,'Activity transition qualification stale')
   if lease_reader:safety.require(set(lease_reader())==set(uuids),'Held8 leases changed during rejected activity-transition qualification')
   raw.update(final=final,eligibility=eligibility,evaluation_now_us=now)
  except Exception as qualification_error:
   qualification_error.measurement_evidence=raw;raise
  raise KnownOwnAttachTransition('Activity transition original zero-own UNKNOWN requires next full fresh measurement',raw) from error
 final=safety.nvml_snapshot()
 if raw_writer:raw_writer(dict(stage='activity-transition-final-before-validation',epoch=time.time(),actual=final))
 f,o=ownership(safety,final,baseline,uuids,service,relative,descendant_check,uid_read)
 safety.require(f==foreign and final['identities']==snapshot['identities'],'Protected PID identity changed during fresh activity-transition qualification')
 if lease_reader:safety.require(set(lease_reader())==set(uuids),'Held8 leases changed during activity-transition qualification')
 if o!=own:
  for uid,_ in foreign:safety.require(uid in bounds and math.isfinite(bounds[uid]) and 0<=bounds[uid]<=15,'Foreign aggregate SM exceeds15% in original qualification')
  raise KnownOwnAttachTransition('Verified same-PID owned UUID residency changed; requires next complete fresh measurement',dict(initial=snapshot,final=final,windows=windows,query_completed_us=query,initial_bounds=bounds,initial_evidence=evidence,initial_own_pid_uuids=sorted(own),final_own_pid_uuids=sorted(o),scope='QUALIFICATION_ONLY_NEW_HIGH_CARD_UNKNOWN_NOT_ACCEPTED'))
 final_bounds,final_evidence=safety.sm_aggregate_bounds(final,foreign,own,windows,int(time.time()*1_000_000),actual_query_completed_us=query)
 for uid,_ in foreign:safety.require(uid in bounds and uid in final_bounds and math.isfinite(bounds[uid]) and math.isfinite(final_bounds[uid]) and max(bounds[uid],final_bounds[uid])<=15,'Foreign aggregate SM exceeds15% or unknown')
 return dict(initial=snapshot,final=final,windows=windows,query_completed_us=query,initial_evidence=evidence,final_evidence=final_evidence)

def sample(safety,baseline,uuids,service,relative,*,lease_reader,descendant_check=descendant,uid_read=None,raw_writer=None):
 def capture(stage,value):
  if raw_writer:raw_writer(dict(stage=stage,epoch=time.time(),actual=value))
  return value
 def snapshot(stage):return capture(stage,safety.nvml_snapshot())
 need=safety.require;need(set(lease_reader())==set(uuids),'Actual held8 UUID leases changed')
 initial=snapshot("initial-before-ownership-validation");foreign,own=ownership(safety,initial,baseline,uuids,service,relative,descendant_check,uid_read)
 index={g['index']:g['uuid'] for g in initial['gpus']};visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
 need(len(visible)==4 and {index.get(v,v) for v in visible}==set(uuids),'Actual visible GPU allocation differs from leased UUIDs')
 windows,query_completed_us=safety.collect_process_windows(initial,foreign)
 capture("original-windows-before-sm-validation",dict(windows=windows,query_completed_us=query_completed_us))
 try:initial_bounds,initial_evidence=safety.sm_aggregate_bounds(initial,foreign,own,windows,query_completed_us)
 except RuntimeError as error:
  raw_failure=dict(initial=initial,windows=windows,query_completed_us=query_completed_us,original_error=str(error))
  error.measurement_evidence=raw_failure
  if str(error) in ('No positive known-own sample proves active process sampler','High-device-SM needs supported fresh process samples'):
   # Validate ALL targets, including later-card breaches, before any retry.
   try:
    eligibility=qualify_zero_own(safety,initial,foreign,own,windows,query_completed_us)
    final=snapshot("final-before-ownership-validation");f,o=ownership(safety,final,baseline,uuids,service,relative,descendant_check,uid_read)
    now=int(time.time()*1_000_000)
    need(0<=now-query_completed_us<=1_500_000,'Rejected-window postfailure qualification stale')
    need(f==foreign and final['identities']==initial['identities'],'Identity/UID changed during zero-own qualification')
    need(set(lease_reader())==set(uuids),'Held8 leases changed during zero-own qualification')
    raw_failure.update(final=final,postfailure_own_pid_uuids=sorted(o),eligibility=eligibility,evaluation_now_us=now)
   except Exception as qualification_error:
    qualification_error.measurement_evidence=raw_failure;raise
   raise KnownOwnAttachTransition('Original V4 zero-own/NOT_FOUND qualified UNKNOWN; requires complete next fresh measurement',raw_failure) from error
  if str(error)!='Unknown PID in process utilization samples':raise
  fresh=snapshot("fresh-before-ownership-validation");fresh_foreign,fresh_own=ownership(safety,fresh,baseline,uuids,service,relative,descendant_check,uid_read)
  if fresh_foreign==foreign and fresh_own!=own and all(fresh['identities'][str(pid)]==baseline['identities'][str(pid)] for _,pid in foreign):
   # Retried only after every previously unknown sampled PID is real owned retriever.
   sampled={(uid,r['pid']) for uid,w in windows.items() for r in w.get('samples',[])}
   if sampled<=(foreign|fresh_own):
    # Original collector cursor/status/finiteness/freshness and every foreign maximum
    # must pass before permitting a new COMPLETE sample; never retry a hidden breach.
    qualified,qualification=safety.sm_aggregate_bounds(initial,foreign,fresh_own,windows,query_completed_us)
    now=int(time.time()*1_000_000)
    qualified_final,qualification_final=safety.sm_aggregate_bounds(fresh,foreign,fresh_own,windows,now,actual_query_completed_us=query_completed_us)
    for uid,_ in foreign:need(uid in qualified and uid in qualified_final and math.isfinite(qualified[uid]) and math.isfinite(qualified_final[uid]) and max(qualified[uid],qualified_final[uid])<=15,'Foreign aggregate SM exceeds15% or unknown')
    raise KnownOwnAttachTransition('Known-owned retriever attached during original process window',dict(initial=initial,final=fresh,windows=windows,query_completed_us=query_completed_us,evaluation_now_us=now,qualification=qualification,qualification_final=qualification_final)) from error
  raise
 for uid,_ in foreign:need(uid in initial_bounds and math.isfinite(initial_bounds[uid]) and 0<=initial_bounds[uid]<=15,'Foreign aggregate SM exceeds15% or unknown')
 final=snapshot("final-before-ownership-validation");final_foreign,final_own=ownership(safety,final,baseline,uuids,service,relative,descendant_check,uid_read)
 # Only a real known-own attach/exit with all protected identities intact gets fresh retry.
 if final['identities']!=initial['identities']:
  if foreign==final_foreign and all(final['identities'][str(pid)]==baseline['identities'][str(pid)] for _,pid in foreign):
   if any(initial_evidence[uid]['method']=='DEVICE_TOTAL_AGGREGATE_UPPER_BOUND' and next(g['sm'] for g in final['gpus'] if g['uuid']==uid)>15 for uid,_ in foreign):
    qualified=qualify_activity_transition(safety,final,baseline,uuids,service,relative,foreign,final_own,descendant_check,uid_read,raw_writer=raw_writer,lease_reader=lease_reader)
    raise KnownOwnAttachTransition('Known owned retriever attached with rising SM; complete supported fresh qualification passed',qualified)
   final_bounds,final_evidence=safety.sm_aggregate_bounds(final,foreign,final_own,windows,int(time.time()*1_000_000),actual_query_completed_us=query_completed_us)
   for uid,_ in foreign:need(uid in final_bounds and math.isfinite(final_bounds[uid]) and 0<=final_bounds[uid]<=15,'Foreign aggregate SM exceeds15% or unknown')
   raise KnownOwnAttachTransition('Known-owned retriever identity attached/exited during measurement; repeat complete fresh sample',dict(initial=initial,final=final,windows=windows,query_completed_us=query_completed_us,initial_bounds=initial_bounds,initial_evidence=initial_evidence))
  raise ValueError('Protected/unknown identity changed during startup measurement')
 final_gpus={g['uuid']:g for g in final['gpus']}
 for uid,e in initial_evidence.items():
  if e['method']=='DEVICE_TOTAL_AGGREGATE_UPPER_BOUND' and final_gpus[uid]['sm']>15:
   qualified=qualify_activity_transition(safety,final,baseline,uuids,service,relative,foreign,final_own,descendant_check,uid_read,raw_writer=raw_writer,lease_reader=lease_reader)
   raise KnownOwnAttachTransition('Known owned retriever activity rose after low-device bound; supported fresh qualification passed',qualified)
 now_us=int(time.time()*1_000_000)
 bounds,evidence=safety.sm_aggregate_bounds(final,foreign,own,windows,now_us,actual_query_completed_us=query_completed_us)
 for uid in bounds:
  bounds[uid]=max(bounds[uid],initial_bounds[uid]);evidence[uid].update(upper_bound=bounds[uid],initial_evidence=initial_evidence[uid],scope='TWO_FOREIGN_BOUND_SNAPSHOTS_MAXIMUM')
 for uid,pid in foreign:
  need(uid in bounds and math.isfinite(bounds[uid]) and 0<=bounds[uid]<=15,'Foreign aggregate SM exceeds15% or unknown')
 return dict(scope='STARTING_RETRIEVER_PARTIAL_RESIDENCY_NOT_ALL4_READY',raw_initial_snapshot=initial,raw_snapshot=final,actual_service_identity=service,own_pid_uuids=sorted(final_own),foreign_pid_uuids=sorted(foreign),foreign_sm_aggregate_upper_bounds=bounds,foreign_sm_evidence=evidence,windows=windows,query_completed_us=query_completed_us,evaluation_now_us=now_us,held_uuid_lease_set=sorted(uuids))

def ownership_progress(error,service,uuids):
 """Only validated monotonic same-PID UUID attachment; not numeric acceptance."""
 raw=error.evidence
 if not service or not isinstance(raw,dict) or raw.get('scope')!='QUALIFICATION_ONLY_NEW_HIGH_CARD_UNKNOWN_NOT_ACCEPTED':return None
 initial,final=raw['initial'],raw['final'];pid=service['pid'];key=str(pid)
 if initial['identities']!=final['identities'] or initial['identities'].get(key)!=service:raise RuntimeError('Owned progress protected actual identities changed')
 before={tuple(x) for x in raw['initial_own_pid_uuids']};after={tuple(x) for x in raw['final_own_pid_uuids']};leased=set(uuids)
 if not before<after or any(p!=pid or u not in leased for u,p in before|after):raise RuntimeError('Owned progress requires monotonic same-service leasedUUID growth')
 for snapshot in (initial,final):
  observed={(a['uuid'],a['pid']) for a in snapshot['applications'] if a['pid']==pid}
  if observed!=(before if snapshot is initial else after):raise RuntimeError('Owned progress actual application residency differs')
 # Original sample has already checked actual UID/job/heldFD and all foreign identities.
 # Retain independent snapshot memory and already validated numeric foreign bounds.
 bounds=raw['initial_bounds']
 if set(bounds)!=leased or any(not math.isfinite(v) or not 0<=v<=15 for v in bounds.values()):raise RuntimeError('Foreign bounds invalid during owned progress')
 foreign={(a['uuid'],a['pid']) for a in initial['applications'] if a['pid']!=pid}
 if foreign!={(a['uuid'],a['pid']) for a in final['applications'] if a['pid']!=pid}:raise RuntimeError('Foreign residency changed during owned progress')
 for snapshot in (initial,final):
  gpus={g['uuid']:g for g in snapshot['gpus']}
  if set(gpus)!=leased:raise RuntimeError('Allocated GPU set changed during owned progress')
  for u in leased:
   total=gpus[u]['total_mib'];memory=sum(a['used_mib'] for a in snapshot['applications'] if a['uuid']==u and a['pid']!=pid)
   if not math.isfinite(total) or total<=0 or not math.isfinite(memory) or not 0<=memory/total<=.15:raise RuntimeError('Foreign memory exceeds15% during owned progress')
 return dict(before=before,after=after,new=after-before)

def bounded_sample(call,*,max_attempts=6,seconds=30,pause=.25,progress_service=None,progress_uuids=None):
 started=time.monotonic();history=[];ordinary=0;progress=set();seen_residency=set();calls=0
 while True:
  if calls and time.monotonic()-started>=seconds:raise TimeoutError('Startup retry exhausted original30s call budget')
  calls+=1
  try:
   value=call()
   if time.monotonic()-started>seconds:raise TimeoutError('Complete startup measurement exceeded original30s call budget')
   value['known_own_attach_retry_history']=history;return value
  except KnownOwnAttachTransition as e:
   try:
    qualified=ownership_progress(e,progress_service,progress_uuids) if progress_service else None
    added=qualified['new'] if qualified else None
    if qualified and not seen_residency<=qualified['before']:raise RuntimeError('Owned startup historical UUID residency regressed')
   except BaseException as qualification_error:
    qualification_error.measurement_evidence=e.evidence;qualification_error.retry_history=list(history);raise
   history.append({'attempt':calls,'error':str(e),'epoch':time.time(),'raw_failed_measurement':e.evidence,'retry_kind':'INITIALIZATION_ONLY_VERIFIED_UUID_PROGRESS' if added else 'ORIGINAL_ORDINARY_UNKNOWN','new_progress_pid_uuids':sorted(added) if added else []});e.retry_history=list(history)
   if added:
    if added&progress:raise RuntimeError('Owned startup UUID progress repeated') from e
    progress.update(added);seen_residency=qualified['after']
    if len(progress)>8:raise RuntimeError('Owned startup finite8UUID progression exceeded') from e
   else:ordinary+=1
   if ordinary>=max_attempts or time.monotonic()-started>=seconds:raise
   time.sleep(pause)
  except Exception as error:
   error.retry_history=list(history);raise
