"""Independent raw reader for actual3046; no submission, no proof generation."""
from pathlib import Path
import hashlib,json
P=Path('outputs/remote-evidence/v7-burst-3046')
def require(x,msg):
 if not x:raise RuntimeError(msg)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
r=json.loads((P/'result.json').read_text());ready=json.loads((P/'ready.json').read_text())
trace=[json.loads(x) for x in (P/'guard.jsonl').read_text().splitlines()]
freeze=json.loads(Path('outputs/qwen25-32b-v7-burst-probe-v1-freeze.json').read_text())
require(r['files']==freeze['files'],'Deployed probe files differ')
require(r['source_commit']=='413a10c50f32febaafc83db8170e2b005e0e979f','Source413 mismatch')
require(ready['torch_version']=='2.4.0+cu121','Actual Torch differs')
owner=ready['pid'];uuids={v['uuid'] for v in ready['devices']};require(len(uuids)==8,'Eight UUIDs missing')
for index,row in enumerate(ready['devices']):
 require(row['device']==index and row['allocated']==row['reserved']==0,'Context ordinal/release wrong')
 require({p['pid'] for p in row['actual_gpu_pid_uuids']}=={owner},'Context not same owner')
 require({p['uuid'] for p in row['actual_gpu_pid_uuids']}=={d['uuid'] for d in ready['devices'][:index+1]},'Actual ordinal UUID map differs')
base=r['baseline'];foreign={(x['uuid'],x['pid']) for x in base['applications'] if x['uuid'] in uuids}
require(len({p for _,p in foreign})==8,'Baseline8 missing')
reference=None;pending=None;measured=[];high=[]
for row in trace:
 require(row['sealed_sha256']=='c7b16f1dd5d7cc133e55f034025c7dc8181ddcafaf59bdccfda02ceaf51f081d','C7 source differs')
 require(row['wrapper_sha256']=='12a2990031e29724b804907e55714eac3bb3896ed31c268a1c4ffeb1656516d0','V7 source differs')
 if row['attempt']==1:pending=None
 for cap in row['captures']:
  snap=cap['raw_snapshot'];identity=(snap['identities'],cap['uids'])
  require(not cap['uid_errors'],'Unknown actual UID')
  if reference is None:reference=identity
  require(identity==reference,'Any observed owner/foreign identity or UID changed')
  require(set(snap['identities'])==set(cap['uids']),'UID missing')
  actual={(p['uuid'],p['pid']) for p in snap['applications'] if p['uuid'] in uuids}
  require(actual==foreign|{(uid,owner) for uid in uuids},'Unknown/missing GPU owner')
  for pid,v in base['identities'].items():require(snap['identities'][pid]==v,'Baseline identity differs')
  require(cap['uids'][str(owner)]==0,'Owner UID not native root')
  g={g['uuid']:g for g in snap['gpus'] if g['uuid'] in uuids};require(set(g)==uuids,'Actual GPU missing')
  if all(v['sm']>15 for v in g.values()):high.append({u:v['sm'] for u,v in g.items()})
  for uid,v in g.items():
   mem=sum(p['used_mib'] for p in snap['applications'] if (p['uuid'],p['pid']) in foreign and p['uuid']==uid)
   require(mem<=.15*v['total_mib'],'Foreign memory exceeds15')
 if row.get('error'):
  require(row['error']=='Device SM increased after upper-bound sample' and not row['accepted'] and row['retry_authorized'],'Unapproved real error')
  require(row['foreign_sm_status']=='UNKNOWN_REJECTED_TRANSITION','Old bound interpreted as accepted')
  first,last=[{g['uuid']:g['sm'] for g in cap['raw_snapshot']['gpus']} for cap in (row['captures'][0],row['captures'][-1])]
  pending={u for u in uuids if first[u]<=15 and last[u]>15};require(pending,'No actual low-high pair')
 if row.get('fresh_measurement_uuids'):
  require(row['accepted'] and set(row['fresh_measurement_uuids'])==pending,'Fresh acceptance not bound to rejection')
  details={}
  for uid in pending:
   ev=row['fresh_foreign_sm_evidence'][uid];bound=row['fresh_foreign_sm_upper_bounds'][uid]
   require(0<=bound<=15 and ev['upper_bound']==bound,'Foreign bound invalid')
   if ev['method']=='FRESH_SUPPORTED_NONZERO_PROCESS_WINDOW':
    w=ev['window'];require(w['status']=='SUCCESS' and owner in ev['positive_own_pids'],'No actual supported ownpositive sample')
    require(any(s['pid']==owner and s['sm']>0 for s in w['samples']),'Own positive missing')
    require(all(0<=int(row['captures'][-1]['epoch']*1e6)-s['timestamp_us']<=1500000 for s in w['samples']),'Actual process sample stale')
    require(len({s['timestamp_us'] for s in w['samples']})==1,'Multiple completed periods')
   else:
    require(ev['method']=='DEVICE_TOTAL_AGGREGATE_UPPER_BOUND','Unsupported evidence method')
    final={g['uuid']:g for g in row['captures'][-1]['raw_snapshot']['gpus']};require(final[uid]['sm']<=15,'Fresh device bound not safe')
   details[uid]=dict(method=ev['method'],upper_bound=bound)
  measured.append(details);pending=None
require(len(measured)==3 and high,'Missing all3 real typed/fresh coverage or all8 high sample')
for phase in (1,2,3):
 done=json.loads((P/f'done-{phase}.json').read_text())
 require(done['pid']==owner and done['phase']==phase and not done['errors'],'Wrong phase/owner or pulse failed')
 require({v['uuid'] for v in done['devices']}==uuids and all(v['allocated']==v['reserved']==0 for v in done['devices']),'All8 release not zero')
 trigger=json.loads((P/f'trigger-{phase}.json').read_text());require(trigger['actual_snapshot']['identities']==reference[0],'Trigger identity differs')
require(r['cleanup_record']['actual_cgroup_empty'] and not r['cleanup_record']['remaining'],'Cleanup cgroup not empty')
final=r['final'];require({(p['uuid'],p['pid']) for p in final['applications'] if p['uuid'] in uuids}==foreign,'Own PID remains or baseline differs')
for pid,v in base['identities'].items():require(final['identities'][pid]==v,'Final foreign identity differs')
require(r['parent_cuda_initialized'] is False and r['baseline_preserved'],'Parent/baseline final failure')
out=dict(scope='READONLY_RAW_TELEMETRY_ONLY_NOT_TEACHER_OR_ALGORITHM_ACCEPTANCE',status='RAW_PHYSICAL_CHECKS_PASS',owner_pid=owner,uuid_count=8,actual_all8_high_samples=high,three_typed_rejected_to_fresh_accepted=measured,release_receipts=24,own_cleanup_actual_empty=True,raw_sha256={n:sha(P/n) for n in ('result.json','guard.jsonl','ready.json','done-1.json','done-2.json','done-3.json')})
Path('outputs/v7-burst-3046-producer-raw-review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
