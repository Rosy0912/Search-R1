"""Read actual snapshot originals; structural saved-record join, not full PT/HF inspection."""
import gzip,json,hashlib,datetime,re,importlib.util
from pathlib import Path
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/runs/formal-3261')
HERE=Path(__file__).resolve().parent
SNAP=HERE/'actual-snapshots/3261-20261005T201537333521Z';FILES=SNAP/'original-files'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_bytes())
manifest_path=FILES/'full_state/global_step_5/manifest.json';manifest=load(manifest_path)
need(sha(manifest_path)=='8c6e17b2af0026b1267a2422c1437aa00461da46e0a1911caea3633bfef473d3','actual checkpoint manifest')
need((manifest['actual_outer_updates'],manifest['global_step'],manifest['next_global_step'],manifest['horizon'],manifest['warmup_steps'],manifest['world_size'])==(5,5,6,200,0,8),'saved contract')
need(manifest['source_commit']=='188f4e5e7b63cf479290d0888a4a075499886750' and manifest['job_id']=='3261' and manifest['driver_pid']==1533312 and manifest['uid']==0,'actual identity')
summary=load(SNAP/'observation-summary.json');observed=load(SNAP/'observed-remote-original.json')
joined=[];large=[]
for key,rec in manifest['files'].items():
 relative=Path(rec['path']).relative_to(ROOT);original=FILES/relative
 if not original.exists():large.append({'key':key,**rec});continue
 need(sha(original)==rec['sha256'],'copied record SHA mismatch '+key)
 entry=summary['files'][str(relative)];need(entry['sha256']==rec['sha256'] and entry['unchanged_during_read'] and entry['same_inode'] and entry['before']['uid']==entry['after']['uid']==0,'original read unstable '+key)
 joined.append({'key':key,'sha256':rec['sha256'],'bytes':original.stat().st_size})
need(set(x['key'] for x in large)==({'driver/driver.pt'}|{'actor/training_state/rank_'+str(r)+'.pt' for r in range(8)}|{'actor/model-0000'+str(r)+'-of-00003.safetensors' for r in range(1,4)}),'unexpected unavailable binary set '+repr([x['key'] for x in large]))
rank_joins=[]
workerpids=[]
for rank in range(8):
 path=FILES/('checkpoints/actor/global_step_5/training_state/rank_'+str(rank)+'.json');report=load(path);pid=report['pid'];proc=observed['processes'][str(pid)]
 need((report['version'],report['algorithm'],report['status'],report['rank'],report['world_size'],report['step'],report['scheduler_last_epoch'],report['model_dtensor_entries'])==(3,'pure_opsd','EXACT_FULL_STATE_RELOAD_OK',rank,8,5,5,435),'actual rank report contract')
 need(report['path']==manifest['files']['actor/training_state/rank_'+str(rank)+'.pt']['path'] and report['sha256']==report['state_sha256']==manifest['files']['actor/training_state/rank_'+str(rank)+'.pt']['sha256'],'actual binary-report join')
 need(proc['uids']==[0]*4 and proc['in_observed_job'] and proc['state'] in ('R','S','D','I'),'actual saved worker own live process')
 matches=[x for x in observed['unfiltered_application_csv_rows'] if int(x[1])==pid]
 need(len(matches)==1,'actual one physical GPU per saved rank')
 teacher=report['teacher'];need(teacher['parameter_count']==3085938688 and teacher['execution_device']=='cuda' and teacher['storage_device']=='cpu' and teacher['counters']['cuda_update_entries']==5,'actual teacher update record')
 if rank:need(teacher['model_content_digest']==rank_joins[0]['teacher_model_digest'] and teacher['buffer_content_digest']==rank_joins[0]['teacher_buffer_digest'],'rank teacher digest differs')
 rank_joins.append({'rank':rank,'pid':pid,'start_ticks':proc['start_ticks'],'gpu_uuid':matches[0][0].strip(),'report_sha256':sha(path),'state_sha256':report['state_sha256'],'state_digest':report['state_digest'],'object_ids':report['object_ids'],'teacher_model_digest':teacher['model_content_digest'],'teacher_buffer_digest':teacher['buffer_content_digest']});workerpids.append(pid)
need(len(set(workerpids))==8 and len(set(x['gpu_uuid'] for x in rank_joins))==8,'nonunique8rank identity')
driver=observed['processes']['1533312'];need(driver['in_observed_job'] and driver['uids']==[0]*4 and driver['start_ticks']=='1829085316','actual driver live unchanged')
dr=load(FILES/'full_state/global_step_5/driver.json');need(dr['status']=='EXACT_DRIVER_STATE_RELOAD_OK' and dr['pid']==1533312 and dr['job_id']=='3261' and dr['step']==5 and dr['state_sha256']==manifest['files']['driver/driver.pt']['sha256'],'driver record binding')
rows=[json.loads(x) for x in gzip.decompress((SNAP/'resources-prefix-original.ndjson.gz').read_bytes()).splitlines()]
need(len(rows)==469 and all(x['job_id']=='3261' for x in rows),'all resource prefix actualjob')
known=set(range(1531212,1531220))|{1531763}|set(workerpids);seen=set()
for row in rows:
 need(len(row['gpus'])==8 and len(set(x['uuid'] for x in row['gpus']))==8,'all resource eight GPU inventory')
 for app in row['gpu_processes']:
  need(app['pid'] in known and app['uid']=='0' and app['in_slurm_job'] is True,'foreign observed GPU process')
  seen.add(app['pid'])
need(seen==known,'not all hardware/worker/retrieval PIDs witnessed')
times=[datetime.datetime.fromisoformat(x['time']) for x in rows];deltas=[(b-a).total_seconds() for a,b in zip(times,times[1:])]
need(min(deltas)>0,'resource time nonmonotonic')
resource={'records':len(rows),'start_utc':rows[0]['time'],'end_utc':rows[-1]['time'],'min_sample_gap_seconds':min(deltas),'max_sample_gap_seconds':max(deltas),'sampled_max_gpu_mib':max(g['used_mib'] for row in rows for g in row['gpus']),'paused_sample_gpu_mib':[g['used_mib'] for g in rows[-1]['gpus']],'sampled_foreign_gpu_processes':0,'observed_gpu_pids':sorted(seen),'gzip_sha256':sha(SNAP/'resources-prefix-original.ndjson.gz'),'scope':'SAMPLED_PREFIX_NOT_CONTINUOUS_PEAK_OR_JOB_MEMORY_LIMIT_PROOF'}
output={'status':'PASS_ACTUAL_FIRST5_LOG_SAVED_RECORD_RESOURCE_JOINS_ONLY_AWAITING_FULL_CPU_STATE_HF','source_commit':manifest['source_commit'],'job_id':'3261','driver_pid':1533312,'driver_start_ticks':driver['start_ticks'],'checkpoint_manifest_sha256':sha(manifest_path),'pause_nonce':manifest['pause_nonce'],'snapshot_original_sha256':sha(SNAP/'observed-remote-original.json'),'joined_local_original_checkpoint_records':joined,'binary_records_pending_actual_cpu_full_sha_and_contents':large,'rank_joins':rank_joins,'resource_prefix':resource,'first5_full_implementation_acceptance':False,'effect_improvement':False,'limitations':['JSON producer claims joined to pinned PT files; full binary state contents and official base/HF tensors await actual CPU inspector','All original observed GPU application processes were own3261, but sample gaps reach recorded maximum; no claim of continuous proof','memory.current/events apply system slurmd hierarchy, not isolated job-specific memory enforcement; no inference of per-job OOM limit','Checkpoint record exact reload claims reflect same live-object source path; no alternate-job restart acceptance']}
with (HERE/'ACTUAL3261_FIRST5_SAVED_RECORD_RESOURCE_INDEPENDENT_JOIN.json').open('x') as f:json.dump(output,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'report_sha256':sha(HERE/'ACTUAL3261_FIRST5_SAVED_RECORD_RESOURCE_INDEPENDENT_JOIN.json'),'resources':resource,'rank_mapping':[{k:r[k] for k in ('rank','pid','gpu_uuid')} for r in rank_joins]},sort_keys=True))
