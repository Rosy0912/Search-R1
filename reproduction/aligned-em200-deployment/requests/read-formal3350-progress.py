"""Bound read-only snapshot; no resume, source edits, imports, or training claims."""
from pathlib import Path
import os,json,hashlib,subprocess,datetime,stat
R=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
SOURCE='080ebbc46b89d61567cd33e1a5e2f1b40afd8b84';TOKEN='5fc47201ff6b4212ab6b73dee2bfefb5'
RUN=R/'runs'/('formal-aligned-em200-'+TOKEN+'-3350');LIMIT=256*1024*1024
MARKERS=('FRESH200_ATTEMPT_JSON=','BAPO_TRAIN_METRICS_JSON=','BAPO_TRAIN_EM_AUDIT_JSON=','BASELINE_EVALUATION_METRICS_JSON=','FRESH200_STEP5_PAUSED_JSON=','FRESH200_STEP5_RELEASED_JSON=','FRESH200_COMPLETED_RUN_JSON=','SEARCH_R1_FINAL_METRICS_JSON=')
def require(v,m):
    if not v:raise RuntimeError(m)
def snapshot(path):
    require(path.resolve()==path and not path.is_symlink(),'noncanonical file')
    with path.open('rb') as f:
        st=os.fstat(f.fileno());require(stat.S_ISREG(st.st_mode) and st.st_size<=LIMIT,'oversized or nonregular file')
        raw=f.read(st.st_size);after=os.fstat(f.fileno());require(len(raw)==st.st_size and after.st_ino==st.st_ino and after.st_size>=st.st_size,'snapshot truncated/replaced')
    now=path.stat();require(now.st_ino==st.st_ino and now.st_dev==st.st_dev,'path replaced while reading')
    return raw,{'path':str(path),'snapshot_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'device':st.st_dev,'inode':st.st_ino,'file_bytes_after':now.st_size,'scope':'captured prefix; later appended bytes not inspected'}
def complete_lines(raw):
    end=raw.rfind(b'\n')+1
    return raw[:end].decode('utf-8').splitlines(),{'unparsed_incomplete_tail_bytes':len(raw)-end,'unparsed_incomplete_tail_sha256':hashlib.sha256(raw[end:]).hexdigest()}
require(os.uname().nodename=='h100-5' and RUN.resolve()==RUN,'wrong node/run')
raw,bind=snapshot(R/'scheduler-head/formal-rayip-v2/formal-submit-receipt.json');receipt=json.loads(raw)
require(receipt.get('source')==SOURCE and receipt.get('job')=='3350' and receipt.get('token')==TOKEN and receipt.get('run')==str(RUN),'wrong submission')
require(receipt.get('submission_namespace')=='formal-rayip-v2' and receipt.get('deployment_directory')==str(R/'tools/deployment/formal-rayip-v2') and receipt.get('previous_failed_job')=='3348','wrong retrynamespace')
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':'3350','source':SOURCE,'run':str(RUN),'submission':bind,'scope':'READONLY_LIVE_SNAPSHOT_NOT_FULL_LEDGER_OR_TRAINING_ACCEPTANCE'}
for key,args in [('slurm',['scontrol','show','job','3350','-o']),('queue',['squeue','-h','-j','3350','-o','%i|%T|%R|%k'])]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);result[key]={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
result['logs']={}
p=RUN/'train.log'
if p.exists():
 raw,b=snapshot(p);lines,tail=complete_lines(raw);groups={k:[] for k in MARKERS}
 for line in lines:
  for marker in MARKERS:
   if marker in line:groups[marker].append(json.loads(line.split(marker,1)[1]))
 result['logs']['train']={'binding':b,'tail':tail,'marker_counts':{k:len(v) for k,v in groups.items()},'latest':{k:(v[-1] if v else None) for k,v in groups.items()},'completed_metric_history':groups['BAPO_TRAIN_METRICS_JSON='],'em_proof_history':groups['BAPO_TRAIN_EM_AUDIT_JSON='],'debug_evaluation_history':groups['BASELINE_EVALUATION_METRICS_JSON=']}
p=RUN/'resource-watch.jsonl'
if p.exists():
 raw,b=snapshot(p);lines,tail=complete_lines(raw);rows=[json.loads(line) for line in lines if line.strip()];counts={};events={};notable=[]
 for row in rows:
  reason=str(row.get('reason'));counts[reason]=counts.get(reason,0)+1
  event=str(row.get('event'));events[event]=events.get(event,0)+1
  if row.get('pause') is True or row.get('error') is not None or row.get('event') is not None or (row.get('reason') is not None and row['reason']!='within_threshold'):notable.append(row)
 result['logs']['resources']={'binding':b,'tail':tail,'rows':len(rows),'reason_counts':counts,'event_counts':events,'historical_notable_rows':notable,'latest':rows[-1] if rows else None}
print(json.dumps(result,allow_nan=False))
