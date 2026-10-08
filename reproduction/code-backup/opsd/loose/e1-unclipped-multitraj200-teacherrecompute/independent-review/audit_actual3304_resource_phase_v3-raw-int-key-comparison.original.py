"""One-off local read-only original data audit. No remote or GPU calls."""
from pathlib import Path
import datetime,gzip,hashlib,json,math,os,stat,types
H=Path(__file__).resolve().parents[1];D=H/'job3304-first5-resource-phase-originals-v3';O=H/'independent-review/actual3304-resource-phase-v3-local-originals';O.mkdir(exist_ok=False)
PIDS=(2548974,2549227,2549228,2549229,2549231,2549232,2549230,2549233)

def sha(b):return hashlib.sha256(b).hexdigest()
def pairs(items):
 out={}
 for k,v in items:assert k not in out;out[k]=v
 return out
def load(b):return json.loads(b,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def stable(path):
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  before=os.fstat(fd)
  with os.fdopen(os.dup(fd),'rb') as f:b=f.read()
  after=os.fstat(fd);current=path.lstat()
  def ident(st):return (st.st_dev,st.st_ino,st.st_uid,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
  assert ident(before)==ident(after)==ident(current) and stat.S_ISREG(before.st_mode)
  return b,dict(path=str(path),sha256=sha(b),bytes=len(b),device=before.st_dev,inode=before.st_ino,uid=before.st_uid)
 finally:os.close(fd)

raw,manifest_rec=stable(D/'manifest.json');m=load(raw);assert manifest_rec['sha256']=='b2300073d9920e3d26a02edea8905c9fbfe374334e60c8c6b6a9c8c2a9f9d9ac'
assert m['source_commit']=='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6' and m['job_id']=='3304' and m['driver_pid']==2548319
assert m['manifest_sha256']=='30b38d6d96abd56ff956baf8a4ba954319c62049aff8263e0c9700037e2c8577' and m['pause_nonce']=='20bf4e0dddc9ac6eed8b0d1a9da656ad60d99e8d8bbf07f93ef8356abdfd5764'
actual_manifest_bytes,actual_manifest_rec=stable(H/'job3304-first5-live-binding-original/manifest.json');actual=load(actual_manifest_bytes);assert actual_manifest_rec['sha256']==m['manifest_sha256']
inputs={};blobs={}
for name in ('all-resources.ndjson','all-resources.ndjson.gz','training-resources.ndjson','training-resources.ndjson.gz'):
 b,r=stable(D/name);assert r['bytes']==m['files'][name]['bytes'] and r['sha256']==m['files'][name]['sha256'];inputs[name]=r;blobs[name]=b
assert gzip.decompress(blobs['all-resources.ndjson.gz'])==blobs['all-resources.ndjson'] and gzip.decompress(blobs['training-resources.ndjson.gz'])==blobs['training-resources.ndjson']
allraw=blobs['all-resources.ndjson'];sliceraw=blobs['training-resources.ndjson'];plan=m['plan'];assert allraw.endswith(b'\n') and sliceraw.endswith(b'\n')
assert sha(allraw[:100995983])=='c0a6dccaec7502045c468278e05c6f0de1a69038188d35e5b7b986a75b86f0fb'
lines=allraw.splitlines(keepends=True);assert len(lines)==1970;before_bad=(H/'actual3304-resource-neighbors-20261006T123907851465Z/resource-adjacent_before.ndjson').read_bytes();bad=(H/'actual3304-resource-neighbors-20261006T123907851465Z/resource-problem.ndjson').read_bytes();after_bad=(H/'actual3304-resource-neighbors-20261006T123907851465Z/resource-adjacent_after.ndjson').read_bytes();assert lines[64]==before_bad and lines[65]==bad and lines[66]==after_bad and sha(bad)=='6b9cd6ed80820bf21689c6d495fcc1201aee8cdd583a0c8fe253985c681feb91'
bound=1791287254032340562;epoch=datetime.datetime(1970,1,1,tzinfo=datetime.timezone.utc)
rows=[];times=[];offset=0;first=None;problems=[];aux_unavailable=[];uuids=None
for index,line in enumerate(lines):
 row=load(line);assert row['job_id']=='3304';t=datetime.datetime.fromisoformat(row['time']);assert t.tzinfo and t.utcoffset()==datetime.timedelta(0);delta=t-epoch;ns=(delta.days*86400+delta.seconds)*10**9+delta.microseconds*1000
 assert not times or ns>times[-1]
 if first is None and ns>=bound:first=(index,offset)
 gpu=row['gpus'];current={g['uuid'] for g in gpu};assert len(gpu)==len(current)==8
 assert uuids is None or current==uuids;uuids=current
 assert all(math.isfinite(float(g['util'])) and 0<=float(g['util'])<=100 and math.isfinite(float(g['used_mib'])) and float(g['used_mib'])>=0 for g in gpu)
 badapps=[a for a in row['gpu_processes'] if a.get('unavailable') or a.get('in_slurm_job') is not True or str(a.get('uid'))!='0']
 if 'query_error' in row or badapps:problems.append(dict(line=index+1,offset=offset,time=row['time'],applications=badapps,query_error=row.get('query_error')))
 unavailable=[a for a in row.get('job_processes',[]) if a.get('unavailable')]
 if unavailable:aux_unavailable.append(dict(line=index+1,count=len(unavailable),records=unavailable))
 rows.append(row);times.append(ns);offset+=len(line)
assert first==(69,3034485) and offset==176887890 and sliceraw==allraw[3034485:]
assert len(sliceraw.splitlines())==1901 and len(problems)==1 and problems[0]['line']==66 and not any(p['offset']>=3034485 for p in problems)
assert times[68]<bound<=times[69] and plan['slice_first_time']==rows[69]['time'] and plan['slice_last_time']==rows[-1]['time']
assert max((b-a)/1e9 for a,b in zip(times,times[1:]))==5.756452 and max((b-a)/1e9 for a,b in zip(times[69:],times[70:]))==3.566539
before=m['before_context'];after=m['after_context']
for key in ('driver','controller','workers'):assert before[key]==after[key]
assert before['driver']['pid']==2548319 and before['driver']['start']=='1834784786' and before['controller']['pid']==2533919 and before['controller']['start']=='1834728058'
assert tuple(r['pid'] for r in before['workers'])==PIDS
for r in [before['driver'],before['controller']]+before['workers']:assert r['uid']==0 and '/job_3304/step_0\n' in r['cgroup']
assert before['manifest']['sha256']==after['manifest']['sha256']==m['manifest_sha256'] and before['manifest']['opened']==after['manifest']['opened']
assert before['ready']['sha256']==after['ready']['sha256'] and before['ready']['opened']==after['ready']['opened']
for context in (before,after):assert context['paused']['manifest_sha256']==m['manifest_sha256'] and context['paused']['pause_nonce']==m['pause_nonce']
assert m['original_sampler']['opened']['inode']==138683997 and m['original_sampler']['opened']['device']==64768 and m['original_sampler']['bytes']==176887890
assert m['original_sampler']['unclosed_tail_bytes']==0
pins={'receipt':'7042f5a93158541e9ff0044bf4c21849d8de16eed159c9787fdd7ae8406c564e','gate':'09561719673a7c273ad39b3c9363b1a250eab0a94c20c80f40dd1dfc05589e16','formal':'ffb3bdc64f91c4a60905cd15994132c63c6494952c4ca0818c0cf7eee119d017'}
assert {k:v['sha256'] for k,v in m['dependencies'].items()}==pins
runtime=m['caller_provenance']['runtime'];assert runtime['sha256']=='ebd52ef36f644ce3d49b4e966f88314c0ddf40dc93e31c1b02308e38e281433e' and m['caller_provenance']['execution_pid']==2605227 and m['caller_provenance']['execution_uid']==0
assert m['caller_provenance']['actual_environment_job_id'] is None and m['caller_provenance']['actual_environment_step_id'] is None
neighbors=H/'actual3304-resource-neighbors-20261006T123907851465Z';original_neighbor=load((neighbors/'observation.json').read_bytes())
for role,name in [('environment','evidence/actual-training-wrapper-environment.json'),('post_capacity_idle','evidence/pre-training-double15-after-capacity.json'),('capacity_exit','evidence/capacity-micro4/exit.json')]:
 expected=original_neighbor['files'][name]['physical'];observed=m['actual_boundary'][role];assert observed['sha256']==expected['sha256'] and observed['opened']==expected['opened']
assert m['actual_boundary']['environment']['opened']['mtime_ns']==bound
receipt_bytes,receipt_rec=stable(H/'first5-1b16c2227caf/receipt_payload.py');assert receipt_rec['sha256']==pins['receipt'];receipt=types.ModuleType('independent_actual_phase_receipt');receipt.__file__=receipt_rec['path'];exec(compile(receipt_bytes,receipt_rec['path'],'exec',dont_inherit=True),receipt.__dict__)
idle=load((neighbors/'evidence/pre-training-double15-after-capacity.json').read_bytes());service=idle[0]['applications'][0]['owner'];reports=[dict(pid=p,rank=i) for i,p in enumerate(PIDS)]
joined=receipt.resource_join(D/'training-resources.ndjson',actual,reports,service);assert joined==m['peer_frozen_resource_join']
for w in before['workers']:
 for sample in joined['samples']:assert sample['identity'][w['pid']]['start']==w['start']
all_failure=None
try:receipt.resource_join(D/'all-resources.ndjson',actual,reports,service)
except ValueError as exc:all_failure=str(exc)
assert all_failure=='Foreign resource GPU PID'
for name,rec in inputs.items():_,afterrec=stable(D/name);assert afterrec==rec
_,afterrec=stable(D/'manifest.json');assert afterrec==manifest_rec
result=dict(version=1,status='PASS_ACTUAL_FIRST5_RESOURCE_TRAINING_CONTIGUOUS_RANGE_NOT_WHOLE_RUN_OR_EFFECT',source_commit=m['source_commit'],job_id='3304',driver_pid=2548319,pause_nonce=m['pause_nonce'],checkpoint_manifest_sha256=m['manifest_sha256'],phase_manifest=manifest_rec,actual_checkpoint_manifest=actual_manifest_rec,inputs=inputs,whole_original_bytes=176887890,whole_original_lines=1970,historical_prefix_bytes=100995983,historical_prefix_sha256='c0a6dccaec7502045c468278e05c6f0de1a69038188d35e5b7b986a75b86f0fb',exact_preserved_bad_row=dict(line=66,offset=2869512,sha256=sha(bad)),training_contiguous_range=dict(start_offset=3034485,end_offset=176887890,bytes=173853405,first_line=70,last_line=1970,lines=1901,first_time=rows[69]['time'],last_time=rows[-1]['time'],maximum_sample_gap_seconds=3.566539),whole_maximum_sample_gap_seconds=5.756452,full_unfiltered_gpu_errors=problems,auxiliary_job_process_unavailable=aux_unavailable,source_pinned_frozen_resource_join=joined,full_all_resource_join_true_failure=all_failure,originals_stable_before_after=True,limits=['Scope only actual complete-prefix snapshot through lastLF; no unclosed tail bytes were observed. It is not future writer lifetime or terminal proof.','Sampler is sampled metadata, not continuous peak/utilization or every request proof.','Historical raw process State/cgroup were not recorded; original false row remains false and its failure is not waived.','Actual creator/aftercontext live PID/start/cgroup checks are raw observations at capture time, not a promise that processes remain alive after transport.','Native rank ownership ordering joins creator source-validated saved-rank metadata context and independently known PID ordering; full native PT/HF/data/trajectory acceptance is a separate review.','Frozen formal whole-run resource_join still fails on original pretraining false row; no child/wrapper/Slurm terminal or final effect PASS.'],scope='Local full-byte/gzip/range/time/error/PID-start replay of actual3304 phase originals; no SSH/GPU/source edits/signing/release.')
path=O/'ACTUAL_RESOURCE_PHASE_INDEPENDENT_LOCAL_BYTES_REVIEW.json'
with path.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(report=str(path),sha256=sha(path.read_bytes()),status=result['status'],whole_all_error=all_failure,training_rows=1901,auxiliary_process_error_rows=len(aux_unavailable))))
