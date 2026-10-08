"""Local synthetic fixtures and explicitly identified read-only actual originals."""
from pathlib import Path
import copy,datetime,gzip,hashlib,json,os,stat,types
import pytest
HERE=Path(__file__).resolve().parent
H=HERE.parent

def module(path,name):
 m=types.ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec',dont_inherit=True),m.__dict__);return m
s=module(HERE/'snapshot_training_resources_once.py','phase_candidate')
receipt=module(H/'first5-1b16c2227caf/receipt_payload.py','actual_frozen_receipt')
gate=module(H.parent/'opsd-unclipped-multitraj200-teacherrecompute/search_r1/baselines/checkpoint_gate.py','actual_frozen_gate')
ORIGINALS=H/'actual3304-resource-neighbors-20261006T123907851465Z'

def encode(rows):return b''.join((json.dumps(x,sort_keys=True,allow_nan=False)+'\n').encode() for x in rows)
def sample(second):
 uuids=['gpu-'+str(i) for i in range(8)];uid=os.getuid()
 return dict(job_id='3304',time='2026-10-06T11:47:'+str(second).zfill(2)+'+00:00',gpus=[dict(uuid=u,util=10.,used_mib=40000.) for u in uuids],gpu_processes=[dict(pid=300+i,process_start_ticks=str(1000+i),uid=uid,uuid=u,in_slurm_job=True) for i,u in enumerate(uuids)]+[dict(pid=900,process_start_ticks='2000',uid=uid,uuid=u,in_slurm_job=True) for u in uuids])
def fixture():
 rows=[sample(v) for v in (26,29,35,38)];rows[0]['gpu_processes'][0]['in_slurm_job']=False
 raw=encode(rows);historical=len(encode(rows[:2]));return rows,raw,historical,s.sha(raw[:historical])
def plan(rows=None):
 original,_,_,_=fixture();rows=original if rows is None else rows;raw=encode(rows);h=len(encode(rows[:2]));return s.phase_plan(raw,s.BOUNDARY_NS,h,s.sha(raw[:h]))

def test_exact_contiguous_range_and_all_bad_preserved(tmp_path):
 rows,raw,h,p=fixture();training,report=s.phase_plan(raw,s.BOUNDARY_NS,h,p)
 assert training==raw[h:]==encode(rows[2:]);assert report['all_problem_rows'][0]['line_number']==1
 assert report['all_sha256']==s.sha(raw) and report['slice_start_offset']==h and report['slice_last_line']==4
 assert gzip.decompress(gzip.compress(raw,mtime=0))==raw
 reports=[dict(pid=300+i,rank=i) for i in range(8)];service=dict(pid=900,start='2000');manifest=dict(job_id='3304',uid=os.getuid())
 allpath=tmp_path/'all.ndjson';allpath.write_bytes(raw)
 with pytest.raises(ValueError,match='Foreign'):receipt.resource_join(allpath,manifest,reports,service)
 selected=tmp_path/'slice.ndjson';selected.write_bytes(training)
 joined=receipt.resource_join(selected,manifest,reports,service)
 assert len(joined['samples'])==2 and len(joined['native_rank_pids'])==8

@pytest.mark.parametrize('mutation', ['foreign','unavailable','query','uuid','util','uid'])
def test_any_training_problem_still_rejected(mutation):
 rows,_,_,_=fixture();target=rows[2]
 if mutation=='foreign':target['gpu_processes'][0]['in_slurm_job']=False
 if mutation=='unavailable':target['gpu_processes'][0]['unavailable']=True
 if mutation=='query':target['query_error']='actual failure'
 if mutation=='uuid':target['gpus'][0]['uuid']='other'
 if mutation=='util':target['gpus'][0]['util']=-1
 if mutation=='uid':target['gpu_processes'][0]['uid']=os.getuid()+1
 with pytest.raises(ValueError,match='Training resource'):plan(rows)

@pytest.mark.parametrize('mutation',['job','same-time','backward','no-training','one-training'])
def test_global_clock_and_job_boundaries(mutation):
 rows,_,_,_=fixture()
 if mutation=='job':rows[3]['job_id']='3305'
 if mutation=='same-time':rows[3]['time']=rows[2]['time']
 if mutation=='backward':rows[3]['time']=rows[1]['time']
 if mutation=='no-training':rows=rows[:2]
 if mutation=='one-training':rows=rows[:3]
 with pytest.raises(ValueError):plan(rows)

@pytest.mark.parametrize('raw',[b'{"job_id":"3304","job_id":"3304"}\n',b'{"x":NaN}\n',b'{"x":Infinity}\n'])
def test_duplicate_nonfinite_json_rejected(raw):
 with pytest.raises(ValueError):s.unique(raw)

@pytest.mark.parametrize('value',[False,1,'2026-10-06T11:47:35','2026-10-06T11:47:35+08:00'])
def test_exact_utc_boundary(value):
 with pytest.raises((ValueError,TypeError)):s.row_time_ns(value)

def test_historical_sha_and_unclosed_tail_rejected():
 _,raw,h,p=fixture()
 with pytest.raises(ValueError,match='drift'):s.phase_plan(raw,s.BOUNDARY_NS,h,'0'*64)
 with pytest.raises(ValueError,match='drift'):s.phase_plan(raw+b'{',s.BOUNDARY_NS,h,p)
 with pytest.raises(ValueError,match='Typed'):s.phase_plan(raw,True,h,p)

def test_read_physical_partial_tail_is_recorded_and_complete_prefix_retained(tmp_path):
 p=tmp_path/'raw';p.write_bytes(b'{"a":1}\n{"pending":')
 raw,proof=s.read_physical(p,100,prefix=True)
 assert raw==b'{"a":1}\n' and proof['unclosed_tail_bytes']==11 and proof['unclosed_tail_sha256']==s.sha(b'{"pending":')
 with pytest.raises(ValueError,match='SHA'):s.read_physical(p,100,'0'*64)
 with pytest.raises(ValueError,match='bound'):s.read_physical(p,1)
 link=tmp_path/'link';link.symlink_to(p)
 with pytest.raises(ValueError,match='Physical'):s.read_physical(link,100)

def test_exclusive_publication_refuses_replay_and_symlink(tmp_path):
 p=tmp_path/'output';s.exclusive_file(p,b'original');assert p.read_bytes()==b'original' and p.stat().st_mode&0o777==0o444
 with pytest.raises(FileExistsError):s.exclusive_file(p,b'replacement')
 link=tmp_path/'link';link.symlink_to(p)
 with pytest.raises(FileExistsError):s.exclusive_file(link,b'replacement')
 assert p.read_bytes()==b'original'

def test_once_directory_is_durably_reserved_before_member_write(monkeypatch,tmp_path):
 events=[];actual=s.os.fsync
 def observed(fd):
  st=os.fstat(fd);events.append((stat.S_ISDIR(st.st_mode),st.st_ino));actual(fd)
 monkeypatch.setattr(s.os,'fsync',observed)
 out=tmp_path/'once';s.reserve_once_directory(out)
 assert events==[(True,tmp_path.stat().st_ino),(True,out.stat().st_ino)]
 s.exclusive_file(out/'raw',b'preserved failure original')
 assert events[2][0] is False
 with pytest.raises(FileExistsError):s.reserve_once_directory(out)
 assert (out/'raw').read_bytes()==b'preserved failure original'

def test_bad_training_plan_retains_all_original_without_manifest_or_retry(tmp_path):
 rows,_,_,_=fixture();rows[2]['gpu_processes'][0]['in_slurm_job']=False
 raw=encode(rows);h=len(encode(rows[:2]));out=tmp_path/'once'
 s.start_resource_archive(out,raw)
 with pytest.raises(ValueError,match='Training resource'):s.phase_plan(raw,s.BOUNDARY_NS,h,s.sha(raw[:h]))
 assert (out/'all-resources.ndjson').read_bytes()==raw
 assert gzip.decompress((out/'all-resources.ndjson.gz').read_bytes())==raw
 assert not (out/'manifest.json').exists() and not (out/'training-resources.ndjson').exists()
 with pytest.raises(FileExistsError):s.start_resource_archive(out,raw)

def actual_boundary_inputs():
 obs=json.loads((ORIGINALS/'observation.json').read_bytes());files=obs['files']
 def read(name):return json.loads((ORIGINALS/name).read_bytes()),files[name]['physical']
 env,ep=read('evidence/actual-training-wrapper-environment.json');idle,ip=read('evidence/pre-training-double15-after-capacity.json');ex,xp=read('evidence/capacity-micro4/exit.json')
 return env,ep,idle,ip,ex,xp

def test_actual_post_capacity_boundary_originals_join(monkeypatch):
 monkeypatch.setattr(s.os,'getuid',lambda:0);owner,uuids=s.boundaries(*actual_boundary_inputs())
 assert owner['pid']==2543106 and owner['start']=='1834765439' and len(uuids)==8

@pytest.mark.parametrize('mutation',['usedbool','totalnan','exitbool','ownerunknown','ownerbool','ownerstart','uuidmissing','changedservice','clock','inode'])
def test_boundary_strict_negative_typed_identity(monkeypatch,mutation):
 monkeypatch.setattr(s.os,'getuid',lambda:0);args=list(copy.deepcopy(actual_boundary_inputs()));env,ep,idle,ip,ex,xp=args
 if mutation=='usedbool':idle[0]['inventory'][0]['used']=False
 if mutation=='totalnan':idle[0]['inventory'][0]['total']=float('nan')
 if mutation=='exitbool':ex['returncode']=False
 if mutation=='ownerunknown':idle[0]['applications'][0]['owner']['state']='unknown'
 if mutation=='ownerbool':idle[0]['applications'][0]['owner']['pid']=True
 if mutation=='ownerstart':idle[0]['applications'][0]['owner']['start']=1834765439
 if mutation=='uuidmissing':idle[0]['inventory'].pop()
 if mutation=='changedservice':idle[1]['applications'][0]['owner']['start']='999'
 if mutation=='clock':ep['opened']['mtime_ns']+=1
 if mutation=='inode':ep['opened']['inode']+=1
 with pytest.raises(ValueError):s.boundaries(*args)

def test_actual_raw_adjacent_identity_and_limitation():
 before=json.loads((ORIGINALS/'resource-adjacent_before.ndjson').read_bytes());bad=json.loads((ORIGINALS/'resource-problem.ndjson').read_bytes());after=json.loads((ORIGINALS/'resource-adjacent_after.ndjson').read_bytes())
 a=next(x for x in before['gpu_processes'] if x['pid']==2544824);b=next(x for x in bad['gpu_processes'] if x['pid']==2544824)
 assert a['process_start_ticks']==b['process_start_ticks']=='1834772278' and a['in_slurm_job'] is True and b['in_slurm_job'] is False
 assert not any(x['pid']==2544824 for x in after['gpu_processes'])
 assert 'cgroup' not in a and 'state' not in a # Cannot invent historical raw cgroup/State.

def test_actual_paused_source_markers_parse_without_gpu():
 m=json.loads((H/'job3304-first5-live-binding-original/manifest.json').read_bytes())
 assert s.sha((H/'job3304-first5-live-binding-original/manifest.json').read_bytes())==s.ACTUAL_MANIFEST_SHA
 log=(H/'job3304-snapshot-10/train.log').read_text();rows=[];paused=[]
 for line in log.splitlines():
  for marker,target in [('OPSD_E1_OUTER_JSON=',rows),('OPSD_E1_STEP5_PAUSED_JSON=',paused)]:
   if marker in line:target.append(gate.parse_log_marker(line,marker,driver_pid=s.ACTUAL_DRIVER,job_id=s.JOB,worker_pids=(2548974,2549227,2549228,2549229,2549231,2549232,2549230,2549233),checkpoint_warning_path=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py')))
 assert [r['actual_outer_updates'] for r in rows]==[1,2,3,4,5]
 assert len(paused)==1 and paused[0]['pause_nonce']==s.ACTUAL_NONCE and paused[0]['manifest_sha256']==s.ACTUAL_MANIFEST_SHA
 assert all(r['source_commit']==s.SOURCE and r['driver_pid']==s.ACTUAL_DRIVER for r in rows)


def test_unchanged_production_module_bytes():
 assert s.sha((H/'first5-1b16c2227caf/receipt_payload.py').read_bytes())==s.RECEIPT_SHA
 assert s.sha((H.parent/'opsd-unclipped-multitraj200-teacherrecompute/search_r1/baselines/checkpoint_gate.py').read_bytes())==s.GATE_SHA
 assert s.sha((H/'formal-71d2501d4512/formal_supervisor.py').read_bytes())==s.FORMAL_SHA
