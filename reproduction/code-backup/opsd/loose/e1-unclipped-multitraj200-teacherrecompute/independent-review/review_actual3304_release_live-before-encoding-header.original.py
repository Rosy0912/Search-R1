from pathlib import Path
import json,hashlib,base64,types,re,csv,datetime
H=Path(__file__).resolve().parents[1];D=H/'job3304-release-live-originals-v1';R=H/'job3304-first5-complete-originals-v3/raw';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes());wire=read(D/'transport.stdout.original')
assert type(read(D/'exit.original.json')['returncode']) is int and read(D/'exit.original.json')=={'returncode':0,'waited':True}
files={}
for name,rec in wire['files'].items():
 raw=base64.b64decode(rec['base64'],validate=True);p=D/name;assert raw==p.read_bytes() and len(raw)==rec['bytes'] and hashlib.sha256(raw).hexdigest()==rec['sha256'];files[name]=rec
assert set(files)=={'acceptance-5-receipt.json','train.log','driver-current-prefix.out'}
receipt='d1db79938fd1a3b3520e4d1da673525dd260302d2402e063fd7734fcb11721fc';assert sha(D/'acceptance-5-receipt.json')==receipt
assert (D/'acceptance-5-receipt.json').read_bytes()==(H/'first5-actual-root-release-v1/signed-envelope.json').read_bytes()
assert sha(H/'ACTUAL3304_SIGNED_FIRST5_PUBLICATION_INDEPENDENT_REVIEW.json')=='e094d37b1d2d7adf4f8ca95339d4c629b961c85e5daf3985af699a517f1f82d3'
pid=2548319;starts={2548974:'1834785562',2549227:'1834786108',2549228:'1834786110',2549229:'1834786111',2549231:'1834786113',2549232:'1834786114',2549230:'1834786112',2549233:'1834786115'}
def proc(q,stat,status,cgroup,expected):
 assert int(stat.split(' ',1)[0])==q and stat.rsplit(')',1)[1].split()[19]==expected
 assert re.findall(r'^Uid:\s+(.*)$',status,re.M)==['0\t0\t0\t0'] and re.search(r'^State:\s+([RSDI])',status,re.M) and '/job_3304/step_0\n' in cgroup
 return dict(pid=q,start=expected,cgroup=cgroup)
assert wire['driver_pid']==pid and wire['driver_start']=='1834784786'
driver=proc(pid,wire['driver_stat'],wire['driver_status'],wire['driver_cgroup'],'1834784786');workers=[]
for q,start in starts.items():
 obs=wire['workers'][str(q)];workers.append(proc(q,obs['stat'],obs['status'],obs['cgroup'],start))
for name in ['source_head','source_status','slurm','gpu','gpu_apps']:assert type(wire[name]['returncode']) is int and wire[name]['returncode']==0
assert wire['source_head']['stdout'].strip()=='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6' and not wire['source_status']['stdout'].strip()
slurm=wire['slurm']['stdout'];assert 'JobId=3304 ' in slurm and 'UserId=root(0)' in slurm and 'JobState=RUNNING' in slurm and 'Restarts=0' in slurm and 'NodeList=h100-10' in slurm and 'gres:gpu:h100:8' in slurm
rows=list(csv.reader(wire['gpu']['stdout'].splitlines()));assert len(rows)==8 and [int(x[0]) for x in rows]==list(range(8));uuids={int(x[0]):x[1].strip() for x in rows};old=read(H/'ACTUAL3304_SIGNED_FIRST5_PUBLICATION_INDEPENDENT_REVIEW.json')['uuid_map'];assert uuids=={int(k):v for k,v in old.items()}
apps=list(csv.reader(wire['gpu_apps']['stdout'].splitlines()));expected={(uuids[rank],q) for rank,q in enumerate(starts)}|{(uuid,2543106) for uuid in uuids.values()};assert len(apps)==16 and {(x[0].strip(),int(x[1])) for x in apps}==expected
path=files['driver-current-prefix.out']['physical'];assert path.startswith('/tmp/o4e-3304/ray/session_') and path.endswith('-2548319.out')
gp=R/'source/search_r1/baselines/checkpoint_gate.py';raw=gp.read_bytes();assert hashlib.sha256(raw).hexdigest()=='09561719673a7c273ad39b3c9363b1a250eab0a94c20c80f40dd1dfc05589e16';g=types.ModuleType('fixed_gate');g.__file__=str(gp);exec(compile(raw,str(gp),'exec',dont_inherit=True),g.__dict__);warn=(R/'native-source/fully_sharded_data_parallel.py').read_bytes()
streams={}
for name in ['driver-current-prefix.out','train.log']:
 markers={'outer':[],'pause':[],'release':[]}
 for line in (D/name).read_text().splitlines():
  for kind,marker in [('outer','OPSD_E1_OUTER_JSON='),('pause','OPSD_E1_STEP5_PAUSED_JSON='),('release','OPSD_E1_STEP5_RELEASED_JSON=')]:
   if marker in line:markers[kind].append(g.parse_log_marker(line,marker,driver_pid=pid,job_id='3304',worker_pids=tuple(starts),checkpoint_warning_path=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py'),fsdp_warning_source_bytes=warn))
 assert [v['actual_outer_updates'] for v in markers['outer']]==[1,2,3,4,5] and len(markers['pause'])==1 and len(markers['release'])==1
 assert markers['release'][0]==dict(driver_pid=pid,actual_outer_updates=5,next_global_step=6,receipt_sha256=receipt)
 for v in markers['outer']:g.validate_outer(v['metrics'],profile='multitraj200')
 streams[name]=markers
assert streams['driver-current-prefix.out']==streams['train.log']
observed=wire['outer6_members_observed'];expected_names={'questions.json','student-manifest.json','student.jsonl'}|{f'teacher-rank-{r}'+x for r in range(8) for x in ['.jsonl','.manifest.json']}
assert set(observed)==expected_names and all(type(v) is int and v>0 for v in observed.values())
out=H/'ACTUAL3304_FIRST5_RELEASE_CONSUMED_SAME_PROCESS_OUTER6_STARTED_INDEPENDENT_REVIEW.json';assert not out.exists();v=dict(version=1,status='PASS_ACTUAL_SIGNED5_CONSUMED_SAME_PROCESS_OUTER6_STARTED_COMPLETION_PENDING',reviewer='/root/author200_review',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observation_utc=wire['utc'],source_commit='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6',job_id='3304',receipt_sha256=receipt,previous_signed_publication_review_sha256='e094d37b1d2d7adf4f8ca95339d4c629b961c85e5daf3985af699a517f1f82d3',original_files={f.name:sha(f) for f in D.iterdir() if f.is_file()},driver=driver,workers=workers,gpu_uuid_map=uuids,known_nvml_apps=16,same_driver_release_record=streams['driver-current-prefix.out']['release'][0],strict_driver_aggregate_records_equal=True,completed_actual_outers=5,outer6_members_observed=observed,completed_outer6=False,original_driver_fd1_path=path,source_clean=True,slurm_state='RUNNING',slurm_restarts=0,scope='Actual historical signature and same-driver consumption/physical process continuity; sixth workload started only.',limits=['Sixth paired manifest and complete outer6 JSON were not present at this capture; no actual sixth optimizer budget/completion inferred.','Outer6 raw files here are observed names/sizes only, not another raw-content/archive acceptance.','Retrieval PID appears in all8 NVML rows; its historical start was verified in the preceding actual publisher guards, not freshly re-read in this capture.','Source continuity/no restart and historical saved object identity support original live continuation; no new-process state restoration, final200, terminal or resource release claimed.']);out.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n');md=H/'ACTUAL3304_FIRST5_RELEASE_CONSUMED_SAME_PROCESS_OUTER6_STARTED_INDEPENDENT_REVIEW.md';assert not md.exists();md.write_text('实际签名首5收据已被原driver消费，唯一 RELEASED5→6 与签名SHA完全一致；driver PID/start和八worker PID/start均保持，源fcec干净，job3304 RUNNING/Restarts0。driver与aggregate严格相同1–5及释放事件，第6轮学生和教师档案已出现。\n\n本次确认第6轮已启动，完整6轮更新JSON与paired manifest当时尚未出现，不能认定第6轮完成；不冒称200或终态通过。\n');print(json.dumps({'report':str(out),'sha256':sha(out),'md_sha256':sha(md)}))
