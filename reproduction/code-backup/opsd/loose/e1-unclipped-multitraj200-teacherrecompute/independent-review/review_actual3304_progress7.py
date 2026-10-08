# coding: utf-8
from pathlib import Path
import json,hashlib,base64,types,re,csv,datetime
H=Path(__file__).resolve().parents[1];D=H/'job3304-release-live-originals-v3';R=H/'job3304-first5-complete-originals-v3/raw';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes());wire=read(D/'transport.stdout.original')
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
 for line in (D/name).read_text(encoding='utf8').splitlines():
  for kind,marker in [('outer','OPSD_E1_OUTER_JSON='),('pause','OPSD_E1_STEP5_PAUSED_JSON='),('release','OPSD_E1_STEP5_RELEASED_JSON=')]:
   if marker in line:markers[kind].append(g.parse_log_marker(line,marker,driver_pid=pid,job_id='3304',worker_pids=tuple(starts),checkpoint_warning_path=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py'),fsdp_warning_source_bytes=warn))
 assert [v['actual_outer_updates'] for v in markers['outer']]==list(range(1,8)) and len(markers['pause'])==1 and len(markers['release'])==1
 assert markers['release'][0]==dict(driver_pid=pid,actual_outer_updates=5,next_global_step=6,receipt_sha256=receipt)
 for v in markers['outer']:
  assert type(v['actual_outer_updates']) is int and type(v['global_step']) is int and type(v['driver_pid']) is int and v['global_step']==v['actual_outer_updates']
  assert v['driver_pid']==pid and v['job_id']=='3304' and v['source_commit']=='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6' and v['config_sha256']=='0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990'
  g.validate_outer(v['metrics'],profile='multitraj200')
  import math
  i=v['actual_outer_updates'];m=v['metrics']
  for rank in range(8):
   assert m[f'actor/optimizer_steps/rank_{rank}']==10 and m[f'actor/samples/rank_{rank}']==320 and m[f'teacher/cuda_update_entries_cumulative_per_rank/rank_{rank}']==i
   assert math.isclose(m[f'actor/optimizer_lr_used_mean/rank_{rank}'],1e-6*min((i-1)/57,1),rel_tol=1e-14,abs_tol=1e-18)
  assert math.isclose(m['actor/lr'],1e-6*min(i/57,1),rel_tol=1e-14,abs_tol=1e-18)
  assert m['diagnostic/student_archived_count']==2560 and m['diagnostic/teacher_archived_count']==512 and m['diagnostic/paired_archive_complete']==1
 streams[name]=markers
assert streams['driver-current-prefix.out']==streams['train.log']
for name in ['driver-current-prefix.out','train.log']:assert (D/name).read_bytes().startswith((H/'job3304-release-live-originals-v2'/name).read_bytes())
root=read(H/'ROOT_ACTUAL3304_LIVE_PROGRESS7.json');assert root['actual_completed_outer_iterations']==7 and root['actual_optimizer_calls_per_rank']==70
m=streams['driver-current-prefix.out']['outer'][-1]['metrics'];r=root['rows'][-1]
for k,v in {'reward':m['critic/score/mean'],'loss_rank0':m['actor/opsd_loss'],'finish_ratio':m['env/finish_ratio'],'used_lr':m['actor/optimizer_lr_used'],'post_outer_lr':m['actor/lr'],'outer_seconds':m['timing_s/step']}.items():assert r[k]==v
assert root['original_files']=={f.name:sha(f) for f in D.iterdir() if f.is_file()}
for cmd in ['source_head','source_status','slurm','gpu','gpu_apps']:assert type(wire[cmd]['returncode']) is int and wire[cmd]['returncode']==0
assert type(read(D/'exit.original.json')['returncode']) is int and read(D/'exit.original.json')['waited'] is True
out=H/'ACTUAL3304_LIVE_PROGRESS7_INDEPENDENT_REVIEW.json';assert not out.exists();v=dict(version=1,status='PASS_ACTUAL_LIVE_PROGRESS7_ONLY',reviewer='/root/author200_review',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observation_utc=wire['utc'],source_commit='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6',config_sha256='0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990',job_id='3304',receipt_sha256=receipt,root_progress_sha256=sha(H/'ROOT_ACTUAL3304_LIVE_PROGRESS7.json'),observer_reviewed_source_sha256=sha(H/'observe_live_driver_metrics.py'),original_files={f.name:sha(f) for f in D.iterdir() if f.is_file()},driver=driver,workers=workers,gpu_uuid_map=uuids,known_nvml_apps=16,driver_aggregate_1_to_7_and_pause_release_identical=True,complete_eof_record_in_aggregate_accepted_without_adding_lf=not (D/'train.log').read_bytes().endswith(b'\n'),completed_actual_outers=7,per_rank_cumulative_observed_optimizer_calls=70,source_clean=True,slurm_state='RUNNING',slurm_restarts=0,latest_metrics={k:m[k] for k in ['critic/rewards/mean','critic/score/mean','env/finish_ratio','env/ratio_of_valid_action','env/number_of_valid_action','env/number_of_valid_search','actor/opsd_loss','actor/grad_norm','actor/optimizer_lr_used','actor/lr','timing_s/step','timing_s/gen','timing_s/update_actor','diagnostic/student_archived_count','diagnostic/teacher_archived_count','diagnostic/paired_archive_complete']},memory_MiB=[int(x[2]) for x in rows],software_notes={'legal_metrics_empty_reason':"Observer filters key substring 'legal'; actual correct key is env/ratio_of_valid_action, not missing or zero behavior.",'requested_minimal_observer_hardening':['Exact int returncodes and exact int outer/global headers','Actual original driver physical FD path matching','Pin prior signed-publication review full SHA'],'current_actual_values_typed_and_physical_path_independently_verified':True},scope='Bounded actual7 progress and current original evidence provenance; current observer exact byte static read, not future evidence qualification.',limitations=['The progress observer is a local read-only report, never a signature/first5 release authority.','Archive completion/count metrics are actual1..7 records; no repeated later raw-token/whole-state content acceptance.','This capsule outer6 member names/sizes are historical6 metadata, not actual7 physical archive-content proof.','Memory is one historical phase NVML sample, not peak/terminal released memory.','Reward changes across training batches do not prove paired efficacy;200/final/no201 lifetime/whole formal/resource release remain pending.'])
with out.open('x',encoding='utf8') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
md=out.with_suffix('.md');assert not md.exists()
with md.open('x',encoding='utf8') as f:f.write('实际已完成7轮，原driver和八worker PID/start、源及配置保持，两日志1–7及签收事件完全一致；每rank累计70次Adam。第7轮reward0.186719、完成率97.58%、合法动作98.99%，loss0.145731。实际使用LR6/57×1e-6，轮后7/57×1e-6。\n\n观察器的legal字段为空来自筛选名称错误，真实动作指标位于env/ratio_of_valid_action。当前证据数值/物理身份已独核；建议补严格类型/driver路径/历史review pin，后续进度工具不能替代最终门禁。\n')
print(json.dumps({'report':str(out),'sha256':sha(out),'md_sha256':sha(md),'latest':v['latest_metrics']}))
