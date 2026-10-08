# coding: utf-8
from pathlib import Path
import json,hashlib,base64,types,re,csv,datetime
H=Path(__file__).resolve().parents[1];D=H/'job3304-release-live-originals-v5';R=H/'job3304-first5-complete-originals-v3/raw';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes());wire=read(D/'transport.stdout.original')
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
 assert [v['actual_outer_updates'] for v in markers['outer']]==list(range(1,11)) and len(markers['pause'])==1 and len(markers['release'])==1
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
for name in ['driver-current-prefix.out','train.log']:assert (D/name).read_bytes().startswith((H/'job3304-release-live-originals-v3'/name).read_bytes())
root=read(H/'ROOT_ACTUAL3304_LIVE_PROGRESS10_V5CAP.json');assert root['actual_completed_outer_iterations']==10 and root['actual_optimizer_calls_per_rank']==100
m=streams['driver-current-prefix.out']['outer'][-1]['metrics'];r=root['rows'][-1]
for k,v in {'reward':m['critic/score/mean'],'loss_rank0':m['actor/opsd_loss'],'finish_ratio':m['env/finish_ratio'],'used_lr':m['actor/optimizer_lr_used'],'post_outer_lr':m['actor/lr'],'outer_seconds':m['timing_s/step']}.items():assert r[k]==v
assert root['original_files']=={f.name:sha(f) for f in D.iterdir() if f.is_file()}
for cmd in ['source_head','source_status','slurm','gpu','gpu_apps']:assert type(wire[cmd]['returncode']) is int and wire[cmd]['returncode']==0
assert type(read(D/'exit.original.json')['returncode']) is int and read(D/'exit.original.json')['waited'] is True
error_markers=['OutOfMemoryError','CUDA out of memory','Traceback (most recent call last)','RayTaskError','NCCL error','Watchdog caught collective operation timeout','SIGKILL'];error_counts={name:{pat:(D/name).read_text().count(pat) for pat in error_markers} for name in ['driver-current-prefix.out','train.log']};assert all(v==0 for x in error_counts.values() for v in x.values())
for row in streams['driver-current-prefix.out']['outer']:
 mm=row['metrics']
 for rank in range(8):assert mm[f'actor/skipped_empty_minibatches/rank_{rank}']==0 and mm[f'actor/opsd_loss_finite/rank_{rank}']==1 and mm[f'actor/grad_norm_finite/rank_{rank}']==1
actor=H.parent/'opsd-unclipped-multitraj200-teacherrecompute/search_r1/baselines/actor.py';distill=H.parent/'opsd-unclipped-multitraj200-teacherrecompute/search_r1/baselines/distillation.py';mathcfg_review=H/'ACTUAL3304_FIRST5_CONFIG_RUNTIME_MATH_SUBREVIEW_V2.json';assert sha(mathcfg_review)=='dd965599da7717a97a27adb5892c5bd58062ca436efaf0716c043e43066ceb63'
GiB=1024**3;peaks={str(rank):dict(allocated_bytes=m[f'actor/cuda_peak_allocated_bytes/rank_{rank}'],reserved_bytes=m[f'actor/cuda_peak_reserved_bytes/rank_{rank}']) for rank in range(8)}
gpu_values=[dict(index=int(x[0]),used_MiB=int(x[2]),total_MiB=int(x[3]),used_GiB=int(x[2])/1024,free_fraction=(int(x[3])-int(x[2]))/int(x[3]),utilization_percent=int(x[4])) for x in rows]
out=H/'ACTUAL3304_COMPLETED10_CONTINUITY_AND_MEMORY_INDEPENDENT_REVIEW.json';assert not out.exists()
v=dict(version=1,status='PASS_ACTUAL_COMPLETED10_CONTINUOUS_UPDATES_WITH_FIRST5_STATE_ANCHOR_ONLY',reviewer='/root/author200_review',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observation_utc=wire['utc'],source_commit='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6',config_sha256='0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990',job_id='3304',receipt_sha256=receipt,accepted_first5_peer_sha256='ce71235e4f9193ece01e08f76447f1b62c72bfa35535951de7b433e9167a7ca5',accepted_first5_content_archive_sha256='dbbd863cc7265edf89bf3e5ba584ff952d11c2e76321a3c6d1cfd66e486c95ee',actual_cfg_math_runtime_subreview_sha256=sha(mathcfg_review),previous_progress7_independent_sha256=sha(H/'ACTUAL3304_LIVE_PROGRESS7_INDEPENDENT_REVIEW.json'),root_progress_sha256=sha(H/'ROOT_ACTUAL3304_LIVE_PROGRESS10_V5CAP.json'),observer_source_sha256=root['observer_source_sha256'],original_files={f.name:sha(f) for f in D.iterdir() if f.is_file()},driver=driver,workers=workers,gpu_uuid_map=uuids,known_nvml_apps=16,driver_aggregate_1_to_10_and_pause_release_identical=True,completed_actual_outers=10,per_rank_cumulative_observed_optimizer_calls=100,per_rank_optimizer_per_outer=[10]*10,per_rank_samples_per_outer=320,actual_student_count_metrics_per_outer=2560,actual_teacher_count_metrics_per_outer=512,teacher_cuda_update_clock10_all8=True,actual_empty_minibatch_skips_all_outers_all_ranks=0,pure_loss={'method':'pure_opsd','pg':False,'reference_kl':False,'component_clip':None,'teacher_logits_recompute':True,'source_and_config_provenance':'Exact frozen fcec/config0f0fc joins acceptedfirst5 and actual prior cfg/math review;current source files SHA retained','all8_every_outer_loss_grad_finite':True},source_clean=True,slurm_state='RUNNING',slurm_restarts=0,latest_metrics={k:m[k] for k in ['critic/rewards/mean','critic/score/mean','env/finish_ratio','env/ratio_of_valid_action','actor/opsd_loss','actor/grad_norm','actor/optimizer_lr_used','actor/lr','timing_s/step','timing_s/gen','timing_s/update_actor']},latest_rank_mean_loss=sum(m[f'actor/opsd_loss_mean/rank_{rank}'] for rank in range(8))/8,reward_outer6_to_10=[x['metrics']['critic/rewards/mean'] for x in streams['driver-current-prefix.out']['outer'][5:]],memory_observation={'gpu_nvml_samples':gpu_values,'completed_outer10_reported_torch_peaks':peaks,'allocated_GiB_range':[min(x['allocated_bytes']/GiB for x in peaks.values()),max(x['allocated_bytes']/GiB for x in peaks.values())],'reserved_GiB_range':[min(x['reserved_bytes']/GiB for x in peaks.values()),max(x['reserved_bytes']/GiB for x in peaks.values())],'source_files':{str(actor):sha(actor),str(distill):sha(distill)},'source_semantics':'actor max_memory_allocated/reserved are current PyTorch peak counters; actor does not reset per outer. teacher on_device wraps diagnostic+loss/update and moves fixed teacher back CPU before observations. Metrics are not exclusive attribution to active tensor or teacher diagnostic.','exact_phase_at_nvml_sample':'Not determined from this capsule','low_free_margin_observed':True,'preflight_15_percent_not_lifetime_guarantee':True,'app_and_inventory_queries_not_simultaneous':True},error_markers_in_supplied_driver_train_prefixes=error_counts,scope='Bounded actual10 update and original-process continuity, preserving prior real5 HF/full-state/data audit; memory and error markers qualified only within supplied prefixes.',limitations=['No HF10 or full-state checkpoint10 exists in the agreed save schedule, and no such reload is claimed.','Full first5 saveddata2560/permutation/RNG/native/HF evidence remains the historical state anchor; no saved10 data cursor or full raw6..10 token replay is fabricated.','Archive per-outer counters/paired completion are actual logs, not repeated all later content acceptance.','No received actual worker.stderr provenance in this capsule; absent markers in driver/train do not prove every process stream error-free.','High memory is a genuine historical sample with about2.18percentfree at rank7, not an observedOOM; allocation/cache/phase cause cannot be uniquely proved here.','Different training questions mean reward/loss trends are not paired final efficacy;200/final51713/no201 lifetime/terminal/whole formal/resource release remain pending.','Original resourcebad66 is still preserved and not reclassified.'])
with out.open('x',encoding='utf8') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
md=out.with_suffix('.md');assert not md.exists()
with md.open('x',encoding='utf8') as f:f.write('实际10轮连续更新通过：原driver、八worker及配置不变，两日志1–10和唯一首5签收逐字段相同，每rank累计100次Adam、所有loss/梯度有限，无空微批跳过。第10reward0.19375、完成率97.38%、合法动作98.65%，loss rank0=0.107150/八rank均值0.118077。\n\n显存高是实际现象：GPU6/7约77.86/77.91GiB，空余仅约2.24%/2.18%；已收driver/train未见OOM或异常标记。第10PyTorch峰allocated约46.0–46.5GiB，reserved约70.0–71.3GiB，峰值非逐轮reset，不能等同全活跃tensor或认定泄漏。当前NVML时相未确定，实际stderr另需原件。无需据此热改或重启。首5HF/状态原验收仍有效，不冒称HF10、后续raw全重放、200或最终效果通过。\n')
print(json.dumps({'report':str(out),'sha256':sha(out),'md_sha256':sha(md),'allocated_GiB':v['memory_observation']['allocated_GiB_range'],'reserved_GiB':v['memory_observation']['reserved_GiB_range']}))
