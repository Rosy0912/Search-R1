"""Independent local replay of captured job3302 GPU records; never runs GPU/source."""
import base64,datetime,hashlib,json,math,pathlib,re,subprocess
H=pathlib.Path(__file__).resolve().parents[1]
O=H/'job3302-snapshot-2'
REPO=H.parent/'opsd-unclipped-multitraj200-nativehead'
SOURCE='b611864d3f6ad048189b11b9b9613854e789da5d'
def need(c,s):
 if not c: raise ValueError(s)
def sha(b):return hashlib.sha256(b).hexdigest()
def unique(b):
 def pairs(rows):
  d={}
  for k,v in rows:need(k not in d,'Duplicate JSON key');d[k]=v
  return d
 return json.loads(b,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def number(v):return type(v) in (int,float) and math.isfinite(v)
def close(a,b):return math.isclose(a,b,rel_tol=2e-12,abs_tol=1e-10)
wire_raw=(O/'transport.stdout.original').read_bytes();wire=unique(wire_raw)
need((O/'transport.stderr.original').read_bytes()==b'','Transport stderr')
need(wire['job_id']=='3302' and wire['run_root'].endswith('/formal-3302'),'Actual job/run')
need(unique((O/'actual-observation.json').read_bytes())=={k:v for k,v in wire.items() if k!='files'},'Observation vs original wire')
pins={}
for name,r in wire['files'].items():
 raw=base64.b64decode(r['base64'],validate=True)
 need(sha(raw)==r['sha256'] and (O/name).read_bytes()==raw,'Original full bytes '+name)
 pins[name]=r['sha256']
need(len(pins)==6,'Actual snapshot2 has six files, not future larger capsule')
raw=(O/'evidence/capacity-micro4/torchrun.log').read_bytes();rows={}
for n,line in enumerate(raw.decode().splitlines(),1):
 if not line.startswith('{'):continue
 d=unique(line)
 if 'short_native_teacher_row_equivalence' not in d:continue
 rank=d['rank'];need(type(rank)is int and 0<=rank<8 and rank not in rows,'Unique actual rank')
 rows[rank]=(n,d)
need(set(rows)==set(range(8)),'All8 actual full records')
need(subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==SOURCE,'Source HEAD')
need(subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()=='','Source clean')
source_review=unique((H/'INDEPENDENT_SOFTWARE_REVIEW.json').read_bytes())
source_pins={}
for name in ['reproduction/baselines/e1-multitraj200/microbatch_probe.py','search_r1/baselines/distillation.py']:
 b=(REPO/name).read_bytes();need(sha(b)==source_review['reviewed_source_files'][name],'Exact fixed source')
 source_pins[name]=sha(b)
rank_records=[];global_a=global_b=global_d=0.;names=None
for rank,(lineno,rec) in sorted(rows.items()):
 q=rec['short_native_teacher_row_equivalence'];stats=q['parameter_statistics']
 need(rec['job_id']=='3302' and rec['source_commit']==SOURCE and rec['hostname']=='h100-10'
      and type(rec['world_size']) is int and rec['world_size']==8 and rec['local_rank']==rank
      and rec['local_micro']==4 and rec['response_only_logits'] is False
      and rec['teacher_forward_micro_batch']==2 and rec['component_clip'] is None
      and rec['oom'] is False and rec['passed'] is False,'Rank actual contract')
 need(q['response_only_logits'] is False and q['baseline_teacher_forward_micro_batch']==4
      and q['production_teacher_forward_micro_batch']==2 and q['passed'] is False,'Single teacher-row factor')
 need(type(q['checked_gradient_shards'])is int and q['checked_gradient_shards']==37 and len(stats)==37,'Actual37 shards')
 current=[s['name'] for s in stats];need(len(set(current))==37,'Unique shard names')
 if names is None:names=current
 need(current==names,'Shard name order')
 for s in stats:
  need(type(s['shard_elements'])is int and s['shard_elements']>0,'Actual shard size')
  need(all(number(s[k]) and s[k]>=0 for k in ['teacher4_squared_norm','teacher2_squared_norm','difference_squared_norm','maximum_abs_difference']),'Finite shard statistics')
 a=math.fsum(s['teacher4_squared_norm'] for s in stats)
 b=math.fsum(s['teacher2_squared_norm'] for s in stats)
 delta=math.fsum(s['difference_squared_norm'] for s in stats)
 need(close(a,q['local_teacher4_squared_gradient_norm']) and close(b,q['local_teacher2_squared_gradient_norm']),'Local sums vs producer')
 global_a+=a;global_b+=b;global_d+=delta
 need(rec['old_logprob_capacity']['passed'] is True
      and rec['old_logprob_capacity']['local_micro']==8
      and rec['old_logprob_capacity']['global_micro']==64
      and rec['old_logprob_capacity']['logprob_shape']==[8,4096]
      and rec['old_logprob_capacity']['entropy_shape']==[8,4096]
      and rec['old_logprob_capacity']['logprob_finite'] is True
      and rec['old_logprob_capacity']['entropy_finite'] is True,'LP8 actual pass')
 need('optimizer_step_records' not in rec and not any(x['label'].startswith('step') for x in rec['memory_phases']),'Failure precedes synthetic Adam loop')
 need(q['maximum_relative_gradient_l2']==.02 and q['minimum_gradient_cosine']==.9999
      and q['maximum_relative_loss']==.002 and q['maximum_teacher_q_mean_l1_difference']==1e-4,'Unchanged gates')
 rank_records.append(dict(rank=rank,raw_line_number=lineno,checked_shards=37,
  local_teacher4_squared_norm=a,local_teacher2_squared_norm=b,local_difference_squared_norm=delta,
  local_relative_l2=math.sqrt(delta/a),global_relative_l2_recorded=q['shard_gradient_relative_l2'],
  global_cosine_recorded=q['shard_gradient_cosine'],teacher4_loss=q['teacher4_full_head_loss'],
  teacher2_loss=q['teacher2_full_head_loss'],loss_relative_difference=q['loss_relative_difference'],
  teacher_q_mean_l1=q['teacher_q_mean_probability_l1_difference'],teacher_logits_max_abs=q['teacher_FP32_logits_max_absolute_difference'],
  maximum_gradient_abs_difference=max(s['maximum_abs_difference'] for s in stats),lp8_pass=True,oom=False))
need(global_a>0 and global_b>0,'Nonzero global gradients')
dot=(global_a+global_b-global_d)/2
relative_l2=math.sqrt(global_d/global_a);cosine=dot/math.sqrt(global_a*global_b)
for rank,(n,rec) in rows.items():
 q=rec['short_native_teacher_row_equivalence']
 need(close(global_a,q['global_teacher4_squared_gradient_norm']) and close(global_b,q['global_teacher2_squared_gradient_norm'])
      and close(relative_l2,q['shard_gradient_relative_l2']) and close(cosine,q['shard_gradient_cosine']),'Global eight-rank allreduce matches independent sum')
 need(q['loss_relative_difference']<.002 and q['teacher_q_mean_probability_l1_difference']<1e-4,'Other gates')
need(relative_l2<=.02 and cosine<.9999,'Specific actual cosine rejection')
fields=dict(x.split('=',1) for x in wire['live']['scontrol']['stdout'].split() if '=' in x)
for k,v in {'JobId':'3302','JobState':'FAILED','ExitCode':'1:0','Restarts':'0','RunTime':'00:11:21',
            'StartTime':'2026-10-06T18:30:16','EndTime':'2026-10-06T18:41:37','NodeList':'h100-10','UserId':'root(0)'}.items():
 need(fields.get(k)==v,'Literal terminal '+k)
need(type(wire['live']['scontrol']['returncode'])is int and wire['live']['scontrol']['returncode']==0,'Terminal command')
capacity_exit=unique((O/'evidence/capacity-micro4/exit.json').read_bytes())
need(type(capacity_exit['returncode'])is int and capacity_exit['returncode']==1 and type(capacity_exit['rl_outer_updates'])is int and capacity_exit['rl_outer_updates']==0,'Capacity actual exit1/0RL')
need((O/'job-exit.txt').read_bytes()==b'1\n','Actual wrapper exit1')
need(unique((O/'evidence/hardware-exit.json').read_bytes())=={'returncode':0},'Hardware prior exit0')
need(wire['live']['gpu_apps']['returncode']==0 and wire['live']['gpu_apps']['stdout']==''
     and wire['live']['gpu']['returncode']==0,'Captured postexit NVML empty apps')
report=dict(version=1,status='ACTUAL_GPU_PROBE_REJECTED_TEACHER_ROWS_NUMERIC_GATE_INDEPENDENT_REVIEW',
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id='3302',source_commit=SOURCE,
 snapshot_path=str(O),captured_utc=wire['utc'],original_transport_sha256=sha(wire_raw),original_files_sha256=pins,
 raw_torchrun_log_sha256=sha(raw),raw_torchrun_log_bytes=len(raw),reviewed_source_files=source_pins,
 actual_rank_records=rank_records,global_recomputed=dict(teacher4_squared_norm=global_a,teacher2_squared_norm=global_b,
  difference_squared_norm=global_d,dot_from_norm_identity=dot,relative_l2=relative_l2,cosine=cosine,
  angle_degrees=math.degrees(math.acos(min(1,cosine)))),
 failure=dict(unchanged_cosine_minimum=.9999,actual_cosine=cosine,all_eight_ranks_rejected=True,
  other_recorded_numeric_thresholds_pass=True,nonzero_finite_gradients=True,lp8_pass_all_eight=True,
  oom_observed=False,worst_width_actor_two_adam_capacity_gate_reached=False,synthetic_adam_calls_completed=0,rl_outers_completed=0),
 statistic_algorithm_review=dict(global_sum_correct=True,independent_all8_all37_shards_matches_gpu=True,
  dot_identity_matches=True,field_naming_warning='shard_gradient_relative_l2/cosine are global eight-rank values; local metrics are separately recomputed.'),
 literal_terminal=dict(state=fields['JobState'],exit_code=fields['ExitCode'],duration_seconds=681,restarts=0,
  start_bjt=fields['StartTime'],end_bjt=fields['EndTime'],capacity_returncode=1,wrapper_returncode=1,
  observed_gpu_applications_empty=True,scope='One captured successful NVML read after failure; not a replacement for complete teardown/process/resource-history acceptance.'),
 causal_scope=dict(proven='Both variants use the same native full student head and unchanged loss/masks/data/weights. The only configured algorithm execution factor is teacher base-model row batch4 versus2. Actual teacher logits/q differ and the resulting gradients violate the fixed cosine threshold.',
  mathematical='Independent teacher rows preserve real-arithmetic fullKL definition. Floating-point FP32 eager teacher batching can change q; BF16 student backward can amplify the resulting changes. This run is numeric validation, not a trajectory/effect experiment.',
  not_proven='No repeated same-batch native GPU baseline is captured here, so row-factor comparison does not uniquely separate batching rounding from run-order/nondeterministic-kernel effects. It does not prove the old clipped author200 collapse cause, parameter-indexing error, harmful Adam update, or inevitable unclipped-training failure.'),
 minimal_candidate=dict(proposal='Restore the unsplit full teacher4 batch, retain native student head, actor local4/LP8, component_clip=None, exact fullKL target, and existing numeric/capacity thresholds.',
  no_gate_relaxation=True,required_actual_next='Fresh pinned candidate: unchanged-path short native check, then requested worst-width actor4 two-Adam memory/finite/nonzero gate and actual first5. Teacher4 increases eager attention memory; prior LP8 PASS and output estimate do not prove actor4 capacity.',
  skip_rule_boundary='Do not substitute skipped optimizer/outer steps for required10 actual Adam calls per completed outer; do not add PG/reference-KL brakes to pure OPSD.'),
 scope_limits=['Captured six snapshot2 originals and eight actual GPU JSON records replayed locally; larger50-original terminal capsule was not assumed.',
  'No SSH, GPU, submit, signature, private key, deployment, or frozen source/old evidence edits performed.',
  'No formal training, fresh5/state save-reload, effect improvement, or final200 accepted.'])
p=H/'ACTUAL3302_TEACHER_ROWS_FAILURE_INDEPENDENT_REVIEW.json'
with p.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
md=H/'ACTUAL3302_TEACHER_ROWS_FAILURE_INDEPENDENT_REVIEW.md'
with md.open('x') as f:
 f.write('# Actual3302 teacher-row numeric failure independent review\n\nThe original eight GPU records and all six transported snapshot2 files match their full SHA. Independent summation of 8 × 37 gradient-shard records reproduces the producer: relative L2 '+format(relative_l2,'.16g')+', cosine '+format(cosine,'.16g')+'. The cosine is below the unchanged .9999 gate; loss relative difference1.4151e-5 and teacher-q mean L1 1.1949e-5 pass their own bounds. All eight LP8 checks pass and no OOM is reported.\n\nBoth student forwards use the same native full head; only teacher row batch4→2 changes in the configured comparison. The actual teacher logits/q differ. This supports rejecting this teacher batching optimization under the existing native numeric gate. Real-arithmetic equivalence is preserved, but native rounding/ordering sensitivity is not proven harmless; no same-batch GPU repeat control exists here. It does not establish the unique cause of historical training collapse.\n\nLiteral Slurm observation is FAILED1:0, duration681seconds, zero restarts; capacity exit1 and wrapper exit1 join. The failure precedes the synthetic two-Adam loop and formal training: zero synthetic Adam and zero RL outers. Captured postexit NVML applications are empty; complete teardown acceptance is a separate evidence scope.\n\nThe minimum next candidate restores full teacher4 with actor4 and native head, keeps pure unclipped fullKL and all existing thresholds. Its higher teacher attention memory still requires the real worst-width capacity check and first5 validation. Skipped steps cannot count as the required10 actual Adam calls per completed outer. This review performed only local reading/recomputation, without source edits, SSH, GPU, job control, signing or deployment.\n')
print(json.dumps(dict(json=str(p),sha256=sha(p.read_bytes()),md_sha256=sha(md.read_bytes()),global_recomputed=report['global_recomputed'],terminal=report['literal_terminal'])))
