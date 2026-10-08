"""Independent local reading and numeric replay of captured job3303 originals only."""
import base64,datetime,hashlib,json,math,pathlib,subprocess
H=pathlib.Path(__file__).resolve().parents[1];O=H/'job3303-snapshot-3'
REPO=H.parent/'opsd-unclipped-multitraj200-fullteacher'
SOURCE='66ea4c9093697c80657d8c57a2058ebb7c9b078f'
def need(c,s):
 if not c:raise ValueError(s)
def sha(b):return hashlib.sha256(b).hexdigest()
def unique(b):
 def pairs(rows):
  d={}
  for k,v in rows:need(k not in d,'Duplicate key');d[k]=v
  return d
 return json.loads(b,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def number(v):return type(v) in (int,float) and math.isfinite(v)
def close(a,b):return math.isclose(a,b,rel_tol=2e-12,abs_tol=1e-10)
wire_raw=(O/'transport.stdout.original').read_bytes();wire=unique(wire_raw)
need((O/'transport.stderr.original').read_bytes()==b'','Transport stderr')
need(wire['job_id']=='3303' and wire['run_root'].endswith('/formal-3303'),'Actual job')
observation=unique((O/'actual-observation.json').read_bytes())
need({k:observation[k] for k in wire if k!='files'}=={k:v for k,v in wire.items() if k!='files'},'Actual observation core')
need(observation['observed_complete_driver_outer_records']==0 and observation['last_driver_record'] is None,'No observed RL outer')
pins={}
for name,r in wire['files'].items():
 raw=base64.b64decode(r['base64'],validate=True)
 need(sha(raw)==r['sha256'] and (O/name).read_bytes()==raw,'Exact original '+name);pins[name]=r['sha256']
need(len(pins)==7,'Actual snapshot3 seven transported files')
aggregate=unique((O/'evidence/capacity-micro4/micro-4.json').read_bytes())
need(aggregate['passed'] is False and aggregate['local_micro']==4 and len(aggregate['ranks'])==8,'Actual failed aggregate')
rawlog=(O/'evidence/capacity-micro4/torchrun.log').read_bytes();rows={};decoder=json.JSONDecoder(object_pairs_hook=lambda rows:unique(json.dumps(dict(rows))))
# Torchrun stdout contains two adjacent complete JSON objects on physical line125.
# Decode only complete adjacent object boundaries, retain exact byte spans; do not
# accept unknown suffixes or infer ranks from repeated messages.
byte_offset=0
for lineno,rawline in enumerate(rawlog.splitlines(keepends=True),1):
 line=rawline.decode();position=0
 if not line.startswith('{'):byte_offset+=len(rawline);continue
 while position<len(line.rstrip('\r\n')):
  d,end=decoder.raw_decode(line,position)
  need('short_native_full_path_repeatability' in d,'Only actual rank capacity JSON on object lines')
  rank=d['rank'];need(type(rank)is int and 0<=rank<8 and rank not in rows,'Unique actual rank')
  span=line[position:end].encode();need(unique(span)==d,'Strict same-byte JSON')
  rows[rank]=(lineno,byte_offset+len(line[:position].encode()),sha(span),d)
  position=end
  need(position==len(line.rstrip('\r\n')) or line[position]=='{','Unknown JSON suffix')
 byte_offset+=len(rawline)
need(set(rows)==set(range(8)),'All8 raw records captured, no inferred rank')
need(subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==SOURCE,'Frozen source HEAD')
need(subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()=='','Source clean')
source_review=unique((H/'INDEPENDENT_SOFTWARE_REVIEW.json').read_bytes());source_pins={}
for name in ['reproduction/baselines/e1-multitraj200/microbatch_probe.py','search_r1/baselines/distillation.py']:
 b=(REPO/name).read_bytes();need(sha(b)==source_review['reviewed_source_files'][name],'Reviewed exact source');source_pins[name]=sha(b)
rank_records=[];sum_a=[];sum_b=[];sum_d=[];names=None;elements=0
for rec in aggregate['ranks']:
 rank=rec['rank'];lineno,offset,span_sha,logged=rows[rank]
 need(all(logged[k]==v for k,v in rec.items()),'Raw record vs aggregate full fields')
 need(set(logged)-set(rec)=={'error_type','error','traceback','oom','elapsed_seconds'},'Only actual exception suffix fields')
 need(logged['error_type']=='RuntimeError' and logged['error']=='Eight-rank capacity headroom gate failed' and logged['oom'] is False,'Specific headroom failure')
 need(rec['job_id']=='3303' and rec['source_commit']==SOURCE and rec['hostname']=='h100-10' and rec['allocation']['job_id']=='3303' and '/job_3303/step_0' in rec['allocation']['own_job_cgroup'] and rec['rank']==rec['local_rank'] and rec['world_size']==8 and rec['local_micro']==4 and rec['response_only_logits'] is False and rec['teacher_forward_micro_batch']==4 and rec['component_clip'] is None and rec['passed'] is False,'Frozen native full4 contract')
 q=rec['short_native_full_path_repeatability'];stats=q['parameter_statistics']
 need(q['comparison_kind']=='SAME_NATIVE_FULL_HEAD_FULL_TEACHER4_PATH_REPEATABILITY' and q['response_only_logits'] is False and q['first_teacher_forward_micro_batch']==4 and q['repeat_teacher_forward_micro_batch']==4 and q['passed'] is True,'Same-path short4 actual pass')
 need(q['checked_gradient_shards']==37 and len(stats)==37,'Actual37shards')
 current=[s['name'] for s in stats];need(len(set(current))==37,'Unique named gradient shards')
 if names is None:names=current
 need(current==names,'Same gradient names')
 for s in stats:
  need(type(s['shard_elements'])is int and s['shard_elements']>0,'Shard elements');elements+=s['shard_elements']
  need(all(number(s[k]) and s[k]>=0 for k in ['native_first_squared_norm','native_repeat_squared_norm','difference_squared_norm','maximum_abs_difference']),'Finite stats')
  need(s['native_first_squared_norm']==s['native_repeat_squared_norm'] and s['difference_squared_norm']==0 and s['maximum_abs_difference']==0,'Every captured shard repeated exact')
 a=math.fsum(s['native_first_squared_norm'] for s in stats);b=math.fsum(s['native_repeat_squared_norm'] for s in stats);d=math.fsum(s['difference_squared_norm'] for s in stats)
 need(close(a,q['local_native_first_squared_gradient_norm']) and close(b,q['local_native_repeat_squared_gradient_norm']),'Local norm sum')
 sum_a.append(a);sum_b.append(b);sum_d.append(d)
 need(q['native_first_full_head_loss']==q['native_repeat_full_head_loss'] and q['loss_absolute_difference']==0 and q['loss_relative_difference']==0 and q['teacher_FP32_logits_max_absolute_difference']==0 and q['teacher_q_mean_probability_l1_difference']==0,'Loss/logits/q repeated exact')
 need(q['maximum_relative_gradient_l2']==.02 and q['minimum_gradient_cosine']==.9999 and q['maximum_relative_loss']==.002 and q['maximum_teacher_q_mean_l1_difference']==1e-4,'Unchanged gates')
 lp=rec['old_logprob_capacity'];need(lp['passed'] is True and lp['local_micro']==8 and lp['global_micro']==64 and lp['logprob_shape']==[8,4096] and lp['entropy_shape']==[8,4096] and lp['logprob_finite'] is True and lp['entropy_finite'] is True and lp['teacher_storage_runtime']['parameter_devices']==['cpu'],'Actual LP8 pass')
 active=rec['teacher_active_runtime'];need(active['parameter_dtypes']==['torch.float32'] and active['parameter_devices']==['cuda:'+str(rank)] and active['active_device']=='cuda:'+str(rank) and active['attention']=='eager' and active['matmul_tf32'] is False and active['cudnn_tf32'] is False,'Actual FP32 eager CUDA teacher')
 need(all(v==0 for v in rec['teacher_audit'].values()) and rec['teacher_returned_runtime']['parameter_devices']==['cpu'],'Frozen teacher no trainable/grad/optimizer membership and returned CPU')
 need(rec['optimizer_steps']==2 and [x['step'] for x in rec['optimizer_step_records']]==[1,2],'Exactly2 synthetic Adam')
 for x in rec['optimizer_step_records']:
  need(all(number(x[k]) and x[k]>0 for k in ['seconds','loss','grad_norm','tracked_parameter_delta']),'Actual finite updates with tracked coordinate change')
 need(set(rec['decoder_forward_pre_hook_counts'].values())=={4} and len(rec['decoder_forward_pre_hook_counts'])==36,'36decoder two actual forward/backward-checkpoint calls')
 need(rec['student_checkpoint_training_runtime'] is True and rec['student_mixed_precision_parameter_dtype']=='torch.bfloat16' and rec['student_master_parameter_dtypes']==['torch.float32'],'Actual student native precision/checkpoint')
 phases=rec['memory_phases'];need(all(x['num_ooms']==0 and x['num_alloc_retries']==0 for x in phases),'No allocator OOM or retry')
 need(all(k in [x['label'] for x in phases] for k in ['step1_adam_done','step2_adam_done']),'Actual phase evidence')
 need(rec['minimum_required_free_fraction']==.15 and rec['headroom_estimate_includes_nccl_slack_bytes']==1024**3,'Original15%gate/slack')
 total=rec['gpu_total_bytes'];estimate=max(rec['external_baseline_used_bytes']+rec['peak_reserved_bytes']+1024**3,lp['estimated_peak_device_used_bytes'])
 need(estimate==rec['estimated_peak_device_used_bytes'] and estimate>.85*total,'Independent memory formula rejects')
 rank_records.append(dict(rank=rank,raw_line_number=lineno,raw_record_byte_offset=offset,raw_record_sha256=span_sha,
  checked_gradient_shards=37,local_first_squared_norm=a,local_repeat_squared_norm=b,difference_squared_norm=d,
  raw_cosine=q['shard_gradient_cosine'],relative_l2=q['shard_gradient_relative_l2'],teacher_q_l1=0.,
  short_same_path_pass=True,lp8_pass=True,synthetic_adam_calls=2,optimizer_step_records=rec['optimizer_step_records'],
  gpu_total_bytes=total,peak_allocated_bytes=rec['peak_allocated_bytes'],peak_reserved_bytes=rec['peak_reserved_bytes'],
  external_baseline_used_bytes=rec['external_baseline_used_bytes'],estimated_peak_device_used_bytes=estimate,
  estimated_free_fraction=1-estimate/total,headroom_required_bytes=.15*total,headroom_shortfall_bytes=estimate-.85*total,
  sampled_minimum_device_free_bytes=min(x['device_free'] for x in phases),oom=False))
a=math.fsum(sum_a);b=math.fsum(sum_b);d=math.fsum(sum_d);cos=(a+b-d)/(2*math.sqrt(a*b));rel=math.sqrt(d/a)
need(a>0 and a==b and d==0 and cos==1 and rel==0,'Independent global repeated identical norm math')
for _,_,_,rec in rows.values():
 q=rec['short_native_full_path_repeatability'];need(close(a,q['global_native_first_squared_gradient_norm']) and close(b,q['global_native_repeat_squared_gradient_norm']) and q['shard_gradient_relative_l2']==0 and abs(q['shard_gradient_cosine']-1)<=1e-12,'Actual allreduce roundoff only')
fields=dict(x.split('=',1) for x in wire['live']['scontrol']['stdout'].split() if '=' in x)
for k,v in {'JobId':'3303','JobState':'FAILED','ExitCode':'1:0','Restarts':'0','RunTime':'00:09:00','StartTime':'2026-10-06T19:02:55','EndTime':'2026-10-06T19:11:55','NodeList':'h100-10','UserId':'root(0)'}.items():need(fields.get(k)==v,'Literal terminal '+k)
need(type(wire['live']['scontrol']['returncode'])is int and wire['live']['scontrol']['returncode']==0,'Terminal command0')
need(unique((O/'evidence/capacity-micro4/exit.json').read_bytes())=={'returncode':1,'rl_outer_updates':0},'Actual capacityexit1/0RL')
need((O/'job-exit.txt').read_bytes()==b'1\n' and unique((O/'evidence/hardware-exit.json').read_bytes())=={'returncode':0},'Actual wrapper1/hardware0')
need(wire['live']['gpu_apps']['returncode']==0 and wire['live']['gpu_apps']['stdout']=='' and wire['live']['gpu']['returncode']==0,'Postexit NVML empty apps')
report=dict(version=1,status='ACTUAL_GPU_PROBE_REJECTED_CAPACITY_HEADROOM_INDEPENDENT_REVIEW',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id='3303',source_commit=SOURCE,snapshot_path=str(O),captured_utc=wire['utc'],original_transport_sha256=sha(wire_raw),original_files_sha256=pins,reviewed_source_files=source_pins,
 actual_rank_records=rank_records,global_short_recomputed=dict(native_first_squared_norm=a,native_repeat_squared_norm=b,difference_squared_norm=d,relative_l2=rel,cosine_from_recorded_norm_identity=cos,raw_gpu_dot_cosine=1.0000000000000004,checked_shards=8*37,checked_shard_elements=elements),
 failure=dict(category='Conservative15percent device headroom, not short repeatability or OOM',all_eight_ranks_rejected=True,all_eight_same_path_short_pass=True,lp8_pass_all_eight=True,synthetic_adam_calls_per_rank=2,synthetic_adam_calls_summed_over_ranks=16,rl_outers_completed=0,oom_observed=False,threshold_unchanged=True),
 memory=dict(total_gib=rank_records[0]['gpu_total_bytes']/1024**3,peak_tensor_allocated_gib=rank_records[0]['peak_allocated_bytes']/1024**3,estimated_used_gib_min=min(x['estimated_peak_device_used_bytes'] for x in rank_records)/1024**3,estimated_used_gib_max=max(x['estimated_peak_device_used_bytes'] for x in rank_records)/1024**3,estimated_free_fraction_min=min(x['estimated_free_fraction'] for x in rank_records),estimated_free_fraction_max=max(x['estimated_free_fraction'] for x in rank_records),required_free_fraction=.15,scope='PyTorch peak reserved + pre-model external usage +1GiB slack; not a continuous NVML peak and not full vLLM coexistence'),
 literal_terminal=dict(state=fields['JobState'],exit_code=fields['ExitCode'],duration_seconds=540,restarts=0,start_bjt=fields['StartTime'],end_bjt=fields['EndTime'],capacity_returncode=1,wrapper_returncode=1,observed_postexit_gpu_applications_empty=True,scope='Successful captured Slurm and NVML terminal reads; no full resource teardown/cost claim'),
 roundoff=dict(raw_cosine=1.0000000000000004,excess=4.440892098500626e-16,independent_delta_is_exact_zero=True,reason='GPU dot and norm reductions accumulated separately; identical vectors have cosine1 mathematically. Frozen formal consumer additionally insists <=1, so a separately reviewed roundoff-bound representation is needed for a future successful probe.',allowed_new_candidate_scope='Preserve raw cosine and add bounded1e-12 roundoff normalization only; reject larger excursions/nonfinite. This does not change KL loss, gradient, or .9999 numeric threshold.'),
 candidate_math_review=dict(prospective=True,teacher_target_fp32_logits_gib=4*4096*151936*4/1024**3,corresponding_fp32_hidden_features_gib=4*4096*2048*4/1024**3,proposal='Retain full teacher base4/native full student head and frozenFP32 teacher. Recompute the frozen teacher output projection from the same hidden features inside chunk checkpoint backward instead of keeping every FP32 target-logit tensor alive.',real_arithmetic_full_kl_unchanged_if=['Same hidden features, teacher weights, exact masks, temperature, component_clip=None and global valid-token denominator.','Teacher frozen/eval/nativeFP32 same-device and whole base4 unchanged.','Checkpoint binds each chunk explicitly; no late-binding loop closure or retained logits in hooks.','Backward runs while teacher remains resident; raw target/q and all student parameter gradients checked against stored target.'],not_proven='Theoretical9.2734GiB retained target-logit size is not guaranteed peak-reserved reduction. Must run actual unchanged numeric/15%capacity gates and fresh5; no implementation or new GPU result reviewed here.'),
 scope_limits=['Only local read/recomputation of seven snapshot3 transported original files plus eight actual GPU rank records and frozen source.','Torchrun physical line125 contains two adjacent complete rank JSON objects; both exact raw spans retained. An exploratory json.loads whole-line read rejected ExtraData; no source or evidence was changed.','Tracked positive parameter coordinate per rank proves each synthetic Adam changed that rank; it does not prove every parameter changed.','No formal training, fresh5/save-reload, trajectory effect, final200, new job, signature, private key, SSH, GPU or source modifications in this review.'])
p=H/'ACTUAL3303_FULLTEACHER_CAPACITY_FAILURE_INDEPENDENT_REVIEW.json'
with p.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
md=H/'ACTUAL3303_FULLTEACHER_CAPACITY_FAILURE_INDEPENDENT_REVIEW.md'
with md.open('x') as f:f.write('# Actual3303 capacity failure independent review\n\nAll seven transported snapshot3 originals match full SHA. Independent replay joins all eight actual rank records, including two adjacent JSON objects on one physical torchrun line. The8×37 recorded gradient shards have identical first/repeated squared norms, zero delta and zero maximum difference. Full teacher4/native student head repeated loss and teacher logits/q also have zero difference. Independent global norm identity gives cosine1 and relativeL2=0. The raw GPU cosine1.0000000000000004 is4.44e-16 separate-reduction roundoff, preserved here.\n\nAll eight LP8 checks pass. All eight ranks complete two actual synthetic Adam calls with finite loss/gradient norm and a positive tracked-coordinate update; this is16 rank-local calls, zero RL outers. Frozen teacher has no gradient tensors, trainable parameters or optimizer membership. The worst-width probe uses actor4/input6144/response4096/fullteacher4 and actual backward checkpointing.\n\nThe fixed15% memory margin fails on every rank. Conservative used memory is71.4842–72.2446GiB of79.1808GiB; free margin is about8.75–9.72%, below15%. Peak tensor allocation is47.9246GiB, while allocator reservation and external baseline plus1GiB slack make the conservative estimate larger. No allocator OOM or retry is reported. Slurm is literallyFAILED1:0,540seconds, zero restarts; capacity and wrapper exit1 join. Captured postexit NVML applications are empty. This is not successful formal training.\n\nRecomputing the frozen teacher head inside each checkpoint from the same hidden features can preserve fullKL mathematically while avoiding retention of roughly9.2734GiB of FP32 target logits. It must keep whole teacher base4, native student head, weights, masks, temperature and denominator unchanged and bind chunks explicitly. Actual numeric equivalence and15% headroom are still required. Theoretical saved tensor size does not prove peak memory reduction or long-training correctness. No production code, old evidence, job, signature or GPU was changed by this review.\n')
print(json.dumps({'json':str(p),'sha256':sha(p.read_bytes()),'md_sha256':sha(md.read_bytes()),'global':report['global_short_recomputed'],'memory':report['memory']}))
