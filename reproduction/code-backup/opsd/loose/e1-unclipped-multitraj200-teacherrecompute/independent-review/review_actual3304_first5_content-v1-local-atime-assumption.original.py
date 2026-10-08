from pathlib import Path
import json,hashlib,os,sys,types,math,datetime
H=Path(__file__).resolve().parents[1];D=H/'job3304-first5-complete-originals-v3';R=D/'raw';P=R/'evidence/first5-cpu-inspection-once-v1';SOURCE='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6';REPO=H.parent/'opsd-unclipped-multitraj200-teacherrecompute';OUT=H/'independent-review/actual3304-first5-content-local-originals';OUT.mkdir(exist_ok=False)
canon=lambda d:json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
def shafile(p):
 h=hashlib.sha256();a=p.stat()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
 assert a==p.stat(),('Local original changed',p)
 return h.hexdigest()
def read(p):return json.loads(p.read_bytes())
def compiled(path,pin,name):
 raw=path.read_bytes();assert sha(raw)==pin
 m=types.ModuleType(name);m.__file__=str(path);exec(compile(raw,str(path),'exec',dont_inherit=True),m.__dict__);return m
inv=read(D/'inventory.json');assert inv['collector_source_sha256']=='c5636c9c6a12b40a397ff58fd09d6b8ac5afc8bfa058b6691f2a51c36847ff97'
assert inv['source_commit']==SOURCE and inv['job_id']=='3304' and inv['uid']==0 and inv['hostname'].split('.')[0]=='h100-10'
ex=read(D/'transport-exit.original.json');assert type(ex['returncode']) is int and ex['returncode']==0 and shafile(D/'transport.stdout.original')==ex['stdout_sha256'] and shafile(D/'transport.stderr.original')==ex['stderr_sha256']
verified={}
for name,rec in inv['files'].items():
 p=R/name;assert p.is_file() and p.stat().st_size==rec['bytes'] and shafile(p)==rec['sha256'],name
 verified[name]=rec['sha256'];assert rec['uid']==0 and rec['stat_before'][:3]==rec['stat_after'][:3]==[rec['device'],rec['inode'],0]
assert len(verified)==186 and len(inv['tensor_metadata'])==12
m=read(R/'full_state/global_step_5/manifest.json');ms=shafile(R/'full_state/global_step_5/manifest.json');assert ms=='30b38d6d96abd56ff956baf8a4ba954319c62049aff8263e0c9700037e2c8577'
for k,v in [('version',1),('global_step',5),('actual_outer_updates',5),('next_global_step',6),('horizon',200),('warmup_steps',57),('world_size',8),('driver_pid',2548319),('uid',0)]:assert type(m[k]) is int and m[k]==v
assert m['source_commit']==SOURCE and m['job_id']=='3304' and m['method']=='pure_opsd'
assert len(m['files'])==141
cfg=read(R/'full_state/global_step_5/resolved-config.json');assert sha(canon(cfg).encode())==m['config_sha256']=='0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990'
receipt=compiled(H/'first5-1b16c2227caf/receipt_payload.py','7042f5a93158541e9ff0044bf4c21849d8de16eed159c9787fdd7ae8406c564e','first5receipt');native=read(P/'native.json');hf=read(P/'hf.json');reports=[read(R/f'checkpoints/actor/global_step_5/training_state/rank_{r}.json') for r in range(8)]
receipt.check_cpu_evidence(native,hf,m,ms,reports)
assert native['inspector_source_sha256']=='77afa00b2b3e0fdcf674f5022f8b6cc39e12818bc7e825f8e58128fecab0f68b'
assert hf['inspector_source_sha256']=='b2e02be33d04d925af04a6a8f4efea4ba93997146197d54729d0a9715754c419'
for rec in m['files'].values():assert native['all_files'][rec['path']]==rec['sha256']
changes=[]
for rank in range(8):
 n=native['ranks'][rank];r=reports[rank];assert n['rank']==rank and n['pid']==r['pid'];assert r['version']==3 and r['status']=='EXACT_FULL_STATE_RELOAD_OK' and r['algorithm']=='pure_opsd'
 assert len(n['optimizer_moments'])==37 and n['unique_local_model_numel']==3085938688//8
 for mo in n['optimizer_moments']:assert type(mo['step']) is int and mo['step']==50 and mo['dtype']=='torch.float32' and len(mo['exp_avg_sha256'])==len(mo['exp_avg_sq_sha256'])==64
 for key,lay in n['model_layouts'].items():
  assert lay['global_dtype']==lay['local_dtype']=='torch.float32' and lay['mesh_device_type']=='cuda' and lay['mesh_coordinate']==[rank] and lay['mesh_ranks']==list(range(8)) and lay['placements']==['S(0)']
  assert lay['local_shape']==[lay['global_shape'][0]//8]+lay['global_shape'][1:]
 changed=sum(v for k,v in n['changed_elements_vs_original_fp32'].items() if k!='lm_head.weight');assert changed>0;changes.append(changed)
 assert n['teacher_cuda_update_entries']==5 and n['teacher_content_digest']==native['base_named_parameter_digest']==hf['base_named_parameter_digest']=='af4d2590defbcfa8deef89518f09ed5e368aef244d043b3ee20ca28624ada159'
 assert n['rng_content']['python_state_valid'] is True and n['rng_content']['cuda'][0]['device']==0
 assert len(n['component_digests'])==5 and n['object_ids']==r['object_ids'] and n['state_digest']==r['state_digest']
assert len({r['pid'] for r in reports})==8 and len({r['actor_config_digest'] for r in native['ranks']})==1
assert all(not any(info.values()) for info in hf['loading_info']) and len(hf['hf_map'])==435 and hf['actual_unique_parameters']==3085938688
assert hf['tied_embeddings'] is True and hf['total_changed_elements']>0
sam=native['driver']['sampler_content'];assert sam['seed'] is None and sam['shuffle'] is True and sam['cursor']==sam['consumed_cursor']==2560 and sam['size']==169028 and sam['same_source_cpu_state_restore_verified'] is True
assert native['driver']['trajectory_archive_state']==read(R/'checkpoints/algorithm/global_step_5/state.json')['trajectory_archive']
collector=compiled(H/'collect_first5_originals_candidate.py','c5636c9c6a12b40a397ff58fd09d6b8ac5afc8bfa058b6691f2a51c36847ff97','collector')
bodies={name:(R/name).read_bytes() for name in inv['files'] if not name.endswith('.jsonl')};assert collector.cpu_status(bodies,m,inv['local_request'])==inv['cpu_pipeline_originals']
step=read(P/'cpu-step-original.json');assert step['pid']==2591511 and step['cuda_is_initialized'] is False and step['cuda_visible_devices']=='' and step['torch']=='2.4.0+cu121' and '/job_3304/step_1\n' in step['cgroup']
for name in ['native-exit-original.json','hf-exit-original.json','srun-exit.original.json','controller-exit.original.json']:
 v=read(P/name);assert type(v['returncode']) is int and v['returncode']==0
for name in ['srun-exit.original.json','controller-exit.original.json']:assert read(P/name)['waited'] is True
for name,pin in read(P/'inspection-completed-review-required.json')['files'].items():assert shafile(P/name)==pin,name
# Exact frozen standalone parser over both unfiltered source streams.
gate=compiled(R/'source/search_r1/baselines/checkpoint_gate.py','09561719673a7c273ad39b3c9363b1a250eab0a94c20c80f40dd1dfc05589e16','gate');workerpids=tuple(r['pid'] for r in reports);warn=(R/'native-source/fully_sharded_data_parallel.py').read_bytes();outersets=[];evalsets=[]
for path in [R/'train.log',R/'driver-current-prefix.out']:
 outer=[];paused=[];evaluations=[];released=[]
 for line in path.read_text().splitlines():
  for marker,store in [('OPSD_E1_OUTER_JSON=',outer),('OPSD_E1_STEP5_PAUSED_JSON=',paused),('OPSD_E1_STEP5_RELEASED_JSON=',released),('EVAL_JSON=',evaluations)]:
   if marker in line:store.append(gate.parse_log_marker(line,marker,driver_pid=m['driver_pid'],job_id='3304',worker_pids=workerpids,checkpoint_warning_path=Path(collector.TORCH_PATH)/'utils/checkpoint.py',fsdp_warning_source_bytes=warn))
 assert len(paused)==1 and not released and [v['actual_outer_updates'] for v in outer]==[1,2,3,4,5]
 for i,v in enumerate(outer,1):
  assert type(v['global_step']) is int and v['global_step']==i and v['source_commit']==SOURCE and v['config_sha256']==m['config_sha256'];gate.validate_outer(v['metrics'],world=8,profile='multitraj200')
  metrics=v['metrics']
  for rank in range(8):assert abs(metrics[f'actor/used_learning_rate/rank_{rank}']-1e-6*(i-1)/57)<1e-20
  assert abs(metrics['actor/lr']-1e-6*i/57)<1e-20
 outersets.append(outer);evalsets.append(evaluations)
assert canon(outersets[0])==canon(outersets[1]);assert canon(evalsets[0])==canon(evalsets[1])
# Compile only exact observed helper bytes; never project pyc or arbitrary report code.
modules=read(H/'native-source-inventory-fcecbea5d939-ACTUAL.json')['modules'];helperpins={}
for pkg in ['search_r1','search_r1.baselines']:
 mod=types.ModuleType(pkg);mod.__path__=[str(REPO/Path(pkg.replace('.','/')))];sys.modules[pkg]=mod
for leaf in ['full_state','holdout','sharedholdout','trajectory_archive']:
 name='search_r1.baselines.'+leaf;local=REPO/'search_r1/baselines'/f'{leaf}.py';pin=modules[name]['sha256'];assert shafile(local)==pin
 raw=(R/'source/search_r1/baselines'/f'{leaf}.py').read_bytes() if (R/'source/search_r1/baselines'/f'{leaf}.py').exists() else local.read_bytes();assert sha(raw)==pin
 mod=types.ModuleType(name);mod.__file__=str(local);mod.__package__='search_r1.baselines';sys.modules[name]=mod;exec(compile(raw,str(local),'exec',dont_inherit=True),mod.__dict__);helperpins[name]=pin
archive=sys.modules['search_r1.baselines.trajectory_archive'];from transformers import AutoTokenizer
import torch
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
tokenizer=AutoTokenizer.from_pretrained(str(R/'checkpoints/actor/global_step_5'),local_files_only=True)
print(json.dumps({'phase':'RAW_SHA_NATIVE_HF_EXIT_AND_LOGS_CLOSED','raw_files':len(verified),'changed_each_rank':changes,'student_total_pending':12800}),flush=True)
archives=[];answers=[]
def rec(name):
 v=inv['files'][name];return {'path':v['physical'],'sha256':v['sha256'],'bytes':v['bytes']}
for outer in range(1,6):
 base=f'trajectories/outer_{outer:06d}/';qdoc=read(R/(base+'questions.json'));qs=qdoc['questions'];archive.validate_questions(qs);paired=read(R/(base+'paired-manifest.json'));studentmanifest=read(R/(base+'student-manifest.json'));identity=paired['identity']
 assert identity==qdoc['identity']==studentmanifest['identity'] and identity['outer_step']==outer and identity['job_id']=='3304' and identity['source_commit']==SOURCE and identity['config_sha256']==m['config_sha256'] and identity['driver_pid']==m['driver_pid'] and identity['teacher_model_path']==cfg['baselines']['distillation']['teacher_path'] and identity['directory']==m['run_root']+'/'+base.rstrip('/')
 assert paired['request']==rec(base+'questions.json') and paired['student_manifest']==rec(base+'student-manifest.json') and studentmanifest['student']==rec(base+'student.jsonl') and studentmanifest['request']==rec(base+'questions.json')
 assert studentmanifest['recorded_before_balancing'] is True and paired['student_count']==studentmanifest['student_count']==2560 and paired['teacher_count']==512 and studentmanifest['response_count_per_question']==5 and paired['teacher_diagnostic_answers_used_as_targets'] is studentmanifest['teacher_diagnostic_answers_used_as_targets'] is False
 for q in qs:
  assert q['original_prompt']==[dict(role='user',content=archive.E1_INSTRUCTION+'\nQuestion: '+q['question'])]
  assert q['original_student_input_ids']==tokenizer.apply_chat_template(q['original_prompt'],tokenize=True,add_generation_prompt=True)
 count=0;tokens=0
 with (R/(base+'student.jsonl')).open('rb') as f:
  for i,line in enumerate(f):
   assert line.endswith(b'\n');row=archive.parse_json(line);q=qs[i//5];t=row['tensors'];width=len(t['responses']);count+=1
   assert row['question_uid']==q['question_uid'] and type(row['question_position']) is int and row['question_position']==i//5 and type(row['response_index']) is int and row['response_index']==i%5 and row['tensor_sha256']==archive.digest(t) and row['training_targets_are_teacher_q_on_student_history'] is True
   assert set(t)==set(archive.TENSOR_FIELDS)|{'loss_mask'} and len(t['position_ids'])==len(t['input_ids']) and len(t['step_ids'])==width
   assert all(type(v) is int and v>=-1 for v in t['step_ids']) and all(type(v) is int and v>=0 for v in t['position_ids']+t['input_ids']) and all(type(v) is int and v in (0,1) for v in t['attention_mask']+t['info_mask'])
   assert t['input_ids']==t['prompts']+t['responses'] and t['loss_mask']==t['info_mask'][-width:] and t['info_mask'][:-width]==t['attention_mask'][:-width] and len(t['attention_mask'])==len(t['info_mask'])==len(t['input_ids']) and all(l<=a for l,a in zip(t['loss_mask'],t['attention_mask'][-width:]))
   assert row['prompt_text']==tokenizer.decode(t['prompts'],skip_special_tokens=False) and row['response_text']==tokenizer.decode(t['responses'],skip_special_tokens=False)
   archive.validate_student_turns(row,q,tokenizer);tokens+=sum(t['loss_mask'])
 assert count==2560
 teacher_members=[];gen=0
 for rank in range(8):
  mn=base+f'teacher-rank-{rank}.manifest.json';tn=base+f'teacher-rank-{rank}.jsonl';rep=read(R/mn);linked=paired['teacher_rank_members'][rank]
  assert linked['rank']==rank and linked['manifest']==rec(mn) and linked['teacher']==rec(tn) and rep['teacher']==rec(tn) and rep['identity']==identity and rep['request_sha256']==rec(base+'questions.json')['sha256']
  assert type(rep['version']) is int and rep['version']==1 and rep['kind']=='OPSD_REAL_FIXED_TEACHER_SHORT_ANSWER_DIAGNOSTIC_RANK' and type(rep['pid']) is int and rep['pid']==reports[rank]['pid'] and rep['uid']==0 and rep['hostname'].split('.')[0]=='h100-10' and rep['world_size']==8 and rep['rank']==rank and rep['question_count']==64
  assert rep['rng_restored'] is rep['model_state_restored'] is True and rep['rng_witness']['before_sha256']==rep['rng_witness']['after_sha256'] and rep['model_state_witness']['before_sha256']==rep['model_state_witness']['after_sha256']
  rows=[archive.parse_json(line) for line in (R/tn).read_bytes().splitlines()];assert len(rows)==64
  for j,row in enumerate(rows):
   q=qs[rank*64+j];assert row['question_uid']==q['question_uid'] and row['question_position']==q['question_position'] and row['question']==q['question'] and row['data_source']==q['data_source'] and row['input_messages']==archive.teacher_diagnostic_messages(q) and row['accepted_answers']==q['accepted_answers'] and row['accepted_answers_provenance']==q['accepted_answers_provenance']
   assert row['generated_by']=='fixed_official_teacher.model.generate' and row['privileged_training_answers'] is True and row['not_used_as_student_or_distillation_or_evaluation_input'] is True and row['is_full_teacher_search_trajectory'] is False and 0<len(row['output_token_ids'])<=128 and row['diagnostic_loss_mask']==[0]*len(row['output_token_ids'])
   assert row['output_attention_mask']==[1]*len(row['output_token_ids']) and row['input_attention_mask']==[1]*len(row['input_token_ids']) and row['input_token_ids']==tokenizer.apply_chat_template(row['input_messages'],tokenize=True,add_generation_prompt=True)
   assert row['input_text']==tokenizer.decode(row['input_token_ids'],skip_special_tokens=False) and row['output_text']==tokenizer.decode(row['output_token_ids'],skip_special_tokens=False)
   assert row['decoding']==dict(do_sample=False,max_new_tokens=128,num_beams=1,use_cache=False,temperature_applies=False,independent_of_student_history=True)
  assert rep['generated_tokens']==sum(len(row['output_token_ids']) for row in rows) and rep['input_tokens']==sum(len(row['input_token_ids']) for row in rows)
  gen+=rep['generated_tokens'];teacher_members.append(dict(manifest=rec(mn),teacher=rec(tn)))
 a=dict(version=1,status='RAW_PAIRED_DIAGNOSTIC_ARCHIVE_AND_ACTUAL_WORKER_BINDINGS_VALIDATED_NOT_EFFECT_PASS',actual_worker_bindings_verified=True,identity=identity,paired_manifest=rec(base+'paired-manifest.json'),request=rec(base+'questions.json'),student_manifest=rec(base+'student-manifest.json'),student=rec(base+'student.jsonl'),student_count=2560,teacher_count=512,teacher_rank_members=teacher_members)
 archives.append(a);answers.append({'outer':outer,'valid_student_tokens':tokens,'teacher_generated_tokens':gen});print(json.dumps({'phase':'ACTUAL_ALL_RAW_ARCHIVE_STREAM_VALIDATED','outer':outer,'student':count,'teacher':512}),flush=True)
actual_archive=dict(status='RAW_FIRST5_PAIRED_ARCHIVES_AND_NATIVE_PID_BINDINGS_VALIDATED_NOT_EFFECT_PASS',actual_outers=5,student_count=12800,teacher_count=2560,archives=archives,official_tokenizer_replayed=True)
receipt.check_archive_evidence(native,actual_archive)
assert not torch.cuda.is_initialized()
result=dict(version=1,status='PASS_ACTUAL_FIRST5_CONTENT_AND_PAIRED_ARCHIVES_ONLY_NO_SIGNATURE_RELEASE_OR_EFFECT_CLAIM',reviewer='/root/author200_review',source_commit=SOURCE,job_id='3304',driver_pid=m['driver_pid'],checkpoint_manifest_sha256=ms,pause_nonce=m['pause_nonce'],inventory_sha256=shafile(D/'inventory.json'),raw_files=verified,metadata_only_tensor_count=12,checkpoint_tensor_scope='Actual native inspector freshly hashed/loaded full PT and HF bytes; local capsule contains declared-SHA metadata only for 12 tensors, not downloaded tensors',native_sha256=shafile(P/'native.json'),hf_sha256=shafile(P/'hf.json'),changed_elements_per_rank_vs_original_fp32=changes,hf_unique_parameters=hf['actual_unique_parameters'],hf_total_changed_elements=hf['total_changed_elements'],loading_info=hf['loading_info'],native_sampler=sam,cpu_step=step,actual_archive=actual_archive,archive_token_counters=answers,actual_outer_records=outersets[0],actual_evaluation_records=evalsets[0],helper_pins=helperpins,limits=native['limitations']+['Teacher independent records are privileged short answers, not full teacher search trajectories or a no-gold teacher/student comparison.','Raw archives describe pre-balancing student rows; actual balanced loss-mask binding is source/runtime path evidence, not a second serialized per-rank witness.','Training-range resources are separately qualified; whole preflight resource bad row remains.','No root/peer signature, release, sixth iteration, final200 or improved effects are claimed by this content report.'])
p=H/'ACTUAL3304_FIRST5_CONTENT_AND_ARCHIVES_INDEPENDENT_REVIEW.json';assert not p.exists();p.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n');print(json.dumps({'report':str(p),'sha256':shafile(p)}),flush=True)
