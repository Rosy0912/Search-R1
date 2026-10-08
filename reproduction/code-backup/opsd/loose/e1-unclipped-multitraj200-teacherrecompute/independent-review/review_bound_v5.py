"""Local exact binding review only: no control/main/remote/signature execution."""
import pathlib,json,hashlib,os,subprocess,datetime,sys,copy
H=pathlib.Path(__file__).resolve().parents[1];REPO=H.parent/'opsd-unclipped-multitraj200-teacherrecompute'
SOURCE='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
sha=lambda b:hashlib.sha256(b).hexdigest()
def need(c,s):
 if not c:raise ValueError(s)
def read(p):
 def pairs(rows):
  d={}
  for k,v in rows:need(k not in d,'Duplicatekey');d[k]=v
  return d
 return json.loads(p.read_bytes(),object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('NonfiniteJSON')))
bpath=H/'runtime-tools-bound-fcecbea5d939-v5.json';spec=read(bpath);pin=sha(bpath.read_bytes())
need(pin=='8cec4d887f656cecf7d533b91ac020da169ded4142a14890351501c9c76b2228','Actual boundfile pin')
review=read(H/'INDEPENDENT_SOFTWARE_REVIEW.json');candidate=read(H/'independent-review/TEACHER_RECOMPUTE_CANDIDATE_BYTES_INDEPENDENT_REVIEW.json')
need(spec['source_commit']==SOURCE and spec['source_files']==review['reviewed_source_files']==candidate['reviewed_source_files'] and len(spec['source_files'])==19,'Sourcebinding')
need(subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip(),'Actual clean source')
for n,p in spec['source_files'].items():need(sha((REPO/n).read_bytes())==p and sha(subprocess.check_output(['git','-C',str(REPO),'show',SOURCE+':'+n]))==p,'Actual source tree '+n)
need(spec['remote_repo']==spec['remote_root']+'/repo-'+SOURCE[:12],'New sourcephysicalnamespace')
records={};all_inodes=[];member_count=0
for role in ('formal','native','first5'):
 b=spec[role];d=pathlib.Path(b['local_directory']);need(d.parent==H and d.resolve()==d and not d.is_symlink(),'Physical localbundle')
 need(d.stat().st_uid==os.getuid() and d.stat().st_mode&0o777==0o555,'Immutable directory')
 need(b['remote_directory']==spec['remote_root']+'/tools/'+d.name,'Actual remote bundle')
 need(set(p.name for p in d.iterdir())==set(b['files'])|{b['manifest_name']},'Exact memberinventory')
 p=d/b['manifest_name'];need(sha(p.read_bytes())==b['manifest_sha256'],'Manifestfullpin');m=read(p)
 need(type(m['version'])is int and m['version']==1 and m['source_commit']==SOURCE and m['tools']==b['files'],'Manifesttypedsource/toolpins')
 if role!='native':need(m['source_files']==spec['source_files'],'Manifest source19')
 else:need(m['saved_adam_steps']==50 and m['horizon']==200 and m['warmup_steps']==57 and m['saved_lr']==1e-6*(5/57),'First5 exactclock')
 for name,sp in dict(b['files'],**{b['manifest_name']:b['manifest_sha256']}).items():
  f=d/name;need(f.resolve()==f and not f.is_symlink() and f.stat().st_uid==os.getuid() and f.stat().st_mode&0o777==0o444,'Physical immutable member')
  raw=f.read_bytes();need(sha(raw)==sp,'Memberpin');all_inodes.append((f.stat().st_dev,f.stat().st_ino));member_count+=1
  if name.endswith('.py'):compile(raw,str(f),'exec',dont_inherit=True)
 records[role]=dict(path=str(d),manifest_sha256=b['manifest_sha256'],files=b['files'],manifest_schema=m)
need(member_count==16 and len(set(all_inodes))==16,'No member inodealias')
replace={'BLOCKED_UNBOUND_SOURCE_COMMIT':SOURCE}
for name in spec['formal']['files']:
 raw=(H/'formal-tools-candidate'/name).read_bytes()
 for a,b in replace.items():raw=raw.replace(a.encode(),b.encode())
 need(raw==(pathlib.Path(spec['formal']['local_directory'])/name).read_bytes(),'Formalexactrender '+name)
for name in spec['native']['files']:
 raw=(H/'first5-native-candidate'/name).read_bytes()
 for a,b in replace.items():raw=raw.replace(a.encode(),b.encode())
 need(raw==(pathlib.Path(spec['native']['local_directory'])/name).read_bytes(),'Nativeexactrender '+name)
replace.update(BLOCKED_NATIVE_TOOLS_PATH=spec['native']['remote_directory'],BLOCKED_NATIVE_INSPECTOR_SHA=spec['native']['files']['inspect_payloads.py'],BLOCKED_NATIVE_MANIFEST_SHA=spec['native']['manifest_sha256'],BLOCKED_GATE_SOURCE_SHA=spec['source_files']['search_r1/baselines/checkpoint_gate.py'])
for name in spec['first5']['files']:
 raw=(H/'first5-receipt-candidate'/name).read_bytes()
 for a,b in replace.items():raw=raw.replace(a.encode(),b.encode())
 need(raw==(pathlib.Path(spec['first5']['local_directory'])/name).read_bytes(),'First5exactrender '+name)
need(spec['native']['files']['inspect_payloads.py']=='77afa00b2b3e0fdcf674f5022f8b6cc39e12818bc7e825f8e58128fecab0f68b','Repaired native77af unchanged')
cp=H/'control-tools-bound-fcecbea5d939-v5.json';controls=read(cp)
need(controls['source_commit']==SOURCE and controls['binding_sha256']==pin and set(controls['files'])=={'deploy_tools.py','deploy_native_cpu.py','submit_reviewed.py','publish_gpu_init.py','publish_first5_release.py'},'Fivecontrols')
replace={'BLOCKED_UNBOUND_SOURCE_COMMIT':SOURCE,'BLOCKED_BOUND_SPEC_FILENAME':bpath.name,'BLOCKED_BOUND_REVIEW_JSON':'INDEPENDENT_BOUND_TOOLS_REVIEW_V5.json','BLOCKED_BOUND_REVIEW_MD':'INDEPENDENT_BOUND_TOOLS_REVIEW_V5.md','BLOCKED_NATIVE_INVENTORY_FILENAME':'native-source-inventory-'+SOURCE[:12]+'-ACTUAL.json','BLOCKED_FORMAL_MANIFEST_SHA':spec['formal']['manifest_sha256'],'BLOCKED_RECEIPT_MANIFEST_SHA':spec['first5']['manifest_sha256'],'BLOCKED_GATE_SOURCE_SHA':spec['source_files']['search_r1/baselines/checkpoint_gate.py'],'BLOCKED_LOCAL_RECEIPT_MANIFEST':pathlib.Path(spec['first5']['local_directory']).name+'/'+spec['first5']['manifest_name'],'BLOCKED_NATIVE_MANIFEST_RELATIVE':pathlib.Path(spec['native']['local_directory']).name+'/'+spec['native']['manifest_name'],'BLOCKED_FORMAL_MANIFEST_RELATIVE':'tools/'+pathlib.Path(spec['formal']['local_directory']).name+'/'+spec['formal']['manifest_name']}
for name,row in controls['files'].items():
 p=pathlib.Path(row['path']);need(p.parent==H and p.resolve()==p and not p.is_symlink() and p.name==pathlib.Path(name).stem+'-bound-'+SOURCE[:12]+'-v5.py','Actual siblingcontrol')
 raw=p.read_bytes();need(sha(raw)==row['sha256'],'Controlfullpin')
 expected=(H/name).read_bytes()
 for a,b in replace.items():expected=expected.replace(a.encode(),b.encode())
 need(expected==raw and b'BLOCKED_' not in raw,'Exactcontrolrender');compile(raw,str(p),'exec',dont_inherit=True)
# Execute only pure schema validation from the exact bound formal module. No main/control.
formal_dir=pathlib.Path(spec['formal']['local_directory']);sys.path.insert(0,str(formal_dir));fp=formal_dir/'formal_supervisor.py'
ns={'__name__':'independent_bound_schema_only','__file__':str(fp)};exec(compile(fp.read_bytes(),str(fp),'exec'),ns)
def fixture():
 numeric=dict(comparison_kind='NATIVE_FULL_HEAD_FULL_TEACHER4_STORED_VS_RECOMPUTED_TARGETS',
  response_only_logits=False,stored_teacher_forward_micro_batch=4,recomputed_teacher_forward_micro_batch=4,
  stored_teacher_logits_recompute=False,recomputed_teacher_logits_recompute=True,passed=True,
  global_native_stored_squared_gradient_norm=1.,global_native_recomputed_squared_gradient_norm=1.,
  shard_gradient_relative_l2=0.,shard_gradient_cosine=1.,shard_gradient_cosine_raw=1.0000000000000004,
  shard_gradient_cosine_roundoff_applied=True,maximum_cosine_roundoff=1e-12,loss_relative_difference=0.,
  teacher_q_mean_probability_l1_difference=0.,teacher_q_forward_mean_probability_l1_difference=0.,
  teacher_q_backward_mean_probability_l1_difference=0.,
  teacher_projection_calls=dict(stored_forward=2,stored_backward=0,recomputed_forward=2,recomputed_backward=2),
  teacher_projection_forward_feature_sha256=['a'*64,'b'*64],teacher_projection_backward_feature_sha256=['a'*64,'b'*64])
 return dict(passed=True,source_commit=SOURCE,job_id='DECLARED_LOCAL_SCHEMA_FIXTURE',world_size=8,
  optimizer_steps=2,component_clip=None,response_only_logits=False,teacher_forward_micro_batch=4,
  teacher_logits_recompute=True,local_micro=4,input_width=6144,response_width=4096,
  minimum_required_free_fraction=.15,estimated_peak_device_used_bytes=60.,gpu_total_bytes=100.,
  short_native_teacher_logits_recompute_equivalence=numeric,
  old_logprob_capacity=dict(passed=True,local_micro=8,global_micro=64,logprob_shape=[8,4096],
   entropy_shape=[8,4096],logprob_finite=True,entropy_finite=True,estimated_peak_device_used_bytes=40.))
validate=ns['validate_capacity_rank'];validate(fixture(),'DECLARED_LOCAL_SCHEMA_FIXTURE');cases=[]
for label,mutate in [
 ('teacher2',lambda r:r.update(teacher_forward_micro_batch=2)),
 ('recompute_disabled',lambda r:r.update(teacher_logits_recompute=False)),
 ('backward_projection_absent',lambda r:r['short_native_teacher_logits_recompute_equivalence']['teacher_projection_calls'].update(recomputed_backward=0)),
 ('wrong_backward_feature',lambda r:r['short_native_teacher_logits_recompute_equivalence'].update(teacher_projection_backward_feature_sha256=['c'*64,'b'*64])),
 ('raw_cosine_not_preserved',lambda r:r['short_native_teacher_logits_recompute_equivalence'].update(shard_gradient_cosine_raw=1.01)),
 ('false_roundoff_flag',lambda r:r['short_native_teacher_logits_recompute_equivalence'].update(shard_gradient_cosine_roundoff_applied=False)),
 ('excess_roundoff',lambda r:r['short_native_teacher_logits_recompute_equivalence'].update(shard_gradient_cosine_raw=1+2e-12)),
 ('failedcosine',lambda r:r['short_native_teacher_logits_recompute_equivalence'].update(shard_gradient_cosine_raw=.999899,shard_gradient_cosine=.999899,shard_gradient_cosine_roundoff_applied=False)),
 ('boolnorm',lambda r:r['short_native_teacher_logits_recompute_equivalence'].update(global_native_stored_squared_gradient_norm=True)),
 ('bad_backward_q',lambda r:r['short_native_teacher_logits_recompute_equivalence'].update(teacher_q_backward_mean_probability_l1_difference=.00011,teacher_q_mean_probability_l1_difference=.00011)),
 ('headroom_failed',lambda r:r.update(estimated_peak_device_used_bytes=86.)),
 ('wrongkind',lambda r:r['short_native_teacher_logits_recompute_equivalence'].update(comparison_kind='SAME_NATIVE_FULL_HEAD_FULL_TEACHER4_PATH_REPEATABILITY'))]:
 r=fixture();mutate(r)
 try:validate(r,'DECLARED_LOCAL_SCHEMA_FIXTURE')
 except (ValueError,KeyError):cases.append(label)
 else:raise ValueError('Boundschemaaccepted '+label)
for legacy in ['short_native_teacher_row_equivalence','short_native_response_head_equivalence','short_native_full_path_repeatability']:
 r=fixture();r[legacy]=r.pop('short_native_teacher_logits_recompute_equivalence')
 try:validate(r,'DECLARED_LOCAL_SCHEMA_FIXTURE')
 except KeyError:cases.append(legacy)
 else:raise ValueError('Legacyaccepted')
# Exact copied candidates remain the reviewed production/test/tool bytes; no rerun101 suite.
need(sha((H/'formal-tools-candidate/formal_supervisor.py').read_bytes())==candidate['reviewed_external_tools']['formal-tools-candidate/formal_supervisor.py'],'Reviewed candidate drift')
need(sha(bpath.read_bytes())==pin,'Bindingendstable')
report=dict(version=1,status='PASS_INDEPENDENT_SOURCE_BOUND_SOFTWARE_ONLY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=SOURCE,revision='v5',runtime_binding_sha256=pin,runtime_binding=dict(path=str(bpath),sha256=pin),source_software_review=dict(path=str(H/'INDEPENDENT_SOFTWARE_REVIEW.json'),sha256=sha((H/'INDEPENDENT_SOFTWARE_REVIEW.json').read_bytes())),reviewed_source_files=spec['source_files'],reviewed_bundles=records,control_binding=dict(path=str(cp),sha256=sha(cp.read_bytes())),reviewed_control_files=controls['files'],exact_template_substitutions_only=True,immutable_bundle_members=16,physical_owner_uid=os.getuid(),all_bundle_members_distinct_inodes=True,native_repaired_reader_77af_unchanged=True,bound_formal_schema_cpu_only=dict(valid_localfixture_accepted=True,negative_cases_rejected=cases,scope='Actual bound consumer compiled exactbytes; schema-only declaredfixture, no GPU/Adam/RL; candidate176+independent8+10CPU evidence not rerun.'),candidate_software_review=dict(path=str(H/'independent-review/TEACHER_RECOMPUTE_CANDIDATE_BYTES_INDEPENDENT_REVIEW.json'),sha256=sha((H/'independent-review/TEACHER_RECOMPUTE_CANDIDATE_BYTES_INDEPENDENT_REVIEW.json').read_bytes())),remaining_proven_bound_software_blockers=[],scope='Only exact source/tool/rendered-control software qualification. Does not prove new actual job/model/GPU/teacher-target recompute equivalence/capacity/retrieval/fresh5/effect/200/final/resource release.',required_actual_next=['NativeCPUsource/config211imports+official/common/tool proof.','Realnewowned8H100jobpreGPUidentity.','Fullteacher4stored-vs-recomputedtargetGPUequivalence and actor4/LP8actualcapacity; no loweredthreshold.','Realfirst5independentHF/state/archive/RNG/data/sameprocess6release;200/final51713/terminal/resource.'],no_remote_gpu_signature_private_key_deployment_or_production_main_execution=True)
p=H/'INDEPENDENT_BOUND_TOOLS_REVIEW_V5.json'
with p.open('x') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
md=H/'INDEPENDENT_BOUND_TOOLS_REVIEW_V5.md'
with md.open('x') as f:f.write('# Independent V5 bound tools software review\n\nPASS_INDEPENDENT_SOURCE_BOUND_SOFTWARE_ONLY, actual sourcefcecbea5 and runtime binding8cec4d887f65. All19 clean source files match the committed software review and frozen candidate. Three immutable bundles contain16 exact physical UID-owned members with distinct inodes. Actual manifest pins are formal71d2501d4512, nativedc1905562f2f and first51b16c2227caf. All runtime/control bytes equal reviewed templates plus actual source/bundle/review substitutions. Five controls bind V5 names.\n\nThe exact bound formal consumer accepts the declared local stored-target versus recomputed-target fixture with preserved raw cosine1+4.44e-16 and correctly recorded normalized value/flag. It rejects missing backward projection, wrong feature identity, malformed/excess roundoff, teacher2, disabled recompute, lower cosine, bad q, failed15% headroom and all three historical numeric object fields. Repaired native77af is unchanged. Existing50Adam/warmup57/200 state and trajectory readers remain source-bound.\n\nPrior176+independent8+10 actual CPU tests are unchanged and not repeated. No proven binding/software blocker remains. Actual new native source/config/assets/import proof, owned8H100 initialization, numeric/capacity probe, fresh5 fullstate/HF/archives and same-process continuation are still required. No control/main, SSH, GPU, job, signature, private key or deployment action was executed here. Old3303FAILED is not upgraded.\n')

print(json.dumps(dict(json=str(p),sha256=sha(p.read_bytes()),md_sha256=sha(md.read_bytes()),runtime_binding_sha256=pin,formal_manifest=spec['formal']['manifest_sha256'],native_manifest=spec['native']['manifest_sha256'],first5_manifest=spec['first5']['manifest_sha256'],negative_cases=len(cases))))
