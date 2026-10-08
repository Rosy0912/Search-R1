from pathlib import Path
import hashlib,json,subprocess,ast
H=Path(__file__).resolve().parent.parent
REPO=H.parent/'opsd-unclipped-multitraj200-teacherrecompute'
SOURCE='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
hash=lambda raw:hashlib.sha256(raw).hexdigest()
bp=H/'runtime-tools-bound-fcecbea5d939-v5.json';cp=H/'control-tools-bound-fcecbea5d939-v5.json'
braw=bp.read_bytes();craw=cp.read_bytes();b=json.loads(braw);c=json.loads(craw)
assert hash(braw)=='8cec4d887f656cecf7d533b91ac020da169ded4142a14890351501c9c76b2228'
assert hash(craw)=='4ceb114a715b65fff2b47d3920c96b6da8b1420a68680ed5eb1657ee7647f80d'
assert c['binding_sha256']==hash(braw) and b['source_commit']==c['source_commit']==SOURCE
head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip();status=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True);assert head==SOURCE and not status
assert len(b['source_files'])==19
for n,pin in b['source_files'].items():assert hash((REPO/n).read_bytes())==pin
previous=json.loads((H/'CONTROL_TEACHERRECOMPUTE_DIFF_REVIEW.json').read_bytes())
for n,r in previous['template_pins'].items():assert hash((H/n).read_bytes())==r['candidate_sha256']
for role in ('formal','native','first5'):
 r=b[role];d=Path(r['local_directory']);assert d.parent.resolve()==H.resolve() and r['remote_directory']==b['remote_root']+'/tools/'+d.name
 for n,pin in {**r['files'],r['manifest_name']:r['manifest_sha256']}.items():assert hash((d/n).read_bytes())==pin
replace={'BLOCKED_UNBOUND_SOURCE_COMMIT':SOURCE,'BLOCKED_BOUND_SPEC_FILENAME':bp.name,'BLOCKED_BOUND_REVIEW_JSON':'INDEPENDENT_BOUND_TOOLS_REVIEW_V5.json','BLOCKED_BOUND_REVIEW_MD':'INDEPENDENT_BOUND_TOOLS_REVIEW_V5.md','BLOCKED_NATIVE_INVENTORY_FILENAME':'native-source-inventory-'+SOURCE[:12]+'-ACTUAL.json','BLOCKED_FORMAL_MANIFEST_SHA':b['formal']['manifest_sha256'],'BLOCKED_RECEIPT_MANIFEST_SHA':b['first5']['manifest_sha256'],'BLOCKED_GATE_SOURCE_SHA':b['source_files']['search_r1/baselines/checkpoint_gate.py'],'BLOCKED_LOCAL_RECEIPT_MANIFEST':Path(b['first5']['local_directory']).name+'/'+b['first5']['manifest_name'],'BLOCKED_NATIVE_MANIFEST_RELATIVE':Path(b['native']['local_directory']).name+'/'+b['native']['manifest_name'],'BLOCKED_FORMAL_MANIFEST_RELATIVE':'tools/'+Path(b['formal']['local_directory']).name+'/'+b['formal']['manifest_name']}
records={}
assert set(c['files'])=={'deploy_native_cpu.py','deploy_tools.py','publish_first5_release.py','publish_gpu_init.py','submit_reviewed.py'}
for n,r in c['files'].items():
 raw=(H/n).read_bytes()
 for a,z in replace.items():raw=raw.replace(a.encode(),z.encode())
 p=Path(r['path']);actual=p.read_bytes();assert p.parent.resolve()==H.resolve() and actual==raw and hash(raw)==r['sha256'];compile(raw,str(p),'exec',dont_inherit=True)
 assert b'BLOCKED_' not in raw
 for old in [b'667c',b'b611864d3f6a',b'66ea4c909369',b'3299',b'3302',b'3303']:assert old not in raw
 records[n]={'path':str(p.resolve()),'sha256':hash(raw),'bytes':len(raw)}
for n in ('collect_pre_gpu.py','snapshot_actual.py'):
 tree=ast.parse((H/n).read_bytes());opts=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='add_argument' and x.args and isinstance(x.args[0],ast.Constant) and x.args[0].value=='--binding'];assert len(opts)==1 and any(k.arg=='required' and isinstance(k.value,ast.Constant) and k.value.value is True for k in opts[0].keywords)
assert "path=HERE.parent/'opsd-unclipped-multitraj200-teacherrecompute/search_r1/baselines/checkpoint_gate.py'" in (H/'publish_first5_release.py').read_text()
transport=(H/'transport.py').read_text();assert 'p.parent.stat().st_uid==os.getuid()==0' in transport and 'os.O_EXCL|os.O_NOFOLLOW' in transport and 'os.link(t,p,follow_symlinks=False)' in transport and 'os.fsync(fd)' in transport
publish=(H/'deploy_tools.py').read_text();assert publish.index("for p,pin in row['files'].items():publish")<publish.index("publish(dest+'/'+row['manifest_name'],m.read_bytes())")
for n in ('publish_gpu_init.py','publish_first5_release.py'):
 raw=(H/n).read_text();assert 'e1-300/review-signing/private.pem' in raw and 'openssl' in raw and ('LOCAL_KEY' in raw or 'PRIVATE' in raw)
init=(H/'publish_gpu_init.py').read_text();first=(H/'publish_first5_release.py').read_text();assert 'review.get(\'job_id\') == job' in init and "review.get('job_id')==args.job_id" in first
assert 'INITIALIZATION_APPROVAL' in init and 'ACTUAL_FIVE_RELEASE_ONLY_NOT_200_OR_EFFECT_ACCEPTANCE' in first
assert b'INDEPENDENT_BOUND_TOOLS_REVIEW_V5.json' in Path(c['files']['deploy_tools.py']['path']).read_bytes()
print(json.dumps({'status':'PASS_LOCAL_BOUND_V5_STATIC_INTERFACE_NOT_PRODUCTION','source_commit':SOURCE,'source_clean':True,'source_files_count':19,'runtime_binding_sha256':hash(braw),'control_binding_sha256':hash(craw),'bound_files':records,'replacements':replace,'all12_previously_reviewed_template_bytes_unchanged':True,'collectors_explicit_binding_required':True,'transport_owned_UID0_and_no_clobber_unchanged':True,'tool_manifest_published_after_members':True,'actual_job_ready_review_and_signature_binding_unchanged':True,'current_job_id_file_exists':(H/'actual-job-id.txt').exists(),'scope':'Local read/hash/AST/syntax only; no imports of control main, SSH, GPU, signing, publishing or submitting; no six unchanged tests repeated'},indent=2))
