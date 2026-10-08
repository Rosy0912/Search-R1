"""Read-only actual code/metadata binding review; no remote, GPU, assets, extraction or entry main."""
import base64,hashlib,importlib.util,json,shlex,stat,subprocess,tarfile
from pathlib import Path,PurePosixPath
ROOT=Path.cwd(); FREEZE=ROOT/'work/tree-fresh200-freeze-f7-v8'; REVIEW=ROOT/'work/tree-fresh200-f7-v8-v6-context-free-review'; OUT=REVIEW/'cpu-execution-binding'
sha=lambda b:hashlib.sha256(b).hexdigest()
def read_regular(p):
 p=Path(p); assert stat.S_ISREG(p.lstat().st_mode) and not p.is_symlink(); return p.read_bytes()
def bound(p,h):
 raw=read_regular(p); assert sha(raw)==h,(str(p),sha(raw),h); return raw
packet_raw=bound(FREEZE/'CPU_EXECUTION_PACKET.json','c8355641bfa1bb35ef057db050c775cec7b75ffcc58adb1a2a4127dd752e1f95'); packet=json.loads(packet_raw)
route_args_raw=bound(FREEZE/'CPU_MANAGEMENT_ROUTE_ARGV.json','4837d17caf3c49014cdac3f75e8a4a211ac11c81629a98f4c6bb06ba9baaa018'); route_args=json.loads(route_args_raw)
assert packet['node']=='h100-9' and packet['actual_remote_executed'] is False and packet['asset_transfer'] is False and packet['gpu_or_slurm_execution'] is False
stage=Path(packet['stage_directory']); assert str(stage)=='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/preparation-f7-v8-20261006'
envraw=bound(packet['stage_envelope_path'],packet['stage_envelope_sha256']); assert len(envraw)==802934; env=json.loads(envraw)
assert env['status']=='REVIEWED_CODE_METADATA_CPU_STAGE_ONLY' and set(env['files'])==set(packet['stage_files'])==set(packet['local_sources']) and len(env['files'])==10
staged={}; records={}
for name,record in env['files'].items():
 assert name==PurePosixPath(name).name and '\\' not in name and '..' not in name
 data=base64.b64decode(record['base64'],validate=True); assert len(data)==record['bytes'] and sha(data)==record['sha256']; assert packet['stage_files'][name]=={'bytes':len(data),'sha256':sha(data)}
 assert read_regular(packet['local_sources'][name])==data
 staged[name]=data;records[name]={'local_source':packet['local_sources'][name],'remote_target':str(stage/name),'sha256':sha(data),'bytes':len(data),'matches_envelope':True}
assert sum(map(len,staged.values()))<=32*1024**2
inputs=json.loads(staged['PREPARATION_INPUTS.json']); assert sha(staged['PREPARATION_INPUTS.json'])==packet['preparation_inputs_sha256']=='c5c714ffb0a6eb82f59cd53373c3fe0cd77a5b025a9ecf94deae5e285113b61e'
archive_raw=staged['CODE_PACKAGE.tar.gz'];assert len(archive_raw)==481889 and sha(archive_raw)==packet['code_archive_sha256']=='44e9c413dccbe17a274fd5bebeadb656643a56f207acfb7b8dcb155b80bc7e12'
import io
files={};tar_records={}
with tarfile.open(fileobj=io.BytesIO(archive_raw),mode='r:gz') as bundle:
 members=bundle.getmembers();assert len(members)==252
 for member in members:
  name=member.name;p=PurePosixPath(name)
  assert member.isfile() and not member.issym() and not member.islnk() and name not in files
  assert str(p)==name and not p.is_absolute() and '..' not in p.parts and '.' not in p.parts and '\\' not in name
  assert not member.linkname and not member.pax_headers
  data=bundle.extractfile(member).read();assert len(data)==member.size;data.decode('utf-8');assert b'\x00' not in data
  files[name]=data;tar_records[name]={'sha256':sha(data),'bytes':len(data),'regular':True,'utf8_text':True,'no_link':True}
manifest_raw=files['manifest.json']; assert sha(manifest_raw)==packet['code_manifest_sha256']=='aafbf0ff818f38b6cfab3e97b99889902b28ff26a63ffc102296ef6f828151be'
manifest=json.loads(manifest_raw);assert set(files)==set(manifest['files'])|{'manifest.json'}
for n,h in manifest['files'].items():assert sha(files[n])==h
package=FREEZE/'package-code-only'; assert read_regular(package/'manifest.json')==manifest_raw
materialized={str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}; assert materialized==set(files)
for n,data in files.items(): assert read_regular(package/n)==data
oldraw=bound(FREEZE/'LOCAL_DRAFT_MANIFEST.json','2162cad0fb2dcb9d5d474d8f44506eb9f658096c705fc2a98bf0570ca371bc03');old=json.loads(oldraw)
deltas={k:{'before':old.get(k),'after':manifest.get(k)} for k in set(old)|set(manifest) if old.get(k)!=manifest.get(k)}
assert set(deltas)=={'scope','status','probability_calibration','source_interface_review_scope'}
assert manifest['source_interface_review_scope']=='f7/V8 source and CPU capability independently accepted; no GPU or training acceptance'
oldpc=dict(old['probability_calibration']);newpc=dict(manifest['probability_calibration']);assert oldpc.pop('source_interface_reviewed') is False and newpc.pop('source_interface_reviewed') is True and oldpc==newpc
assert old['scope']=='LOCAL_DRAFT_NOT_RELEASED' and manifest['scope']=='FROZEN_PENDING_INDEPENDENT_FRESH200_MU_PRE_DISPATCH_REVIEW'
source=ROOT/'work/tree-behavior-mu-ppo'; head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(); assert head=='f7e762af365a2beb9429e4cb3b6d5f0d808d4d7d'; assert subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True)==''
tree_rows=subprocess.check_output(['git','ls-tree','-r','HEAD'],cwd=source,text=True).splitlines();allnames={};
for row in tree_rows:
 meta,name=row.split('\t',1);mode,kind,oid=meta.split(' ');allnames[name]=(mode,kind,oid)
root_allowed={'.gitignore','LICENSE','NOTICE','Makefile','README.md','setup.py','pyproject.toml','requirements.txt'}
suffix_allowed={'.cfg','.ini','.json','.md','.patch','.py','.sbatch','.sh','.toml','.txt','.yaml','.yml'}
plain_allowed={'LICENSE','NOTICE','.gitignore','Makefile','verl/single_controller/version/version','verl/version/version'}
selected={n for n in allnames if (n in root_allowed or n.startswith(('search_r1/','verl/'))) and (n in plain_allowed or Path(n).suffix.lower() in suffix_allowed)}
assert len(allnames)==737 and len(selected)==236 and len(set(allnames)-selected)==501
assert selected==set(manifest['source_selection']['selected_files'])
assert set(allnames)-selected==set(manifest['source_selection']['excluded_files'])
assert {n[12:] for n in files if n.startswith('source/repo/')}==selected
for n in selected:
 assert allnames[n][0] in ('100644','100755') and allnames[n][1]=='blob'
 data=subprocess.check_output(['git','show','HEAD:'+n],cwd=source);assert files['source/repo/'+n]==data and manifest['source_selection']['selected_files'][n]==sha(data)
assert manifest['source_selection']['selected_count']==236 and manifest['source_selection']['excluded_count']==501
interface=json.loads(staged['SOURCE_INTERFACE_RECEIPT.json']);context=json.loads(staged['SOURCE_CONTEXT_FREE_RECEIPT.json']);audit=json.loads(staged['SOURCE_BYTE_AUDIT.json']);helper=json.loads(staged['HELPER_SOURCE_REVIEW.json'])
assert len(audit['source_files'])==19 and len(audit['wrapper_files'])==15 and len(audit['runtime_entries'])==4
for n,e in audit['source_files'].items(): assert sha(files['source/repo/'+n])==e['sha256'] and e['matches_git_head'] is True
for n,h in interface['source_files'].items():assert sha(files['source/repo/'+n])==h
for n,e in audit['wrapper_files'].items():assert sha(files[n])==e['sha256'] and len(files[n])==e['bytes'] and e['matches'] is True
for n,e in audit['runtime_entries'].items():assert sha(files[n])==e['sha256'] and e['matches_wrapper'] is True
assert set(manifest['wrapper_files'])==set(audit['wrapper_files'])
assert context['artifacts']['BYTE_AUDIT.json']==sha(staged['SOURCE_BYTE_AUDIT.json'])
assert context['artifacts']['HELPER_V6_RECEIPT.json']==sha(staged['HELPER_SOURCE_REVIEW.json'])
assert helper['status']=='FRESH200_PREPARE_AND_SUBMIT_HELPERS_SOURCE_CPU_REVIEWED' and helper['actual_gpu_accepted'] is False
for n,h in helper['helper_files'].items(): assert sha(staged[n])==h==audit['helper_files'][n]['sha256']
for k,n in [('source_interface_receipt','SOURCE_INTERFACE_RECEIPT.json'),('context_free_review','SOURCE_CONTEXT_FREE_RECEIPT.json'),('context_byte_audit','SOURCE_BYTE_AUDIT.json')]: assert inputs[k]=={'path':str(stage/n),'sha256':sha(staged[n])}
assert inputs['archive']==str(stage/'CODE_PACKAGE.tar.gz') and inputs['target']=='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/fresh200-mu-f7-v8' and inputs['max_uncompressed_bytes']==33554432
assert inputs['manifest_sha256']==sha(manifest_raw) and inputs['archive_sha256']==sha(archive_raw)
common_path=Path(packet['local_sources']['fresh200_dispatch_common.py']);spec=importlib.util.spec_from_file_location('actual_bind_common',common_path);common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
common.validate_manifest(manifest)
local_bound={str(stage/n):data for n,data in staged.items()};
def json_bound_fixture(entry):
 data=local_bound[entry['path']];assert sha(data)==entry['sha256'];return json.loads(data)
common.json_bound=json_bound_fixture;source_result=common.validate_reviews(inputs,manifest,for_dispatch=False);assert source_result['source_review_accepted'] is True
base=['env','CUDA_VISIBLE_DEVICES=','PYTHONOPTIMIZE=0','HF_HUB_OFFLINE=1','HF_DATASETS_OFFLINE=1','/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python','-B']
stageargv=shlex.split(packet['stage_remote_command']);assert stageargv[:7]==base and stageargv[7]=='-c' and len(stageargv)==10
stage_source=bound(FREEZE/'stage_code_metadata_cpu.py',packet['stage_source_sha256']);assert stageargv[8].encode()==stage_source and stageargv[9]==sha(envraw)
for argvk,commandk in [('prepare_argv','prepare_remote_command'),('tokenizer_preflight_argv','tokenizer_preflight_remote_command')]:
 assert packet[argvk][:7]==base and shlex.split(packet[commandk])==packet[argvk] and shlex.join(packet[argvk])==packet[commandk]
prepare=packet['prepare_argv'];assert prepare[7]==str(stage/'prepare_tree_fresh200_package.py');prepareopts=dict(zip(prepare[8::2],prepare[9::2]));assert len(prepareopts)==8
assert prepareopts=={'--archive':inputs['archive'],'--archive-sha':sha(archive_raw),'--target':inputs['target'],'--manifest-sha':sha(manifest_raw),'--inputs':str(stage/'PREPARATION_INPUTS.json'),'--inputs-sha':sha(staged['PREPARATION_INPUTS.json']),'--helper-review':str(stage/'HELPER_SOURCE_REVIEW.json'),'--helper-review-sha':sha(staged['HELPER_SOURCE_REVIEW.json'])}
plan=json.loads(staged['HISTORICAL_RAW_CPU_PLAN.json']); token=packet['tokenizer_preflight_argv'];assert token[7]==inputs['target']+'/tokenizer_raw_preflight.py';tokenopts=dict(zip(token[8::2],token[9::2]));assert len(tokenopts)==3
assert tokenopts=={'--plan':str(stage/'HISTORICAL_RAW_CPU_PLAN.json'),'--plan-sha':sha(staged['HISTORICAL_RAW_CPU_PLAN.json']),'--output':inputs['target']+'/CPU_PREP_CONFIG/HISTORICAL_RAW_CPU_PREFLIGHT.json'}
assert plan['reviewed_source_commit']==head and plan['raw_helper']=={'path':inputs['target']+'/source/repo/search_r1/baselines/tree_raw_tokens.py','sha256':sha(files['source/repo/search_r1/baselines/tree_raw_tokens.py'])}
assert [r['job_id'] for r in plan['runs']]==['3295','3296'] and sum(len(r['rank_ready']) for r in plan['runs'])==16
for group in plan['runs']:
 for rank,e in enumerate(group['rank_ready']): assert e['path']==f'/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-probability-probe-{group["job_id"]}/gpu-records/rank-{rank:02d}/ready.json' and e['bytes']>0 and len(e['sha256'])==64
# Route is inspected, never imported or called.
route_path=ROOT/'work/remote.py';route_raw=bound(route_path,'6f715639275c826afc1736863a12a2da0268b98617f174f3d29efa18b682b30b')
proof={'status':'ACTUAL_CODE_METADATA_EXECUTION_BYTE_BINDINGS_PASSED_READ_ONLY','packet_sha256':sha(packet_raw),'management_route_argv_sha256':sha(route_args_raw),'route_source_sha256':sha(route_raw),'stage_envelope_sha256':sha(envraw),'stage_envelope_bytes':len(envraw),'stage_files':records,'manifest_sha256':sha(manifest_raw),'archive_sha256':sha(archive_raw),'archive_bytes':len(archive_raw),'archive_regular_file_count':252,'archive_total_uncompressed_bytes':sum(map(len,files.values())),'all_members_regular_utf8_text_no_symlink_hardlink_or_pax':True,'tar_files':tar_records,'full_selected_source_files_equal_clean_HEAD':True,'source_selected_count':236,'source_excluded_names_metadata_only_not_loaded':501,'no_model_data_tensor_payload_in_archive':True,'wrapper_count':15,'source_interface_count':len(interface['source_files']),'source_audit_count':19,'runtime_entry_count':4,'source_refs_match_actual_staged_bytes':True,'actual_manifest_validate_manifest_passed':True,'actual_source_review_validate_reviews_passed':True,'local_draft_manifest_sha256':sha(oldraw),'manifest_deltas':deltas,'stage_source_sha256':sha(stage_source),'stage_remote_argv':stageargv,'prepare_argv':prepare,'tokenizer_preflight_argv':token,'raw_plan_sha256':sha(staged['HISTORICAL_RAW_CPU_PLAN.json']),'historical_raw_rows_expected':5120,'remote_destination_existence_not_observed':True,'remote_no_clobber_enforced_by_reviewed_source':True,'actual_remote_executed':False,'actual_slurm_executed':False,'actual_gpu_accepted':False,'actual_official_assets_read':False,'actual_preparation_executed':False}
OUT.mkdir(exist_ok=True);(OUT/'ACTUAL_BYTE_BINDINGS_PROOF.json').write_text(json.dumps(proof,sort_keys=True,indent=2)+'\n')
print('READ_ONLY BYTE BINDING PASS: archive252/source236/wrapper15/interface15/audit19/runtime4/envelope10; no remote/GPU/official asset bytes read')
print('ACTUAL_BYTE_BINDINGS_PROOF.json',sha((OUT/'ACTUAL_BYTE_BINDINGS_PROOF.json').read_bytes()))
