"""Exact same-run formal admission; no submission or GPU operation."""
import sys
sys.dont_write_bytecode=True
import hashlib,json,pathlib,re,importlib.util
SCOPE='SEARCH_R1_PROFILE2_RUNTIME_GEOMETRY_FULL_HF_SEMANTIC_AND_CLEANUP_PASS_NOT_TEACHER_NOT_TOPD'
def need(ok,message):
 if not ok:raise ValueError(message)
def job_id(v):
 need(type(v) is int and v>0 or type(v) is str and re.fullmatch('[1-9][0-9]*',v) is not None,'Invalid actual decimal job identity')
 return int(v)
def read_ref(ref,root):
 p=pathlib.Path(ref['path']);need(not p.is_symlink() and root in p.resolve().parents,'Only own regular evidence path')
 raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Evidence SHA drift')
 return json.loads(raw)
def terminal_ok(t,job):
 need(job_id(t['job_id'])==job and t['returncode']==0,'Wrong/failed actual Slurm observation')
 raw=t['stdout'];need(re.findall(r'\bJobId=([0-9]+)\b',raw)==[str(job)],'scontrol job differs')
 need(re.search(r'\bJobState=COMPLETED\b',raw) and re.search(r'\bExitCode=0:0\b',raw),'Actual Slurm terminal is not COMPLETED0:0')
def profile_admission(c,g,root):
 root=pathlib.Path(root);job=job_id(g['prior_profile_job_id'])
 result=read_ref(c['profile_resource_proof'],root);review=read_ref(g['independent_profile_review'],root)
 need(review['status']=='PASS' and review['scope']==SCOPE and job_id(review['profile_job_id'])==job,'Independent actual profile review pending/wrong scope')
 need(review['reviewed_profile_result_sha256']==c['profile_resource_proof']['sha256'],'Review belongs to another result')
 need(review['prior_profile_terminal']==g['prior_profile_terminal'],'Terminal refs differ')
 terminal_ok(read_ref(g['prior_profile_terminal'],root),job)
 need(job_id(result['job'])==job and result['status']=='COMPLETE_UNREVIEWED' and result['cleanup_passed'] is True and result['error'] is None,'Actual profile not complete/clean')
 need(result['scope']=='ACTUAL_SEARCH_R1_PROFILE2_UNREVIEWED_NOT_100_NOT_TOPD','Wrong algorithm result')
 config=read_ref(review['profile_effective_config'],root)
 need(job_id(config['job'])==job and config['mode']=='profile2' and config['expected_outer_steps']==2,'Actual config job/budget differs')
 need(review['profile_effective_config']['sha256']==result['effective_config_sha256'],'Actual config SHA differs')
 need(result['evidence_root']==config['evidence_root'] and c['profile_resource_proof']['path']==str(pathlib.Path(config['evidence_root'])/'train-result.json'),'Result path is another run')
 need(g['prior_run_cleanup']==result['cleanup'],'Cleanup belongs to another run')
 cleanup=read_ref(result['cleanup'],root)
 for key in ('own_processes_remaining','baseline_identity_missing','original_gpu_baseline_missing','own_gpu_records_remaining','parent_gpu_records','guardian_errors','sampler_errors','failure_history'):
  need(cleanup[key]==[],'Actual cleanup incomplete '+key)
 need(cleanup['cleanup']['actual_cgroup_empty'] is True and cleanup['error'] is None and cleanup['primary_error'] is None and cleanup['parent_cuda_initialized'] is False and cleanup['terminal_monitors_drained_before_authorized_exit'] is True,'Actual terminal/cleanup failed')
 outer=read_ref(result['actual_outer_evidence'],root)
 need(outer['actual_completed_outer_steps']==[1,2] and outer['export_step']==2 and outer['checkpoint_directory']==config['final_checkpoint']==result['export']['directory'],'Wrong actual two-outer export')
 verifier=read_ref(review['profile_resource_verifier'],root);need(job_id(verifier['job_id'])==job,'Wrong resource verifier job')
 refs=verifier['rank_records'];need(len(refs)==8,'All8 actual rank receipts required')
 records=[]
 for rank,ref in enumerate(refs):
  need(ref['path']==str(pathlib.Path(config['evidence_root'])/'actor-resource-audit'/('actor-updates-rank%d-outer2.json'%rank)),'Actual rank record from another run')
  record=read_ref(ref,root)
  need(job_id(record['job'])==job and record['rank']==rank and record['hostname']=='h100-11' and record['cgroup']==config['cgroup'],'Actual rank receipt belongs to another job/rank/cgroup')
  records.append(record)
 # Source origin is the exact actual profile tool manifest, no shadow imports.
 manifest=read_ref(config['tools_manifest'],root);tool=pathlib.Path(config['tools'])/'resource_audit.py';expected=manifest['files']['resource_audit.py'];need(hashlib.sha256(tool.read_bytes()).hexdigest()==expected['sha256'],'Actual counter source drift')
 spec=importlib.util.spec_from_file_location('actual_profile_resource_audit',tool);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 counts=module.verify_updates(records,2)
 need(counts==verifier['actual_update_evidence']==result['actual_update_evidence'],'Actual rank counter verifier/result mismatch')
 semantic=read_ref(review['semantic_report'],root)
 need(semantic['status']=='FULL_HF_FINITE_SEMANTIC_CHANGED_UNREVIEWED' and semantic['scope']=='CPU_WEIGHT_SCAN_ONLY_NOT_TEACHER_QUALITY_NOT_TOPD_ACCEPTANCE' and job_id(semantic['job_id'])==job and semantic['outer_steps']==2,'Whole-HF semantic scan missing/wrong run')
 need(semantic['train_result']==c['profile_resource_proof'] and semantic['effective_config']==review['profile_effective_config'] and semantic['cleanup']==result['cleanup'] and semantic['outer_evidence']==result['actual_outer_evidence'],'Semantic scan refs do not bind actual profile')
 need(semantic['source_manifest']==config['source_manifest'] and semantic['tools_manifest']==config['tools_manifest'] and c['source_manifest']==config['source_manifest'],'Profile semantic/source identities differ')
 weights=semantic['semantic'];need(weights['all_finite'] is True and weights['change_beyond_base_to_teacher_dtype_cast'] is True and weights['changed_tensor_names'] and weights['compared_elements_including_tied_aliases']==32763876352 and weights['output_vocab_size']==152064,'Missing full finite semantic change')
 inv=semantic['actual_hf_inventory'];need(inv['directory']==result['export']['directory'] and inv['files']==result['export']['files'] and inv['shards']==result['export']['shards'] and inv['tensor_count']==result['export']['tensor_count']==771,'Whole HF inventory differs')
 for name,record in inv['files'].items():
  p=pathlib.Path(inv['directory'])/name;need(not p.is_symlink() and p.stat().st_size==record['size'],'HF file path/size drift')
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
  need(h.hexdigest()==record['sha256'],'Fresh full HF SHA drift')
 md=review['review_md'];p=pathlib.Path(md['path']);need(not p.is_symlink() and root in p.resolve().parents and hashlib.sha256(p.read_bytes()).hexdigest()==md['sha256'],'Independent review Markdown drift')
 return {'job_id':job,'counts':counts,'result_sha256':c['profile_resource_proof']['sha256'],'semantic_ref':review['semantic_report'],'terminal_ref':g['prior_profile_terminal']}
