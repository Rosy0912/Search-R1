"""Review already-retrieved local metadata only. No remote/main/GPU/model/data reads."""
import hashlib,importlib,json,subprocess,sys,yaml
from pathlib import Path
root=Path.cwd();f=root/'work/tree-fresh200-freeze-f7-v8';d=root/'work/tree-fresh200-f7-v8-v6-context-free-review';out=d/'actual-cpu-stage12-independent'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):p=Path(p);return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def boundjson(p,h=None):
 p=Path(p)
 if h is not None:assert sha(p)==h
 return json.loads(p.read_text())
binding=boundjson(d/'CPU_EXECUTION_BINDING_REVIEW.json','fdf074bdfe8ae64d20d60c01553103dcb9f8df34323618fc3d5b5d2b585bec37')
packet=boundjson(f/'CPU_EXECUTION_PACKET.json',binding['execution_packet']['sha256']);inputs=boundjson(f/'PREPARATION_INPUTS.json',binding['preparation_inputs']['sha256']);manifest=boundjson(f/'package-code-only/manifest.json',binding['code_manifest']['sha256']);declared=boundjson(f/'package-code-only/CONFIGURATION_OVERRIDES.json',manifest['files']['CONFIGURATION_OVERRIDES.json'])
stagetool=boundjson(f/'CPU_STAGE_ACTUAL_TOOL_RESULT.json');assert stagetool['exit_code']==0;stage=json.loads(stagetool['output']);assert stage['status']=='CODE_METADATA_STAGED_ONLY_NOT_EXECUTED' and stage['directory']==binding['stage_directory'] and stage['envelope_sha256']==binding['stage_envelope']['sha256'] and stage['files']==packet['stage_files']
assert all(stage[k] is False for k in ['asset_transfer','cuda_initialized','slurm_submitted']) and stage['optimizer_writes']==0
preptool=boundjson(f/'CPU_PREPARATION_ACTUAL_TOOL_RESULT.json');assert preptool['exit_code']==0;lines=preptool['output'].splitlines();assert len(lines)==2 and lines[0].startswith('BASELINE_PREBUILT_MANIFEST_JSON=');split=json.loads(lines[0].split('=',1)[1]);end=json.loads(lines[-1])
prep=boundjson(f/'actual-cpu/PREPARATION.json','a6ce9101dc2529d451123f4c23f65dca090b1db76517d446df5cec51d9705df8');configpath=f/'actual-cpu/resolved.yaml';assert sha(configpath)=='77570d7e7806407c2262c75b73b94ca23786edb52732257b3b251fb025a684a9';config=yaml.safe_load(configpath.read_text())
assert end=={'status':'SAME_NODE_FRESH200_CODE_ASSETS_AND_CPU_CONFIG_VERIFIED_NOT_SUBMITTED','path':binding['preparation_target']+'/CPU_PREP_CONFIG/PREPARATION.json','sha256':sha(f/'actual-cpu/PREPARATION.json'),'submitted':False}
assert prep['status']==end['status'] and prep['stage']=='fresh_mu_calibration' and prep['package']==binding['preparation_target'] and prep['code_commit']==binding['source_clean_head']
assert prep['archive_sha256']==binding['code_archive']['sha256'] and prep['manifest_sha256']==binding['code_manifest']['sha256'] and prep['inputs_sha256']==binding['preparation_inputs']['sha256'] and prep['helper_review_sha256']==binding['helper_review_receipt']['sha256']
assert prep['resolved_config_path']==binding['preparation_target']+'/CPU_PREP_CONFIG/resolved.yaml' and prep['resolved_config_sha256']==sha(configpath)
assert prep['actual_hydra_resolved'] is True and prep['optimizer_writes']==0 and all(prep[k] is False for k in ['actual_3b_training_accepted','actual_gpu_accepted','cuda_initialized','long_training_ready','original_sampler_actor_gate_accepted','production_thresholds_accepted','submitted'])
assert prep['model_identity']==manifest['model_identity'] and prep['official_model_path']==manifest['official_model_path']
assert hashlib.sha256(json.dumps(prep['model_identity']['files'],sort_keys=True).encode()).hexdigest()==prep['model_identity']['sha256']=='744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0'
assert prep['common_data_identity']==manifest['common_data_identity']==split['content_identity']['sha256']==split['manifest_sha256']=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
assert split['input_mode']=='prebuilt_manifest' and split['train_rows']==169028 and split['holdout_rows']==512
assert split['manifest']['seed']==42 and split['manifest']['protocol']=='search-e1-common-300' and split['manifest']['preparation']['selection']=='source-order-seed42-first256-dev-next32-stage'
assert len(split['manifest']['preparation']['raw_sources'])==9 and split['manifest']['splits']['validation']['source_counts']=={'nq':256,'hotpotqa':256}
counts={'train':169028,'validation':512,'stage':64,'final':51713};assert {k:split['manifest']['splits'][k]['rows'] for k in counts}==counts
assert all(v=={'ids':0,'questions':0} for v in split['overlaps'].values()) and len(split['overlaps'])==6
assert split['effective_paths']=={k:'/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/common-data/'+k+'.parquet' for k in counts}
assert prep['source_reviews']=={'source_interface':inputs['source_interface_receipt'],'context_free':inputs['context_free_review'],'context_byte_audit':inputs['context_byte_audit'],'source_review_accepted':True}
expected=declared['expected_resolved_values'];assert prep['checked_resolved_values']==expected
values=[]
for name,value in expected.items():
 actual=config
 for part in name.split('.'):actual=actual[part]
 assert actual==value and (type(value) is not bool or type(actual) is bool),name
 values.append({'field':name,'expected':value,'actual':actual,'matches':True})
assert declared['trial_only'] is True and declared['production_thresholds']=={'target_old_policy_kl':0.2,'negative_tail_limit':None,'initial_actor_repeat_max_abs_delta':None} and declared['configuration_overrides']==manifest['configuration_overrides']
assert prep['trial_thresholds']==manifest['trial_thresholds']=={'target_old_policy_kl':.2,'negative_tail_limit':.01,'initial_actor_repeat_max_abs_delta':0.}
assert prep['budget']=={'actor_micro_global':16,'complete_outers':200,'epochs':1,'leaves':5,'mini_global':128,'minimum_successful_optimizer_writes_per_complete_outer':1,'old_ref_micro_global':32,'optimizer_opportunities_per_rank_per_complete_outer':20,'questions':512,'successful_optimizer_writes_counted_separately':True,'total_leaves':2560,'warmup_outers':57,'zero_write_or_abandoned_batch_credits_outer':False}
assert prep['datasets']=={'final':split['effective_paths']['final'],'final_rows':51713,'train':split['effective_paths']['train'],'train_rows':169028,'validation':split['effective_paths']['validation'],'validation_rows':512}
assert prep['guard_optins']=={'behavior_mu':True,'policy_safety':True,'raw_emission':True,'raw_tokens':True}
# Execute only the frozen pure configuration validator against the actual returned YAML, with dict+attribute access. No suite or opportunity/update experiment.
class AttributeDict(dict):
 __getattr__=dict.__getitem__
def convert(value):
 if isinstance(value,dict):return AttributeDict({k:convert(v) for k,v in value.items()})
 if isinstance(value,list):return [convert(v) for v in value]
 return value
source=root/'work/tree-behavior-mu-ppo';assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==binding['source_clean_head'];assert subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True)==''
sys.path.insert(0,str(source));fresh=importlib.import_module('search_r1.baselines.tree_fresh200');assert Path(fresh.__file__).resolve()==(source/'search_r1/baselines/tree_fresh200.py').resolve();fresh.validate_fresh200_config(convert(config))
failed=boundjson(f/'CPU_PREPARATION_CONNECTION_UNKNOWN_TOOL_RESULT.json');assert failed['tool_result']['exit_code']==255 and 'Connection closed' in failed['tool_result']['output']
first=boundjson(f/'CPU_PREPARATION_FIRST_READONLY_PROBE.json');assert first['exit_code']==0;firstvalue=json.loads(first['output']);assert firstvalue['staging_receipt']['exists'] is True and all(firstvalue[k]['exists'] is False for k in ['candidate_manifest','preparation_report','resolved_config'])
notstarted=boundjson(f/'CPU_PREPARATION_NOT_STARTED_READONLY_PROOF.json');notstarted_result=notstarted.get('tool_result',notstarted);assert notstarted_result['exit_code']==0;absent=json.loads(notstarted_result['output']);assert absent=={'target_exists':False,'target_is_symlink':False,'target_entries':None,'live_exact_cpu_prepare_processes':[],'unreadable_proc_pids':[]}
source_receipt=boundjson(d/'RECEIPT.json','6f2ceaebd1ff6c04e9330b404e17ff7db9df7493db5c686cbc3366cb3253fb44');audit=boundjson(d/'BYTE_AUDIT.json','e13b4828490988d089581500352207e109ac38324a4f44542471503a95e52edf');assert sha(d/'HELPER_V6_RECEIPT.json')==binding['helper_review_receipt']['sha256']
for n,e in audit['source_files'].items():assert sha(source/n)==e['sha256']
for n,e in audit['wrapper_files'].items():assert sha(e['path'])==e['sha256']
for n,e in audit['helper_files'].items():assert sha(e['path'])==e['sha256']
proof={'status':'ACTUAL_STAGE_AND_CPU_PREPARATION_METADATA_INDEPENDENTLY_REVIEWED_NO_ZERO_READY','stage_actual_receipt':stage,'preparation_report':rec(f/'actual-cpu/PREPARATION.json'),'actual_resolved_yaml':rec(configpath),'actual_checked_config_values':values,'checked_config_field_count':len(values),'actual_config_frozen_pure_validator_passed':True,'source_clean_head':binding['source_clean_head'],'model_identity_matches_manifest':True,'common_split_metadata_identity_matches':True,'common_split_counts':counts,'validation_source_counts':{'nq':256,'hotpotqa':256},'all_split_overlap_metadata_zero':True,'actual_cpu_report_optimizer_writes':0,'opportunity_accounting_contract_matches':prep['budget'],'source_counting_CPU_suites_not_repeated':True,'source_and_wrapper_helper_frozen_bytes_still_match':True,'preparation_reexecution_evidence':{'initial_unknown_connection_error_preserved':True,'post_failure_no_materialized_target_and_no_live_exact_helper_observed':True,'no_claim_all_initial_read_only_instructions_never_executed':True,'same_CPU_argv_binding':binding['management_route_argv']['sha256']},'actual_artifacts':{name:rec(f/name) for name in ['CPU_STAGE_ACTUAL_TOOL_RESULT.json','CPU_PREPARATION_ACTUAL_TOOL_RESULT.json','CPU_PREPARATION_CONNECTION_UNKNOWN_TOOL_RESULT.json','CPU_PREPARATION_FIRST_READONLY_PROBE.json','CPU_PREPARATION_NOT_STARTED_READONLY_PROOF.json']},'reviewer_actual_remote_executed':False,'reviewer_actual_gpu_executed':False,'reviewer_actual_official_assets_read':False,'actual_stage_observed_via_retained_tool_result':True,'actual_CPU_preparation_observed_via_bound_source_report':True,'actual_historical_tokenizer_final_report_reviewed':False,'actual_zero_READY':False,'actual_gpu_accepted':False,'actual_3b_training_accepted':False,'actual_new_complete_outers':0,'actual_new_optimizer_writes':0,'production_thresholds_accepted':False,'original_sampler_actor_gate_accepted':False,'fresh_tree_capacity_probability_accepted':False,'zero_runtime_identity_blocker':'V8 missing runtime cluster/job/node typed-allocation qualification; exact CPU mismatched reply admission counterexample independently retained'}
(out/'ACTUAL_STAGE12_CPU_METADATA_PROOF.json').write_text(json.dumps(proof,sort_keys=True,indent=2)+'\n');print('ACTUAL STAGE/CPU PREP METADATA PASS; fields',len(values),'source pure config validator PASS; no READY or tokenizer completion asserted');print('ACTUAL_STAGE12_CPU_METADATA_PROOF.json',sha(out/'ACTUAL_STAGE12_CPU_METADATA_PROOF.json'))
