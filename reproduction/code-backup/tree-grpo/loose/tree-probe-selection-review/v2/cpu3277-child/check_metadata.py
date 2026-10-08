"""Review already-saved CPU3277 selection metadata only; no tensor asset reads."""
import hashlib,json,math
from pathlib import Path
ROOT=Path.cwd();HERE=ROOT/'work/tree-probe-selection-review/v2/cpu3277-child';SEL=ROOT/'outputs/tree-instruct-300/cpu3277/selection.json';EXPECTED='bca59a4508a0627a04b3f1f64e632b69488208650fa375aac52ab7d16029f995'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def ck(n,b,d=None):
 assert b,(n,d);checks.append(dict(name=n,passed=True,detail=d))
ck('original_selection_exact_SHA',sha(SEL)==EXPECTED)
s=json.loads(SEL.read_bytes())
M=ROOT/'work/tree-safety110-a632ba9-v2/manifest.json';C=ROOT/'outputs/tree-instruct-300/cpu3271/context-analysis-3271.json';R=ROOT/'outputs/tree-instruct-300/healthy110-3268/output/recovery-loaded.json'
ck('frozenpackage_exact_SHA',sha(M)=='84e3e18682afa880f1053a0025eda595fca332d550ec1f6905bc1334049b5a48')
ck('saved_context_CPU3271_exact_SHA',sha(C)=='5c018a631e3615bfa0eb9b3bcf5e8f6a886a0fc2daed89d345e3eabca7ed74e4')
m=json.loads(M.read_bytes());c=json.loads(C.read_bytes());inv=s['inventory']
base='/dataset/shaoshitong/search-r1-rl/tree-grpo';run=base+'/runs/tree-safety300-3268';pkg=base+'/instruct300/candidates/safety110-a632ba9-84e3e186'
expected_src={pkg+'/'+p:v for p,v in m['files'].items() if p.startswith('source/repo/')}
actual_src={x['path']:x['sha256'] for x in s['source_files']}
ck('source_files_unique_and_identity_duplicate_same',len(actual_src)==len(s['source_files'])==700 and s['identity']['source_files']==s['source_files'])
ck('all699_source_repo_entries_exact_manifest_metadata',{p:h for p,h in actual_src.items() if p.startswith(pkg+'/source/repo/')}==expected_src,len(expected_src))
recovery=[x for x in c['diagnostic_source_files'] if x['path']==run+'/output/recovery-loaded.json']
ck('only_extra_source_is_bound_recovery_metadata',len(recovery)==1 and {p:h for p,h in actual_src.items() if p not in expected_src}=={recovery[0]['path']:recovery[0]['sha256']} and sha(R)==recovery[0]['sha256'])
ck('selection_frozen_and_zero_execution_scope',s['schema_version']==1 and s['frozen'] is True and s['status']=='SELECTION_FROZEN_FROM_CPU_RESULTS' and s['actual_model_forwards']==s['actual_optimizer_writes']==0 and s['probability_consistency_accepted'] is False and s['long_training_ready'] is False)
raw={p:x for p,x in inv.items() if x['kind']=='raw'};mon={p:x for p,x in inv.items() if x['kind']=='monitor'}
ck('inventory72_raw64_monitor8_exact',len(inv)==72 and len(raw)==64 and len(mon)==8 and set(inv)==set(raw)|set(mon))
ck('eight_rank_raw_calls_complete_and_paths_local_metadata',all((x['rank'],x['generation_call'])==(r,k) and x['path']==run+f'/output/sampling-consistency/rank-{r:02d}-call-{k:05d}.pt' for r in range(8) for k in range(1,9) for x in [raw[f'rank-{r:02d}-call-{k:05d}.pt']]))
raw_ev={x['snapshot_path']:x for x in c['evidence'] if x['kind']=='raw_same_turn'};mon_ev={x['path']:x for x in c['evidence'] if x['kind']=='final_leaf_monitor_subset'}
ck('all64_snapshot_path_SHA_rowcount_match_saved_context_evidence',len(raw_ev)==64 and all(x['path'] in raw_ev and (x['sha256'],x['row_count'])==(raw_ev[x['path']]['snapshot_sha256'],raw_ev[x['path']]['rows']) for x in raw.values()))
ck('all_raw_original_micro4_inventory_complete_adjacent',all(x['original_micro4_groups']==[list(range(i,min(i+4,x['row_count']))) for i in range(0,x['row_count'],4)] for x in raw.values()))
plain=dict(temperature=1.,top_p=1.,top_k=-1,min_p=0.,presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.,n=1)
ck('raw_inventory_plain_distribution_no_processors',all(all(x['distribution'].get(k)==v for k,v in plain.items()) and not x['distribution'].get('logits_processors') for x in raw.values()))
ck('raw64_rows_total11528_and_effective3186179_cross_metadata',sum(x['row_count'] for x in raw.values())==s['raw_complete_counts']['rows']==c['raw_sampling_evidence']['raw_rows']==11528 and s['raw_complete_counts']['effective_tokens']==c['raw_sampling_evidence']['raw_attention_valid_response_token_occurrences']==3186179)
raw_meta={x['path']:x['sha256'] for x in s['raw_metadata']};expected_meta={x['path']:x['sha256'] for x in raw_ev.values()}
ck('all64_raw_metadata_JSON_path_SHA_match_saved_context',len(s['raw_metadata'])==len(raw_meta)==64 and raw_meta==expected_meta)
ck('monitor8_rank_path_SHA_rowcount32_cross_context',len(mon_ev)==8 and all(mon[f'rank-{r:02d}-monitor.pt']['rank']==r and mon[f'rank-{r:02d}-monitor.pt']['row_count']==32 and mon[f'rank-{r:02d}-monitor.pt']['path']==run+f'/output/probability-calibration/rank-{r:02d}-monitor.pt' and mon[f'rank-{r:02d}-monitor.pt']['sha256']==mon_ev[mon[f'rank-{r:02d}-monitor.pt']['path']]['sha256'] for r in range(8)))
asset_map={x['path']:x['sha256'] for x in inv.values()}
ck('inventory_complete_asset_aggregate_matches_identity',s['identity']['assets_sha256']==hashlib.sha256(json.dumps(asset_map,sort_keys=True).encode()).hexdigest())
ck('identity_package_Ready_config_core_parent_HF_state_fixed',s['identity']['source_sha256']==sha(M)==c['package_manifest_sha256'] and s['identity']['ready_sha256']==c['ready_sha256']=='ba948bdec0b71a3c8ff5c61db1360840134e57fccaf71fb154449f30c300c468' and s['identity']['config_sha256']==c['identity']['config_sha256']=='e25214a90cf6ac063174787b7aa2c607ef5c942218fd31c2af0cfdf2c206b71b' and s['identity']['core_commit']=='3d10326fe8cde3bf7a62bc9c7567b308cc6afc3a' and s['identity']['parent_hf_sha256']=='56df6f31c42decee1b2302875b54db32d5cfa07db5e858446502c549cb0243ef' and s['identity']['parent_state_manifest_sha256']=='e59d8011df3b1248c0c548afeb9d4ad6e564c5f9b0f84f2147bb3f99ad182366')
ck('v2_helper_and_context_pins',s['helper_sha256']=='70d07788f4a372329d9e9c3ecb73cc5ba3bbccf4da0e39d52c6a8679c31f443b' and s['context_helper_sha256']==c['helper_source_sha256']=='28e665eb822651ff44559b2e9c6110d01542c5e6433db8dd4d07c659af7131ac')
criteria=['max_raw_abs_delta','highest_above_dot01_fraction','longest_full_attention_context','position_or_prompt_padding_anomaly','hash_first_low_delta_control'];summ=[]
ck('rank_arrays_exact_zero_to_seven',[x['rank'] for x in s['selections']]==[x['rank'] for x in s['selection_categories']]==list(range(8)))
for selection,cats in zip(s['selections'],s['selection_categories']):
 r=selection['rank'];g=selection['raw_groups'];rows=cats['categories']
 assert [x['category'] for x in rows]==criteria
 assert [x['status'] for x in rows]==['SELECTED','SELECTED','SELECTED','ABSENT','SELECTED']
 assert rows[3]==dict(category=criteria[3],reason='No source row meets frozen criterion',status='ABSENT')
 assert len(g)==4 and sum(len(x['row_indices']) for x in g)==16
 assert len({(x['asset'],x['source_group']) for x in g})==4
 assert all(inv[x['asset']]['kind']=='raw' and inv[x['asset']]['rank']==r and x['row_indices']==inv[x['asset']]['original_micro4_groups'][x['source_group']] and len(x['row_indices'])==4 for x in g)
 assert selection['monitor']==dict(asset=f'rank-{r:02d}-monitor.pt',row_indices=list(range(32)))
 bygroup={(x['asset'],x['source_group']):x for x in cats['selected_group_criteria']}
 assert list(bygroup)==sorted(bygroup) and [{k:v for k,v in x.items() if k!='criteria'} for x in cats['selected_group_criteria']]==g
 expected_groupcriteria={}
 for row in rows:
  if row['status']=='ABSENT':continue
  key=(row['asset'],row['source_group']);expected_groupcriteria.setdefault(key,[]).append(row['category'])
  assert key in bygroup and row['row'] in bygroup[key]['row_indices'] and row['source_group']==row['row']//4
  assert inv[row['asset']]['rank']==r and row['effective_tokens']>0 and math.isfinite(row['max_abs']) and row['max_abs']>=0 and len(row['row_tensor_sha256'])==64 and all(x in '0123456789abcdef' for x in row['row_tensor_sha256'])
  assert row['above_dot01_fraction']==row['above_dot01']/row['effective_tokens'] and 0<=row['above_dot01']<=row['effective_tokens'] and row['full_attention_tokens']>=row['effective_tokens'] and row['position_anomaly'] is False
 assert expected_groupcriteria=={key:x['criteria'] for key,x in bygroup.items()}
 assert (rows[0]['asset'],rows[0]['source_group'])!=(rows[1]['asset'],rows[1]['source_group'])
 assert rows[4]['max_abs']<=.01 and rows[4]['above_dot01']==0
 summ.append(dict(rank=r,raw_groups=4,raw_rows=16,monitor_rows=32,tail_is_other_original_group=True,anomaly='ABSENT',low_delta_control_max_abs=rows[4]['max_abs']))
ck('all_rank_16_complete_micro4_and_categories_consistent',True,summ)
ck('anomaly_absent_supported_by_complete_context_metadata',c['raw_sampling_evidence']['raw_rows_recorded_positions_differ_from_zero_based_valid_sequence']==c['raw_sampling_evidence']['raw_prompt_rows_with_internal_or_right_attention_padding']==0)
ck('selected_max_across_ranks_agrees_saved_global_raw_max',max(x['categories'][0]['max_abs'] for x in s['selection_categories'])==c['raw_sampling_evidence']['raw_same_turn_actor_minus_sampled']['max_abs'])
ck('CPU3277_cost_identity_and_no_cuda',s['cost']['slurm_job_id']=='3277' and s['cost']['cuda_initialized'] is False and s['cost']['threads']==4)
result=dict(status='CPU3277_SELECTION_METADATA_INDEPENDENTLY_CONSISTENT',selection_sha256=EXPECTED,checks=checks,check_count=len(checks),inventory_count=72,raw_inventory_count=64,monitor_inventory_count=8,source_file_count=700,full_package_source_metadata_bound=699,raw_JSON_metadata_bound=64,raw_rows=11528,raw_effective_tokens=3186179,selected_raw_rows=128,selected_monitor_rows=256,ranks=summ,global_row_selection_ranking_recomputed=False,selected_row_full_tensor_SHA_recomputed=False,PT_assets_loaded=False,actual_model_forwards=0,actual_optimizer_writes=0,actual_GPU_or_remote_calls=0,scope='Saved selection JSON and existing frozen package/context/recovery metadata consistency only. No tensor assets or all-candidate row table; maxima/fraction/longest/hash tie global rankings not independently recomputed. Actual CPU3277 Slurm/companions reviewed separately by parent.')
(HERE/'CPU3277_CHILD_METADATA_PROOF.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','check_count','inventory_count','source_file_count','raw_rows','selected_raw_rows','selected_monitor_rows']}))
