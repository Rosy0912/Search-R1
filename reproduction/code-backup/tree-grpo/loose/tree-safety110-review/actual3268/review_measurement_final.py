import json,hashlib,pathlib,math
D=pathlib.Path(__file__).resolve().parent;W=D.parents[2];O=W/'outputs/tree-instruct-300'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
poll=json.loads((O/'PPO_CONTEXT_CPU_JOB3271_PROGRESS2_RAW.json').read_bytes());reports={}
for name in ['raw-stats-3271.json','tree-trajectory-3271.json','context-analysis-3271.json']:
 raw=(O/'cpu3271'/name).read_bytes();v=poll['reports'][name];assert sha(O/'cpu3271'/name)==v['sha256'] and len(raw)==v['bytes'] and raw==v['text'].encode();reports[name]=json.loads(raw)
s=poll['commands']['scontrol show job 3271']['stdout'];assert 'JobId=3271 ' in s and 'JobState=COMPLETED' in s and 'ExitCode=0:0' in s and 'RunTime=00:01:01' in s and 'NumCPUs=4' in s and 'TresPerNode=gres:gpu:0' in s
r,t,c=[reports[n] for n in reports];ready='ba948bdec0b71a3c8ff5c61db1360840134e57fccaf71fb154449f30c300c468'
assert r['ready_sha256']==c['ready_sha256']==ready
assert c['helper_source_sha256']=='28e665eb822651ff44559b2e9c6110d01542c5e6433db8dd4d07c659af7131ac'
assert c['identity']['job_id']=='3268' and c['identity']['code_commit']=='a632ba91840fb116aacfc0e770681a860647b295'
assert c['package_manifest_sha256']=='84e3e18682afa880f1053a0025eda595fca332d550ec1f6905bc1334049b5a48'
assert c['parent_manifest_sha256']=='e59d8011df3b1248c0c548afeb9d4ad6e564c5f9b0f84f2147bb3f99ad182366'
raw=c['raw_sampling_evidence'];assert raw['raw_rows']==sum(a['physical_rows'] for a in r['calls'])==11528
assert raw['raw_attention_valid_response_token_occurrences']==sum(a['valid_response_token_occurrences'] for a in r['calls'])==3186179
assert raw['unique_exact_context_token_keys']+raw['duplicate_raw_occurrences']==3186179
assert raw['raw_rows_recorded_positions_equal_zero_based_valid_sequence']==11528 and raw['raw_rows_recorded_positions_differ_from_zero_based_valid_sequence']==raw['raw_prompt_rows_with_internal_or_right_attention_padding']==0
by_path={e['path']:e for e in c['evidence'] if e['kind']=='raw_same_turn'}
assert len(by_path)==len(r['evidence'])==64
for e in r['evidence']:
 for k in ['sha256','snapshot_path','snapshot_sha256']:assert e[k]==by_path[e['path']][k]
for a,b in zip(r['calls'],c['captured_raw_generation_call_modes']):
 assert a['call']==b['generation_call'] and b['ranks']==list(range(8))
 z=b['global_diagnostic_decision'];assert not z['allowed'] and z['invalid_tokens']==0
 assert a['valid_response_token_occurrences']==z['valid_tokens'] and a['max_abs_delta']==z['max_abs_delta']
for k in ['complete_actor_inputs','monitor_subset']:
 q=c[k];assert q['exact_context_matched_token_occurrences']+q['exact_context_unmatched_token_occurrences']==q['effective_supervised_token_occurrences']
 assert q['unique_final_context_token_keys']+q['duplicated_final_context_token_occurrences']==q['effective_supervised_token_occurrences']
 assert q['token_exact_context_coverage']==q['exact_context_matched_token_occurrences']/q['effective_supervised_token_occurrences']
 w=q['tree_weights'];assert w['invalid_effective_weights']==0 and w['weighted_exact_context_coverage']==w['matched_effective_weight_mass']/w['all_effective_weight_mass']
 assert all(v==0 for key,v in q['finite_numeric_evidence'].items() if key.startswith('invalid_'))
assert c['actual_forward_passes']==c['actual_optimizer_writes']==0 and c['analysis_cost']['analysis_slurm_job_id']=='3271' and c['analysis_cost']['cuda_initialized'] is False
assert t['records'][0]['trajectory_rows']==3840 and t['records'][0]['all_actor_effective_inputs_finite'] and t['records'][0]['actual_optimizer_writes']==0
final=json.loads((D/'FINAL_METADATA.json').read_bytes())
for v in final['files'].values():assert hashlib.sha256(v['text'].encode()).hexdigest()==v['sha256']
result=json.loads(final['files']['run-result.json']['text']);cost=json.loads(final['files']['cost.json']['text']);assert result['training_body_exit_code']==0 and result['entry_config_verified'] is True and result['cleanup_residual']=={} and result['final_guard_healthy'] is True and result['optimizer_committed']==result['new_completed_outer']==0
samples=[json.loads(l) for l in final['files']['resource-samples.jsonl']['text'].splitlines()];assert len(samples)==230
for v in samples:
 assert not v.get('event') and v['job_id']=='3268' and len(v['gpus'])==8
 for g in v['gpus']:assert g['foreign_memory_fraction']<=.15 and g['foreign_sm_upper']<=15
for pid,stat in final['protected'].items():assert stat[stat.rfind(')')+2:].split()[19]==('1502352181' if int(pid)<1690000 else '1502398762')
terminal=json.loads((D/'TERMINAL_METADATA_PROOF.json').read_bytes());assert terminal['slurm_exit']=='0:0' and terminal['slurm_wall_seconds']==1181
sc=json.loads((D/'POLL8.json').read_bytes())['slurm']['stdout'];assert 'JobState=COMPLETED' in sc and 'ExitCode=0:0' in sc
proof={'status':'ACTUAL_ZERO_WRITE_MEASUREMENT_REPORTS_AND_TERMINAL_RECOMPUTED','context_report_sha256':sha(O/'cpu3271/context-analysis-3271.json'),'raw_report_sha256':sha(O/'cpu3271/raw-stats-3271.json'),'trajectory_report_sha256':sha(O/'cpu3271/tree-trajectory-3271.json'),'cpu_terminal':{'job':3271,'state':'COMPLETED','exit':'0:0','seconds':61,'cpus':4,'gpus':0},'context_coverage':{k:{j:c[k][j] for j in ['leaf_rows_denominator','effective_supervised_token_occurrences','exact_context_matched_token_occurrences','exact_context_unmatched_token_occurrences','token_exact_context_coverage','tree_weights']} for k in ['complete_actor_inputs','monitor_subset']},'raw_tokens':3186179,'raw_physical_rows':11528,'above_dot01':345919,'above_fraction':345919/3186179,'max_abs_delta':r['all_max_abs'],'scope':'No repeated PT analysis; independently hash-bound actual CPU report aggregation and actual GPU metadata checked'}
(D/'FINAL_RECOMPUTED_PROOF.json').write_text(json.dumps(proof,indent=2)+'\n')
receipt={'status':'ACTUAL_ZERO_WRITE_CALIBRATION_MEASUREMENT_ACCEPTED_PROBABILITY_CONSISTENCY_REFUSED','stage':'healthy110_zero_write_calibration','job_id':'3268','code_commit':c['identity']['code_commit'],'package_manifest_sha256':c['package_manifest_sha256'],'config_sha256':c['identity']['config_sha256'],'ready_sha256':ready,'parent_completed_outer':110,'parent_manifest_sha256':c['parent_manifest_sha256'],'observed_outer_label':111,'actual_new_completed_outer_updates':0,'actual_3b_optimizer_writes':0,'measurement_complete':True,'probability_consistency_accepted':False,'probability_consistency_status':'REFUSED_TRIAL_DOT01','thresholds_calibrated':False,'actual_3b_training_accepted':False,'actual_five_accepted':False,'long_training_ready':False,'final300_accepted':False,'causal_root_cause_established':False,'raw_statistics':{'calls':8,'rank_snapshots':64,'physical_rows':11528,'valid_token_occurrences':3186179,'max_abs_delta':r['all_max_abs'],'above_dot01':345919},'trajectory_rows':3840,'tree_structure_and_actor_input_invariants_verified':True,'raw_context_coverage':proof['context_coverage'],'final_monitor_global_max_abs_delta':terminal['final_monitor_global_decision']['global_monitor_max_abs_delta'],'guard_sample_count':230,'background_eight_pid_starttimes_preserved':True,'cleanup_residual':{},'gpu_terminal':{'state':'COMPLETED','exit':'0:0','seconds':1181,'gpu_hours':1181*8/3600,'allocated_cpu_hours':1181*192/3600,'sampled_gpu_peak_gib':terminal['sampled_gpu_peak_GiB']},'cpu_analysis_cost':c['analysis_cost'],'cpu_terminal':proof['cpu_terminal'],'restored_fixed512':json.loads((D/'RESTORED512_RECEIPT.json').read_bytes()),'native_scope':'8-rank synthetic FSDP fullstate120→130 plus24CUDAKL; native toy writes distinct from actual3B0writes','limitations':['Full-context matching covers54.5233% actual leaf tokens; unmatched contexts unknown, duplicate raw occurrences kept as min/max rather than selected sampled policy.','Recorded positions zero-based for all11528 rawrows; no internal/right prompt padding observed; actualengineKV/position execution not exported.','Historical current monitor logprobs not saved; CPU readback does not rerun actual3B CUDA forward.','Monitor256rows is subset of3840leaves and cannot be added to fullleaf totals.','Numerical cause, tolerance changes, calibrated thresholds,3B updates/five/long/final300 not accepted.','Launch manifest initial FAILED sentinel is frozen pre-body snapshot per supervisor line97/102, not terminal result.','3270 failed0sec due reviewer SyntaxError retained separately; corrected3271 completed0:0.'],'evidence':[{'path':str(p.resolve()),'sha256':sha(p)} for p in [D/'FINAL_RECOMPUTED_PROOF.json',O/'PPO_CONTEXT_CPU_JOB3271_PROGRESS2_RAW.json',O/'cpu3271/raw-stats-3271.json',O/'cpu3271/tree-trajectory-3271.json',O/'cpu3271/context-analysis-3271.json',D/'FINAL_METADATA.json',D/'POLL8.json',D/'STARTUP_METADATA_PROOF.json',D/'NATIVE_FULL_REPLAY_CPU_PROOF.json',D/'RESTORED512_RECEIPT.json']]}
(D/'ACTUAL3268_MEASUREMENT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(sha(D/'ACTUAL3268_MEASUREMENT_RECEIPT.json'))
