import hashlib,importlib.util,json,math,sys
from pathlib import Path
B=Path.cwd();sys.path.insert(0,str(B/'work/bapo-fresh200-controller-initfix-20261006'))
spec=importlib.util.spec_from_file_location('h',B/'work/fresh200-allrank-history-3315-v2/history_validator.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
from search_r1.baselines.fresh200_contract import validate_rank_updates
records=[];bindings={};expect={}
for n,idx,healthidx in [(56,63,78),(57,64,82)]:
 paths=[B/'work/evidence'/p for p in ['fresh200-controller-initfix-formal3315-metrics%d-attempt.actual.json'%idx,'fresh200-formal3315-health%d.actual.json'%healthidx,'fresh200-formal3315-outer%d-full-metrics-from-health.actual.json'%n,'fresh200-formal3315-outer%d-update-replay.actual.json'%n]]
 x,health,full,parent=[h.strict_json(p.read_bytes()) for p in paths]
 for p in paths:bindings[str(p.relative_to(B))]=hashlib.sha256(p.read_bytes()).hexdigest()
 e=[e for e in x['latest']['FRESH200_ATTEMPT_JSON='] if e['attempt_id']==n][0];assert e['source_commit']==h.SOURCE and e['driver_pid']==760122 and e['accepted_complete_outer'] is True and e['status']=='COMPLETE'
 validate_rank_updates(e['rank_write_ledger'],n)
 ranks=[r for r in x['latest']['ACTOR_RANK_UPDATE_JSON='] if r['scheduler_last_epoch']==n];assert len(ranks)==8
 merged,gaps,cov=h.check_rank_metrics(ranks,e['reward_metrics'],n-1,True,e['rank_write_ledger']);assert not gaps
 rawmarked=[h.strict_json(line.split('BAPO_TRAIN_METRICS_JSON=',1)[1]) for line in health['train.log_tail'].splitlines() if 'BAPO_TRAIN_METRICS_JSON=' in line];actual=[m for m in rawmarked if m['step']==n];assert len(actual)==1 and actual[0]==full['record']
 h.check_rank_metrics(ranks,actual[0]['metrics'],n-1,True,e['rank_write_ledger'])
 for key in ['bapo/mask_binary_errors','bapo/mask_on_observation_tokens','bapo/mask_on_padding_tokens','bapo/model_tokens_missing_loss','bapo/new_logprob_nonfinite','bapo/old_logprob_nonfinite','critic/kl_coeff']:assert e['reward_metrics'][key]==0
 for ev in e['rank_write_ledger']:
  h.finite(ev);assert ev['optimizer_steps']==10 and ev['processed_opportunities']==10 and ev['skipped_opportunities']==0
  assert len(ev['policy_monitor'])==40 and all(v['allowed'] is True and v['overflow_count']==0 for v in ev['policy_monitor'])
 counts={k:sum(len(r['metrics'][k]) for r in ranks) for k in ['actor/pg_loss','actor/grad_norm']};assert counts=={'actor/pg_loss':80,'actor/grad_norm':80}
 assert e['progress']['actual_adam_writes_per_rank']==[1050 if n==56 else 1060]*8 and e['progress']['completed_outers']==n
 expect[str(n)]=hashlib.sha256(json.dumps(e,sort_keys=True,allow_nan=False).encode()).hexdigest()
 records.append(dict(outer=n,progress=e['progress'],lr_before=ranks[0]['lr_before'],lr_after=ranks[0]['lr_after'],scheduler_epoch=n,raw_counts=counts,pg_loss_mean=math.fsum(merged['actor/pg_loss'])/len(merged['actor/pg_loss']),grad_mean=math.fsum(merged['actor/grad_norm'])/len(merged['actor/grad_norm']),grad_extrema=[min(merged['actor/grad_norm']),max(merged['actor/grad_norm'])],numeric_coverage_gaps=gaps,allrank_monitor_records=320,driver_full_marker_matches_health=True,recorded_costs={k:v for k,v in actual[0]['metrics'].items() if k.startswith('timing_s/')}))
print(json.dumps(dict(status='PASS_LOCAL_ACTUAL_WARMUP56_57_ALLRANK_REPLAY',outers=records,ledger_semantic_expectations=expect,evidence_bindings=bindings,whole200_accepted=False),allow_nan=False))
