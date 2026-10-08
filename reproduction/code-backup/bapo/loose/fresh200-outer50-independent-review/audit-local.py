import json,sys,hashlib,math,importlib.util,time
from pathlib import Path
BASE=Path.cwd();REPO=BASE/'work/bapo-fresh200-controller-initfix-20261006';sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location('history',BASE/'work/fresh200-allrank-history-3315-v2/history_validator.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
paths=[BASE/'work/evidence'/p for p in ['fresh200-controller-initfix-formal3315-metrics50-attempt.actual.json','fresh200-controller-initfix-formal3315-metrics49-rank.actual.json','fresh200-formal3315-outer50-actor-records.actual.json']]
a,r,parent=[h.strict_json(p.read_bytes()) for p in paths];e=a['latest']['FRESH200_ATTEMPT_JSON='][-1]
ranks=[x for x in r['latest']['ACTOR_RANK_UPDATE_JSON='] if x['scheduler_last_epoch']==50]
assert len(ranks)==8
from search_r1.baselines.fresh200_contract import validate_rank_updates
assert validate_rank_updates(e['rank_write_ledger'],50)
merged,gaps,cov=h.check_rank_metrics(ranks,e['reward_metrics'],49,True,e['rank_write_ledger']);assert not gaps
for k in ['bapo/mask_binary_errors','bapo/mask_on_observation_tokens','bapo/model_tokens_missing_loss','bapo/mask_on_padding_tokens','bapo/new_logprob_nonfinite','bapo/old_logprob_nonfinite','critic/kl_coeff']:assert e['reward_metrics'][k]==0
counts={k:sum(len(rank['metrics'][k]) for rank in ranks) for k in ['actor/pg_loss','actor/grad_norm']};assert counts=={'actor/pg_loss':80,'actor/grad_norm':80}
scopes=[]
for ev in e['rank_write_ledger']:
 h.finite(ev)
 reports=ev['policy_monitor'];assert len(reports)==40
 assert all(p['allowed'] is True and p['overflow_count']==0 for p in reports)
 assert all(p['monitored_local_rows']==(32 if p['monitoring_scope']=='full_current_minibatch' else 320) for p in reports)
 scopes.append({'rank':ev['rank'],'monitor_count':len(reports),'mini_rows':32,'full_rollout_rows':320})
print(json.dumps(dict(status='PASS_LOCAL_OUTER50_ALL_RANK_NUMERICAL_AGGREGATE_MASK_REPLAY',raw_counts=counts,numerical_coverage_gaps=gaps,means={k:math.fsum(merged[k])/len(merged[k]) for k in ['actor/pg_loss','actor/grad_norm','actor/entropy_loss','actor/pg_clipfrac','actor/ppo_kl']},grad_extrema=[min(merged['actor/grad_norm']),max(merged['actor/grad_norm'])],lr_before=ranks[0]['lr_before'],lr_after=ranks[0]['lr_after'],monitors=scopes,evidence_bindings={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},whole200_accepted=False),allow_nan=False))
