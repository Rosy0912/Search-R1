"""Local original-byte curve/config audit, not checkpoint or terminal acceptance."""
from pathlib import Path
from types import ModuleType
from datetime import datetime, timezone
import hashlib, json, gzip, base64, math, os, stat, statistics

B = Path(__file__).resolve().parent
P = B / 'curve-originals-20261006T081658003420Z'
S = B / 'actual-snapshots/3261-20261006T040534209402Z'
def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(path, pin=None):
    assert path.resolve(strict=True) == path and path.is_file()
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno()); assert stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid()
        raw=stream.read(); after=os.fstat(stream.fileno())
    stamp=lambda s: (s.st_dev,s.st_ino,s.st_uid,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    assert stamp(before)==stamp(after)==stamp(path.stat()) and len(raw)==before.st_size
    assert pin is None or sha(raw)==pin
    return raw
def loads(raw):
    def pairs(items):
        out={}
        for k,v in items: assert k not in out; out[k]=v
        return out
    out=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    json.dumps(out,allow_nan=False);return out

summary_raw=read(P/'summary.json'); summary=loads(summary_raw)
wire_raw=read(P/'transport.stdout.original',summary['transport_sha256']); wire=loads(wire_raw)
assert summary['transport_sha256']=='b4d790b6b62e2e833c0bce4dc4bdd5c33c035fc5fb2b07ccb0388d56cb1c1807'
assert summary['actual_transport_returncode']==0 and summary['node']==wire['node']=='h100-10'
assert set(summary['files'])=={'driver.out','train.log','resolved-config.json','driver200.json'}
raws={}; original_pins={}
for name,rec in summary['files'].items():
    raw=read(P/name,rec['sha256']); assert len(raw)==rec['bytes']
    wrec=wire['files'][name];packed=base64.b64decode(wrec['gzip_base64'],validate=True)
    assert sha(packed)==rec['gzip_sha256']==wrec['gzip_sha256'] and gzip.decompress(packed)==raw
    assert {k:v for k,v in wrec.items() if k!='gzip_base64'}==rec
    raws[name]=raw; original_pins[name]=dict(path=rec['path'],sha256=rec['sha256'],bytes=rec['bytes'])
caller_path=B/'review_progress_typed.py'; caller_raw=read(caller_path,'d9c1e303b32e74c747bf9158aefdbc5bb173d22df49eae14ab44d12a69af2eca')
m=ModuleType('independent_curve_context_only');m.__file__=str(caller_path)
exec(compile(caller_raw,str(caller_path),'exec',dont_inherit=True),m.__dict__)
g,c,tm,tmeta=m.runtime_helpers()
ss=loads(read(S/'observation-summary.json'))
ctx,_=m.source_context(S,ss,g,c,tm,tmeta)
assert raws['resolved-config.json']==m.captured_member(S,ss,'full_state/global_step_5/resolved-config.json')[0]
config=loads(raws['resolved-config.json']); assert sha(raws['resolved-config.json'])==m.CONFIG
streams={role:c.parse_records(raws[name],ctx) for role,name in [('driver','driver.out'),('aggregate','train.log')]}
joined=c.join_log_prefixes(streams['driver'],streams['aggregate'],ctx,200)
assert joined['authoritative_last_outer']==joined['aggregate_last_outer']==200
coverage=c.check_evaluation_sequence(streams['driver'],200,True)
coverage_aggregate=c.check_evaluation_sequence(streams['aggregate'],200,True)
omit_line=lambda es:[{k:v for k,v in e.items() if k!='console_line'} for e in es]
assert omit_line(coverage['events'])==omit_line(coverage_aggregate['events'])
expected=[('dev',0,512),('stage',5,64)]
for step in [50,100,150,200]:expected += [('dev',step,512),('stage',step,64)]
expected += [('final',200,51713)]
assert [(e['kind'],e['step'],e['count']) for e in coverage['events']]==expected
metrics_keys=['actor/grad_norm','actor/lr','actor/opsd_loss','critic/rewards/mean','env/finish_ratio','env/ratio_of_valid_action','timing_s/step']
curves=[]
for step,row in enumerate(streams['driver']['outers'],1):
    metrics=row['metrics']; assert row['global_step']==step and row['source_commit']==m.SOURCE and row['config_sha256']==m.CONFIG
    for rank in range(8):
        assert metrics['actor/optimizer_steps/rank_'+str(rank)]==16
        assert metrics['actor/samples/rank_'+str(rank)]==64
        assert metrics['actor/opsd_loss_finite/rank_'+str(rank)]==metrics['actor/grad_norm_finite/rank_'+str(rank)]==1
        assert metrics['teacher/cuda_update_entries_cumulative_per_rank/rank_'+str(rank)]==step
        for k in ('teacher/trainable_parameters_per_rank','teacher/gradient_tensors_per_rank','teacher/student_optimizer_membership_per_rank'):
            assert metrics[k+'/rank_'+str(rank)]==0
    assert metrics['actor/lr']==5e-6 and all(type(metrics[k]) in (float,int) and math.isfinite(metrics[k]) for k in metrics_keys)
    curves.append(dict(iteration=step,**{k:metrics[k] for k in metrics_keys}))
root_raw=read(P/'curve-data-root.json'); root=loads(root_raw)
assert root['outers']==curves and root['driver_original_sha256']==original_pins['driver.out']['sha256'] and root['train_original_sha256']==original_pins['train.log']['sha256']
reward=[r['critic/rewards/mean'] for r in curves]
dev=[];stage=[]
for e in coverage['events']:
    values={k:e['original_metrics']['val/'+k+'/all'] for k in ('em','f1','format_rate','answer_format','search_calls','generated_tokens')}
    item=dict(step=e['step'],count=e['count'],**values)
    if e['kind']=='dev':dev.append(item)
    elif e['kind']=='stage':stage.append(item)
    else:final=item
actor=config['actor_rollout_ref']['actor'];rollout=config['actor_rollout_ref']['rollout'];dist=config['baselines']['distillation']
actual_config=dict(actor_path=config['actor_rollout_ref']['model']['path'],teacher_path=dist['teacher_path'],learning_rate=actor['optim']['lr'],warmup=actor['optim']['lr_warmup_steps'],gradient_clip=actor['grad_clip'],mini_batch_global=actor['ppo_mini_batch_size'],micro_batch_global=actor['ppo_micro_batch_size'],epochs=actor['ppo_epochs'],questions_per_outer=config['data']['train_batch_size'],responses_per_question=rollout['n_agent'],temperature=rollout['temperature'],top_p=rollout['top_p'],top_k=rollout['top_k'],distillation_beta=dist['beta'],distillation_temperature=dist['temperature'],component_clip=dist['component_clip'],teacher_prompt_mode=dist['teacher_prompt_mode'],teacher_device=dist['teacher_device'],privileged_answers=dist['privileged_answers'],reference_kl_enabled=actor['use_kl_loss'],reference_kl_coefficient=actor['kl_loss_coef'],entropy_coefficient=actor['entropy_coeff'],max_turns=config['max_turns'],tool=config['retriever'],total_outer_budget=config['trainer']['total_training_steps'])
report=dict(version=1,status='PASS_INDEPENDENT_ACTUAL200_CURVE_AND_FIXED_SET_AGGREGATE_AUDIT_ONLY',reviewer='author200_review',utc=datetime.now(timezone.utc).isoformat(),scope='Actual full copied logs and resolved configuration only; not full final/checkpoint200/terminal/resource acceptance.',originals=original_pins,transport_sha256=sha(wire_raw),summary_sha256=sha(summary_raw),root_curve_data_sha256=sha(root_raw),exact_root_outer_curve_match=True,source_commit=m.SOURCE,config_sha256=m.CONFIG,training_job_id='3261',actual_contiguous_outers=200,each_rank_actual_adam_calls=3200,all8_each_outer16_calls_finite=True,teacher_all200_each8rank_frozen_optimizer_exclusion_gradient_zero=True,log_join=joined,fixed_evaluation_coverage=coverage,dev_fixed512=dev,stage_fixed64=stage,final51713=final,first_rewards=reward[:5],mean_reward_first2=statistics.mean(reward[:2]),mean_reward_3to200=statistics.mean(reward[2:]),mean_reward_last10=statistics.mean(reward[-10:]),last_outer=curves[-1],actual_configuration=actual_config,negative_effect=dict(first_training_suspicion_step=3,earliest_post_update_fixed64_observation_step=5,first_same512_post_update_observation_step=50,dev200_minus_dev0_em=dev[-1]['em']-dev[0]['em'],dev200_minus_dev0_f1=dev[-1]['f1']-dev[0]['f1'],limitations=['Training questions change per outer; early reward decline is a warning, not a paired causal comparison.','First same512 post-update evaluation is50; exact within1..50 fixed-set onset is unobserved.','No matched official-base final51713 run is supplied by this packet.','actor/opsd_loss is clipped vocabulary-component objective, can be negative and is not trueKL.']),limits=['No SSH, new CPU/GPU job, deployment, signature, training control or source edit performed.','Known aggregate framing is preserved by exact new90 parser; original frozenformal error is not ignored or upgraded to success.','Driver200 JSON is saved metadata only; full checkpoint200/native/HF whole load not accepted here.','OriginalSlurm/wrapper/child terminal, source big assets and resource release are separate missing or pending gates.'])
out=B/'ACTUAL200_CURVE_AND_FIXED_EFFECT_INDEPENDENT_REVIEW.json'
with out.open('x') as stream:json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
print(json.dumps(dict(report=str(out),report_sha256=sha(read(out)),first_rewards=reward[:3],mean_reward_3to200=report['mean_reward_3to200'],dev_fixed512=dev,stage_fixed64=stage,final51713=final,last_outer=curves[-1]),ensure_ascii=False))
