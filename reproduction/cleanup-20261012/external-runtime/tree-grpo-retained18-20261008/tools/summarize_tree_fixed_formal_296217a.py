"""Read-only extraction of actual complete Tree training/evaluation evidence."""
import argparse,hashlib,json,math,sys
from pathlib import Path
from validate_tree_stage import parse_console
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--hf-cpu-run',type=Path,required=True);a=p.parse_args()
assert not a.output.exists();run=a.run.resolve()
raw=(run/'stage.log').read_text();records=parse_console(run/'stage.log',500)
result=json.loads((run/'run-result.json').read_text());assert result['status']=='ACTUAL_STAGE_COMPLETED_UNREVIEWED' and result['training_body_exit_code']==0 and result['entry_config_verified']
assert result['cleanup_residual']=={}
assert result['code_commit']=='296217a4b676a26a92fc729f5d9882b785dbfa96'
hf_cpu=a.hf_cpu_run.resolve()
assert str(hf_cpu).startswith('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-fixed-formal-cpu-'+str(result['job_id'])+'-')
hf=json.loads((hf_cpu/'hf500-full-reload.json').read_text());cpu=json.loads((hf_cpu/'result.json').read_text())
assert cpu['status']=='HF500_FULL_RELOAD_COMPLETE_PENDING_REVIEW' and cpu['exit_code']==0 and str(cpu['parent_job_id'])==str(result['job_id'])
assert hf['status']=='HF_RELOAD_PARAMETER_DELTA_ACCEPTED' and hf['all_parameters_finite'] and hf['cpu_forward_logits_finite'] and hf['tokenizer_vocab_equal'] and hf['bos_eos_equal']
assert hf['changed_parameters']>0 and hf['changed_tensors']>0 and hf['max_abs_parameter_delta']>0
assert Path(hf['checkpoint']).resolve()==run/'output/checkpoints/tree_grpo-qwen2.5-3b/actor/global_step_500'
assert not any(hf['hf_loading_info'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
assert 'JobState=COMPLETED' in (hf_cpu/'slurm-cpu-terminal.txt').read_text() and 'ExitCode=0:0' in (hf_cpu/'slurm-cpu-terminal.txt').read_text()
assert hashlib.sha256((run/'resolved.yaml').read_bytes()).hexdigest()==hashlib.sha256((run/'output/entry-audit/entry-config.yaml').read_bytes()).hexdigest()

final=[]
for line in raw.splitlines():
 if 'SEARCH_R1_FINAL_METRICS_JSON=' in line:final.append(json.JSONDecoder().raw_decode(line.split('SEARCH_R1_FINAL_METRICS_JSON=',1)[1])[0])
assert len(final)==1;metrics=final[0];assert all(isinstance(v,(int,float)) and math.isfinite(v) for v in metrics.values())
sources=sorted(k[len('val/evaluated_count/'):] for k in metrics if k.startswith('val/evaluated_count/'));assert len(sources)==7 and 'all' not in sources
assert sum(metrics['val/evaluated_count/'+s] for s in sources)==metrics['val/qa_count/all']==51713
keys=['f1','em','answer_format','format_rate','generated_tokens','search_calls'];table=[]
for s in sources:
 count=metrics['val/evaluated_count/'+s];assert int(count)==count==metrics['val/qa_count/'+s] and count>0
 row=dict(source=s,questions=int(count),**{k:metrics[f'val/{k}/{s}'] for k in keys});table.append(row)
 for k in keys[:4]:assert 0<=row[k]<=1
for k in keys:
 weighted=sum(r['questions']*r[k] for r in table)/51713;assert math.isclose(weighted,metrics[f'val/{k}/all'],rel_tol=1e-9,abs_tol=1e-9)
reports=[json.loads(f.read_text()) for f in sorted((run/'output/validation-cost').glob('*.json'))]
assert len(reports)==12
assert all(x['status']=='VALIDATION_COST_COMPLETE' and x['dataset_rows']==x['evaluated_rows'] for x in reports)
pre=[x for x in reports if x['phase']=='pretrain'];periodic=[x for x in reports if x['phase']=='periodic'];end=[x for x in reports if x['phase']=='final']
assert len(pre)==len(end)==1 and pre[0]['evaluated_rows']==64 and end[0]['evaluated_rows']==51713
assert sorted(x['global_step'] for x in periodic)==list(range(50,501,50)) and all(x['evaluated_rows']==64 for x in periodic)
for x in reports:
 ledger=run/'output/validation-cost'/f"validation-{x['pid']}-{x['validation_sequence']}.jsonl";assert hashlib.sha256(ledger.read_bytes()).hexdigest()==x['ledger_sha256']
rank_files=sorted((run/'output/tree-rank-audit').glob('*.json'));assert len(rank_files)==4000
rank_steps={};rank_calls=0
for f in rank_files:
 x=json.loads(f.read_text());key=(x['outer_step'],x['rank']);assert key not in rank_steps and x['world_size']==8 and x['local_rows']==384 and x['local_minibatch']==32 and x['local_microbatch']==8 and x['epochs']==1
 assert x['code_commit']==result['code_commit'] and x['job_id']==result['job_id'] and x['successful_optimizer_calls']==12 and x['successful_optimizer_calls']==x['finite_grad_norms']
 assert all(math.isfinite(v) and v>0 for v in x['existing_grad_norms']);rank_steps[key]=x['successful_optimizer_calls'];rank_calls+=x['successful_optimizer_calls']
profiles=sorted((run/'output/tree-rank-audit/memory-profile').glob('step-*.json'));assert len(profiles)==4000
assert set(rank_steps)=={(s,r) for s in range(1,501) for r in range(8)}
fields=['rollout/generated_tokens','rollout/prefill_tokens','rollout/retrieval_calls','rollout/trajectories','rollout/padded_rows','rollout/context_mismatches']
totals={k:sum(x['metrics'][k] for x in records) for k in fields}
assert totals['rollout/trajectories']==500*3072
summary=dict(status='ACTUAL_FORMAL_EVIDENCE_EXTRACTED_NOT_INDEPENDENT_ACCEPTANCE',run=str(run),job_id=result['job_id'],code_commit=result['code_commit'],actual_outer_updates=500,qa_evaluated=51713,datasets=table,all={k:metrics[f'val/{k}/all'] for k in keys},training_ledger_totals=totals,direct_rank_records=4000,memory_profile_records=4000,hf_full_cpu_reload=hf,hf_cpu_result=cpu,total_successful_rank_optimizer_calls=rank_calls,validation_reports=reports,cost=json.loads((run/'cost.json').read_text()),limitations='Original nonpad generated-token convention can exclude EOS when PAD==EOS. Submitted prefill/reused-prefix events are not unique or cache-aware tokens. No fair GRPO comparison established by this extractor; check hardware, backend, data/split/config/budget identity independently. Fixed terminal-answer F1/EM and response legality; numerical source296217a. Independent Tree/profile/split/cleanup/cost acceptance still required.')
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps(dict(status=summary['status'],job_id=summary['job_id'],questions=51713,output=str(a.output))),flush=True)
