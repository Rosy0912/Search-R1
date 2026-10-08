import pathlib,json,hashlib,gzip,re,math
p=pathlib.Path('work/tree-recovery-review/restored512');b=json.loads((p/'ACTUAL_REPORT_LEDGER_SCORE_RAW.json').read_text());files={x['path']:x for x in b['files']};reportfile=next(x for x in b['files'] if x['path'].endswith('-1.json'));r=json.loads(reportfile['text']);ledgerfile=files[reportfile['path']+'l'];events=[json.loads(x) for x in ledgerfile['text'].splitlines()]
for f in b['files']:assert hashlib.sha256(f['text'].encode()).hexdigest()==f['sha256']
assert r['status']=='VALIDATION_COST_COMPLETE' and r['global_step']==100 and r['job_id']=='3234' and r['dataset_rows']==r['evaluated_rows']==512
assert r['pid']==358612 and r['validation_sequence']==1 and r['observer_source_sha256']=='ac8f2783830d12c5db2de3d19e9f2d19b594aa82d240849b4d24220ddd482598'
assert ledgerfile['sha256']==r['ledger_sha256'];assert [e['event'] for e in events]==list(range(1,len(events)+1))
assert all(e['pid']==358612 and e['global_step']==100 and e['validation_sequence']==1 and e['phase']=='periodic' for e in events)
assert events[0]['kind']=='validation_start' and events[0]['dataset_rows']==512 and events[-1]['kind']=='validation_completed' and events[-1]['evaluated_rows']==512
assert not any(e['kind'].endswith('_failed') for e in events)
from collections import Counter
kinds=Counter(e['kind'] for e in events)
def total(kind,key):return sum(e[key] for e in events if e['kind']==kind)
counters=dict(generation_attempts=kinds['generation_start'],generation_completed=kinds['generation_completed'],generation_failed=0,submitted_rows=total('generation_start','rows'),returned_rows=total('generation_completed','rows'),submitted_prefill_token_events=total('generation_start','prefill_token_events'),returned_nonpad_model_token_events=total('generation_completed','nonpad_model_token_events'),real_sample_rows=total('real_sample_batch','rows'),active_row_events=total('gpu_padding_submission','active_rows'),submitted_padding_row_events=total('gpu_padding_submission','padding_rows'),http_attempts=kinds['http_start'],http_completed=kinds['http_completed'],http_failed=0,http_query_events=total('http_start','queries'),http_returned_query_rows=total('http_completed','query_result_rows'))
assert counters==r['counters'],(counters,r['counters'])
for first,last in [('generation_start','generation_completed'),('http_start','http_completed')]:
 assert [e['call'] for e in events if e['kind']==first]==list(range(1,65))
 assert [e['call'] for e in events if e['kind']==last]==list(range(1,65))

line=b['step100_original_lines'][0];metrics={k:float(v) for k,v in re.findall(r'([\w/]+):([-+\deE.]+)',line)}
counts={ds:int(metrics['val/qa_count/'+ds]) for ds in ('nq','hotpotqa')};assert counts=={'nq':256,'hotpotqa':256}
assert all(metrics['val/evaluated_count/'+ds]==counts[ds] for ds in counts)
f1=sum(metrics['val/f1/'+ds]*n for ds,n in counts.items())/512;em=sum(metrics['val/em/'+ds]*n for ds,n in counts.items())/512
assert math.isclose(f1,metrics['val/f1/all'],rel_tol=0,abs_tol=1e-15) and metrics['val/f1/all']==.48918324738300495 and em==metrics['val/em/all']==203/512
assert metrics['val/format_rate/all']==0 and metrics['val/answer_format/all']==494/512
bundle=json.loads(gzip.decompress(pathlib.Path('outputs/tree-instruct-300/RECOVERY3234_RESTORED_EVAL_BUNDLE.json.gz').read_bytes()))
for path,item in bundle['files'].items():assert hashlib.sha256(item['text'].encode()).hexdigest()==item['sha256']
assert bundle['files'][reportfile['path']]['sha256']==reportfile['sha256']
proof=dict(status='PASS_RESTORED_FIXED512_QUALITY_ONLY',job_id='3234',code_commit='502e9fafa8e3aba722dbfcb493abc2b9d3ece326',config_sha256='e54d322ba3635beb8d930c4f8282997f1867b4dbf5a0ca727e54a569bb198f35',data_identity='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',model_identity='744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0',recovery_parent_completed_step=100,recovery_parent_manifest_sha256='c20922f74f7dc77e7a233fcdcf329f2914bfaff335ed34d02209e1e6d09a3e86',restored_fixed512_quality_accepted=True,actual_recovery105_accepted=False,actual_recovery300_accepted=False,observer_pid=358612,observer_sequence=1,observer_phase='periodic',global_step=100,evaluated_rows=512,dataset_counts=counts,dataset_f1={ds:metrics['val/f1/'+ds] for ds in counts},dataset_em={ds:metrics['val/em/'+ds] for ds in counts},f1_all=metrics['val/f1/all'],weighted_recomputed_f1=f1,weighted_roundoff_delta=f1-metrics['val/f1/all'],em_all=em,em_correct=203,quality_threshold=.34242827316810346,weighted_score_equal_parent100=True,answer_format_rate=metrics['val/answer_format/all'],strict_format_rate=0,report_sha256=reportfile['sha256'],ledger_sha256=ledgerfile['sha256'],ledger_events=len(events),counters=counters,wall_seconds=r['wall_seconds'],trainer_process_cpu_seconds=r['trainer_process_cpu_seconds'],proof_sha256=hashlib.sha256((p/'ACTUAL_REPORT_LEDGER_SCORE_RAW.json').read_bytes()).hexdigest(),limits='Recomputed actual source aggregate counts/weighted scores and complete ledger; no independent per-answer rescore, new optimizer acceptance, or bitwise executor equivalence. PAD151643 andEOS151645 distinct in this actual run; generic PAD==EOS note does not apply.')
(p/'RESTORED512_RECEIPT.json').write_text(json.dumps(proof,indent=2));print('RESTORED512 RECEIPT',hashlib.sha256((p/'RESTORED512_RECEIPT.json').read_bytes()).hexdigest())
