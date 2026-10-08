from pathlib import Path
import json,hashlib,torch,math
report=Path('work/evidence/fresh200-numeric-diag3311-numeric-probe.actual.json');d=json.loads(report.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sidecar=next(Path('work/evidence/fresh200-numeric-diag3311-terminal-raw').rglob('numeric-probe.selected-logits.pt'));assert sha(sidecar)==d['selected_logit_sidecar']['sha256']and sidecar.stat().st_size==d['selected_logit_sidecar']['bytes']
s=torch.load(sidecar,map_location='cpu',weights_only=True);assert s['source_commit']==d['source_commit']and s['raw_sha256']==d['raw_sha256']and s['numeric_probe_sha256']==d['numeric_probe_sha256']
key=lambda r:(r['precision'],r['mode'],r['row'],r['position']);records={key(r):r for r in d['results']};assert len(records)==len(s['records'])==36
assert len({key(r)for r in s['records']})==36 and {key(r)for r in s['records']}==set(records)
assert s['schema_version']==1 and s['vocab_size']==151936
labels={(p['row'],p['position']):p['target_token']for p in d['points']}
replays=[]
for r in s['records']:
 k=key(r);q=records[k];assert r['label']==labels[(r['row'],r['position'])];assert r['original_dtype']==('torch.bfloat16'if r['precision']=='BF16_ORIGINAL'else'torch.float32');v=r['original_logits_float32'];assert v.shape==(151936,)and v.dtype==torch.float32 and bool(torch.isfinite(v).all());assert hashlib.sha256(v.numpy().tobytes()).hexdigest()==q['original_logits']['float32_bytes_sha256'];label=r['label'];lp=float(torch.log_softmax(v.double(),0)[label]);assert abs(lp-q['original_logits']['cpu_fp64_logprob'])<1e-10
 h=torch.tensor(q['hidden_float32_values'],dtype=torch.float32);assert hashlib.sha256(h.numpy().tobytes()).hexdigest()==q['hidden_float32_sha256']
 replays.append(dict(precision=r['precision'],mode=r['mode'],row=r['row'],position=r['position'],label=label,cpu_f64_replay_lp=lp,producer_f64_lp=q['original_logits']['cpu_fp64_logprob'],original_lp=q['actor_or_eager_lp'],original_lp_minus_cpu_f64=q['actor_or_eager_lp']-lp,selected_logit=float(v[label]),max_logit=float(v.max()),logit_range=float(v.max()-v.min()),fp32_head_lp=q['selected_hidden_fp32_head']['cpu_fp64_logprob'],bf16_singlequery_lp=None if q['same_hidden_bf16_single_query_head']is None else q['same_hidden_bf16_single_query_head']['cpu_fp64_logprob']))
for point in d['points']:
 row=point['row'];print('POINT',row,point['position'],'native',point['native'])
 for precision in ('BF16_ORIGINAL','FP32_EAGER_CONFOUNDED'):
  qs=[q for q in d['results']if q['row']==row and q['precision']==precision];base=next(q for q in qs if q['mode']=='FULL');hb=torch.tensor(base['hidden_float32_values'],dtype=torch.float64)
  for q in qs:
   h=torch.tensor(q['hidden_float32_values'],dtype=torch.float64)
   print(precision,q['mode'],'LP',round(q['actor_or_eager_lp'],9),'CEerr',round(q['actor_or_eager_lp']-q['original_logits']['cpu_fp64_logprob'],9),'FP32head',round(q['selected_hidden_fp32_head']['cpu_fp64_logprob'],9),'BF16single',None if q['same_hidden_bf16_single_query_head']is None else round(q['same_hidden_bf16_single_query_head']['cpu_fp64_logprob'],9),'hiddenmaxdelta',round(float((h-hb).abs().max()),9),'logitmax',q['original_logits']['maximum'])
summary={'scope':'ACTUAL_IMMUTABLE_3311_CPU_FP64_REPLAY_NOT_NATIVE_ACCEPTANCE','source_commit':d['source_commit'],'report_sha256':sha(report),'sidecar_sha256':sha(sidecar),'sidecar_bytes':sidecar.stat().st_size,'count':36,'weights_only_load':True,'independently_replayed_all_original_logit_hashes_and_fp64LP':True,'all_hidden_float32_hashes_verified':True,'maximum_same_logit_CE_abs_difference':max(abs(r['original_lp_minus_cpu_f64'])for r in replays),'native_acceptance':False,'production_release':False,'records':replays}
Path('outputs/fresh200/3311-INDEPENDENT-CPU-FP64-LOGIT-REPLAY.json').write_text(json.dumps(summary,indent=2)+'\n')
