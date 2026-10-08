import json,pathlib,base64,hashlib,importlib.util,collections,math
b=pathlib.Path('outputs/evidence');d=json.load(open(b/'native-x-validation150-20261007-0506-stream.json'));assert d['exists']
p=b/'native-x-validation150-predictions.jsonl';raw=base64.b64decode(d['data_base64']);assert len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['sha256'];assert raw.endswith(b'\n');p.write_bytes(raw)
root=pathlib.Path('work/cosearch-author-review/verl/verl/utils/reward_score')
mods={}
for n in ['search_r1_like_qa_em.py','search_qa_f1_with_format_penalty.py']:
 assert hashlib.sha256((root/n).read_bytes()).hexdigest()==d['source_sha256'][n]
 spec=importlib.util.spec_from_file_location(n,root/n);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);mods[n]=m
em=mods['search_r1_like_qa_em.py'];f1=mods['search_qa_f1_with_format_penalty.py'];rows=[json.loads(v) for v in raw.splitlines()];assert len(rows)==64 and all(v['step']==150 for v in rows)
scored=[]
for i,v in enumerate(rows):
 target=v['gts']['target'];ans=em.extract_solution(v['output']);score=em.em_check(ans,target) if ans is not None else 0;ff=f1.compute_f1(ans,target) if ans is not None else 0
 assert math.isfinite(v['f1']) and abs(ff-v['f1'])<1e-7
 scored.append(dict(row=i,em=score,normalized_prediction=em.normalize_answer(ans) if ans is not None else None,normalized_targets=[em.normalize_answer(t) for t in target],f1=ff))
old=[json.loads(v) for v in (b/'native-x-validation50-predictions.jsonl').read_text().splitlines()]
def binding(v):return json.dumps([v['input'],v['gts']],sort_keys=True,ensure_ascii=False)
match=collections.Counter(map(binding,old))==collections.Counter(map(binding,rows))
result=dict(run=d['run'],job=3211,checkpoint_step=150,questions=64,correct=sum(v['em'] for v in scored),em_percent=100*sum(v['em'] for v in scored)/64,mean_f1=sum(v['f1'] for v in scored)/64,valid_count=sum(bool(v['valid']) for v in rows),predictions_sha256=d['sha256'],author_em_source_sha256=d['source_sha256']['search_r1_like_qa_em.py'],author_f1_source_sha256=d['source_sha256']['search_qa_f1_with_format_penalty.py'],score_method='author extract_solution last answer tag + normalize_answer + exact equality against any target alias; missing answer scores0; not substring EM',f1_recomputed_matches_all64=True,no_new_generation=True,no_GPU_updates=True,remote_local_prediction_sha_equal=True,remote_local_author_sources_sha_equal=True,same_question_target_multiset_as_checkpoint50=match,same_question_target_order_as_checkpoint50=list(map(binding,old))==list(map(binding,rows)),verified=True,rows=scored,notification_sent=False)
(b/'native-x-validation150-em-offline.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
