from pathlib import Path
from collections import Counter
import json,hashlib,datetime
E=Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-analysis-output/opsd-ablation-execution-20261008/em-loss-relation-20261009')
A=E/'extraction-attempt-0001'
O=E/'formal512-existing-evidence-20261009-v2'
def sha(b):return hashlib.sha256(b).hexdigest()
def ref(p,b=None):
 if b is None:b=p.read_bytes()
 return {'path':str(p),'bytes':len(b),'sha256':sha(b)}
bound=json.loads((A/'stats-inputs.private.json').read_bytes())
expected={'cohort_features':'9e3307b671463ba958dffbefd53cbb58814028980b68d1c002c82eb6244c044e','extraction_summary':'7b5defbcabf598c64668f54b8771f2327adb7ff6c940f53954b6162e24512311','root_execution':'fe64203ca1bb1418968c6b453f544af1ee2a46a1c9a7a73645aafcf7562f39da'}
proofs=[]
for k in ['extraction_summary','root_execution']:
 r=bound[k];assert r['sha256']==expected[k];p=Path(r['path']);b=p.read_bytes();assert len(b)==r['bytes'] and sha(b)==r['sha256'];proofs.append(ref(p,b))
 if k=='extraction_summary':summary=json.loads(b)
 else:execution=json.loads(b)
assert summary['accepted'] is True and summary['counts']=={'pairs':4529,'valid_tokens':940295,'cohort_rows':7680,'outers':15,'optimizer_updates':151}
assert execution['returncode']==0
r=bound['cohort_features'];assert r['sha256']==expected['cohort_features'];p=Path(r['path']);h=hashlib.sha256();n=0;size=0;maps={};source_shas=set();source_models=set();available=Counter();refs={}
with p.open('rb') as f:
 for b in f:
  assert b.endswith(b'\n');h.update(b);size+=len(b);x=json.loads(b);n+=1
  assert x['record_kind']=='cohort_feature' and x['schema']=='opsd-first5-cohort-feature-v1'
  assert x['group'] in ['C0','C1','M'] and 0<=x['outer_index']<5
  key=(x['group'],x['outer_index']);m=maps.setdefault(key,{})
  assert x['question_id'] not in m
  m[x['question_id']]=(x['question_semantic_row_sha256'],x['data_source'])
  assert len(x['raw_rewards'])==5 and all(type(v) is int and v in [0,1] for v in x['raw_rewards'])
  assert x['siblings_success_count']==sum(x['raw_rewards']) and x['sampling_success_fraction']==sum(x['raw_rewards'])/5 and x['sampling_pass_at5']==int(sum(x['raw_rewards'])>0)
  if not x['loss_available']:assert x['loss_missing_reason'] is not None
  source_shas.add(x['train_source_sha256']);source_models.add(x['source_model_identity']);available[(x['group'],x['outer_index'],str(x['loss_available']),str(x['loss_missing_reason']))]+=1
  if key not in refs:refs[key]=x['source_refs']['questions']
  else:assert refs[key]==x['source_refs']['questions']
assert n==7680 and h.hexdigest()==r['sha256'] and size==r['bytes'];proofs.append(dict(path=str(p),bytes=size,sha256=h.hexdigest()))
assert len(maps)==15 and all(len(m)==512 for m in maps.values())
assert len(source_shas)==1 and len(source_models)==1
for o in range(5):assert maps['C0',o]==maps['C1',o]==maps['M',o]
full={}
for g in ['C0','C1','M']:
 full[g]={}
 for o in range(5):
  assert not (set(full[g]) & set(maps[g,o]));full[g].update(maps[g,o])
 assert len(full[g])==2560
vf=O/'validation-source-question-identities.private.jsonl';vb=vf.read_bytes();assert sha(vb)=='beeafb7aee715e7a03f7b90fc6c9950951a405e987dfcc8784701df3cbdc7598'
vrows=[json.loads(l) for l in vb.splitlines() if l.strip()];validation={x['question_id']:(x['row_semantic_sha256'],x['data_source']) for x in vrows}
assert len(vrows)==len(validation)==512;proofs.append(ref(vf,vb))
formal_summary=json.loads((O/'summary.private.json').read_bytes());assert formal_summary['accepted'] is True and formal_summary['equal_question_row_dataset_identity_across_all_4_models'] is True
vals=set(validation);vsem={x[0] for x in validation.values()};results=[]
for g in ['C0','C1','M']:
 for o in range(5):
  m=maps[g,o];overlap=sorted(vals & set(m))
  results.append({'group':g,'outer_index':o,'TRAIN_question_ids':len(m),'formal_validation_question_ids':512,'question_id_overlap':len(overlap),'same_qid_semantic_and_dataset_matches':sum(m[q]==validation[q] for q in overlap),'same_qid_semantic_or_dataset_mismatches':sum(m[q]!=validation[q] for q in overlap),'full_four_field_semantic_SHA_overlap':len(vsem & {x[0] for x in m.values()}),'source_questions_ref':refs[g,o]})
payload={'schema':'actual-first5-TRAIN-formal512-identity-overlap-v1','accepted':True,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Existing accepted local whole first-five cohort extraction and actual baseline512 reconstructed semantic identities; no new Node read, data transfer, QA scoring, model/GPU/HTTP/training, source modification, or outcome-loss join.','source_proofs':proofs,'TRAIN_source_file_SHA256':next(iter(source_shas)),'formal_validation_file_contract_SHA256':'75785f9c9b32c1b35878f6922be57923f431c52d11aa410c380f159d69febd76','source_model_manifest_SHA256':next(iter(source_models)),'same_outer_three_group_qid_semantic_dataset_mappings_exact':True,'distinct_first5_qids_per_group':{g:len(m) for g,m in full.items()},'per_outer_results':results,'whole_first5_question_id_overlap_per_group':{g:len(vals & set(m)) for g,m in full.items()},'identical_four_field_semantic_SHA_overlap_per_group':{g:len(vsem & {x[0] for x in m.values()}) for g,m in full.items()},'question_text_disjoint_independently_verified':False,'limit':'Canonical four-field semantic row SHA includes question_id. Zero qid and full-row SHA intersections are not proof that different qids never share question text. No separate TRAIN question-text hash was emitted, so text duplicates across different IDs remain unknown. Different-cohort loss/outcome pairs are not fabricated.','loss_scope':'Only original selected TRAIN trajectory pre-update fixed-prefix loss exists in this table; formal greedy512 has no token probability/loss logs and no same-qid TRAIN matches in these first-five outers.','actual_no_per_question_loss_formal_EM_join_performed':True,'missing_loss_not_set_to_zero':True}
q=O/'TRAIN-formal512-overlap.private.json';q.write_text(json.dumps(payload,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps({'accepted':True,'output':ref(q),'actual_first5_same_qid_overlap':payload['whole_first5_question_id_overlap_per_group'],'same_semantic_row_overlap':payload['identical_four_field_semantic_SHA_overlap_per_group'],'text_disjoint_not_claimed':True,'shared_three_group_mapping_verified':True},ensure_ascii=False))
