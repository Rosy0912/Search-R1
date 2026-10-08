import random,shutil,itertools,tempfile,collections,os
base=Path('/dataset/shaoshitong/search-r1-rl/bapo/assets-root/assets')
parent=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-20261004')
parent.mkdir(parents=True,exist_ok=True)
target=parent/'common-data'
if target.exists():
 print(json.dumps({'state':'EXISTING_NO_MUTATION','manifest_sha256':file_hash(target/'manifest.json'),'path':str(target)}),flush=True)
 raise SystemExit(0)
rows_by_source={};raws=[]
for expected in EXPECTED_RAW:
 source,split=expected['source'],expected['split'];path=base/source/(split+'.jsonl')
 assert file_hash(path)==expected['sha256'],path
 raw=[json.loads(line)for line in path.read_text().splitlines()];assert len(raw)==expected['rows']
 rows_by_source[source,split]=[dict(question_id=f'{source}:{split}:{i}',question=r['question'].strip(),answers=r['golden_answers'],data_source=source,raw_id=r['id'])for i,r in enumerate(raw)]
 raws.append(dict(source=source,split=split,path=f'raw/{source}/{split}.jsonl',sha256=expected['sha256'],rows=len(raw)))
splits={k:[]for k in ('train','validation','stage','final')}
for source in E1_SOURCES:
 if source in ('nq','hotpotqa'):
  selected=list(rows_by_source[source,'train']);random.Random(42).shuffle(selected)
  splits['validation']+=selected[:256];splits['stage']+=selected[256:288];splits['train']+=selected[288:]
 final='test'if source in ('nq','triviaqa','popqa','bamboogle')else'dev';splits['final']+=rows_by_source[source,final]
canonical=lambda r:' '.join(r['question'].casefold().rstrip('?').split())
held={k:set(map(canonical,splits[k]))for k in ('validation','stage','final')}
excluded=[];retained=[]
for row in splits['train']:
 reasons=[k for k,v in held.items()if canonical(row)in v]
 if reasons:excluded.append(dict(question_id=row['question_id'],canonical_question=canonical(row),overlap_with=reasons))
 else:retained.append(row)
splits['train']=retained
assert len(splits['validation'])==512 and len(splits['stage'])==64 and len(splits['final'])==51713
sets={k:set(map(canonical,v))for k,v in splits.items()}
intersections={a+'__'+b:len(sets[a]&sets[b])for a,b in itertools.combinations(splits,2)};assert not any(intersections.values()),intersections
staging=Path(tempfile.mkdtemp(prefix='common-data-build-',dir=parent));records={}
for raw in raws:
 dst=staging/raw['path'];dst.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(base/raw['source']/(raw['split']+'.jsonl'),dst);assert file_hash(dst)==raw['sha256']
for name,rows in splits.items():
 packed=[dict(data_source=r['data_source'],prompt=[dict(role='user',content=E1_INSTRUCTION+'\nQuestion: '+r['question'])],ability='fact-reasoning',reward_model=dict(style='rule',ground_truth=dict(target=r['answers'])),extra_info=dict(index=r['question_id'],split=name))for r in rows]
 frame=pd.DataFrame(packed);dst=staging/(name+'.parquet');frame.to_parquet(dst,index=False);reread=pd.read_parquet(dst)
 assert frame_identity(frame)==frame_identity(reread)
 records[name]=dict(path=dst.name,sha256=file_hash(dst),rows=len(reread),source_counts=dict(collections.Counter(reread.data_source)),question_ids_sha256=ordered_question_ids_hash(reread),content_identity=frame_identity(reread),canonical_questions_sha256=canonical_questions_hash(reread))
 with (staging/(name+'.jsonl')).open('w')as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
audit=dict(initial_train_candidates=169039,retained_train_rows=len(retained),excluded_train_rows=excluded,canonical_intersections=intersections,canonical_policy="' '.join(question.casefold().rstrip('?').split())",preserve_complete_final=True)
(staging/'exclusions-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
manifest=dict(version=1,protocol='search-e1-common-300',seed=42,prompt_key='prompt',preparation=dict(dataset_id='RUC-NLPIR/FlashRAG_datasets',dataset_revision='bcafb8dd07d453be3cbeeeb3f78be1841bddf92c',prepare_sha256=PACKAGE_HASHES['prepare'],core_sha256=PACKAGE_HASHES['core'],selection='source-order-seed42-first256-dev-next32-stage',source_order=list(E1_SOURCES),raw_sources=raws),splits=records,exclusion_audit=dict(path='exclusions-audit.json',sha256=file_hash(staging/'exclusions-audit.json')))
(staging/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
os.rename(staging,target)
print(json.dumps(dict(state='CREATED',path=str(target),manifest_sha256=file_hash(target/'manifest.json'),splits=records,exclusions=audit),ensure_ascii=False),flush=True)
