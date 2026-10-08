from pathlib import Path
import json,hashlib,os,struct,ast,time
w=Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia');b=w/'work/search-e1-analysis-output/opsd-ablation-execution-20261008';out=b/'goal-continuation-20261009-11/C0-debug-coverage';out.mkdir(exist_ok=False);start=time.time();inventory={};invproof=[]
def meta(p):s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mode=s.st_mode,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def proof(p,raw=None):raw=p.read_bytes() if raw is None else raw;return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
for suffix in ['0-7','8-9']:
 root=b/f'raw-c0-closed{suffix}-direct-20261009';sp=root/'source-full-sha-private.json';lp=root/'local-full-sha-private.json';s=json.loads(sp.read_text());l=json.loads(lp.read_text());assert s['accepted'] and l['accepted'] and l['exact_inventory'];sm={x['relative']:x for x in s['files']}
 for x in l['files']:
  sr=sm[x['relative']];assert sr['sha256']==x['sha256'] and sr['bytes']==x['bytes'];assert sr['path'] not in inventory;inventory[sr['path']]={'source':sr,'local':x,'path':Path(x['path'])}
 invproof+=[proof(sp),proof(lp)]
small=[]
def read_json(src):
 x=inventory[src];p=x['path'];assert meta(p)==x['local']['metadata'];raw=p.read_bytes();assert meta(p)==x['local']['metadata'];digest=hashlib.sha256(raw).hexdigest();assert digest==x['source']['sha256'];small.append(dict(source_path=src,local_path=str(p),bytes=len(raw),sha256=digest));return json.loads(raw)
fact=b/'goal-continuation-20261009-7/ten-outer-committed-facts.stdout.json';facts=json.loads(fact.read_text());outerfacts=sorted([x for x in facts['outers'] if x['variant']=='C0'],key=lambda x:x['outer']);base='/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-C0/runs/outers';teacher=[]
for o in outerfacts:
 n=o['outer'];root=f'{base}/outer-{n:05d}/teacher-diagnostics';sel=read_json(root+'/selection.json');done=read_json(root+'/complete.json');child=read_json(root+'/child-complete.json');assert sel['count']==len(sel['selected'])==done['count']==child['count']<=8;assert done['accepted'] and child['accepted'];assert done['child']['sha256']==inventory[root+'/child-complete.json']['source']['sha256'];assert all(z['actual_wait_returncode']==0 for z in done['exit_proof']['child_exit_evidence']);assert done['exit_proof']['accepted']
 for obj in [sel,done,child]:assert obj['diagnostic_teacher_autoregressive'] is True and obj['original_training_objective'] is False
 assert child['training_or_validation_denominator_contribution']==0 and done['training_or_validation_denominator_contribution']==0;assert child['rng_isolation'].startswith('Independent child');assert len(child['cases'])==sel['count'];caseproofs=[]
 for i,ref in enumerate(child['cases']):
  x=inventory[ref['path']];assert x['source']['sha256']==ref['sha256'] and x['source']['bytes']==ref['bytes'];p=x['path'];assert meta(p)==x['local']['metadata'];d=json.loads(p.read_text());assert meta(p)==x['local']['metadata'];assert d['selected']==sel['selected'][i];assert d['diagnostic_teacher_autoregressive'] and not d['original_training_objective'];assert d['trace']['teacher_forced'] is False and not d['trace']['generation_options_modified'] and not d['trace']['return_objects_modified'];assert not d['trace']['logging_errors'];assert d['outer_index']==n and d['run_id']==facts['outers'][0]['run_id'];assert d['http']['path'] in inventory;assert d['http']['sha256']==inventory[d['http']['path']]['source']['sha256'];caseproofs.append(dict(source_path=ref['path'],local_path=str(p),sha256=ref['sha256'],bytes=ref['bytes'],flags_verified=True,selected_pair_index=d['selected']['pair_index'],stratum=d['selected']['stratum'],http_sha256=d['http']['sha256'],sha_scope='Existing accepted source/local inventory digest; current metadata continuity, not fresh case-content SHA'))
 teacher.append(dict(outer=n,count=sel['count'],available_strata=sel['available_strata'],selected_strata=sel['selected_strata'],missing_strata=sel['missing_strata'],selection_rule=sel['selection_rule'],seed=sel['seed'],not_applicable=sel['not_applicable'],actual_child_wait_codes=[z['actual_wait_returncode'] for z in done['exit_proof']['child_exit_evidence']],cases=caseproofs))
# Only exact committed logs, no failed attempts. Read JSON records for actual top32 coverage;
# reuse previous fresh SHA verification rather than hashing them again.
refs_path=b/'goal-continuation-20261009-8/c0-clipmass-derived/referenced-files-proof.private.json';refs=json.loads(refs_path.read_text());assert refs['accepted'];selected=[x for x in refs['files'] if x['category']=='committed_token_log'];assert len(selected)==3012;top={n:dict(outer=n,committed_pairs=0,valid_rows=0,top32_positions=0,without_top32=0,partial_distribution_rows=0,cap_values=set(),vocabulary_requested_rows=0,vocabulary_saved_rows=0,vocabulary_cap_skip_rows=0) for n in range(10)};dumps=[];distributions=['student','teacher','outer_old','mixture'];seen=set()
for ref in selected:
 x=inventory[ref['source_path']];p=x['path'];assert meta(p)==x['local']['metadata'];assert ref['sha256']==x['source']['sha256'];before=meta(p)
 with p.open() as f:
  h=json.loads(next(f));n=h['outer_index'];s=top[n];s['committed_pairs']+=1;s['cap_values'].add(h['topk_cap']);M=sum(bool(x) for x in h['policy_mask']);positions=set(i for i,m in enumerate(h['policy_mask']) if m);rows=0
  for line in f:
   r=json.loads(line)
   if 'full_vocab_dump' in r:
    s['vocabulary_requested_rows']+=1
    if r['full_vocab_dump']=='cap_skip':s['vocabulary_cap_skip_rows']+=1
    else:
     assert r['full_vocab_dump']=='complete';s['vocabulary_saved_rows']+=1;assert len(r['files'])==4;files=[]
     for fp in r['files']:
      z=inventory[fp['path']];np=z['path'];assert fp['sha256']==z['source']['sha256'] and fp['bytes']==z['source']['bytes'];assert meta(np)==z['local']['metadata'];before_np=meta(np)
      with np.open('rb') as g:
       assert g.read(6)==b'\x93NUMPY';major,minor=g.read(2);ln=struct.unpack('<H' if major==1 else '<I',g.read(2 if major==1 else 4))[0];hdr=ast.literal_eval(g.read(ln).decode('latin1').strip());offset=g.tell()
      assert meta(np)==before_np;assert hdr['descr']=='<f4' and hdr['shape']==(151936,) and not hdr['fortran_order'];assert np.stat().st_size==offset+151936*4;assert fp['shape']==[151936] and fp['dtype']=='float32';files.append(dict(source_path=fp['path'],local_path=str(np),sha256=fp['sha256'],bytes=fp['bytes'],shape=list(hdr['shape']),dtype=hdr['descr'],header_bytes_read=offset,distribution=fp['distribution'],content_sha_scope='Accepted source/local fullSHA, not rehashed in this audit; only NPY header and metadata reread'))
     dumps.append(dict(outer=n,token_index=r['token_index'],pair_index=h['pair_index'],optimizer_update=h['optimizer_update'],files=files,recorded_study_used_bytes=r['study_used_bytes'],scope=r['scope'],probabilities_persisted=r['probabilities_persisted']))
    continue
   rows+=1;s['valid_rows']+=1;assert r['token_index'] in positions
   flags=[all(k in r for k in [dist+'_top32_ids',dist+'_top32_logp']) for dist in distributions]
   if any(flags):
    assert all(flags);s['top32_positions']+=1
    for dist in distributions:assert len(r[dist+'_top32_ids'])==len(r[dist+'_top32_logp'])==32
   else:s['without_top32']+=1
  assert rows==M
 assert meta(p)==before
for n,s in top.items():
 assert s['cap_values']=={8192};assert s['top32_positions']==min(8192,s['valid_rows']);assert s['valid_rows']==outerfacts[n]['counts']['valid_tokens'];s['cap_values']=sorted(s['cap_values']);s['top32_cap_excluded_positions']=s['without_top32'];s['explicit_top32_cap_skip_record']='not present in producer; absence of allfourtop32 fields on policy scalar rows after deterministic cap is counted, not fabricated cap_skip field'
assert len(dumps)==2 and sum(x['count'] for x in teacher)==80
payload=sum(f['bytes'] for d in dumps for f in d['files']);all_npy=[x for x in inventory.values() if x['path'].suffix=='.npy'];assert len(all_npy)==8 and payload==sum(x['source']['bytes'] for x in all_npy)
result=dict(schema='C0-closed-debug-coverage-local-review-v1',accepted=True,started_unix=start,ended_unix=time.time(),run_id='bfc195746b6d41aaa74db5ac4d7a8be7',variant='C0',inputs=dict(inventories=invproof,committed_facts=proof(fact),prior_committed_scalar_sha_proof=proof(refs_path)),teacher_autoregressive=dict(outers=teacher,total_cases=80,max_per_outer=8,flags_verified=True,independent_child_fixedseed42=True,no_training_validation_denominator=True,q_limitation='Diagnostic own generated history != teacher-forced originalD training q; commonBF16 no-adapter source100; success not proof of eachtrainingteacher distribution quality',source_weight_loaded_finite='Existing actual child source_identity/scoring_identity and frozenpublished source proof; no model reload here'),top32=dict(outers=list(top.values()),valid_tokens=sum(s['valid_rows'] for s in top.values()),saved_positions=sum(s['top32_positions'] for s in top.values()),distinct_distribution_vectors_per_position=4,logged_slots_are_policy_positions=True,existing_SHA_scope='Only explicitly committed token references already freshSHA verified in prior scalaraudit; current metadata continuity checked; no second fullhash',not_saved_positions=sum(s['without_top32'] for s in top.values()),row_arrays_verified_shape=True,array_finiteness_revalidation=False),full_vocabulary=dict(requested_tokens=2,saved_tokens=2,cap_skip_tokens=0,saved_files=8,actual_npy_bytes=payload,study_cap_bytes=24*(1<<30),C0_saved_below_cap=True,studywide_allgroups_cap_verification='UNKNOWN until all7complete/sharedledgeraudit; C0subset cannotcertifyfuturewhole-study',records=dumps,preselection='Onlyouter0: eligiblepairminimal SHA256(vocab-v1:42:qid), first2validpositions; source trainer576–583. Loggedpair268/update4 alignsfixedselectedcase, nooutcome-selectedclaim',header_metadata_verified=True,npy_payload_arithmetic_recomputed=False,npy_fullSHA_recomputed=False),small_actual_fresh_SHA_proofs=small,limitations=['Not runtime/model/QA rescoring or RNG replay. No Node/GPU/HTTP/Git/source mutation/tests.', '80caseflags/history/schema inspected, contents not published; accepted case/HTTP SHA provenance reused withmetadata continuity.', 'Fullvocab NPY originalwrite finite flag/source andacceptedSHA retained; thisaudit readsheaders only, notexp/probability/logit arithmetic.', 'Top32 arraysshape only, notfiniteness/value/normalization; cannotreconstructfullKL.', 'C0 only; other6debugcoverage remains pending; allgroups24GiB<=cap not inferred fromC0.'])
(out/'C0-debug-coverage.private.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');note=f'''# C0 已闭合 debug 覆盖（私有）

十个 outer 的独立 teacher 自主诊断共80题，每轮8题；selection/child/parent完成、actual wait rc0、源/配置/选择引用及case flags闭合。明确 diagnostic_teacher_autoregressive=true、original_training_objective=false、training/validation分母贡献0，独立子进程seed42；完整B特权自主历史不等同训练原D history的q，不能据此证明教师每token质量或退化根因。

3012个精确committed token logs：583074有效policy位置，其中每outer8192位置保存四分布top32，共{result['top32']['saved_positions']}位置；其余{result['top32']['not_saved_positions']}位置仅保留全部标量。生产没有显式逐token top32 cap_skip字段，四分布top32整体缺席按确定性cap排除计数，不能虚报字段存在。已核top32向量长度32，不重新验证数值finite或排序。

full-vocab仅outer0预选pair268的前两个validtoken，update4：requested2/saved2/cap_skip0，4分布×2=8NPY，合计{payload}字节（含NPY头），shape151936、FP32小端、非Fortran，头/尺寸与声明一致。完整SHA引用来自已accepted source/local库存，不重hashNPY、不重算概率或KL；C0小于24GiB不证明最终七组共享ledger cap。

本次读取精确committed日志用于实际top32覆盖计数，不扫描failedattempt；复用已验SHA并检查当前metadata连续性。小型selection/child/parent完成票据freshSHA另列；case/HTTP原接受SHA及当前metadata证据保留私有。没有新模型/测试/Node/GPU/HTTP/Git或publisher修改。
''';(out/'notes.zh.md').write_text(note)
manifest=dict(files=[proof(out/'C0-debug-coverage.private.json'),proof(out/'notes.zh.md')]);(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest));print(json.dumps(dict(teacher80=True,top32=result['top32']['saved_positions'],not_saved=result['top32']['not_saved_positions'],npybytes=payload,seconds=time.time()-start)))
