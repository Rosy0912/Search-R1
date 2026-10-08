from pathlib import Path
import json,hashlib
b=Path('work/search-e1-analysis-output/opsd-ablation-execution-20261008');out=b/'goal-continuation-20261009-11/C0-debug-coverage';p=out/'C0-debug-coverage.private.json';d=json.loads(p.read_text());inv={}
for suf in ['0-7','8-9']:
 root=b/f'raw-c0-closed{suf}-direct-20261009';l=json.loads((root/'local-full-sha-private.json').read_text());s=json.loads((root/'source-full-sha-private.json').read_text());sm={x['relative']:x for x in s['files']}
 for x in l['files']:inv[sm[x['relative']]['path']]=(Path(x['path']),sm[x['relative']])
refs=json.loads((b/'goal-continuation-20261009-8/c0-clipmass-derived/referenced-files-proof.private.json').read_text());headers=[]
for r in refs['files']:
 if r['category']=='committed_token_log' and '/outer-00000/' in r['source_path']:
  with inv[r['source_path']][0].open() as f:h=json.loads(next(f))
  headers.append(h)
assert len(headers)==327;chosen=min(headers,key=lambda h:hashlib.sha256(('vocab-v1:42:'+h['identity']['question_id']).encode()).hexdigest());assert chosen['pair_index']==268;assert [i for i,m in enumerate(chosen['policy_mask']) if m][:2]==[0,1];d['full_vocabulary']['actual_preselection_verified_from_committed_headers']=dict(eligible_pairs=327,selected_pair_index=268,selected_qid_digest=hashlib.sha256(chosen['identity']['question_id'].encode()).hexdigest(),first_valid_token_positions=[0,1],selection_digest=hashlib.sha256(('vocab-v1:42:'+chosen['identity']['question_id']).encode()).hexdigest())
for t in d['teacher_autoregressive']['outers']:
 root=f'/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-C0/runs/outers/outer-{t["outer"]:05d}/teacher-diagnostics';done=json.loads(inv[root+'/complete.json'][0].read_text())
 for k,name in [('selection','selection.json'),('child','child-complete.json')]:assert done[k]['sha256']==inv[root+'/'+name][1]['sha256']
 term=done['exit_proof']['actual_terminal'];t['terminal_reference']=dict(path=term['path'],sha256=term['sha256'],bytes=term['bytes'],available_in_two_raw_inventories=term['path'] in inv,scope='Actualwaitrc0/release embedded in freshSHA parent-complete; externalruns/phase-exits terminal file excluded from two rawouter backups, not independently reread here')
d['top32']['deterministic_order_position_selection']='Actualcap8192/fullfourvectors verified; nativeheader cap and sourceorder allocation contract; no fullorder/encodedcache reexecution in this audit';d['limitations'].append('Parent complete embeds actualchildrc0/release. Referenced runs/phase-exits terminal files are outside twoaccepted rawouter backups; independentterminalfile rehash notclaimed.')
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');note=out/'notes.zh.md';txt=note.read_text();txt+='\n补充：逐读outer0全部327个committed header，复算SHA最小eligible qid确为pair268，首两个valid位置0/1；不输出原qid。十个teacher parent完成小票据fresh SHA与child/selection接受库存一致，内嵌actual wait rc0/release；其外部runs/phase-exits终态文件不在这两份rawouter库存中，本次不冒称独立重读外部终态票据。\n';note.write_text(txt)
def pr(p):raw=p.read_bytes();return dict(path=str(p.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
m=dict(files=[pr(p),pr(note)]);(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m))
