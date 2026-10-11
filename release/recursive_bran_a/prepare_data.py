"""Prepare identical A/B splits on HKUST from existing same-cluster raw assets."""
import argparse,json,random,os,hashlib
from pathlib import Path
import pandas as pd
from search_r1.baselines.holdout import (E1_INSTRUCTION,E1_SOURCES,frame_identity,
    ordered_question_ids_hash,file_hash,canonical_question_key,prebuilt_overlap_audit)

def prepare(raw,output,revision):
    output.mkdir(parents=True,exist_ok=False)
    sources=[]
    def load(source,split):
        src=raw/source/(split+'.jsonl');dst=output/'raw'/source/(split+'.jsonl')
        dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
        data=[json.loads(line) for line in src.open()]
        sources.append(dict(source=source,split=split,path=str(dst.relative_to(output)),sha256=file_hash(dst),rows=len(data)))
        return [dict(data_source=source,prompt=[dict(role='user',content=E1_INSTRUCTION+'\nQuestion: '+row['question'])],
            ability='fact-reasoning',reward_model=dict(style='rule',ground_truth=dict(target=row['golden_answers'])),
            extra_info=dict(index=source+':'+split+':'+str(index),split=split)) for index,row in enumerate(data)]
    final=[]
    for source in E1_SOURCES:final+=load(source,'test' if source in ('nq','triviaqa','popqa','bamboogle') else 'dev')
    assert len(final)==51713
    forbidden={canonical_question_key(r['prompt']) for r in final}
    train=[];validation=[];stage=[]
    for source in ('nq','hotpotqa'):
        rows=load(source,'train');rng=random.Random(42);positions=list(range(len(rows)));rng.shuffle(positions)
        seen=set(forbidden);selected=[]
        for i in positions:
            key=canonical_question_key(rows[i]['prompt'])
            if key not in seen:
                selected.append(i);seen.add(key)
            if len(selected)==288:break
        val_ids=set(selected[:256]);stage_ids=set(selected[256:288]);held=set()
        for i in selected:held.add(canonical_question_key(rows[i]['prompt']))
        validation += [rows[i] for i in selected[:256]];stage += [rows[i] for i in selected[256:]]
        train += [r for i,r in enumerate(rows) if canonical_question_key(r['prompt']) not in forbidden|held]
    records={};ids={};questions={}
    for split,rows in [('train',train),('validation',validation),('stage',stage),('final',final)]:
        f=pd.DataFrame(rows);p=output/(split+'.parquet');f.to_parquet(p,index=False);f=pd.read_parquet(p)
        records[split]=dict(path=p.name,sha256=file_hash(p),rows=len(f),source_counts={str(k):int(v) for k,v in f.data_source.value_counts().items()},
            question_ids_sha256=ordered_question_ids_hash(f),content_identity=frame_identity(f))
        ids[split]={r['extra_info']['index'] for r in rows};questions[split]={canonical_question_key(r['prompt']) for r in rows}
    overlap=prebuilt_overlap_audit(ids,questions)
    manifest=dict(version=1,protocol='search-e1-common-300',seed=42,prompt_key='prompt',preparation=dict(
        dataset_id='RUC-NLPIR/FlashRAG_datasets',dataset_revision=revision,selection='source-order-seed42-first256-dev-next32-stage',
        source_order=list(E1_SOURCES),raw_sources=sources,prepare_sha256=file_hash(Path(__file__)),
        core_sha256=file_hash(Path(__file__).parents[2]/'search_r1/baselines/holdout.py')),splits=records,overlaps=overlap)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print(json.dumps({'path':str(output),'manifest_sha256':file_hash(output/'manifest.json'),'rows':{k:v['rows'] for k,v in records.items()}}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--revision',required=True);a=p.parse_args();prepare(a.raw,a.output,a.revision)
