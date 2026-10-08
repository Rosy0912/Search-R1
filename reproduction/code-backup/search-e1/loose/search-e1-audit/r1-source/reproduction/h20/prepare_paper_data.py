"""Immutable FlashRAG inputs; globally unique IDs for mixed GRPO prompts."""
import json
import hashlib
from pathlib import Path
import runpy
import sys
from datasets import Dataset
BASE=Path('/data/search-r1')
sys.path.insert(0,str(BASE/'repo'))
prefix=runpy.run_path(str(BASE/'repo/scripts/data_process/qa_search_train_merge.py'))['make_prefix']
SOURCES=['nq','triviaqa','popqa','hotpotqa','2wikimultihopqa','musique','bamboogle']

def diverse_validation(validation):
    """64 held-out questions: seven sources and four length strata per source."""
    selected=[]
    for source in SOURCES:
        pool=[r for r in validation if r['data_source']==source]
        pool.sort(key=lambda r:(len(r['prompt'][0]['content']),r['extra_info']['source_index']))
        quota=10 if source=='nq' else 9
        for stratum in range(4):
            bucket=pool[len(pool)*stratum//4:len(pool)*(stratum+1)//4]
            bucket.sort(key=lambda r:hashlib.sha256(('search-r1-64-v1:'+r['prompt'][0]['content']).encode()).hexdigest())
            take=quota//4+(stratum<quota%4)
            selected.extend(bucket[:take])
    assert len(selected)==64
    assert len({r['extra_info']['index'] for r in selected})==64
    assert len({r['prompt'][0]['content'] for r in selected})==64
    return selected

def rows(sources, training):
    output=[]
    for source in sources:
        directory=BASE/'assets'/source
        split='train' if training else ('test' if (directory/'test.jsonl').exists() else 'dev')
        with (directory/(split+'.jsonl')).open() as f:
            for original_id,line in enumerate(f):
                example=json.loads(line)
                q=example['question'].strip()
                assert q, 'Empty question'
                example['question']=q if q.endswith('?') else q+'?'
                output.append(dict(data_source=source,prompt=[dict(role='user',content=prefix(example,'base'))],ability='fact-reasoning',reward_model=dict(style='rule',ground_truth=dict(target=example['golden_answers'])),extra_info=dict(split=split,index=len(output),source_index=original_id)))
    assert len({r['extra_info']['index'] for r in output})==len(output)
    return output

if __name__=='__main__':
    out=BASE/'data/paper-grpo'
    out.mkdir(parents=True,exist_ok=True)
    train=rows(['nq','hotpotqa'],True)
    validation=rows(SOURCES,False)
    for name,data in [('train',train),('test',validation)]:
        Dataset.from_list(data).to_parquet(str(out/(name+'.parquet')))
    # Balanced mixed-data CPU/GPU integration acceptance, not a benchmark result.
    mixed=train[:16]+[r for r in train if r['data_source']=='hotpotqa'][:16]
    assert len(mixed)==32 and len({r['extra_info']['index'] for r in mixed})==32
    Dataset.from_list(mixed).to_parquet(str(out/'smoke-train.parquet'))
    tail=validation[:9]
    Dataset.from_list(tail).to_parquet(str(out/'smoke-test.parquet'))
    quick=diverse_validation(validation)
    Dataset.from_list(quick).to_parquet(str(out/'validation-64.parquet'))
    quick_report={'sample_count':64,'method':'seven sources, four prompt-length strata per source; deterministic SHA256 order within strata',
                  'counts':{s:sum(r['data_source']==s for r in quick) for s in SOURCES},
                  'sample_ids':[r['extra_info']['index'] for r in quick]}
    (out/'validation-64-audit.json').write_text(json.dumps(quick_report,indent=2))
    report={'train_counts':{s:sum(r['data_source']==s for r in train) for s in ['nq','hotpotqa']},'eval_counts':{s:sum(r['data_source']==s for r in validation) for s in SOURCES},'unique_train_ids':len(train),'unique_eval_ids':len(validation),'eval_split':{s:next(r['extra_info']['split'] for r in validation if r['data_source']==s) for s in SOURCES}}
    (out/'dataset-audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
