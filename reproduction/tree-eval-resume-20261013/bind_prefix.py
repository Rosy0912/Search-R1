"""Bind externally salvaged ordered scores before any predecessor cancellation."""
import argparse
from collections import Counter
import json
from pathlib import Path
from eval_merge import asset_manifest, atomic_immutable, digest, identities, validate_row
from eval_worker import question_sha


def require(value,message):
    if not value:
        raise RuntimeError(message)


def main():
    p=argparse.ArgumentParser()
    for name in ('raw','receipt','config','model','manifest','output','old-run'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    old=Path(a.old_run).resolve()
    require(old.name=='tree-branpo-fresh200-3360','Wrong predecessor')
    require(Path(a.config).resolve()==old/'resolved.yaml','Wrong old config path')
    require(Path(a.model).resolve()==old/'checkpoints/actor/global_step_200','Wrong trained HF200 path')
    require(digest(a.config)=='c9415c0096853f78d870e205fc956f20045cde985836de5225f609eaf091d4b3','Old config identity changed')
    receipt=json.loads(Path(a.receipt).read_text())
    rows=json.loads(Path(a.raw).read_text())
    require(isinstance(rows,list) and receipt['sha256']==digest(a.raw),'Salvage bytes changed')
    require(receipt['pid']==4011013 and receipt['batch_size']==32
            and receipt['stable_batch_aligned_count']==len(rows)
            and receipt['first_snapshot_count']>=len(rows)
            and receipt['second_snapshot_count']>=len(rows)
            and len(rows)%32==0,'Salvage snapshots/cursor invalid')
    frozen='/instruct300/candidates/branpo-fresh200-8b6f99d1b3353145200eb3fa7c16a579a79a987c/source/repo/'
    require(frozen in receipt['source_file'] and receipt['source_file'].endswith('/verl/trainer/ppo/ray_trainer.py'),'Salvaged frame is not original frozen evaluation')
    for key in ('observer_first','observer_second'):
        observation=receipt[key]
        require(observation and observation['started_rows'] in (len(rows),len(rows)+32),'Observer cursor does not bound saved scores')
    # This actual final validation is an unshuffled, drop_last=False concatenation.
    # RLHFDataset's tokenizer filtering is disabled in the frozen implementation;
    # its input order exactly equals final parquet order and contains all51713.
    import pandas as pd
    manifest=json.loads(Path(a.manifest).read_text())
    require(asset_manifest(a.config,a.model)==manifest,'HF/config/data asset bytes changed')
    frame=pd.concat([pd.read_parquet(x['path']) for x in manifest['dataset_files']],ignore_index=True)
    require(len(frame)==51713 and dict(Counter(frame.data_source))==manifest['expected_source_counts'],'Final source coverage changed')
    totals={i:0 for i in range(8)}
    attempts=sorted((old/'checkpoints/training_state/attempt-results').glob('attempt-*.json'))
    credited=[]
    for path in attempts:
        v=json.loads(path.read_text())
        if v['status']!='COMPLETED_OPPORTUNITIES_OUTER_CREDITED':
            continue
        credited.append(v['accepted_outer'])
        ranks=v['all_rank_result']['ranks']
        require(sorted(r['rank'] for r in ranks)==list(range(8)),'Eight direct rank results required')
        for r in ranks:
            require(r['optimizer_calls']==r['opportunities_processed']==20 and r['opportunities_skipped']==0,'Old actual update counts changed')
            totals[r['rank']]+=r['optimizer_calls']
    require(sorted(credited)==list(range(1,201)) and all(v==4000 for v in totals.values()),'Actual trained200 prerequisite missing')
    for index,row in enumerate(rows):
        validate_row(row)
        require(row['source']==frame.iloc[index]['data_source'],'Saved score positional source mismatch')
        row['index']=index
        row['question_sha256']=question_sha(frame.iloc[index],manifest['prompt_key'])
    value=dict(rows=rows,**identities(manifest),salvage_receipt=receipt,
               original_job=3360,actual_trained_outer=200,actual_adam_per_rank=totals,
               positional_binding='Frozen _validate append order; unshuffled/drop_last=False full parquet; filtering disabled; source compared every row',
               exact_executor_resume=False)
    atomic_immutable(a.output,value)
    atomic_immutable(str(a.output)+'.done.json',dict(sha256=digest(a.output),count=len(rows),
                      original_job=3360,actual_trained_outer=200,identity=identities(manifest)))
    print(json.dumps(dict(bound_prefix=a.output,sha256=digest(a.output),preserved_count=len(rows),remaining_count=51713-len(rows),identity=identities(manifest))))


if __name__=='__main__':
    main()
