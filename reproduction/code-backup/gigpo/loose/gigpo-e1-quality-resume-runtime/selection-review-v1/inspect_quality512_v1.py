"""Read-only physical/raw-trajectory verification of all eight GPU validations."""
import argparse,gzip,hashlib,json,math,os,re
from pathlib import Path

def require(value,message):
    if not value:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def verify_rows(values,expected_ids):
    require(len(values)==512 and [r['validation_row'] for r in values]==list(range(512)),'Actual validation row coverage/order differs')
    require([r['question_id'] for r in values]==expected_ids,'Actual immutable question order differs')

def inspect(args):
    import pandas as pd
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.trainer.main_ppo import decode_model_response_spans
    from search_r1.baselines.bapo import answer_from_spans,token_f1,legal_action_records
    from search_r1.baselines.evaluation import aggregate_answer_metrics
    from verl.utils.reward_score.qa_em import em_check
    root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
    content='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
    require(sha(root/'common-data/manifest.json')==content,'Common manifest changed')
    manifest=json.loads((root/'common-data/manifest.json').read_text())
    validation=root/'common-data/validation.parquet'
    require(sha(validation)==manifest['splits']['validation']['sha256'],'Actual validation data changed')
    frame=pd.read_parquet(validation);require(len(frame)==512,'Actual validation size differs')
    expected_ids=[str(v['index']) for v in frame.extra_info]
    require(len(set(expected_ids))==512 and frame.data_source.value_counts().to_dict()=={'nq':256,'hotpotqa':256},'Invalid actual question/source coverage')
    require(sha(args.candidates)==args.candidates_sha,'Pinned candidate identities changed')
    scheduler=json.loads(args.scheduler_receipt.read_text())
    tokens=scheduler['slurm'].split()
    require(all(value in tokens for value in ('JobId=3232','JobState=COMPLETED','ExitCode=0:0','NodeList=h100-8','Requeue=0','Restarts=0')),'Missing actual scheduler terminal0 receipt')
    summary=json.loads(args.summary.read_text());require(summary['status']=='ALL_CANDIDATES_GPU512_COMPLETE_UNREVIEWED','Eight GPU evaluations have not all finished')
    results=summary['completed_candidates'];pinned=json.loads(args.candidates.read_text())
    require(len(results)==8 and [r['candidate_step'] for r in results]==[p['step'] for p in pinned]==[5,25,50,75,80,85,100,125],'Missing/repeated candidate')
    tokenizer=AutoTokenizer.from_pretrained(root/'assets/qwen2.5-3b-instruct-aa8e725',local_files_only=True)
    evidence=[]
    for record,candidate in zip(results,pinned):
        run=Path(record['run']);require(run.parent==root/'runs' and 'job3232-' in run.name,'Another GPU job/run')
        physical=json.loads((run/'run-result.json').read_text())
        require(physical=={k:v for k,v in record.items() if k!='run'},'Physical run result differs from summary')
        require(record['status']=='GPU512_EVALUATED' and record['job_id']=='3232' and record['candidate']==candidate['path']
            and record['cleanup_residual_states']=={} and record['final_gpu_snapshot']['processes']==[],'Wrong candidate identity or incomplete own cleanup')
        config=OmegaConf.load(run/'audit/entry-config.yaml')
        require(sha(run/'audit/entry-config.yaml')==record['config_sha256'] and config.trainer.val_only
            and not config.trainer.instruct300.enabled and not config.trainer.resume.enabled
            and config.actor_rollout_ref.model.path==candidate['path']
            and config.actor_rollout_ref.ref.model_path==str(root/'assets/qwen2.5-3b-instruct-aa8e725')
            and config.max_turns==4 and not config.force_final_answer,'Actual loaded evaluation config differs')
        checkpoint=Path(candidate['path']);state=checkpoint/'training_state/manifest.json'
        require(sha(state)==candidate['manifest_sha256'],'Pinned complete state manifest changed')
        frozen=json.loads(state.read_text())
        require(all(frozen[k]==candidate[k] for k in ('job_id','code_commit','config_sha256','data_identity'))
            and frozen['logical_step']==candidate['step'],'Actual original saved state identity differs')
        hf={v['path']:v for v in frozen['files'] if Path(v['path']).parent==checkpoint}
        require(len(hf)==12 and len(record['hf_files'])==12 and {v['path'] for v in record['hf_files']}==set(hf),'HF coverage differs')
        for item in record['hf_files']:
            require(item['path'] in hf and item['sha256']==hf[item['path']]['sha256'] and sha(item['path'])==item['sha256'],'Actual HF bytes differ')
        traces=run/'trajectories';require(json.loads((traces/'complete.json').read_text())==dict(actual_scored_rows=512,training_updates=0),'Missing zero-training complete512 receipt')
        paths=sorted(traces.glob('scored_*.jsonl.gz'))
        require([p.name for p in paths]==[f'scored_{i:04d}.jsonl.gz' for i in range(0,512,32)] and not list(traces.rglob('*.partial')),'Incomplete physical trajectory batches')
        values=[];file_receipts=[]
        for p in paths:
            with gzip.open(p,'rt') as stream:batch=[json.loads(line) for line in stream]
            require(len(batch)==32 and [r['row'] for r in batch]==list(range(32)),'Actual within-batch row coverage differs')
            values.extend(batch);file_receipts.append(dict(path=str(p),sha256=sha(p),rows=len(batch)))
        verify_rows(values,expected_ids);recomputed=[]
        for index,row in enumerate(values):
            n=len(row['responses']);width=n+len(row['prompts'])
            require(len(row['attention_mask'])==width and len(row['info_mask'])==width,'Token/mask widths differ')
            mask=torch.tensor(row['attention_mask'][-n:],dtype=torch.bool)&torch.tensor(row['info_mask'][-n:],dtype=torch.bool)
            spans=decode_model_response_spans(tokenizer,torch.tensor(row['responses']),mask)
            answer=answer_from_spans(spans);targets=frame.iloc[index].reward_model['ground_truth']['target']
            targets=[targets] if isinstance(targets,str) else list(targets)
            metric=dict(source=frame.iloc[index].data_source,f1=max((token_f1(answer,t) for t in targets),default=0.) if answer is not None else 0.,
                em=float(em_check(answer,targets)) if answer is not None else 0.,answer_format=float(answer is not None),
                format_rate=float(answer is not None and legal_action_records([dict(text=s) for s in spans])),generated_tokens=int(mask.sum()),
                search_calls=sum(len(re.findall(r'<search>.*?</search>',s,re.DOTALL)) for s in spans))
            require(metric==row['metrics'],'Actual decoded answer score differs')
            recomputed.append(metric)
        metrics=aggregate_answer_metrics(recomputed)
        require(all(record['metrics'][key]==value for key,value in metrics.items()),'Actual full512 aggregate differs')
        for b in range(16):
            folder=traces/f'batch_{b:03d}';turns=sorted(folder.glob('turn_*.jsonl.gz'))
            require(0<len(turns)<=4 and [p.name for p in turns]==[f'turn_{i:02d}.jsonl.gz' for i in range(1,len(turns)+1)],'Incomplete actual search turns')
            for p in turns:
                with gzip.open(p,'rt') as stream:rows=[json.loads(line) for line in stream]
                require(len(rows)==32 and [r['row'] for r in rows]==list(range(32)),'Incomplete tool trajectory row coverage')
                require(all(isinstance(r.get('model_action'),str) and isinstance(r.get('raw_tool_observation_before_token_limit'),str)
                    and all(type(r[k]) is bool for k in ('active','done','valid_action','is_search')) for r in rows),'Incomplete original action/tool observation')
                file_receipts.append(dict(path=str(p),sha256=sha(p),rows=32))
        actual=json.loads((run/'evaluation-result.json').read_text())
        require(actual['status']=='ACTUAL_GPU512_EVALUATION_COMPLETE' and actual['job']=='3232'
            and actual['source']=='229ee0cf44f754b137dc2226425fbc9fce0b2628'
            and actual['data_identity']==content and actual['config_sha256']==record['config_sha256']
            and actual['hf_directory']==candidate['path'] and actual['frozen_reference']==str(root/'assets/qwen2.5-3b-instruct-aa8e725')
            and actual['metrics']==record['metrics'],'Actual GPU evaluation receipt identity differs')
        for path in (run/'audit').rglob('*.jsonl'):
            with path.open() as stream:
                for line in stream:require(json.loads(line).get('kind') not in ('actor_outer','driver_outer'),'Unexpected actual training update')
        raw=[];seen=set();printed=[]
        for p in run.glob('ray/session_*/logs/worker-*.out'):
            stat=p.stat();identity=(stat.st_dev,stat.st_ino)
            if identity in seen:continue
            seen.add(identity)
            for line in p.read_text(errors='replace').splitlines():
                if 'BASELINE_EVALUATION_METRICS_JSON=' in line:printed.append(json.loads(line.split('BASELINE_EVALUATION_METRICS_JSON=',1)[1]))
            raw.append(dict(path=str(p),device=stat.st_dev,inode=stat.st_ino,bytes=stat.st_size,sha256=sha(p)))
        require(raw and printed==[record['metrics']],'Missing/different authoritative original Ray evaluation metrics')
        outer_printed=[json.loads(line.split('QUALITY_CANDIDATE_EVALUATION_JSON=',1)[1]) for line in (run/'evaluation.log').read_text(errors='replace').splitlines() if 'QUALITY_CANDIDATE_EVALUATION_JSON=' in line]
        require(outer_printed==[actual],'Actual evaluation process log and physical receipt differ')
        evidence.append(dict(step=candidate['step'],run=str(run),metrics=metrics,physical_run_result_sha256=sha(run/'run-result.json'),
            complete_state_manifest_sha256=sha(state),trajectory_files=file_receipts,terminal_raw_workers=raw))
    best=max(results,key=lambda r:(r['metrics']['val/f1/all'],r['candidate_step']))
    require(summary['selected']==best,'Highest F1/latest tie selection differs')
    payload=dict(status='RAW_ALL8_GPU512_PHYSICAL_AND_SCORE_PASS_UNREVIEWED',selection_path=str(args.summary),selection_sha256=sha(args.summary),
        selected_checkpoint=best['candidate'],selected_manifest_sha256=next(p['manifest_sha256'] for p in pinned if p['step']==best['candidate_step']),
        candidate_evidence=evidence,scheduler_receipt_path=str(args.scheduler_receipt),scheduler_receipt_sha256=sha(args.scheduler_receipt),evaluation_job='3232',training_updates=0)
    with args.output.open('x') as stream:json.dump(payload,stream,indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True);p.add_argument('--candidates-sha',required=True);p.add_argument('--scheduler-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    inspect(p.parse_args())
